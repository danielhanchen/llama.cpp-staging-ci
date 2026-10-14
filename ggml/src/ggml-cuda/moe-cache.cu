// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#include "moe-cache.cuh"

// device-side lookup of the MoE expert cache, see ggml_moe_cache_lookup and llama-moe-cache.cpp
//
// state: [0] CLOCK hand, [1] epoch (one per lookup), [2..3] hits, [4..5] misses, [6] filled slots, [7] unused, then key_of, ref and pin [n_slots]
// key_of: il*n_expert + expert (-1 if empty); ref: CLOCK reference bits; pin: epoch of the last lookup using the slot (that lookup does not evict it)
// dst:   slots[n_ids], then a row starting with the number of copied experts, then their experts[n_ids] and slots[n_ids]
//
// assign result rows (ne0 = n_expert_used, T tokens): slots [T], fill count [1], fill experts [T], fill slots [T], host ids [T], hit slots [T]
// hybrid (ggml_moe_cache_assign): counts [n_expert, n_layer + 1] use counters, last row: [0..1] experts left to the host, [2] fetch accumulator (1/65536)
// only a share of the missing experts is admitted (fetch_q/65536, error diffused), the ones with the highest counters, ties by expert id
// (all of them while the cache has empty slots: filling an empty slot evicts nothing)
// the others keep slot -1 and are written to host_ids, after the slots and the copy list

#define MOE_CACHE_HDR   8
#define MOE_CACHE_LANES 32

// the lookup runs in a single block of 32 threads, so a block barrier is a barrier of the warp (also with 64-wide wavefronts)
static __device__ __forceinline__ unsigned moe_cache_ballot(const bool p) {
#if defined(GGML_USE_HIP)
    return (unsigned) __ballot(p);
#else
    return __ballot_sync(0xffffffffu, p);
#endif
}

// ids out of range (never produced by the router) are read as expert 0
static __device__ __forceinline__ int32_t moe_cache_id(const int32_t * ids, const int i, const int n_expert) {
    const int32_t e = ids[i];
    return (uint32_t) e < (uint32_t) n_expert ? e : 0;
}

static __global__ void moe_cache_assign(
        const int32_t * __restrict__ ids, int32_t * __restrict__ map, int32_t * __restrict__ state, int32_t * __restrict__ dst,
        const int n_ids, const int ne0, const int n_expert, const int il, const int n_slots,
        int32_t * __restrict__ counts, const int n_layer, const int fetch_q) {
    extern __shared__ int32_t smem[];
    int32_t * miss_e = smem;         // [n_ids] missing experts, in the order of their first use
    int32_t * victim = smem + n_ids; // [n_ids] slots given to them
    int32_t * take_e = miss_e;       // the missing experts that are admitted (hybrid: [n_ids] after victim)

    const int      lane = threadIdx.x;
    const unsigned full = 0xffffffffu;

    int32_t * key_of = state + MOE_CACHE_HDR;
    int32_t * ref    = key_of + n_slots;
    int32_t * pin    = ref + n_slots;
    int32_t * row    = map + (int64_t) il*n_expert;

    int32_t * slots  = dst;
    int32_t * n_fill = dst + n_ids;
    int32_t * fill_e = dst + n_ids + ne0;
    int32_t * fill_s = fill_e + n_ids;
    int32_t * cnt    = counts != nullptr ? counts + (int64_t) il*n_expert : nullptr;

    uint32_t epoch = (uint32_t) state[1] + 1;
    if (epoch == 0) {
        // wrapped around: no slot may look pinned by this lookup
        for (int s = lane; s < n_slots; s += MOE_CACHE_LANES) {
            pin[s] = 0;
        }
        epoch = 1;
    }
    __syncthreads();
    if (lane == 0) {
        state[1] = (int32_t) epoch;
    }

    // hits: set the reference bit and pin the slot for this lookup
    int n_hit = 0;
    for (int i = lane; i < n_ids; i += MOE_CACHE_LANES) {
        const int32_t s = row[moe_cache_id(ids, i, n_expert)];
        slots[i] = s;
        if (counts != nullptr) {
            fill_s[2*n_ids + i] = s; // hit slots: the slot of an expert cached before this lookup, else -1
        }
        if (s >= 0) {
            ref[s] = 1;
            if ((uint32_t) atomicExch(&pin[s], (int32_t) epoch) != epoch) {
                n_hit++;
                if (cnt != nullptr) {
                    const int32_t e = moe_cache_id(ids, i, n_expert);
                    cnt[e] += cnt[e] < 0x7fffffff;
                }
            }
        }
    }
    __syncthreads();

    // misses: the first use of each missing expert claims it, in the order of the ids
    int n_miss = 0;
    for (int i0 = 0; i0 < n_ids; i0 += MOE_CACHE_LANES) {
        const int i = i0 + lane;
        const int32_t e = i < n_ids && slots[i] < 0 ? moe_cache_id(ids, i, n_expert) : -1;
        bool first = e >= 0;
        for (int k = 0; k < MOE_CACHE_LANES; ++k) {
            const int32_t ek = __shfl_sync(full, e, k, MOE_CACHE_LANES);
            if (k < lane && ek == e) {
                first = false;
            }
        }
        // a previous group of ids may have claimed it already
        first = first && row[e] == -1;
        const unsigned m = moe_cache_ballot(first);
        if (first) {
            row[e] = -2;
            miss_e[n_miss + __popc(m & ((1u << lane) - 1))] = e;
            if (cnt != nullptr) {
                cnt[e] += cnt[e] < 0x7fffffff;
            }
        }
        n_miss += __popc(m);
        __syncthreads();
    }

    // hybrid: admit the n_take most used missing experts, in the order of their first use
    int n_take = n_miss;
    int32_t * stat = counts != nullptr ? counts + (int64_t) n_layer*n_expert : nullptr;
    if (counts != nullptr) {
        take_e = victim + n_ids;
        const int64_t acc = (int64_t) (uint32_t) stat[2] + (int64_t) fetch_q*n_miss;
        n_take = max((int) (acc >> 16), min(n_miss, n_slots - state[6]));
        int n_sel = 0;
        for (int j0 = 0; j0 < n_miss; j0 += MOE_CACHE_LANES) {
            const int j = j0 + lane;
            bool take = false;
            if (j < n_miss) {
                const int32_t ej = miss_e[j];
                const int32_t cj = cnt[ej];
                int rank = 0;
                for (int i = 0; i < n_miss && rank < n_take; ++i) {
                    const int32_t ei = miss_e[i];
                    const int32_t ci = cnt[ei];
                    rank += ci > cj || (ci == cj && ei < ej);
                }
                take = rank < n_take;
                if (!take) {
                    row[ej] = -1; // left to the host, not admitted
                }
            }
            const unsigned m = moe_cache_ballot(take);
            if (take) {
                take_e[n_sel + __popc(m & ((1u << lane) - 1))] = miss_e[j];
            }
            n_sel += __popc(m);
        }
        __syncthreads();
        if (lane == 0) {
            stat[2] = (int32_t) (acc & 0xffff);
            *(unsigned long long *) stat += (unsigned long long) (n_miss - n_take);
        }
    }

    // victims: CLOCK sweep, 32 slots at a time, the slots used by this lookup are skipped
    // there are enough unpinned slots because the number of distinct ids is at most the number of slots
    int h = state[0];
    const int width = min(MOE_CACHE_LANES, n_slots);
    int found = 0;
    while (found < n_take) {
        const bool active = lane < width;
        const int  s      = active ? (h + lane < n_slots ? h + lane : h + lane - n_slots) : 0;
        const bool pinned = active && (uint32_t) pin[s] == epoch;
        const bool r      = active && ref[s] != 0;
        const bool elig   = active && !pinned && !r;

        const unsigned m    = moe_cache_ballot(elig);
        const int      need = n_take - found;
        int      adv  = width;
        unsigned take = m;
        if (__popc(m) >= need) {
            unsigned mm = m;
            for (int k = 0; k < need - 1; ++k) {
                mm &= mm - 1;
            }
            const int last = __ffs(mm) - 1;
            adv  = last + 1;
            take = last == 31 ? m : m & ((2u << last) - 1);
        }
        if (lane < adv && r && !pinned) {
            ref[s] = 0;
        }
        if (take & (1u << lane)) {
            victim[found + __popc(take & ((1u << lane) - 1))] = s;
            pin[s] = (int32_t) epoch;
        }
        found += __popc(take);
        h += adv;
        if (h >= n_slots) {
            h -= n_slots;
        }
        __syncthreads();
    }

    // evict and insert
    int n_new = 0; // empty slots filled
    for (int j = lane; j < n_take; j += MOE_CACHE_LANES) {
        const int32_t s   = victim[j];
        const int32_t e   = take_e[j];
        const int32_t old = key_of[s];
        if (old >= 0) {
            map[old] = -1;
        } else {
            n_new++;
        }
        key_of[s] = il*n_expert + e;
        ref[s]    = 1;
        row[e]    = s;
        fill_e[j] = e;
        fill_s[j] = s;
    }
    __syncthreads();

    for (int i = lane; i < n_ids; i += MOE_CACHE_LANES) {
        if (slots[i] < 0) {
            slots[i] = row[moe_cache_id(ids, i, n_expert)];
        }
        if (counts != nullptr) {
            // after the copy list: the experts left to the host
            fill_s[n_ids + i] = slots[i] < 0 ? moe_cache_id(ids, i, n_expert) : -1;
        }
    }

    for (int off = MOE_CACHE_LANES/2; off > 0; off >>= 1) {
        n_hit += __shfl_xor_sync(full, n_hit, off, MOE_CACHE_LANES);
        n_new += __shfl_xor_sync(full, n_new, off, MOE_CACHE_LANES);
    }
    if (lane == 0) {
        state[6] += n_new;
        state[0] = h;
        *n_fill  = n_take;
        *(unsigned long long *) (state + 2) += (unsigned long long) n_hit;
        *(unsigned long long *) (state + 4) += (unsigned long long) n_take;
    }
}

struct moe_cache_fetch_args {
    const char * src[4];
    char       * dst[4];
    int64_t      nb[4];
};

// copy the missing experts from (mapped) host memory to their slots, block z copies the chunks z, z + gridDim.z, ...
static __global__ void moe_cache_fetch(const int32_t * __restrict__ info, const int n_ids, const int ne0, const moe_cache_fetch_args args, const int64_t chunk) {
    const int32_t * n_fill = info + n_ids;
    const int32_t * fill_e = info + n_ids + ne0;
    const int32_t * fill_s = fill_e + n_ids;

    const int j = blockIdx.x;
    if (j >= *n_fill) {
        return;
    }
    const int     b   = blockIdx.y;
    const int64_t nb  = args.nb[b];

    const char * src = args.src[b] + fill_e[j]*nb;
    char       * dst = args.dst[b] + fill_s[j]*nb;

    for (int64_t beg = blockIdx.z*chunk; beg < nb; beg += gridDim.z*chunk) {
        const int64_t end = min(beg + chunk, nb);

        if ((((uintptr_t) (src + beg)) | ((uintptr_t) (dst + beg)) | (uintptr_t) (end - beg)) % 16 == 0) {
            const int4 * s4 = (const int4 *) (src + beg);
            int4       * d4 = (int4 *) (dst + beg);
            const int64_t n = (end - beg)/16;
            constexpr int unroll = 4;
            int64_t i = threadIdx.x;
            for (; i + (unroll - 1)*blockDim.x < n; i += unroll*blockDim.x) {
                int4 v[unroll];
#pragma unroll
                for (int u = 0; u < unroll; ++u) {
                    v[u] = s4[i + u*blockDim.x];
                }
#pragma unroll
                for (int u = 0; u < unroll; ++u) {
                    d4[i + u*blockDim.x] = v[u];
                }
            }
            for (; i < n; i += blockDim.x) {
                d4[i] = s4[i];
            }
        } else {
            for (int64_t i = beg + threadIdx.x; i < end; i += blockDim.x) {
                dst[i] = src[i];
            }
        }
    }
}

// the slots admitted by an assign (res): its slot where the expert was not cached before (hit slot -1), else -1
static __global__ void moe_cache_admitted(const int32_t * __restrict__ res, int32_t * __restrict__ dst, const int n_ids, const int ne0) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_ids) {
        return;
    }
    const int32_t * hit = res + 4*n_ids + ne0; // see the assign result layout
    dst[i] = hit[i] < 0 ? res[i] : -1;
}

bool ggml_cuda_moe_cache_lookup_supported(const ggml_tensor * op) {
    const int mode    = ggml_get_op_params_i32(op, 10);
    const int n_banks = ggml_get_op_params_i32(op, 1);
    if (mode == 1) {
        return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 && op->src[2]->type == GGML_TYPE_I32 &&
            op->src[3]->type == GGML_TYPE_I32 && ggml_nelements(op->src[0]) <= 4096;
    }
    if (mode == 2) {
        return n_banks >= 1 && n_banks <= 4 && op->src[0]->type == GGML_TYPE_I32;
    }
    if (mode == 3) {
        return op->src[0]->type == GGML_TYPE_I32;
    }
    if (mode != 0 || n_banks < 1 || n_banks > 4) {
        return false;
    }
    const int64_t n_ids = ggml_nelements(op->src[0]);
    return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 && op->src[2]->type == GGML_TYPE_I32 &&
        n_ids <= 4096;
}

// max_z > 0: at most max_z blocks per expert and bank (each copies several chunks), so that an overlapped fetch, which mostly waits
// for host memory, leaves the SMs to the kernels that run beside it
static void moe_cache_fetch_launch(cudaStream_t stream, const ggml_tensor * dst, const ggml_tensor * const * banks, const int n_banks, const int32_t * info, const int n_ids, const int ne0,
        const int max_z = 0) {
    moe_cache_fetch_args args = {};
    int64_t max_nb = 0;
    for (int b = 0; b < n_banks; ++b) {
        const ggml_tensor * bank = banks[b];
        memcpy(&args.src[b], (const int32_t *) dst->op_params + 2 + 2*b, sizeof(void *));
        args.dst[b] = (char *) bank->data;
        args.nb[b]  = bank->nb[2];
        max_nb = std::max(max_nb, (int64_t) bank->nb[2]);
    }

    const int64_t chunk = 64*1024;
    int64_t n_z = (max_nb + chunk - 1)/chunk;
    if (max_z > 0) {
        n_z = std::min<int64_t>(n_z, max_z);
    }
    const dim3 grid(n_ids, n_banks, (unsigned) std::max<int64_t>(n_z, 1));
    moe_cache_fetch<<<grid, 256, 0, stream>>>(info, n_ids, ne0, args, chunk);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 10);

    if (mode == 2) {
        // fetch the experts admitted by the assign of src[0], the result is its slots
        const ggml_tensor * res = dst->src[0];
        const int ne0   = (int) res->ne[0];
        const int n_ids = (int) ggml_nelements(dst);
        GGML_ASSERT(ggml_is_contiguous(res) && ggml_is_contiguous(dst));
        if (n_ids == 0) {
            return;
        }
        cudaStream_t stream = ctx.stream();
        if (ggml_get_op_params_i32(dst, 12) == 0) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, res->data, n_ids*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
            moe_cache_fetch_launch(stream, dst, dst->src + 1, ggml_get_op_params_i32(dst, 1), (const int32_t *) res->data, n_ids, ne0);
            return;
        }
        // overlap: the copies run on a side stream after the assign, ggml_moe_cache_join makes the main stream wait for them;
        // the hits only read slots cached before this lookup, which the copies (into its victims) never touch
        if (ctx.moe_fork == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.moe_fork, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.moe_join, cudaEventDisableTiming));
        }
        cudaStream_t side = ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1);
        CUDA_CHECK(cudaEventRecord(ctx.moe_fork, stream));
        CUDA_CHECK(cudaStreamWaitEvent(side, ctx.moe_fork, 0));
        // GGML_CUDA_MOE_FETCH_OVERLAP_Z: blocks per expert and bank of the overlapped copies (0 = one per 64 KiB chunk)
        static const int max_z = [] {
            const char * env = getenv("GGML_CUDA_MOE_FETCH_OVERLAP_Z");
            return env != nullptr ? std::max(0, atoi(env)) : 0;
        }();
        moe_cache_fetch_launch(side, dst, dst->src + 1, ggml_get_op_params_i32(dst, 1), (const int32_t *) res->data, n_ids, ne0, max_z);
        moe_cache_admitted<<<(n_ids + 255)/256, 256, 0, side>>>((const int32_t *) res->data, (int32_t *) dst->data, n_ids, ne0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ctx.moe_join, side));
        ctx.moe_join_pending = true;
        return;
    }

    if (mode == 3) {
        // join of an overlapped fetch: the main stream waits for its copies, the result is its admitted slots
        const ggml_tensor * a = dst->src[0];
        if (ggml_nelements(dst) == 0) {
            return; // nothing was fetched (see mode 2)
        }
        GGML_ASSERT(ctx.moe_join != nullptr && ggml_is_contiguous(a) && ggml_is_contiguous(dst));
        cudaStream_t stream = ctx.stream();
        CUDA_CHECK(cudaStreamWaitEvent(stream, ctx.moe_join, 0));
        ctx.moe_join_pending = false;
        CUDA_CHECK(cudaMemcpyAsync(dst->data, a->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const ggml_tensor * ids   = dst->src[0];
    ggml_tensor       * map   = dst->src[1];
    ggml_tensor       * state = dst->src[2];

    const int il      = ggml_get_op_params_i32(dst, 0);
    const int n_banks = ggml_get_op_params_i32(dst, 1);

    const int n_ids    = (int) ggml_nelements(ids);
    const int ne0      = (int) ids->ne[0];
    const int n_expert = (int) map->ne[0];
    const int n_slots  = (int) ((ggml_nelements(state) - MOE_CACHE_HDR)/3);

    GGML_ASSERT(ggml_is_contiguous(ids) && ggml_is_contiguous(dst));

    if (n_ids == 0) {
        return; // for example the last layer of a ubatch without outputs
    }

    cudaStream_t stream = ctx.stream();

    if (mode == 1) {
        // hybrid assign: no copies here, see ggml_moe_cache_fetch
        ggml_tensor * counts = dst->src[3];
        float fetch;
        memcpy(&fetch, (const int32_t *) dst->op_params + 11, sizeof(float));
        const int fetch_q = (int) lrintf(fetch*65536.0f);
        GGML_ASSERT(ggml_is_contiguous(counts) && fetch_q >= 0 && fetch_q <= 65536);
        moe_cache_assign<<<1, MOE_CACHE_LANES, 3*n_ids*sizeof(int32_t), stream>>>(
            (const int32_t *) ids->data, (int32_t *) map->data, (int32_t *) state->data, (int32_t *) dst->data,
            n_ids, ne0, n_expert, il, n_slots, (int32_t *) counts->data, (int) map->ne[1], fetch_q);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    moe_cache_assign<<<1, MOE_CACHE_LANES, 2*n_ids*sizeof(int32_t), stream>>>(
        (const int32_t *) ids->data, (int32_t *) map->data, (int32_t *) state->data, (int32_t *) dst->data,
        n_ids, ne0, n_expert, il, n_slots, nullptr, 0, 0);
    CUDA_CHECK(cudaGetLastError());

    moe_cache_fetch_launch(stream, dst, dst->src + 3, n_banks, (const int32_t *) dst->data, n_ids, ne0);
}
