// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"
#include "preload.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "kv-sel.cuh"

#include <atomic>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1       ? flash_attn_mask_to_sparse_indices<1, false> :
                        n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>  :
                                             flash_attn_mask_to_sparse_indices<8, false>;
    ggml_cuda_kernel_launch_pl(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// QSA maskless: the lists of flash_attn_mask_to_sparse_indices built from the selected cells themselves. A query row
// keeps cell c if a slot of its token selects c (idx < n_kv, a dead slot points at a dump row >= n_kv) and its causal mask is
// finite there, which is the set of finite entries of the dense mask (-inf, 0 at the selected cells, + kq_mask). A bitmap over
// the cells makes the lists ascending and duplicate free, so they equal the compacted ones element by element
#define FATTN_SEL_THREADS 1024
#define FATTN_SEL_UNROLL  4
static_assert(FATTN_SEL_THREADS/32 == 32, "ggml_cuda_qsa_maskless_gather_ok counts 32 warp sums");
template <int ncols1>
__launch_bounds__(FATTN_SEL_THREADS, 1)
static __global__ void flash_attn_sel_to_sparse_indices(
        const int32_t * idx_ptr, const half * kq_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int n_kv, const int n_sel,
        const int n_queries, const int n_kv_max, const int64_t s_idx, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    extern __shared__ uint32_t bits[];
    __shared__ int warp_sums[FATTN_SEL_THREADS/WARP_SIZE];

    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;
    const int nwords   = (n_kv + 31) / 32;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    for (int w = tid; w < nwords; w += blockDim.x) {
        bits[w] = 0;
    }
    __syncthreads();

    // the loads of a batch of slots are independent: issue them all before the dependent mask reads and the bitmap updates
    for (int q = q0; q < q1; ++q) {
        const int32_t * idx = idx_ptr + (int64_t(sequence)*n_queries + q)*s_idx;
        const half    * kq  = kq_ptr  + sequence*s33 + q*s31;
        for (int s0 = 0; s0 < n_sel; s0 += FATTN_SEL_UNROLL*FATTN_SEL_THREADS) {
            int32_t c[FATTN_SEL_UNROLL];
#pragma unroll
            for (int u = 0; u < FATTN_SEL_UNROLL; ++u) {
                const int s = s0 + u*FATTN_SEL_THREADS + tid;
                c[u] = s < n_sel ? idx[s] : -1;
                c[u] = c[u] < n_kv ? c[u] : -1;
            }
            bool live[FATTN_SEL_UNROLL];
#pragma unroll
            for (int u = 0; u < FATTN_SEL_UNROLL; ++u) {
                live[u] = c[u] >= 0 && isfinite(__half2float(kq[max(c[u], 0)]));
            }
#pragma unroll
            for (int u = 0; u < FATTN_SEL_UNROLL; ++u) {
                if (live[u]) {
                    atomicOr(&bits[c[u] / 32], uint32_t(1) << (c[u] % 32));
                }
            }
        }
    }
    __syncthreads();

    // each thread owns a contiguous run of words, an exclusive scan of the run counts orders the writes
    const int wpt = (nwords + blockDim.x - 1) / blockDim.x;
    const int w0  = min(tid*wpt, nwords);
    const int w1  = min(w0 + wpt, nwords);
    int count = 0;
    for (int w = w0; w < w1; ++w) {
        count += __popc(bits[w]);
    }

    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    int incl = count;
#pragma unroll
    for (int off = 1; off < WARP_SIZE; off *= 2) {
        const int v = __shfl_up_sync(0xFFFFFFFF, incl, off, WARP_SIZE);
        if (lane >= off) {
            incl += v;
        }
    }
    if (lane == WARP_SIZE - 1) {
        warp_sums[warp] = incl;
    }
    __syncthreads();
    int offset = incl - count;
    int total  = 0;
#pragma unroll
    for (int iw = 0; iw < FATTN_SEL_THREADS/WARP_SIZE; ++iw) {
        const int v = warp_sums[iw];
        offset += iw < warp ? v : 0;
        total  += v;
    }

    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;
    for (int w = w0; w < w1 && offset < n_kv_max; ++w) {
        uint32_t b = bits[w];
        while (b != 0 && offset < n_kv_max) {
            const int bit = __ffs(b) - 1;
            b &= b - 1;
            indices[offset++] = w*32 + bit;
        }
    }

    const int n = min(total, n_kv_max);
    for (int i = n + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = n;
    }
    __syncthreads();

    ggml_cuda_pdl_lc();
}

// the dense mask when the kernel does not gather: -inf, then the causal mask at the selected cells, as fill(-inf) + set_rows(0)
// + add(kq_mask) computes it. One block per (token row, FATTN_SEL_MASK_CHUNK cells): a block writes only its own cells
#define FATTN_SEL_MASK_CHUNK 1024
__launch_bounds__(256, 1)
static __global__ void flash_attn_sel_to_mask(
        const int32_t * idx_ptr, const half * kq_ptr, half * mask_ptr, const int n_kv, const int n_sel, const int n_queries,
        const int64_t s_idx, const int64_t s31, const int64_t s33, const int chunk) {
    ggml_cuda_pdl_sync();

    const int q        = blockIdx.y;
    const int sequence = blockIdx.z;
    const int c0       = blockIdx.x*chunk;
    const int c1       = min(c0 + chunk, n_kv);

    const int32_t * idx  = idx_ptr  + (int64_t(sequence)*n_queries + q)*s_idx;
    const half    * kq   = kq_ptr   + sequence*s33 + q*s31;
    half          * mask = mask_ptr + sequence*s33 + q*s31;

    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x) {
        mask[c] = half(-INFINITY);
    }
    __syncthreads();
    for (int s = threadIdx.x; s < n_sel; s += blockDim.x) {
        const int32_t c = idx[s];
        if (c >= c0 && c < c1) {
            mask[c] = __float2half(0.0f + __half2float(kq[c]));
        }
    }
    __syncthreads();

    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

int ggml_cuda_qsa_maskless_mode() {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return 0;
#else
    static const int mode = getenv("GGML_CUDA_QSA_MASKLESS") ? atoi(getenv("GGML_CUDA_QSA_MASKLESS")) : 1;
    return mode;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

bool ggml_cuda_qsa_maskless_gather_ok(const ggml_cuda_qsa_maskless & qsa) {
    // the bitmap (dynamic shared memory) and the list kernel's static warp sums (1024 threads: 32 ints) within the default 48 KiB
    // per block (no opt-in above it)
    return size_t(qsa.kq->ne[0] + 31)/32*sizeof(uint32_t) + 32*sizeof(int) <= 48*1024;
}

void ggml_cuda_qsa_maskless_compact(
        const ggml_cuda_qsa_maskless & qsa, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max,
        cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(qsa, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const ggml_tensor * idx = qsa.idx;
    const ggml_tensor * kq  = qsa.kq;
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    GGML_ASSERT(n_queries == kq->ne[1]);
    const int n_kv = kq->ne[0];
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, kq->ne[3], 1);
    const dim3 block_dim(FATTN_SEL_THREADS, 1, 1);
    const size_t nbytes_shared = size_t(n_kv + 31)/32*sizeof(uint32_t);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, nbytes_shared, stream);
    const auto kernel = ncols1 == 1 ? flash_attn_sel_to_sparse_indices<1> : flash_attn_sel_to_sparse_indices<8>;
    ggml_cuda_kernel_launch_pl(kernel, launch_params,
        (const int32_t *) idx->data, (const half *) kq->data, indices, counts, n_kv, int(idx->ne[0]), n_queries, n_kv_max,
        int64_t(idx->nb[1]/sizeof(int32_t)), int64_t(kq->nb[1]/sizeof(half)), int64_t(kq->nb[3]/sizeof(half)));
    CUDA_CHECK(cudaGetLastError());
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

void ggml_cuda_qsa_maskless_materialize(const ggml_cuda_qsa_maskless & qsa, half * mask, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(qsa, mask, stream);
    GGML_ABORT("QSA maskless is only planned on NVIDIA CUDA");
#else
    const ggml_tensor * idx = qsa.idx;
    const ggml_tensor * kq  = qsa.kq;
    // GGML_CUDA_QSA_MASK_SPLIT=0: one block per token row (the cell chunk is the whole row)
    static const bool split = !getenv("GGML_CUDA_QSA_MASK_SPLIT") || atoi(getenv("GGML_CUDA_QSA_MASK_SPLIT")) != 0;
    const int chunk = split ? FATTN_SEL_MASK_CHUNK : int(kq->ne[0]);
    const dim3 blocks_num((kq->ne[0] + chunk - 1)/chunk, kq->ne[1], kq->ne[3]);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch_pl(flash_attn_sel_to_mask, launch_params,
        (const int32_t *) idx->data, (const half *) kq->data, mask, int(kq->ne[0]), int(idx->ne[0]), int(kq->ne[1]),
        int64_t(idx->nb[1]/sizeof(int32_t)), int64_t(kq->nb[1]/sizeof(half)), int64_t(kq->nb[3]/sizeof(half)), chunk);
    CUDA_CHECK(cudaGetLastError());
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    // (a verify window wider than the reference picks as the reference window does, ggml_cuda_fattn_verify_ref_rows)
    const int     ref_rows = ggml_cuda_fattn_verify_ref_rows(dst);
    const int64_t n_q      = ref_rows > 0 ? ref_rows : Q->ne[1];
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(n_q, 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    // diagnostic: GGML_CUDA_FATTN_SPARSE_MIN_KV=N replaces the K length threshold below (unset: the rule as is)
    static const int64_t min_kv_env = getenv("GGML_CUDA_FATTN_SPARSE_MIN_KV") ? atoll(getenv("GGML_CUDA_FATTN_SPARSE_MIN_KV")) : -1;
    // op_params[5] == 1 (set by the model on prompt attention whose rows see few of the K rows, DeepSeek-V4.1): the
    // gather from K >= 2 * n_kv_max per query, since every query tile of a prompt reads the whole K otherwise
    const bool    prefer   = dst->op_params[5] == 1;
    const int64_t min_kv = min_kv_env >= 0 ? min_kv_env : prefer ? 2*(int64_t) n_kv_max : std::max<int64_t>(4096, 2*n_gather);

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && dst->src[6] == nullptr &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= min_kv;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// per device, the q8_0 in-place row limit common's fit asks for (0 = the default 16), see ggml_backend_cuda_set_fa_q8_inplace_rows
static std::atomic<int> ggml_cuda_fattn_q8_inplace_rows[GGML_CUDA_MAX_DEVICES];

void ggml_cuda_flash_attn_ext_set_q8_inplace_rows(const int device, const int rows) {
    if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
        ggml_cuda_fattn_q8_inplace_rows[device].store(std::max(0, std::min(rows, 32)), std::memory_order_relaxed);
    }
}

// q8_0 K and V of a head size with a q8_0 instance (ggml_cuda_flash_attn_ext_mma_q8_may_use), on NVIDIA with Turing
// tensor cores, rows of whole blocks starting 4 byte aligned: the MMA kernel reads them in place. The same answer for the launch and for the
// graph allocation (ggml_cuda_flash_attn_ext_get_alloc_size: no f16 copy behind the output). GGML_CUDA_FATTN_Q8_INPLACE=0: off
// likewise q4_0 K and / or V (q4_0 / q4_0, q4_0 / q8_0, q8_0 / q4_0), GGML_CUDA_FATTN_Q4_INPLACE=0: off (staged f16 copy
// as before). Up to GGML_CUDA_FATTN_Q4_INPLACE_MAX_ROWS query rows (default 16: verify windows; prompt ubatches stay staged).
// The allocation answer (for_alloc) of a pair with q4_0 is in place only up to 16 rows, the configs that have a
// q4_0 kernel (ggml_cuda_flash_attn_ext_mma_kvq_compiled); a launch of more rows in place leaves the reserved copy unused.
// Returns 0 (staged) or the in-place code of fattn-mma-f16.cuh: 1 | 4 (K q4_0) | 8 (V q4_0)
int ggml_cuda_flash_attn_ext_mma_kvq_inplace(const int device, const ggml_tensor * dst, const bool for_alloc) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst, for_alloc);
    return 0;
#else
    static const bool enabled = getenv("GGML_CUDA_FATTN_Q8_INPLACE") == nullptr || atoi(getenv("GGML_CUDA_FATTN_Q8_INPLACE")) != 0;
    // only for few query rows (decode, verify windows): every query tile converts the K / V it reads, so a prompt ubatch of 2048
    // rows converts the cache once per tile and loses to the one staged copy (A100, head size 256: 1.3-1.8x the staged time at
    // 512 / 2048 rows, 0.44-0.9x at 2..17 rows). GGML_CUDA_FATTN_Q8_INPLACE_MAX_ROWS: the row limit (default 16)
    static const int64_t max_rows_set = getenv("GGML_CUDA_FATTN_Q8_INPLACE_MAX_ROWS") ? atoll(getenv("GGML_CUDA_FATTN_Q8_INPLACE_MAX_ROWS")) : -1;
    static const bool enabled_q4 = getenv("GGML_CUDA_FATTN_Q4_INPLACE") == nullptr || atoi(getenv("GGML_CUDA_FATTN_Q4_INPLACE")) != 0;
    static const int64_t max_rows_q4 = getenv("GGML_CUDA_FATTN_Q4_INPLACE_MAX_ROWS") ? atoll(getenv("GGML_CUDA_FATTN_Q4_INPLACE_MAX_ROWS")) : 16;
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const auto is_kvq = [](const ggml_type t) { return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0; };
    if (!is_kvq(K->type) || !is_kvq(V->type)) {
        return 0;
    }
    const bool has_q4 = K->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q4_0;
    const int64_t rows = dst->src[0]->ne[1];
    // without GGML_CUDA_FATTN_Q8_INPLACE_MAX_ROWS (read once) the limit is 16, or the device's fit rows where common's fit set
    // them (ggml_backend_cuda_set_fa_q8_inplace_rows): a small device whose graphs are reserved at 31 tokens, so that
    // reserve holds no f16 copy of the whole q8_0 K / V (RTX PRO 6000 Qwen3.8-Flash-Next 12 GB / 262K: compute 561 -> 193 MiB, to the MoE cache)
    const int64_t max_rows = max_rows_set >= 0 ? max_rows_set : std::max<int64_t>(16, ggml_cuda_fattn_q8_inplace_rows[device].load(std::memory_order_relaxed));
    if (has_q4 ? (!enabled_q4 || rows > (for_alloc ? std::min<int64_t>(max_rows_q4, 16) : max_rows_q4)) : (!enabled || rows > max_rows)) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
        return 0;
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    if (dst->src[0]->ne[0] != K->ne[0] ||
            !ggml_cuda_flash_attn_ext_mma_q8_may_use(K->ne[0], V->ne[0], logit_softcap != 0.0f, V_is_K_view)) {
        return 0;
    }
    for (const ggml_tensor * t : {K, V}) {
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 4 != 0 || t->nb[2] % 4 != 0 || t->nb[3] % 4 != 0 ||
                t->nb[1] > INT32_MAX || t->nb[2] > INT32_MAX) {
            return 0;
        }
    }
    return 1 | (K->type == GGML_TYPE_Q4_0 ? 4 : 0) | (V->type == GGML_TYPE_Q4_0 ? 8 : 0);
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// INT8 Q*K^T over a q8_0 K cache read in place by the MMA kernel (fattn-mma-f16.cuh, fattn_kvq_i8). GGML_CUDA_FATTN_I8 unset: on for
// Ada (cc 8.9) and RTX Blackwell (cc 12.x), where long prompts measured faster (f16 x f16 -> f32 MMA runs at half the f16 accumulate
// rate on GeForce cards, INT8 at full rate); off on Turing (INT8 -1..-2% prompt), Ampere, Hopper and datacenter Blackwell, where the
// per-block scale epilogue loses. GGML_CUDA_FATTN_I8=0: never (the f16 kernel on a staged copy); =1: the default set plus GA10x
// (cc 8.6 - 8.7); =2: any NVIDIA Turing+ (tests). Nodes: q8_0 K, head size 128 / 256 (K = V), more than 2 query rows, no logit
// softcap, not two-source (dst->src[6]), not batch invariant, not read in place by the q8_0 / q4_0 kernels
// (ggml_cuda_flash_attn_ext_mma_kvq_inplace: few rows), K not partly in host memory (every query tile would read those cells over the
// bus; the staged copy reads them once), rows 4 byte aligned (pipelined loads: 16 byte at head size 256, 8 byte at 128). The graph
// allocation is unchanged (the f16 copy of K stays reserved behind the output; unused by these launches)
bool ggml_cuda_flash_attn_ext_mma_i8(const int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    static const int mode = getenv("GGML_CUDA_FATTN_I8") ? atoi(getenv("GGML_CUDA_FATTN_I8")) : -1;
    if (mode == 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (K->type != GGML_TYPE_Q8_0 || dst->src[6] != nullptr || Q->ne[1] <= 2 || (dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT)) {
        return false;
    }
    const int64_t D = K->ne[0];
    if ((D != 128 && D != 256) || Q->ne[0] != D || V->ne[0] != D) {
        return false;
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    if (logit_softcap != 0.0f || V_is_K_view) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
        return false;
    }
    if (mode != 2) {
        const bool rtx_blackwell = cc >= GGML_CUDA_CC_BLACKWELL && cc < GGML_CUDA_CC_RUBIN;
        const int  first         = mode == 1 ? GGML_CUDA_CC_AMPERE_GA10X : GGML_CUDA_CC_ADA_LOVELACE;
        if (!rtx_blackwell && (cc < first || cc >= GGML_CUDA_CC_HOPPER)) {
            return false;
        }
    }
    if (ggml_cuda_flash_attn_ext_mma_kvq_inplace(device, dst, false) != 0) {
        return false;
    }
    if (ggml_cuda_buffer_has_host_chunks(K->buffer)) {
        return false;
    }
    const size_t align = D == 256 ? 16 : 8;
    if (K->nb[0] != ggml_type_size(K->type) || K->nb[1] % align != 0 || K->nb[2] % align != 0 || K->nb[3] % align != 0 ||
            K->nb[1] > INT32_MAX || K->nb[2] > INT32_MAX) {
        return false;
    }
    return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// INT8 Q*K^T over an f16 K cache: per launch Q and K are staged as int8 rows with one f32 scale each (ggml_cuda_flash_attn_ext_i8_rows,
// behind the output, reserved by ggml_cuda_flash_attn_ext_get_alloc_size), rotated first per KV head whose rows have outliers (the
// per-row scale needs them spread; GGML_CUDA_FATTN_I8_ROT=0: never, =1: always), and the MMA kernel runs its INT8 tiles on them
// (fattn_kvq_i8r).
// GGML_CUDA_FATTN_I8_F16 unset: as GGML_CUDA_FATTN_I8 (unset: Ada and RTX Blackwell, head size 256; 0: never); 0: never; 1: also GA10x
// and head size 128; 2: any NVIDIA Turing+, head size 128 / 256 (tests). Nodes: f16 K and V, head size 128 / 256 (K = V), more than 16 query rows (decode and verify batches never),
// no logit softcap, not two-source, not batch invariant, K not partly in host memory, 16 byte aligned Q / K rows.
// Returns 0 (no), else 1 + the rotation mode (0 never, 1 always, 2 per KV head)
int ggml_cuda_flash_attn_ext_mma_i8_f16(const int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst);
    return 0;
#else
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_FATTN_I8_F16");
        if (e != nullptr) {
            return atoi(e);
        }
        const char * m = getenv("GGML_CUDA_FATTN_I8");
        return m != nullptr && atoi(m) == 0 ? 0 : -1;
    }();
    static const int rot = getenv("GGML_CUDA_FATTN_I8_ROT") == nullptr ? 2 : std::min(std::max(atoi(getenv("GGML_CUDA_FATTN_I8_ROT")), 0), 2);
    if (mode == 0) {
        return 0;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || Q->type != GGML_TYPE_F32 || dst->src[6] != nullptr || Q->ne[1] <= 16 ||
            (dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT)) {
        return 0;
    }
    const int64_t D = K->ne[0];
    if ((D != 128 && D != 256) || Q->ne[0] != D || V->ne[0] != D) {
        return 0;
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    if (logit_softcap != 0.0f || V_is_K_view) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
        return 0;
    }
    if (mode != 2) {
        const bool rtx_blackwell = cc >= GGML_CUDA_CC_BLACKWELL && cc < GGML_CUDA_CC_RUBIN;
        const int  first         = mode == 1 ? GGML_CUDA_CC_AMPERE_GA10X : GGML_CUDA_CC_ADA_LOVELACE;
        if (!rtx_blackwell && (cc < first || cc >= GGML_CUDA_CC_HOPPER)) {
            return 0;
        }
    }
    // by default head size 256 only: at 128 (dense Qwen3, outlier-heavy K) the rotated per-row INT8 scores measured outside the -fa off
    // KLD band; =1 / =2 take 128 too
    if (mode == -1 && D != 256) {
        return 0;
    }
    if (K->buffer != nullptr && ggml_cuda_buffer_has_host_chunks(K->buffer)) {
        return 0;
    }
    if (K->nb[0] != ggml_type_size(K->type) || Q->nb[0] != ggml_type_size(Q->type)) {
        return 0;
    }
    for (int i = 1; i < 4; ++i) {
        if (K->nb[i] % 16 != 0 || Q->nb[i] % 16 != 0) {
            return 0;
        }
    }
    const int64_t row = ggml_cuda_fattn_i8_row_bytes(D);
    if (ggml_cuda_fattn_i8_k_rows(K->ne[1])*row > INT32_MAX || Q->ne[1]*row > INT32_MAX) {
        return 0;
    }
    return 1 + rot;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// the E = D/32 consecutive values of lane threadIdx.x of a row (16 byte aligned rows: 8 or 16 byte loads)
template <int D, typename T>
static __device__ __forceinline__ void fattn_i8_load_row(const char * __restrict__ p, float * x) {
    constexpr int E = D/WARP_SIZE;
    __align__(16) T tmp[E];
    constexpr int chunk = sizeof(tmp) % 16 == 0 ? 16 : 8;
    p += threadIdx.x*int(sizeof(tmp));
#pragma unroll
    for (int c = 0; c < int(sizeof(tmp))/chunk; ++c) {
        if constexpr (chunk == 16) {
            ((int4 *) tmp)[c] = ((const int4 *) p)[c];
        } else {
            ((int2 *) tmp)[c] = ((const int2 *) p)[c];
        }
    }
#pragma unroll
    for (int i = 0; i < E; ++i) {
        x[i] = ggml_cuda_cast<float>(tmp[i]);
    }
}

// the rotation decision per KV head: rows whose max |v| over their RMS is above this on average (fixed point, 1/1024) have outliers that
// one scale per row cannot resolve, so their head is rotated (the transform spreads them; a Gaussian row of 128 / 256 values sits near
// 2.7 / 2.9). Rows without outliers (bounded values: uniform data 1.7) are not, where the transform would raise the ratio to the Gaussian
// one and cost precision
static constexpr int fattn_i8_rot_thr = 3*1024;
// K rows sampled for the decision: every 8th cell
static constexpr int fattn_i8_rot_step = 8;

// row statistics for the rotation decision: one warp per row (every step-th of ne1), max |v| / RMS in 1/1024 added to
// stats[2*(i3/r3*ne2_kv + i2/r2) + slot] (integer sums: the decision does not depend on the order of the rows)
template <int D, typename T>
static __global__ void k_fattn_i8_stats(const char * __restrict__ src, int * __restrict__ stats, const int ne1, const int step,
        const int64_t nb1, const int64_t nb2, const int64_t nb3, const int r2, const int r3, const int ne2_kv, const int slot) {
    constexpr int E = D/WARP_SIZE;
    const int r = (blockIdx.x*blockDim.y + threadIdx.y)*step;
    if (r >= ne1) {
        return;
    }
    float x[E];
    fattn_i8_load_row<D, T>(src + blockIdx.z*nb3 + blockIdx.y*nb2 + int64_t(r)*nb1, x);
    float amax = 0.0f;
    float ss   = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
        ss  += x[i]*x[i];
    }
    amax = warp_reduce_max(amax);
    ss   = warp_reduce_sum(ss);
    if (threadIdx.x == 0) {
        const float ratio = ss > 0.0f ? fminf(amax * rsqrtf(ss / D), 64.0f) : 0.0f;
        atomicAdd(stats + 2*((blockIdx.z/r3)*ne2_kv + blockIdx.y/r2) + slot, int(ratio*1024.0f + 0.5f));
    }
}

// one warp per row: E = D/32 consecutive values per lane, rotated (rot 1, or rot 2 where the statistics of the row's KV head ask for it):
// the Walsh-Hadamard transform of the row (log2(E) butterfly stages in registers, 5 across the lanes, unnormalised: Q*K^T scales by D,
// folded into Q's scale), then d = amax/127, q = round(v/d), the D quants and d*d_mul (f32, d_mul / D when rotated and is_q) behind them.
// Rows ne1 .. ne1_pad - 1 are zeros
template <int D, typename T>
static __global__ void k_fattn_i8_rows(const char * __restrict__ src, char * __restrict__ dst, const int ne1, const int ne1_pad, const int ne2,
        const int64_t nb1, const int64_t nb2, const int64_t nb3, const float d_mul, const bool is_q, const int rot,
        const int * __restrict__ stats, const int r2, const int r3, const int ne2_kv, const int nK, const int nQ) {
    constexpr int E = D/WARP_SIZE;
    static_assert(E == 4 || E == 8, "bad head size");
    const int r = blockIdx.x*blockDim.y + threadIdx.y;
    if (r >= ne1_pad) {
        return;
    }
    const int lane = threadIdx.x;
    char * out = dst + ((int64_t(blockIdx.z)*ne2 + blockIdx.y)*ne1_pad + r)*ggml_cuda_fattn_i8_row_bytes(D);

    bool rotate = rot == 1;
    if (rot == 2) {
        const int * st = stats + 2*((blockIdx.z/r3)*ne2_kv + blockIdx.y/r2);
        rotate = int64_t(st[0]) > int64_t(fattn_i8_rot_thr)*nK || int64_t(st[1]) > int64_t(fattn_i8_rot_thr)*nQ;
    }

    float x[E];
    if (r < ne1) {
        fattn_i8_load_row<D, T>(src + blockIdx.z*nb3 + blockIdx.y*nb2 + int64_t(r)*nb1, x);
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) {
            x[i] = 0.0f;
        }
    }

    if (rotate) {
#pragma unroll
        for (int h = 1; h < E; h <<= 1) {
#pragma unroll
            for (int i = 0; i < E; ++i) {
                if ((i & h) == 0) {
                    const float a = x[i];
                    const float b = x[i + h];
                    x[i]     = a + b;
                    x[i + h] = a - b;
                }
            }
        }
#pragma unroll
        for (int m = 1; m < WARP_SIZE; m <<= 1) {
#pragma unroll
            for (int i = 0; i < E; ++i) {
                const float y = __shfl_xor_sync(0xFFFFFFFF, x[i], m, WARP_SIZE);
                x[i] = (lane & m) ? y - x[i] : x[i] + y;
            }
        }
    }

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }
    amax = warp_reduce_max(amax);
    const float d = amax / 127;

    uint32_t q[E/4];
#pragma unroll
    for (int k = 0; k < E/4; ++k) {
        q[k] = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int v = d != 0.0f ? int(roundf(x[4*k + l] / d)) : 0;
            q[k] |= (uint32_t(v) & 0xFFu) << (8*l);
        }
    }
    if constexpr (E == 8) {
        *(int2 *) (out + lane*E) = make_int2(q[0], q[1]);
    } else {
        *(int *) (out + lane*E) = q[0];
    }
    if (lane == 0) {
        *(float *) (out + D) = d*(rotate && is_q ? d_mul / D : d_mul);
    }
}

template <int D>
static void ggml_cuda_flash_attn_ext_i8_rows_case(const ggml_tensor * Q, const ggml_tensor * K, const float q_scale, const int rot, char * Q8,
        char * K8, int * S8, cudaStream_t stream) {
    constexpr int rows_per_block = 4;
    const dim3 block(WARP_SIZE, rows_per_block, 1);
    const int k_rows = int(ggml_cuda_fattn_i8_k_rows(K->ne[1]));
    const int ne2_kv = int(K->ne[2]);
    const int r2     = int(Q->ne[2] / K->ne[2]);
    const int r3     = int(Q->ne[3] / K->ne[3]);
    const int nK     = int((K->ne[1] + fattn_i8_rot_step - 1) / fattn_i8_rot_step);
    const int nQ     = int(Q->ne[1])*r2*r3;
    if (rot == 2) {
        CUDA_CHECK(cudaMemsetAsync(S8, 0, 2*K->ne[2]*K->ne[3]*sizeof(int), stream));
        const dim3 grid_SK((nK + rows_per_block - 1)/rows_per_block, K->ne[2], K->ne[3]);
        k_fattn_i8_stats<D, half><<<grid_SK, block, 0, stream>>>
            ((const char *) K->data, S8, int(K->ne[1]), fattn_i8_rot_step, K->nb[1], K->nb[2], K->nb[3], 1, 1, ne2_kv, 0);
        const dim3 grid_SQ((Q->ne[1] + rows_per_block - 1)/rows_per_block, Q->ne[2], Q->ne[3]);
        k_fattn_i8_stats<D, float><<<grid_SQ, block, 0, stream>>>
            ((const char *) Q->data, S8, int(Q->ne[1]), 1, Q->nb[1], Q->nb[2], Q->nb[3], r2, r3, ne2_kv, 1);
    }
    const dim3 grid_K((k_rows + rows_per_block - 1)/rows_per_block, K->ne[2], K->ne[3]);
    k_fattn_i8_rows<D, half><<<grid_K, block, 0, stream>>>
        ((const char *) K->data, K8, int(K->ne[1]), k_rows, int(K->ne[2]), K->nb[1], K->nb[2], K->nb[3], 1.0f, false, rot,
         S8, 1, 1, ne2_kv, nK, nQ);
    const dim3 grid_Q((Q->ne[1] + rows_per_block - 1)/rows_per_block, Q->ne[2], Q->ne[3]);
    k_fattn_i8_rows<D, float><<<grid_Q, block, 0, stream>>>
        ((const char *) Q->data, Q8, int(Q->ne[1]), int(Q->ne[1]), int(Q->ne[2]), Q->nb[1], Q->nb[2], Q->nb[3], q_scale, true, rot,
         S8, r2, r3, ne2_kv, nK, nQ);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_i8_rows(const ggml_tensor * Q, const ggml_tensor * K, const float q_scale, const int rot, char * Q8,
        char * K8, int * S8, cudaStream_t stream) {
    GGML_ASSERT(Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F16);
    GGML_ASSERT((uintptr_t) Q->data % 16 == 0 && (uintptr_t) K->data % 16 == 0);
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0 && Q->ne[3] % K->ne[3] == 0);
    switch (K->ne[0]) {
        case 128:
            ggml_cuda_flash_attn_ext_i8_rows_case<128>(Q, K, q_scale, rot, Q8, K8, S8, stream);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_i8_rows_case<256>(Q, K, q_scale, rot, Q8, K8, S8, stream);
            break;
        default:
            GGML_ABORT("INT8 rows: head size %lld", (long long) K->ne[0]);
    }
}

bool ggml_cuda_flash_attn_ext_mma_q8_inplace(const int device, const ggml_tensor * dst) {
    return ggml_cuda_flash_attn_ext_mma_kvq_inplace(device, dst, true) != 0;
}

// one-row decode over a q8_0 / q4_0 K and V cache with a GQA group of >= 4 query heads per KV head runs the in-place
// MMA kernel (f16 Q*K^T, the GQA group packed into the tile columns, ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2) instead of the
// vector kernel, which leaves most of the memory bandwidth unused at long context (a 4-bit MoE, q8_0 K / V decode at 8k / 30k / 120k: T4
// +7 / +31 / +107%, L4 +2 / +12 / +46%, A100 +1 / +15 / +62%, RTX PRO 6000 (sm_120) -1 / +7 / +30%).
// Taken on NVIDIA Turing+ for nodes the in-place kernels read (ggml_cuda_flash_attn_ext_mma_kvq_inplace: types, alignment), head size
// 128 / 256 (K = V), the GQA layout the MMA kernel packs (mask, no ALiBi, KV padded to 256), no softcap, not two-source, not batch
// invariant (that one-query node stays on the vector kernel, the reference of its verify windows), and a K length of at least
// min_kv rows. min_kv: 12288 on compute capability 12.x, where the vector kernel still wins short contexts (RTX PRO 6000 q8_0 K / V, 3 launches,
// spread <= 0.2%: 2k -0.4%, 8k -0.7%, 16k +2.4%, 30k +7.1%; crossover near 10k), 0 elsewhere (T4 / L4 / A100 already gain at 8k);
// GGML_CUDA_FATTN_DEC_MMA_MIN_KV=N: N on every arch. No f16 copy is reserved for these nodes (in place), as for the vector kernel.
// GGML_CUDA_FATTN_DEC_MMA=0: off (the vector kernel, as before)
static bool ggml_cuda_flash_attn_ext_dec_mma(const int device, const ggml_tensor * dst, const bool gqa_opt_applies) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst, gqa_opt_applies);
    return false;
#else
    static const bool enabled = getenv("GGML_CUDA_FATTN_DEC_MMA") == nullptr || atoi(getenv("GGML_CUDA_FATTN_DEC_MMA")) != 0;
    static const int64_t min_kv_env = getenv("GGML_CUDA_FATTN_DEC_MMA_MIN_KV") ? atoll(getenv("GGML_CUDA_FATTN_DEC_MMA_MIN_KV")) : -1;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!enabled || !gqa_opt_applies || Q->ne[1] != 1 || dst->src[6] != nullptr || (dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) ||
            !ggml_is_quantized(K->type) || !ggml_is_quantized(V->type) || K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    const int64_t D = K->ne[0];
    if ((D != 128 && D != 256) || Q->ne[0] != D || V->ne[0] != D || Q->ne[2] / K->ne[2] < 4) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
        return false;
    }
    const int64_t min_kv = min_kv_env >= 0 ? min_kv_env : (cc >= GGML_CUDA_CC_BLACKWELL && cc < GGML_CUDA_CC_RUBIN ? 12288 : 0);
    if (K->ne[1] < min_kv) {
        return false;
    }
    // q8_0 / q4_0 K and V in any pairing (the kernels read every pair in place up to 16 rows); 0 = staged: keep the vector kernel
    if (ggml_cuda_flash_attn_ext_mma_kvq_inplace(device, dst, false) == 0) {
        return false;
    }
    // logged at the first launch (data set), not at a graph reservation (the parameter fit reserves with its log muted)
    static std::atomic<int> n_logged { 0 };
    if (ggml_cuda_fattn_probe_cur == nullptr && dst->data != nullptr && K->data != nullptr && n_logged++ == 0) {
        GGML_LOG_INFO("%s: one-row decode over the %s K / %s V cache runs the in-place MMA kernel (D %lld, GQA %lld, %lld "
            "cells; GGML_CUDA_FATTN_DEC_MMA=0: the vector kernel)\n", __func__, ggml_type_name(K->type), ggml_type_name(V->type),
            (long long) D, (long long) (Q->ne[2] / K->ne[2]), (long long) K->ne[1]);
    }
    return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// a two-source node (ggml_flash_attn_ext_set_src2) is read in place when its body is (ggml_cuda_flash_attn_ext_mma_kvq_inplace:
// q4_0 / q4_0 or q8_0 / q8_0, the row limits, alignment) and its ring is q8_0 / q8_0 with 4 byte aligned rows: the compiled
// combinations (fattn-mma-f16.cuh). Returns code1 | code2 << 8 (type bits | 1), 0 = both staged to f16
int ggml_cuda_flash_attn_ext_src2_codes(const int device, const ggml_tensor * dst, const bool for_alloc) {
    const ggml_tensor * K2 = dst->src[6];
    const ggml_tensor * V2 = dst->src[7];
    if (K2 == nullptr) {
        return 0;
    }
    const int code1 = ggml_cuda_flash_attn_ext_mma_kvq_inplace(device, dst, for_alloc);
    if (code1 == 0 || ((code1 & 12) != 12 && (code1 & 12) != 0) || K2->type != GGML_TYPE_Q8_0 || V2->type != GGML_TYPE_Q8_0) {
        return 0;
    }
    for (const ggml_tensor * t : {K2, V2}) {
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 4 != 0 || t->nb[2] % 4 != 0 || t->nb[3] % 4 != 0 ||
                t->nb[1] > INT32_MAX || t->nb[2] > INT32_MAX || (uintptr_t) t->view_offs % 4 != 0) {
            return 0;
        }
    }
    // the two-source in-place kernels exist for tiles of up to 16 query columns: more than 16 rows with fewer than 4 query heads per
    // tile (ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2 / _ncols1) is a wider tile, staged to f16
    const ggml_tensor * Q = dst->src[0];
    if (Q->ne[1] > 16) {
        const ggml_tensor * K    = dst->src[1];
        const ggml_tensor * V    = dst->src[2];
        const ggml_tensor * mask = dst->src[3];
        float max_bias = 0.0f;
        memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
        bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
        for (const ggml_tensor * t : {Q, K, V, mask}) {
            if (t == nullptr || ggml_is_quantized(t->type)) {
                continue;
            }
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                use_gqa_opt = use_gqa_opt && t->nb[i] % 16 == 0;
            }
        }
        if (!use_gqa_opt || Q->ne[2] / K->ne[2] <= 2) {
            return 0;
        }
    }
    return code1 | (1 << 8);
}

// a two-source node this backend can run: NVIDIA Turing+ (the MMA kernel), head sizes 64 / 128 / 256 (K = V) or the
// MLA 576 / 512, no logit softcap, the second source of supported K / V types with whole 256-cell tiles, its mask F16 shaped like the
// first's. GGML_CUDA_FATTN_SRC2=0: unsupported (the graph's generic form)
static bool ggml_cuda_fattn_src2_supported(const int cc, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_FATTN_SRC2") == nullptr || atoi(getenv("GGML_CUDA_FATTN_SRC2")) != 0;
    const ggml_tensor * K  = dst->src[1];
    const ggml_tensor * V  = dst->src[2];
    const ggml_tensor * M  = dst->src[3];
    const ggml_tensor * K2 = dst->src[6];
    const ggml_tensor * V2 = dst->src[7];
    const ggml_tensor * M2 = dst->src[8];
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool hs_ok = (K->ne[0] == V->ne[0] && (K->ne[0] == 64 || K->ne[0] == 128 || K->ne[0] == 256)) || (K->ne[0] == 576 && V->ne[0] == 512);
    const auto type_ok = [](const ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 ||
            t == GGML_TYPE_Q8_0 || t == GGML_TYPE_BF16;
    };
    return enabled && GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && hs_ok && logit_softcap == 0.0f &&
        V2 != nullptr && M2 != nullptr && M != nullptr && M2->type == GGML_TYPE_F16 &&
        type_ok(K2->type) && type_ok(V2->type) && K2->ne[0] == K->ne[0] && V2->ne[0] == V->ne[0] &&
        K2->ne[1] == V2->ne[1] && K2->ne[1] > 0 && K2->ne[1] % FATTN_KQ_STRIDE == 0 && K2->ne[2] == K->ne[2] && V2->ne[2] == V->ne[2] &&
        K2->ne[3] == K->ne[3] && V2->ne[3] == V->ne[3] &&
        K2->nb[0] == ggml_type_size(K2->type) && V2->nb[0] == ggml_type_size(V2->type) &&
        M2->ne[0] == K2->ne[1] && M2->ne[1] == M->ne[1] && M2->ne[2] == 1 && M2->ne[3] == M->ne[3] && M2->nb[0] == sizeof(half);
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        if (head_size == (D) && type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {                          \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = type_K_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_K_in;
    const ggml_type type_V = type_V_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_V_in;
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // a two-source node runs on the MMA kernel only
    if (dst->src[6] != nullptr) {
        if (!ggml_cuda_fattn_src2_supported(ggml_cuda_info().devices[device].cc, dst)) {
            return BEST_FATTN_KERNEL_NONE;
        }
        if (K->ne[0] == 576 && !gqa_opt_applies) {
            return BEST_FATTN_KERNEL_NONE;
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        // one-row decode over a quantized K / V cache with GQA >= 4 on the in-place MMA kernel
        if (ggml_cuda_flash_attn_ext_dec_mma(device, dst, gqa_opt_applies)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                // a node that asks for the selected-cell gather (op_params[5] == 1: its K / V cells partly in host memory,
                // see qwen4exp build_attn_qsa) takes the sparse MMA kernel where it applies; the vector kernel walks every cell
                const bool sparse_pref = dst->op_params[5] == 1 && gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_pref) {
                    if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                        if (Q->ne[1] <= 2) {
                            return BEST_FATTN_KERNEL_VEC;
                        }
                    } else {
                        if (Q->ne[1] == 1) {
                            return BEST_FATTN_KERNEL_VEC;
                        }
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;
    bool i8_f16     = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            // no f16 copy when the kernel reads q8_0 K / V in place (or q4_0)
            need_f16_K = dst->src[6] != nullptr ? ggml_cuda_flash_attn_ext_src2_codes(device, dst, true) == 0 :
                ggml_cuda_flash_attn_ext_mma_kvq_inplace(device, dst, true) == 0;
            need_f16_V = need_f16_K;
            // INT8 Q*K^T over an f16 K: the int8 rows of Q and K behind the output
            i8_f16 = ggml_cuda_flash_attn_ext_mma_i8_f16(device, dst) != 0;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V, i8_f16);

    return f16_extra.end - (uintptr_t) dst->data;
}

thread_local const ggml_cuda_fattn_kv_len_ctx * ggml_cuda_fattn_kv_len_cur = nullptr;
thread_local ggml_cuda_fattn_probe *            ggml_cuda_fattn_probe_cur  = nullptr;

static void ggml_cuda_flash_attn_ext_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// a node with a KV length (ggml_flash_attn_ext_set_kv_len) is run as one of
//   PAD:  the padded node as built, where its launch is the live node's (the sparse gather: its lists, split and blocks follow
//         n_kv_max, not the cells, and the padding cells are -inf, never listed);
//   DEV:  the live node's launch choices with the length read on the device (the dense stream-k MMA kernel, ggml_cuda_fattn_dev_len):
//         its CUDA graph is replayed while the length changes;
//   LIVE: the live node (K / V / mask views of kv_len cells) with the length as a constant, captured again when it changes.
// Each computes the bits of the node of kv_len cells. The signature (ggml_cuda_flash_attn_ext_kv_len_sig) changes when the mode,
// the kernel or the fixup of the live launch changes (and with the length in LIVE), so a CUDA graph is replayed only for a launch
// it holds. GGML_CUDA_FA_KV_LEN=0: every node runs LIVE (the kill switch: the bits are the same, the graphs follow the length)
enum ggml_cuda_fattn_kvl_mode {
    GGML_CUDA_FATTN_KVL_NONE = 0,
    GGML_CUDA_FATTN_KVL_PAD  = 1,
    GGML_CUDA_FATTN_KVL_DEV  = 2,
    GGML_CUDA_FATTN_KVL_LIVE = 3,
};

struct ggml_cuda_fattn_kvl_plan {
    int     mode = GGML_CUDA_FATTN_KVL_NONE;
    int64_t sig  = 0;
};

struct ggml_cuda_fattn_live_node {
    ggml_tensor K, V, M, dst;
};

static int64_t ggml_cuda_fattn_kv_len_host(const ggml_tensor * dst) {
    const ggml_tensor * kvl = dst->src[5];
    const int64_t n_pad = dst->src[1]->ne[1];
    const int64_t n     = ggml_get_op_params_i32(kvl, 0);
    return n > 0 && n < n_pad ? n : n_pad;
}

static void ggml_cuda_fattn_make_live(const ggml_tensor * dst, const int64_t n_kv, ggml_cuda_fattn_live_node & ln) {
    ln.K = *dst->src[1];
    ln.K.ne[1] = n_kv;
    ln.V = *dst->src[2];
    ln.V.ne[1] = n_kv;
    ln.dst = *dst;
    ln.dst.src[1] = &ln.K;
    ln.dst.src[2] = &ln.V;
    ln.dst.src[5] = nullptr;
    if (dst->src[3]) {
        ln.M = *dst->src[3];
        ln.M.ne[0] = n_kv;
        ln.dst.src[3] = &ln.M;
    }
}

// the kernel reads the length through a plain device pointer: only from this device's memory (a length left in host memory by
// the placement of the graph's inputs runs LIVE, logged once)
static bool ggml_cuda_fattn_kv_len_on_device(ggml_backend_cuda_context & ctx, const ggml_tensor * kvl) {
    cudaPointerAttributes pa = {};
    const bool ok = kvl->data != nullptr && cudaPointerGetAttributes(&pa, kvl->data) == cudaSuccess &&
        pa.type == cudaMemoryTypeDevice && pa.device == ctx.device;
    (void) cudaGetLastError();
    static std::atomic<int> n_logged { 0 };
    if (!ok && n_logged++ < 4) {
        GGML_LOG_WARN("%s: the KV length '%s' (buffer %s, memory type %d, device %d) is not in the memory of device %d: the flash "
            "attention keeps the length as a constant (re-captured when it changes)\n", __func__, kvl->name,
            kvl->buffer ? ggml_backend_buffer_name(kvl->buffer) : "none", (int) pa.type, pa.device, ctx.device);
    }
    return ok;
}

static ggml_cuda_fattn_probe ggml_cuda_fattn_run_probe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_fattn_probe pr;
    ggml_cuda_fattn_probe * prev = ggml_cuda_fattn_probe_cur;
    ggml_cuda_fattn_probe_cur = &pr;
    ggml_cuda_flash_attn_ext_dispatch(ctx, dst);
    ggml_cuda_fattn_probe_cur = prev;
    return pr;
}

// the dry run of the node's dispatch (the padded node as built: the launch of a node with a KV length where its sparse plan holds)
// loads the kernel of the launch; a sparse launch also loads the kernels that build its lists (both query tile widths), from the
// selected cells where the node gathers them (QSA maskless), else from the mask
void ggml_cuda_flash_attn_ext_preload(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_cuda_fattn_probe pr = ggml_cuda_fattn_run_probe(ctx, dst);
    if (!pr.done) {
        return;
    }
    ggml_cuda_preload_func(pr.kernel);
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    if (!pr.use_sparse || mask == nullptr) {
        return;
    }
    const auto it = ctx.qsa_maskless.find(dst);
    const ggml_cuda_qsa_maskless * qsa = it != ctx.qsa_maskless.end() ? &it->second : nullptr;
    const bool gather = qsa && !qsa->prompt && ggml_cuda_qsa_maskless_gather_ok(*qsa) && ggml_cuda_qsa_maskless_mode() == 1;
    for (const int ncols1 : { 1, 8 }) {
        if (gather) {
            ggml_cuda_qsa_maskless_compact(*qsa, nullptr, nullptr, int32_t(Q->ne[1]), ncols1, 0, ctx.stream());
        } else {
            ggml_cuda_flash_attn_ext_compact_mask(mask, nullptr, nullptr, int32_t(Q->ne[1]), ncols1, 0, ctx.stream());
        }
    }
    if (qsa && !gather) {
        ggml_cuda_qsa_maskless_materialize(*qsa, nullptr, ctx.stream());
    }
}

static ggml_cuda_fattn_kvl_plan ggml_cuda_fattn_kvl_get_plan(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    ggml_cuda_fattn_kvl_plan plan;
    if (dst->src[5] == nullptr) {
        return plan;
    }
    static const bool enabled = getenv("GGML_CUDA_FA_KV_LEN") == nullptr || atoi(getenv("GGML_CUDA_FA_KV_LEN")) != 0;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];
    const int64_t n_kv = ggml_cuda_fattn_kv_len_host(dst);
    // (a two-source node takes the first source's live length as a constant)
    if (!enabled || (dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) || dst->src[6] != nullptr) {
        plan.mode = GGML_CUDA_FATTN_KVL_LIVE;
        plan.sig  = (int64_t(GGML_CUDA_FATTN_KVL_LIVE) << 56) ^ n_kv;
        return plan;
    }

    // the plan follows the shapes, types, flags and parameters of the node and the live length only
    uint64_t h = 0x6b76;
    const auto mix = [&h](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        mix(Q->ne[d]); mix(K->ne[d]); mix(V->ne[d]); mix(M ? M->ne[d] : 0);
        mix(K->nb[d]); mix(V->nb[d]); mix(M ? M->nb[d] : 0);
    }
    mix(K->type); mix(V->type); mix(dst->flags); mix(dst->src[4] != nullptr); mix(n_kv); mix(ctx.device);
    for (int i = 0; i < 5; ++i) {
        mix((uint32_t) dst->op_params[i]);
    }
    mix(V->view_src == K || (V->view_src && V->view_src == K->view_src && V->view_offs == K->view_offs));
    mix((uint64_t) (uintptr_t) dst->src[5]->data);
    static thread_local std::unordered_map<uint64_t, ggml_cuda_fattn_kvl_plan> cache;
    const auto it = cache.find(h);
    if (it != cache.end()) {
        return it->second;
    }

    ggml_cuda_fattn_live_node ln;
    ggml_cuda_fattn_make_live(dst, n_kv, ln);
    const ggml_cuda_fattn_kv_len_ctx kctx = { dst, nullptr, K->ne[1] };
    const ggml_cuda_fattn_kv_len_ctx * prev = ggml_cuda_fattn_kv_len_cur;
    ggml_cuda_fattn_kv_len_cur = &kctx;
    const ggml_cuda_fattn_probe pl = ggml_cuda_fattn_run_probe(ctx, &ln.dst);
    ggml_cuda_fattn_kv_len_cur = prev;

    const uint64_t kh = (uint64_t) (uintptr_t) pl.kernel;
    if (pl.done && pl.use_sparse) {
        ggml_tensor dp = *dst;
        dp.src[5] = nullptr;
        const ggml_cuda_fattn_probe pp = ggml_cuda_fattn_run_probe(ctx, &dp);
        if (pp.done && pp.use_sparse && pp.kernel == pl.kernel && pp.nblocks == pl.nblocks && pp.fixup == pl.fixup) {
            plan.mode = GGML_CUDA_FATTN_KVL_PAD;
            plan.sig  = (int64_t(GGML_CUDA_FATTN_KVL_PAD) << 56) ^ (int64_t) ((kh * 0x9e3779b97f4a7c15ULL) >> 8) ^ (int64_t(pl.nblocks) << 24) ^ pl.fixup;
        }
    } else if (pl.done && pl.dev_ok && ggml_cuda_fattn_kv_len_on_device(ctx, dst->src[5])) {
        plan.mode = GGML_CUDA_FATTN_KVL_DEV;
        static std::atomic<int> n_dev_logged { 0 };
        if (n_dev_logged++ == 0) {
            GGML_LOG_INFO("%s: flash attention reads the KV length on the device ('%s', %s), first at %lld of %lld cells "
                "(GGML_CUDA_FA_KV_LEN=0: off)\n", __func__, dst->src[5]->name, dst->src[5]->buffer ? ggml_backend_buffer_name(dst->src[5]->buffer) : "none",
                (long long) n_kv, (long long) K->ne[1]);
        }
        plan.sig  = (int64_t(GGML_CUDA_FATTN_KVL_DEV) << 56) ^ (int64_t) ((kh * 0x9e3779b97f4a7c15ULL) >> 8) ^ (int64_t(pl.stream_k) << 4) ^ pl.fixup;
    }
    if (plan.mode == GGML_CUDA_FATTN_KVL_NONE) {
        plan.mode = GGML_CUDA_FATTN_KVL_LIVE;
        plan.sig  = (int64_t(GGML_CUDA_FATTN_KVL_LIVE) << 56) ^ n_kv;
    }
    if (cache.size() > 4096) {
        cache.clear();
    }
    cache.emplace(h, plan);
    return plan;
}

int64_t ggml_cuda_flash_attn_ext_kv_len_sig(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    return ggml_cuda_fattn_kvl_get_plan(ctx, dst).sig;
}

// host read counter (LLAMA_KV_SEL_STATS=1, a diagnostic: one extra kernel per attention node whose K / V buffer has host
// chunks). Counters per device: [0] selected bytes on host chunks, [1] staged bytes on host chunks, [2] nodes, [3] nodes with host chunks
static unsigned long long * ggml_cuda_kv_sel_ctr[GGML_CUDA_MAX_DEVICES] = {};
static std::atomic<uint64_t> ggml_cuda_kv_sel_staged{0};
static std::atomic<uint64_t> ggml_cuda_kv_sel_nodes_host{0};
static std::atomic<uint64_t> ggml_cuda_kv_sel_nodes_staged{0};

static __device__ __forceinline__ unsigned long long kv_sel_cell_host_bytes(const int32_t c, const char * K, const size_t nbk,
        const char * V, const size_t nbv, const char * base, const size_t gran, const uint8_t * map) {
    unsigned long long b = 0;
    b += map[(size_t) (K + (size_t) c*nbk - base) / gran] ? nbk : 0;
    if (V != nullptr) {
        b += map[(size_t) (V + (size_t) c*nbv - base) / gran] ? nbv : 0;
    }
    return b;
}

// selected cells from the QSA selection (idx: I32 [n_sel, n_rows], dead slots >= n_kv) or from the mask (F16, finite = read)
static __global__ void kv_sel_count_kernel(const int32_t * idx, const int64_t idx_s1, const int n_sel,
        const half * mask, const int64_t mask_s1, const int n_rows, const int n_kv,
        const char * K, const size_t nbk, const char * V, const size_t nbv, const char * base, const size_t gran, const uint8_t * map,
        unsigned long long * ctr) {
    const int r = blockIdx.y;
    unsigned long long b = 0;
    if (idx != nullptr) {
        for (int j = blockIdx.x*blockDim.x + threadIdx.x; j < n_sel; j += gridDim.x*blockDim.x) {
            const int32_t c = idx[r*idx_s1 + j];
            if (c >= 0 && c < n_kv) {
                b += kv_sel_cell_host_bytes(c, K, nbk, V, nbv, base, gran, map);
            }
        }
    } else {
        for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < n_kv; c += gridDim.x*blockDim.x) {
            if (__half2float(mask[r*mask_s1 + c]) > -INFINITY) {
                b += kv_sel_cell_host_bytes(c, K, nbk, V, nbv, base, gran, map);
            }
        }
    }
    if (b > 0) {
        atomicAdd(ctr, b);
    }
    if (r == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(ctr + 2, 1ull);
    }
}

static void ggml_cuda_kv_sel_count(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const char * base = nullptr;
    size_t gran = 0;
    const uint8_t * map_dev  = nullptr;
    const uint8_t * map_host = nullptr;
    size_t n_map = 0;
    if (K == nullptr || K->buffer == nullptr || !ggml_cuda_buffer_host_chunks(K->buffer, &base, &gran, &map_dev, &map_host, &n_map)) {
        return;
    }
    const int64_t n_kv = K->ne[1];
    const bool same_buf = V != nullptr && V->buffer == K->buffer && V->data != K->data;
    // the chunks the node's K / V cells span
    bool any_host = false;
    uint64_t span_host = 0;
    for (const ggml_tensor * t : { K, same_buf ? V : nullptr }) {
        if (t == nullptr) {
            continue;
        }
        const size_t o0 = (const char *) t->data - base;
        const size_t o1 = o0 + (size_t) n_kv*t->nb[1];
        for (size_t c = o0/gran; c < n_map && c*gran < o1; ++c) {
            if (map_host[c]) {
                any_host = true;
                span_host += std::min(o1, (c + 1)*gran) - std::max(o0, c*gran);
            }
        }
    }
    if (!any_host) {
        return;
    }
    ggml_cuda_kv_sel_nodes_host++;
    const int dev = ctx.device;
    if (ggml_cuda_kv_sel_ctr[dev] == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &ggml_cuda_kv_sel_ctr[dev], 4*sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(ggml_cuda_kv_sel_ctr[dev], 0, 4*sizeof(unsigned long long)));
    }
    const int64_t n_rows = Q->ne[1];
    if (n_rows > 16 || Q->ne[3] > 1) {
        ggml_cuda_kv_sel_staged += span_host; // a prompt node stages every cell once
        return;
    }
    const ggml_cuda_qsa_maskless * qsa = nullptr;
    if (!ctx.qsa_maskless.empty()) {
        const auto it = ctx.qsa_maskless.find(dst);
        if (it != ctx.qsa_maskless.end()) {
            qsa = &it->second;
        }
    }
    const ggml_tensor * mask = dst->src[3];
    const int32_t * idx = qsa ? (const int32_t *) qsa->idx->data : nullptr;
    if (idx == nullptr && (mask == nullptr || mask->type != GGML_TYPE_F16)) {
        ggml_cuda_kv_sel_staged += span_host; // no selection to read: every cell
        return;
    }
    const int n_sel = qsa ? (int) qsa->idx->ne[0] : 0;
    const dim3 grid(4, (unsigned) n_rows, 1);
    kv_sel_count_kernel<<<grid, 256, 0, ctx.stream()>>>(idx, qsa ? qsa->idx->nb[1]/sizeof(int32_t) : 0, n_sel,
        idx ? nullptr : (const half *) mask->data, idx ? 0 : mask->nb[1]/sizeof(half), (int) n_rows, (int) n_kv,
        (const char *) K->data, K->nb[1], same_buf ? (const char *) V->data : nullptr, same_buf ? V->nb[1] : 0, base, gran, map_dev,
        ggml_cuda_kv_sel_ctr[dev]);
    CUDA_CHECK(cudaGetLastError());
}

// staging (see ggml_cuda_kv_sel_stage): one warp per listed cell copies its K row, its V row and the mask values of the
// tile's query rows, then points the list entry at the staging row (lane 0, after every lane has read the cell)
template <typename T>
static __global__ void kv_sel_stage_kernel(const char * __restrict__ K, const size_t nbk, const char * __restrict__ V, const size_t nbv,
        const half * __restrict__ mask, const int64_t s31, int32_t * __restrict__ indices, const int32_t * __restrict__ counts,
        const int ncols1, const int n_rows, const int n_kv_max, char * __restrict__ K_c, char * __restrict__ V_c, half * __restrict__ M_c) {
    const int l    = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const int j    = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (j >= counts[l]) {
        return;
    }
    int32_t * idx = indices + (size_t) l*n_kv_max;
    const int32_t c   = idx[j];
    const int64_t dst = (int64_t) l*n_kv_max + j;
    {
        const T * src = (const T *) (K + (size_t) c*nbk);
        T * d = (T *) (K_c + (size_t) dst*nbk);
        for (size_t i = lane; i < nbk/sizeof(T); i += WARP_SIZE) {
            d[i] = src[i];
        }
    }
    if (V != nullptr) {
        const T * src = (const T *) (V + (size_t) c*nbv);
        T * d = (T *) (V_c + (size_t) dst*nbv);
        for (size_t i = lane; i < nbv/sizeof(T); i += WARP_SIZE) {
            d[i] = src[i];
        }
    }
    for (int r = l*ncols1 + lane; r < min(n_rows, (l + 1)*ncols1); r += WARP_SIZE) {
        M_c[r*s31 + dst] = mask[r*s31 + c];
    }
    __syncwarp();
    if (lane == 0) {
        idx[j] = (int32_t) dst;
    }
}

void ggml_cuda_kv_sel_stage(const char * K, size_t nbk, const char * V, size_t nbv, const half * mask, int64_t s31,
        int32_t * indices, const int32_t * counts, int ntiles, int ncols1, int n_rows, int n_kv_max,
        char * K_c, char * V_c, half * M_c, cudaStream_t stream) {
    ggml_cuda_kv_sel_nodes_staged++;
    constexpr int warps = 8;
    const dim3 grid((n_kv_max + warps - 1)/warps, ntiles, 1);
    const bool v16 = nbk % 16 == 0 && (V == nullptr || nbv % 16 == 0) && (uintptr_t) K % 16 == 0 && (uintptr_t) K_c % 16 == 0 &&
        (V == nullptr || ((uintptr_t) V % 16 == 0 && (uintptr_t) V_c % 16 == 0));
    if (v16) {
        kv_sel_stage_kernel<int4><<<grid, warps*WARP_SIZE, 0, stream>>>(K, nbk, V, nbv, mask, s31, indices, counts, ncols1, n_rows, n_kv_max, K_c, V_c, M_c);
    } else {
        kv_sel_stage_kernel<int><<<grid, warps*WARP_SIZE, 0, stream>>>(K, nbk, V, nbv, mask, s31, indices, counts, ncols1, n_rows, n_kv_max, K_c, V_c, M_c);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_kv_sel_stats(uint64_t * out) {
    out[0] = out[1] = out[2] = out[3] = out[4] = 0;
    for (int d = 0; d < ggml_backend_cuda_get_device_count() && d < GGML_CUDA_MAX_DEVICES; ++d) {
        if (ggml_cuda_kv_sel_ctr[d] == nullptr) {
            continue;
        }
        ggml_cuda_set_device(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long h[4] = {};
        CUDA_CHECK(cudaMemcpy(h, ggml_cuda_kv_sel_ctr[d], sizeof(h), cudaMemcpyDeviceToHost));
        out[0] += h[0];
        out[2] += h[2];
    }
    out[1] = ggml_cuda_kv_sel_staged.load();
    out[3] = ggml_cuda_kv_sel_nodes_host.load();
    out[4] = ggml_cuda_kv_sel_nodes_staged.load();
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    // LLAMA_KV_SEL_STATS=1 counts the bytes this node reads from host chunks (a separate kernel, results unchanged)
    static const bool kv_sel_stats = ggml_cuda_kv_sel_stats_on();
    if (kv_sel_stats) {
        ggml_cuda_kv_sel_count(ctx, dst);
    }

    const ggml_cuda_fattn_kvl_plan plan = ggml_cuda_fattn_kvl_get_plan(ctx, dst);
    if (plan.mode == GGML_CUDA_FATTN_KVL_DEV || plan.mode == GGML_CUDA_FATTN_KVL_LIVE) {
        ggml_cuda_fattn_live_node ln;
        ggml_cuda_fattn_make_live(dst, ggml_cuda_fattn_kv_len_host(dst), ln);
        const ggml_cuda_fattn_kv_len_ctx kctx = { dst, plan.mode == GGML_CUDA_FATTN_KVL_DEV ? (const int *) dst->src[5]->data : nullptr, dst->src[1]->ne[1] };
        const ggml_cuda_fattn_kv_len_ctx * prev = ggml_cuda_fattn_kv_len_cur;
        ggml_cuda_fattn_kv_len_cur = &kctx;
        ggml_cuda_flash_attn_ext_dispatch(ctx, &ln.dst);
        ggml_cuda_fattn_kv_len_cur = prev;
        return;
    }
    ggml_cuda_flash_attn_ext_dispatch(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    // batch invariant node over several queries (speculative verify, the queries of one KV length): every query gets what
    // the node of that query alone gets. The vector kernel at one column per block takes all of them in one launch with the KV
    // split of a single query; any other kernel runs once per query
    const ggml_tensor * Q = dst->src[0];
    // (not a two-source node, which takes the ordinary MMA launch below: correct, not split per query)
    if ((dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) && Q->ne[1] > 1 && dst->src[6] == nullptr) {
        const ggml_tensor * mask = dst->src[3];
        ggml_tensor q1 = *Q;
        q1.ne[1] = 1;
        ggml_tensor m1;
        if (mask) {
            m1 = *mask;
            m1.ne[1] = 1;
        }
        ggml_tensor d1 = *dst;
        d1.src[0] = &q1;
        d1.src[3] = mask ? &m1 : nullptr;
        d1.ne[2]  = 1;
        d1.flags &= ~GGML_TENSOR_FLAG_BATCH_INVARIANT;
        const best_fattn_kernel k1 = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), &d1);
        // GGML_CUDA_FA_VERIFY_MMA=1 (opt-in): where the one query node runs the MMA kernel (Turing / Ampere: f16 KQ tiles,
        // f16 P, half2 VKQ), the verify node runs that kernel once for all its queries instead of once per query. Same kernel family
        // and number formats, only the tiling of the KV loop / softmax rescales and the KV split differ (equivalence
        // rule: reassociation); taken only when the batched node would pick the MMA kernel too
        static const bool verify_mma = [] {
            const char * e = getenv("GGML_CUDA_FA_VERIFY_MMA");
            return e != nullptr && atoi(e) != 0;
        }();
        if (verify_mma && k1 == BEST_FATTN_KERNEL_MMA_F16) {
            ggml_tensor dn = *dst;
            dn.flags &= ~GGML_TENSOR_FLAG_BATCH_INVARIANT;
            if (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), &dn) == BEST_FATTN_KERNEL_MMA_F16) {
                ggml_cuda_flash_attn_ext_mma_f16(ctx, &dn);
                return;
            }
        }
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        // the per query launches need f16 K and V (another type stages a converted copy behind the output, which a
        // launch writing one row of it would place over the rows after it). llama-graph splits the queries of one stream itself and
        // groups f16 K / V only, but a node of several streams (non-unified KV, -np N, a verify batch over 2+ sequences) with a
        // quantized cache reaches here whole: it takes the ordinary kernels below (correct, not bit-equal to a decode) instead of
        // the assert it hit before
        if (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
            if (k1 == BEST_FATTN_KERNEL_VEC && Q->ne[3] == 1) {
                ggml_cuda_flash_attn_ext_vec(ctx, dst);
                return;
            }
            // one launch per (stream, query): the kernels index the output from its shape (contiguous), so each launch gets its
            // stream's K / V / mask and output rows with ne[3] = 1 (with ne[3] kept, stream s of the one row output landed at
            // s*H*D instead of s*n_q*H*D and the streams overwrote each other)
            GGML_ASSERT(Q->ne[3] % K->ne[3] == 0 && Q->ne[3] % V->ne[3] == 0);
            GGML_ASSERT(mask == nullptr || Q->ne[3] % mask->ne[3] == 0);
            for (int64_t s = 0; s < Q->ne[3]; ++s) {
                ggml_tensor k_s = *K;
                k_s.ne[3] = 1;
                k_s.data  = (char *) K->data + (s / (Q->ne[3] / K->ne[3]))*K->nb[3];
                ggml_tensor v_s = *V;
                v_s.ne[3] = 1;
                v_s.data  = (char *) V->data + (s / (Q->ne[3] / V->ne[3]))*V->nb[3];
                for (int64_t i = 0; i < Q->ne[1]; ++i) {
                    ggml_tensor q_i = q1;
                    q_i.ne[3] = 1;
                    q_i.data  = (char *) Q->data + i*Q->nb[1] + s*Q->nb[3];
                    ggml_tensor m_i;
                    if (mask) {
                        m_i = m1;
                        m_i.ne[3] = 1;
                        m_i.data  = (char *) mask->data + i*mask->nb[1] + (s % mask->ne[3])*mask->nb[3];
                    }
                    ggml_tensor d_i = d1;
                    d_i.src[0] = &q_i;
                    d_i.src[1] = &k_s;
                    d_i.src[2] = &v_s;
                    d_i.src[3] = mask ? &m_i : nullptr;
                    d_i.ne[3]  = 1;
                    d_i.data   = (char *) dst->data + i*dst->nb[2] + s*dst->nb[3];
                    ggml_cuda_flash_attn_ext_dispatch(ctx, &d_i);
                }
            }
            return;
        }
    }

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
