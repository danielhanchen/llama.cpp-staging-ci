// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#include "moe-cache.cuh"
#include "ggml-moe-mailbox.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define MOE_MAILBOX_OK 1 // single-graph host misses (ggml_moe_mailbox_publish)
#endif

// device-side lookup of the MoE expert cache, see ggml_moe_cache_lookup and llama-moe-cache.cpp
//
// state: [0] CLOCK hand, [1] epoch (one per lookup), [2..3] hits, [4..5] misses, [6] filled slots, [7] weight of a use (decayed counts), then key_of, ref and pin [n_slots]
// key_of: il*n_expert + expert (-1 if empty); ref: CLOCK reference bits; pin: epoch of the last lookup using the slot (that lookup does not evict it)
// dst:   slots[n_ids], then a row starting with the number of copied experts, then their experts[n_ids] and slots[n_ids]
//
// assign result rows (ne0 = n_expert_used, T tokens): slots [T], fill count [1], fill experts [T], fill slots [T], host ids [T], hit slots [T]
// hybrid (ggml_moe_cache_assign): counts [n_expert, n_layer + 1] use counters, last row: [0..1] experts left to the host, [2] fetch accumulator (1/65536)
// only a share of the missing experts is admitted (fetch_q/65536, error diffused), the ones with the highest counters, ties by expert id
// (all of them while the cache has empty slots: filling an empty slot evicts nothing)
// the others keep slot -1 and are written to host_ids, after the slots and the copy list
//
// decayed counts (ggml_moe_cache_set_scores, LLAMA_MOE_CACHE_POLICY=decay): scores [n_expert, n_layer] with growing weights, each
// distinct selected expert adds w = state[7]; at the first layer of the group (advance) w *= grow/65536 first, and when w reaches
// MOE_CACHE_W_MAX every counter and w are halved (a pass over the whole table, every few dozen tokens), so a counter decays by
// 65536/grow per token without touching it; the victims of the missing experts are the lowest scores of the experts in the next
// MOE_CACHE_WINDOW slots from the hand that this lookup does not use (empty slots: -1), ties by hand order, read in one pass, then the
// hand moves past the furthest victim (more than MOE_CACHE_LANES missing experts: one window per missing expert); the reference bits
// are not used. Integer only: the same ids give the same slots.

#define MOE_CACHE_HDR   8
#define MOE_CACHE_LANES 32
#define MOE_CACHE_WINDOW 64
#define MOE_CACHE_W_INIT (1 << 16)
#define MOE_CACHE_W_MAX  (1 << 22)
#define MOE_CACHE_SCORE_MAX (1 << 30)

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

// k15-fn: ids with strided rows (s1 elements apart, s1 = ne0 when contiguous), e.g. the view of the router's argsort
static __device__ __forceinline__ int32_t moe_cache_id(const int32_t * ids, const int i, const int n_expert, const int ne0, const int s1) {
    const int32_t e = ids[s1 == ne0 ? i : (i / ne0)*s1 + i % ne0];
    return (uint32_t) e < (uint32_t) n_expert ? e : 0;
}

// route tail skip (ggml_moe_cache_set_tail_skip): entry i is skipped if its expert is not cached before this lookup (s < 0, read
// before the lookup writes the map) and every entry of that expert in the window has a rank (i % ne0, ids in routing order) of at least
// tail_rank; such an expert is neither admitted, fetched nor left to the host (slot and host id -1), so its rows are 0
static __device__ __forceinline__ bool moe_cache_tail_skip(const int32_t * ids, const int i, const int32_t s, const int n_ids, const int ne0,
        const int n_expert, const int tail_rank, const int s1) {
    if (tail_rank <= 0 || s >= 0 || i % ne0 < tail_rank) {
        return false;
    }
    const int32_t e = moe_cache_id(ids, i, n_expert, ne0, s1);
    for (int j = 0; j < n_ids; ++j) {
        if (j % ne0 < tail_rank && moe_cache_id(ids, j, n_expert, ne0, s1) == e) {
            return false;
        }
    }
    return true;
}

// a skipped entry while the lookup runs (its final slot is -1)
#define MOE_CACHE_SKIPPED (-3)

static __device__ __forceinline__ void moe_cache_add_u64(int32_t * p, const unsigned long long v) {
    // two 4-byte words (the row of the statistics is only 4-byte aligned)
    const unsigned long long x = ((unsigned long long) (uint32_t) p[1] << 32 | (uint32_t) p[0]) + v;
    p[0] = (int32_t) (uint32_t) x;
    p[1] = (int32_t) (uint32_t) (x >> 32);
}

// k19-g4pref cross-layer prefetch (ggml_moe_cache_prefetch, op mode 8). A prefetch runs the assign below as a plain lookup of predicted
// ids (is_pf): it admits them like a lookup but writes no use counters, scores or lookup statistics, at most max_fill experts (0: all).
// The real lookups and assigns of a group with prefetches count what came of them. mark [n_slots]: MOE_PF_MAGIC while a slot holds an
// expert a prefetch admitted that no lookup has hit since; ev [n_layer*n_expert]: MOE_PF_MAGIC while an expert a prefetch evicted has
// not been missed since; stat: the counters MOE_PF_STAT_* (every counter has one writer kernel: the prefetch or the real lookup)
#define MOE_PF_MAGIC 0x50465046
enum {
    MOE_PF_STAT_LAYERS = 0,  // prefetch calls
    MOE_PF_STAT_CACHED,      // predicted experts that were cached already
    MOE_PF_STAT_FILLS,       // experts admitted (copied) by prefetches
    MOE_PF_STAT_CAPPED,      // predicted missing experts left out by the cap
    MOE_PF_STAT_USEFUL,      // prefetched slots hit by a later lookup (before any eviction)
    MOE_PF_STAT_WASTED_PF,   // prefetched slots evicted before any hit, by a prefetch
    MOE_PF_STAT_WASTED_REAL, // the same, by a real lookup
    MOE_PF_STAT_EVMISS,      // real misses of experts a prefetch evicted
    MOE_PF_STAT_REAL,        // real lookups / assigns
    MOE_PF_STAT_REAL_FILLS,  // experts they admitted
    MOE_PF_STAT_SKIPPED,     // prefetch ops not run: the layer's own lookup came first (host side)
    MOE_PF_STAT_N
};
struct moe_pf_args {
    int32_t *            mark;
    int32_t *            ev;
    unsigned long long * stat;
    int                  is_pf;
    int                  max_fill;
};

// asynchronous refill of lent slots (llama_moe_cache::unlend, ggml_cuda_moe_refill_*). When a prompt gives its lent slots back, the
// uploads that refill them may run on a second stream while the next graph computes. Per registered group (the state tensor) and
// layer: need = the epoch of the last refill queued for the layer, done = the last one whose uploads completed (written by the upload
// stream after them). An assign waits for its own layer before its hits are read, and for the layer of the expert a victim holds
// before that slot is given to another expert, so every read and write of a slot comes after the refill as with a synchronous one:
// the same slots, the same contents, the same results
#define MOE_RF_GROUPS 16
#define MOE_RF_LAYERS 128
static __device__ int32_t moe_rf_tab[MOE_RF_GROUPS][2][MOE_RF_LAYERS];

static __device__ __forceinline__ void moe_rf_wait(const int rf, const int il) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (rf < 0 || rf >= MOE_RF_GROUPS || (unsigned) il >= MOE_RF_LAYERS) {
        return;
    }
    volatile const int32_t * need = moe_rf_tab[rf][0];
    volatile const int32_t * done = moe_rf_tab[rf][1];
    const int32_t n = need[il];
    while (done[il] < n) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(256);
#endif
    }
    __threadfence();
#else
    GGML_UNUSED(rf);
    GGML_UNUSED(il);
#endif
}

static __global__ void moe_cache_assign(
        const int32_t * __restrict__ ids, int32_t * __restrict__ map, int32_t * __restrict__ state, int32_t * __restrict__ dst,
        const int n_ids, const int ne0, const int n_expert, const int il, const int n_slots,
        int32_t * __restrict__ counts, const int n_layer, const int fetch_q,
        int32_t * __restrict__ scores, const int grow, const int advance, const int n_scores, const int frozen = 0,
        const int tail_rank = 0, int32_t * __restrict__ skip_stat = nullptr, int ids_s1 = 0, const int defer = 0,
        const int margin_q = 0, const int margin_add_q = 0, const int lookup_counts = 0, const int lo = 0,
        const moe_pf_args pf = moe_pf_args{nullptr, nullptr, nullptr, 0, 0}, const int rf = -1) {
    if (ids_s1 <= 0) {
        ids_s1 = ne0;
    }
    // a refill of this layer still uploading (see moe_rf_wait): its slots are read as hits below
    if (rf >= 0) {
        if (threadIdx.x == 0) {
            moe_rf_wait(rf, il);
        }
        __syncthreads();
    }
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
    int32_t * sc     = scores != nullptr ? scores + (int64_t) il*n_expert : nullptr;
    // k19-q4xcache (ggml_moe_cache_assign_lookup_counts): an assign that stands in for the plain lookup (every miss fetched) leaves the
    // use counters as the lookup does, so the host / fetch ranking of later hybrid batches is the same as without it
    int32_t * cnt_w  = lookup_counts ? nullptr : cnt;

    // frozen (ggml_moe_cache_assign_freeze, hybrid only): the cached experts are hits, every other one goes to the host; nothing is
    // admitted and the state, counters and scores are not written (only the statistics), so a token gets the same slots in any batch
    if (frozen && counts != nullptr) {
        int n_hit = 0;
        int n_mis = 0;
        int n_skp = 0;
        for (int i = lane; i < n_ids; i += MOE_CACHE_LANES) {
            const int32_t e = moe_cache_id(ids, i, n_expert, ne0, ids_s1);
            const int32_t s = row[e];
            const bool skip = moe_cache_tail_skip(ids, i, s, n_ids, ne0, n_expert, tail_rank, ids_s1);
            slots[i]             = s;
            fill_s[2*n_ids + i]  = s;
            fill_s[n_ids + i]    = s < 0 && !skip ? e : -1;
            n_hit += s >= 0;
            n_mis += s < 0 && !skip;
            n_skp += skip;
        }
        for (int off = MOE_CACHE_LANES/2; off > 0; off >>= 1) {
            n_hit += __shfl_xor_sync(full, n_hit, off, MOE_CACHE_LANES);
            n_mis += __shfl_xor_sync(full, n_mis, off, MOE_CACHE_LANES);
            n_skp += __shfl_xor_sync(full, n_skp, off, MOE_CACHE_LANES);
        }
        if (lane == 0) {
            if (skip_stat != nullptr && n_skp > 0) {
                moe_cache_add_u64(skip_stat, (unsigned long long) n_skp);
            }
            *n_fill = 0;
            *(unsigned long long *) (state + 2) += (unsigned long long) n_hit;
            int32_t * stat = counts + (int64_t) n_layer*n_expert;
            const unsigned long long n_host = ((unsigned long long) (uint32_t) stat[1] << 32 | (uint32_t) stat[0]) + (unsigned long long) n_mis;
            stat[0] = (int32_t) (uint32_t) n_host;
            stat[1] = (int32_t) (uint32_t) (n_host >> 32);
        }
        return;
    }

    uint32_t epoch = (uint32_t) state[1] + 1;
    if (epoch == 0) {
        // wrapped around: no slot may look pinned by this lookup
        for (int s = lane; s < n_slots; s += MOE_CACHE_LANES) {
            pin[s] = 0;
        }
        epoch = 1;
    }
    int32_t w = 0; // weight of a use, decayed counts
    if (sc != nullptr) {
        w = state[7] > 0 ? state[7] : MOE_CACHE_W_INIT;
        if (advance) {
            w = (int32_t) (((int64_t) w*grow) >> 16);
            if (w >= MOE_CACHE_W_MAX) {
                for (int i = lane; i < n_scores; i += MOE_CACHE_LANES) {
                    scores[i] >>= 1;
                }
                w >>= 1;
            }
        }
    }
    __syncthreads();
    if (sc != nullptr && lane == 0) {
        state[7] = w;
    }
    if (lane == 0) {
        state[1] = (int32_t) epoch;
    }

    // hits: set the reference bit and pin the slot for this lookup
    int n_hit = 0;
    int n_skp = 0;
    int n_useful = 0; // k19-g4pref counters (pf.mark / pf.ev set)
    int n_wasted = 0;
    int n_evmiss = 0;
    for (int i = lane; i < n_ids; i += MOE_CACHE_LANES) {
        const int32_t s = row[moe_cache_id(ids, i, n_expert, ne0, ids_s1)];
        const bool skip = moe_cache_tail_skip(ids, i, s, n_ids, ne0, n_expert, tail_rank, ids_s1);
        n_skp += skip;
        slots[i] = skip ? MOE_CACHE_SKIPPED : s;
        if (counts != nullptr) {
            fill_s[2*n_ids + i] = s; // hit slots: the slot of an expert cached before this lookup, else -1
        }
        if (s >= 0) {
            ref[s] = 1;
            if ((uint32_t) atomicExch(&pin[s], (int32_t) epoch) != epoch) {
                n_hit++;
                const int32_t e = moe_cache_id(ids, i, n_expert, ne0, ids_s1);
                if (cnt_w != nullptr) {
                    cnt_w[e] += cnt_w[e] < 0x7fffffff;
                }
                if (sc != nullptr && !pf.is_pf) {
                    sc[e] = min(sc[e] + w, MOE_CACHE_SCORE_MAX);
                }
                if (pf.mark != nullptr && !pf.is_pf && pf.mark[s] == MOE_PF_MAGIC) {
                    pf.mark[s] = 0;
                    n_useful++;
                }
            }
        }
    }
    __syncthreads();

    // misses: the first use of each missing expert claims it, in the order of the ids
    int n_miss = 0;
    for (int i0 = 0; i0 < n_ids; i0 += MOE_CACHE_LANES) {
        const int i = i0 + lane;
        const int32_t e = i < n_ids && slots[i] < 0 ? moe_cache_id(ids, i, n_expert, ne0, ids_s1) : -1;
        bool first = e >= 0;
        for (int k = 0; k < MOE_CACHE_LANES; ++k) {
            const int32_t ek = __shfl_sync(full, e, k, MOE_CACHE_LANES);
            if (k < lane && ek == e) {
                first = false;
            }
        }
        // a previous group of ids may have claimed it already
        first = first && row[e] == -1;
        // a skipped expert (every entry of it is) counts as a use but is not missing: it holds -3 in the map until the end of the lookup
        const bool skip = first && slots[i] == MOE_CACHE_SKIPPED;
        const unsigned m = moe_cache_ballot(first && !skip);
        if (first) {
            row[e] = skip ? MOE_CACHE_SKIPPED : -2;
            if (!skip) {
                miss_e[n_miss + __popc(m & ((1u << lane) - 1))] = e;
            }
            if (cnt_w != nullptr) {
                cnt_w[e] += cnt_w[e] < 0x7fffffff;
            }
            if (sc != nullptr && !pf.is_pf) {
                sc[e] = min(sc[e] + w, MOE_CACHE_SCORE_MAX);
            }
            if (pf.ev != nullptr && !pf.is_pf && !skip && pf.ev[(int64_t) il*n_expert + e] == MOE_PF_MAGIC) {
                pf.ev[(int64_t) il*n_expert + e] = 0;
                n_evmiss++;
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
        // k19-hostrow (LLAMA_MOE_HYBRID_WIDE): a window whose experts may outnumber the slots admits at most the slots this lookup does
        // not pin (its hits); the rest go to the host. Never binds while the distinct ids fit the slots (every earlier caller)
        int n_pin = n_hit;
        for (int off = MOE_CACHE_LANES/2; off > 0; off >>= 1) {
            n_pin += __shfl_xor_sync(full, n_pin, off, MOE_CACHE_LANES);
        }
        n_take = lo > 0 ? max(0, min(n_take, n_slots - lo - n_pin)) : min(n_take, n_slots - n_pin);
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
            // the row of the statistics starts at n_layer*n_expert: only 4-byte aligned, so the 64-bit counter is read and written as two words
            const unsigned long long n_host = ((unsigned long long) (uint32_t) stat[1] << 32 | (uint32_t) stat[0]) + (unsigned long long) (n_miss - n_take);
            stat[0] = (int32_t) (uint32_t) n_host;
            stat[1] = (int32_t) (uint32_t) (n_host >> 32);
        }
    }

    // k19-g4pref: a prefetch admits at most max_fill of its missing experts (the first ones in the order of the ids: rank-major)
    if (pf.is_pf && pf.max_fill > 0 && n_take > pf.max_fill) {
        for (int j = pf.max_fill + lane; j < n_take; j += MOE_CACHE_LANES) {
            row[take_e[j]] = -1;
        }
        if (lane == 0) {
            pf.stat[MOE_PF_STAT_CAPPED] += (unsigned long long) (n_take - pf.max_fill);
        }
        n_take = pf.max_fill;
        __syncthreads();
    }

    // victims: CLOCK sweep, 32 slots at a time, the slots used by this lookup are skipped
    // there are enough unpinned slots because the number of distinct ids is at most the number of slots
    // k19-ptres core (ggml_moe_cache_set_core, lo > 0): the victims come from slots [lo, n_slots) only, ns of them; lo = 0: as before
    const int ns = n_slots - lo;
    int h = state[0];
    if (lo > 0 && (h < lo || h >= n_slots)) {
        h = lo;
    }
    const int width = min(MOE_CACHE_LANES, ns);
    int found = 0;
    if (scores != nullptr && n_take > 0 && n_take <= MOE_CACHE_LANES) {
        // decayed counts, one window for all the missing experts: the scores and hand offsets of the first MOE_CACHE_WINDOW unpinned
        // slots go to shared memory, then n_take rounds of a warp argmin (no global reads)
        int32_t * cv = smem + (counts != nullptr ? 3 : 2)*n_ids; // [MOE_CACHE_WINDOW] scores
        int32_t * co = cv + MOE_CACHE_WINDOW;                    // [MOE_CACHE_WINDOW] offsets from the hand
        int seen = 0;
        for (int off0 = 0; off0 < ns && seen < MOE_CACHE_WINDOW; off0 += MOE_CACHE_LANES) {
            const int  off    = off0 + lane;
            const bool active = off < ns;
            const int  s      = active ? (h + off < n_slots ? h + off : h + off - ns) : 0;
            const bool free_s = active && (uint32_t) pin[s] != epoch;
            const unsigned m  = moe_cache_ballot(free_s);
            const int rank    = seen + __popc(m & ((1u << lane) - 1));
            if (free_s && rank < MOE_CACHE_WINDOW) {
                const int32_t key = key_of[s];
                cv[rank] = key >= 0 ? scores[key] : -1;
                co[rank] = off;
            }
            seen = min(seen + __popc(m), MOE_CACHE_WINDOW);
        }
        __syncthreads();
        int far = 0;
        for (; found < n_take; ++found) {
            int best_v = INT_MAX;
            int best_o = INT_MAX;
            int best_i = -1;
            for (int i = lane; i < seen; i += MOE_CACHE_LANES) {
                if (cv[i] < best_v || (cv[i] == best_v && co[i] < best_o)) {
                    best_v = cv[i];
                    best_o = co[i];
                    best_i = i;
                }
            }
            for (int o = MOE_CACHE_LANES/2; o > 0; o >>= 1) {
                const int v2 = __shfl_xor_sync(full, best_v, o, MOE_CACHE_LANES);
                const int o2 = __shfl_xor_sync(full, best_o, o, MOE_CACHE_LANES);
                const int i2 = __shfl_xor_sync(full, best_i, o, MOE_CACHE_LANES);
                if (v2 < best_v || (v2 == best_v && o2 < best_o)) {
                    best_v = v2;
                    best_o = o2;
                    best_i = i2;
                }
            }
            const int s = h + best_o < n_slots ? h + best_o : h + best_o - ns;
            far = max(far, best_o);
            __syncthreads();
            if (lane == 0) {
                victim[found] = s;
                pin[s]        = (int32_t) epoch;
                cv[best_i]    = INT_MAX; // taken
                co[best_i]    = INT_MAX;
            }
            __syncthreads();
        }
        h = h + far + 1 < n_slots ? h + far + 1 : h + far + 1 - ns;

        // k19-ramevict admission margin (hybrid, LLAMA_MOE_CACHE_ADMIT_MARGIN): a missing expert replaces a resident only if its decayed
        // count beats the victim's by the margin (score > victim*margin_q/65536 + w*margin_add_q/65536); empty slots always admit. The
        // victims were taken in rising score order, the candidates go in falling score order (ties by expert id), so the admitted ones
        // are a prefix of both; the rest go to the host as the experts the fetch share leaves out (the victims keep their experts).
        // Placement only on the device; the host computes the rejected experts (host rounding). Idea from Straydog (MIT)
        if (margin_q > 0 && counts != nullptr && sc != nullptr) {
            int n_acc = 0;
            if (lane == 0) {
                for (int a = 1; a < n_take; ++a) { // insertion sort of the candidates, highest score first
                    const int32_t ea = take_e[a];
                    int b = a - 1;
                    while (b >= 0 && (sc[take_e[b]] < sc[ea] || (sc[take_e[b]] == sc[ea] && take_e[b] > ea))) {
                        take_e[b + 1] = take_e[b];
                        b--;
                    }
                    take_e[b + 1] = ea;
                }
                for (; n_acc < n_take; ++n_acc) {
                    const int32_t key = key_of[victim[n_acc]];
                    if (key < 0) {
                        continue;
                    }
                    const long long need = ((long long) scores[key]*margin_q + (long long) w*margin_add_q) >> 16;
                    if ((long long) sc[take_e[n_acc]] <= need) {
                        break;
                    }
                }
                for (int j = n_acc; j < n_take; ++j) {
                    row[take_e[j]] = -1; // left to the host
                }
                const int n_rej = n_take - n_acc;
                if (n_rej > 0) {
                    moe_cache_add_u64(stat, (unsigned long long) n_rej);      // experts left to the host
                    moe_cache_add_u64(stat + 6, (unsigned long long) n_rej);  // of them by the margin
                }
            }
            n_take = __shfl_sync(full, n_acc, 0, MOE_CACHE_LANES);
            found  = n_take;
            __syncthreads();
        }
    }
    if (scores != nullptr) {
        // decayed counts, more missing experts than lanes (multi-token batches): one window per missing expert
        // there is an unpinned slot in the first n_slots ones from the hand because the distinct ids fit in the slots
        for (; found < n_take; ++found) {
            int best_sc  = INT_MAX;
            int best_off = INT_MAX;
            int seen     = 0;
            for (int off0 = 0; off0 < ns && seen < MOE_CACHE_WINDOW; off0 += MOE_CACHE_LANES) {
                const int  off    = off0 + lane;
                const bool active = off < ns;
                const int  s      = active ? (h + off < n_slots ? h + off : h + off - ns) : 0;
                const bool free_s = active && (uint32_t) pin[s] != epoch;
                const unsigned m  = moe_cache_ballot(free_s);
                const int rank    = __popc(m & ((1u << lane) - 1));
                if (free_s && rank < MOE_CACHE_WINDOW - seen) {
                    const int32_t key = key_of[s];
                    const int     v   = key >= 0 ? scores[key] : -1;
                    if (v < best_sc) { // offsets grow within a lane: the first of equal scores is kept
                        best_sc  = v;
                        best_off = off;
                    }
                }
                seen += min(__popc(m), MOE_CACHE_WINDOW - seen);
            }
            for (int o = MOE_CACHE_LANES/2; o > 0; o >>= 1) {
                const int v2 = __shfl_xor_sync(full, best_sc,  o, MOE_CACHE_LANES);
                const int o2 = __shfl_xor_sync(full, best_off, o, MOE_CACHE_LANES);
                if (v2 < best_sc || (v2 == best_sc && o2 < best_off)) {
                    best_sc  = v2;
                    best_off = o2;
                }
            }
            const int s = h + best_off < n_slots ? h + best_off : h + best_off - ns;
            __syncthreads();
            if (lane == 0) {
                victim[found] = s;
                pin[s]        = (int32_t) epoch;
            }
            h = s + 1 < n_slots ? s + 1 : lo;
            __syncthreads();
        }
    }
    while (found < n_take) {
        const bool active = lane < width;
        const int  s      = active ? (h + lane < n_slots ? h + lane : h + lane - ns) : 0;
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
            h -= ns;
        }
        __syncthreads();
    }

    // evict and insert
    int n_new = 0; // empty slots filled
    for (int j = lane; j < n_take; j += MOE_CACHE_LANES) {
        const int32_t s   = victim[j];
        const int32_t e   = take_e[j];
        const int32_t old = key_of[s];
        if (rf >= 0 && old >= 0) {
            moe_rf_wait(rf, old / n_expert); // the victim's expert may still be uploading into it
        }
        if (old >= 0) {
            map[old] = -1;
        } else {
            n_new++;
        }
        if (pf.mark != nullptr) {
            n_wasted += old >= 0 && pf.mark[s] == MOE_PF_MAGIC;
            pf.mark[s] = pf.is_pf ? MOE_PF_MAGIC : 0;
        }
        if (pf.ev != nullptr) {
            if (pf.is_pf && old >= 0) {
                pf.ev[old] = MOE_PF_MAGIC;
            }
            pf.ev[(int64_t) il*n_expert + e] = 0;
        }
        key_of[s] = il*n_expert + e;
        ref[s]    = 1;
        row[e]    = s;
        fill_e[j] = e;
        fill_s[j] = s;
    }
    __syncthreads();

    for (int i = lane; i < n_ids; i += MOE_CACHE_LANES) {
        bool skip = false;
        if (slots[i] == MOE_CACHE_SKIPPED) {
            // every entry of a skipped expert is skipped, so no other entry reads its map entry here
            skip     = true;
            slots[i] = -1;
            row[moe_cache_id(ids, i, n_expert, ne0, ids_s1)] = -1;
        } else if (slots[i] < 0) {
            slots[i] = row[moe_cache_id(ids, i, n_expert, ne0, ids_s1)];
        }
        if (counts != nullptr) {
            // after the copy list: the experts left to the host; k19-l4c deferred admission (ggml_moe_cache_set_defer): every expert that
            // was not cached before this lookup (the hit slots row), the admitted ones too, which the copies make hits for later batches
            const bool to_host = defer ? fill_s[2*n_ids + i] < 0 : slots[i] < 0;
            fill_s[n_ids + i] = to_host && !skip ? moe_cache_id(ids, i, n_expert, ne0, ids_s1) : -1;
        }
    }

    for (int off = MOE_CACHE_LANES/2; off > 0; off >>= 1) {
        n_hit += __shfl_xor_sync(full, n_hit, off, MOE_CACHE_LANES);
        n_new += __shfl_xor_sync(full, n_new, off, MOE_CACHE_LANES);
        n_skp += __shfl_xor_sync(full, n_skp, off, MOE_CACHE_LANES);
        if (pf.stat != nullptr) {
            n_useful += __shfl_xor_sync(full, n_useful, off, MOE_CACHE_LANES);
            n_wasted += __shfl_xor_sync(full, n_wasted, off, MOE_CACHE_LANES);
            n_evmiss += __shfl_xor_sync(full, n_evmiss, off, MOE_CACHE_LANES);
        }
    }
    if (lane == 0) {
        if (skip_stat != nullptr && n_skp > 0) {
            moe_cache_add_u64(skip_stat, (unsigned long long) n_skp);
        }
        state[6] += n_new;
        state[0] = h;
        *n_fill  = n_take;
        if (pf.is_pf) {
            // a prefetch is not a lookup: the hit / miss statistics of the cache stay those of the real lookups
            pf.stat[MOE_PF_STAT_LAYERS]    += 1;
            pf.stat[MOE_PF_STAT_CACHED]    += (unsigned long long) n_hit;
            pf.stat[MOE_PF_STAT_FILLS]     += (unsigned long long) n_take;
            pf.stat[MOE_PF_STAT_WASTED_PF] += (unsigned long long) n_wasted;
        } else {
            *(unsigned long long *) (state + 2) += (unsigned long long) n_hit;
            *(unsigned long long *) (state + 4) += (unsigned long long) n_take;
            if (pf.stat != nullptr) {
                pf.stat[MOE_PF_STAT_USEFUL]      += (unsigned long long) n_useful;
                pf.stat[MOE_PF_STAT_WASTED_REAL] += (unsigned long long) n_wasted;
                pf.stat[MOE_PF_STAT_EVMISS]      += (unsigned long long) n_evmiss;
                pf.stat[MOE_PF_STAT_REAL]        += 1;
                pf.stat[MOE_PF_STAT_REAL_FILLS]  += (unsigned long long) n_take;
            }
        }
    }
}

struct moe_cache_fetch_args {
    const char * src[4];
    char       * dst[4];
    int64_t      nb[4];
};

// copy the missing experts from (mapped) host memory to their slots, block z copies the chunks z, z + gridDim.z, ...
// admitted != nullptr (overlapped fetch of an assign, see ggml_moe_cache_fetch): the first block also writes the slots of the
// experts that were not cached before the lookup, -1 for the others
static __global__ void moe_cache_fetch(const int32_t * __restrict__ info, const int n_ids, const int ne0, const moe_cache_fetch_args args, const int64_t chunk,
        int32_t * __restrict__ admitted) {
    const int32_t * n_fill = info + n_ids;
    const int32_t * fill_e = info + n_ids + ne0;
    const int32_t * fill_s = fill_e + n_ids;

    if (admitted != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        const int32_t * hit = info + 4*n_ids + ne0; // hit slots, see the assign result layout
        for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
            admitted[i] = hit[i] < 0 ? info[i] : -1;
        }
    }

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

// device seed (ggml_moe_cache_seed_plan), one block: count the uses of each expert, rank the used ones (most used first, ties by expert id),
// keep the hits of the n_seed first, give the missing ones slots below n_victim in hand order (empty slots first, then CLOCK reference
// bit 0, then 1), skipping the slots pinned by this seed or the pos seeds before it in the graph, and the slots of the layers after il,
// whose copies may still read them. Block-wide scans only (no warp intrinsics), so the result does not depend on the warp size.
#define MOE_SEED_THREADS 1024

static __device__ int moe_seed_scan(int * buf, const int v) {
    // inclusive block scan of v over MOE_SEED_THREADS threads, buf: [MOE_SEED_THREADS]
    const int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int off = 1; off < MOE_SEED_THREADS; off <<= 1) {
        const int add = t >= off ? buf[t - off] : 0;
        __syncthreads();
        buf[t] += add;
        __syncthreads();
    }
    const int r = buf[t];
    __syncthreads();
    return r;
}

static __global__ void moe_cache_seed_plan(
        const int32_t * __restrict__ ids, const int n_ids, int32_t * __restrict__ map, int32_t * __restrict__ state,
        int32_t * __restrict__ scores, int32_t * __restrict__ out, const int n_expert, const int il, const int n_slots,
        const int n_victim, const int n_seed, const int n_seed_max, const int pos, const int lo = 0) {
    extern __shared__ int32_t smem[];
    int32_t * cnt  = smem;                    // [n_expert] uses of each expert
    int32_t * sel  = cnt + n_expert;          // [n_seed] the selected experts by rank, -1 none
    int32_t * miss = sel + n_seed;            // [n_seed] the missing ones among them, in rank order
    int32_t * vic  = miss + n_seed;           // [n_seed] their slots
    int32_t * scan = vic + n_seed;            // [MOE_SEED_THREADS]
    __shared__ int s_miss;
    __shared__ int s_found;
    __shared__ int s_new;

    const int t = threadIdx.x;

    int32_t * key_of = state + MOE_CACHE_HDR;
    int32_t * ref    = key_of + n_slots;
    int32_t * pin    = ref + n_slots;
    int32_t * row    = map + (int64_t) il*n_expert;
    int32_t * sc     = scores != nullptr ? scores + (int64_t) il*n_expert : nullptr;

    int32_t * fill_e = out + 1;
    int32_t * fill_s = out + 1 + n_seed_max;

    for (int e = t; e < n_expert; e += MOE_SEED_THREADS) {
        cnt[e] = 0;
    }
    for (int j = t; j < n_seed; j += MOE_SEED_THREADS) {
        sel[j] = -1;
    }
    if (t == 0) {
        s_miss  = 0;
        s_found = 0;
        s_new   = 0;
    }
    __syncthreads();
    // k19-ptres core (lo > 0): the victims come from slots [lo, n_victim) only
    if (n_seed == 0 || n_victim <= lo) {
        if (t == 0) {
            out[0] = 0;
        }
        return;
    }
    for (int i = t; i < n_ids; i += MOE_SEED_THREADS) {
        atomicAdd(&cnt[moe_cache_id(ids, i, n_expert)], 1);
    }

    uint32_t epoch = (uint32_t) state[1] + 1;
    __syncthreads();
    if (epoch == 0) {
        for (int s = t; s < n_slots; s += MOE_SEED_THREADS) {
            pin[s] = 0;
        }
        epoch = 1;
    }
    const int32_t w = sc != nullptr ? (state[7] > 0 ? state[7] : MOE_CACHE_W_INIT) : 0;
    __syncthreads();

    // rank of each used expert
    for (int e = t; e < n_expert; e += MOE_SEED_THREADS) {
        const int32_t c = cnt[e];
        if (c == 0) {
            continue;
        }
        int rank = 0;
        for (int k = 0; k < n_expert && rank < n_seed; ++k) {
            const int32_t ck = cnt[k];
            rank += ck > c || (ck == c && k < e);
        }
        if (rank < n_seed) {
            sel[rank] = e;
        }
    }
    __syncthreads();

    // hits: keep them (reference bit, pin); the missing ones in rank order (sel is filled from rank 0, -1 after the used experts)
    for (int j0 = 0; j0 < n_seed; j0 += MOE_SEED_THREADS) {
        const int j = j0 + t;
        const int32_t e = j < n_seed ? sel[j] : -1;
        const int32_t s = e >= 0 ? row[e] : -1;
        if (s >= 0) {
            ref[s] = 1;
            pin[s] = (int32_t) epoch;
            if (sc != nullptr) {
                sc[e] = min(sc[e] + w, MOE_CACHE_SCORE_MAX);
            }
        }
        const bool m    = e >= 0 && s < 0;
        const int  incl = moe_seed_scan(scan, m ? 1 : 0);
        const int  base = s_miss;
        if (m) {
            miss[base + incl - 1] = e;
        }
        const int total = scan[MOE_SEED_THREADS - 1];
        __syncthreads();
        if (t == 0) {
            s_miss = base + total;
        }
        __syncthreads();
    }
    const int n_miss = s_miss;

    // victims, in hand order by class: 0 empty, 1 reference bit 0, 2 reference bit 1
    const int nv = n_victim - lo;
    const int h = state[0] >= lo && state[0] < n_victim ? state[0] : lo;
    for (int cls = 0; cls < 3 && s_found < n_miss; ++cls) {
        for (int off0 = 0; off0 < nv && s_found < n_miss; off0 += MOE_SEED_THREADS) {
            const int off = off0 + t;
            bool elig = false;
            int  s    = 0;
            if (off < nv) {
                s = h + off < n_victim ? h + off : h + off - nv;
                const int32_t  key = key_of[s];
                const uint32_t p   = (uint32_t) pin[s];
                const bool pinned  = p != 0 && epoch - p <= (uint32_t) pos;
                const int  c       = key < 0 ? 0 : (ref[s] == 0 ? 1 : 2);
                elig = !pinned && c == cls && (key < 0 || key/n_expert <= il);
            }
            const int incl  = moe_seed_scan(scan, elig ? 1 : 0);
            const int base  = s_found;
            const int total = scan[MOE_SEED_THREADS - 1];
            if (elig && base + incl - 1 < n_miss) {
                vic[base + incl - 1] = s;
            }
            __syncthreads();
            if (t == 0) {
                s_found = min(base + total, n_miss);
            }
            __syncthreads();
        }
    }
    const int n_take = s_found;

    // evict and insert
    for (int j = t; j < n_take; j += MOE_SEED_THREADS) {
        const int32_t s   = vic[j];
        const int32_t e   = miss[j];
        const int32_t old = key_of[s];
        if (old >= 0) {
            map[old] = -1;
        } else {
            atomicAdd(&s_new, 1);
        }
        key_of[s] = il*n_expert + e;
        ref[s]    = 1;
        pin[s]    = (int32_t) epoch;
        row[e]    = s;
        if (sc != nullptr) {
            sc[e] = min(sc[e] + w, MOE_CACHE_SCORE_MAX);
        }
        fill_e[j] = e;
        fill_s[j] = s;
    }
    __syncthreads();
    if (t == 0) {
        state[1]  = (int32_t) epoch;
        state[6] += s_new;
        out[0]    = n_take;
    }
}

// the experts admitted by a seed plan, from the experts of the matrix product (src) to the bank, block x: the x-th admitted expert
static __global__ void moe_cache_seed_fill(const int32_t * __restrict__ plan, const int n_seed_max, const char * __restrict__ src,
        char * __restrict__ dst, const int64_t nb, const int64_t chunk) {
    const int j = blockIdx.x;
    if (j >= plan[0]) {
        return;
    }
    const char * s = src + (int64_t) plan[1 + j]*nb;
    char       * d = dst + (int64_t) plan[1 + n_seed_max + j]*nb;
    for (int64_t beg = blockIdx.z*chunk; beg < nb; beg += gridDim.z*chunk) {
        const int64_t end = min(beg + chunk, nb);
        if ((((uintptr_t) (s + beg)) | ((uintptr_t) (d + beg)) | (uintptr_t) (end - beg)) % 16 == 0) {
            const int4 * s4 = (const int4 *) (s + beg);
            int4       * d4 = (int4 *) (d + beg);
            const int64_t n = (end - beg)/16;
            for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
                d4[i] = s4[i];
            }
        } else {
            for (int64_t i = beg + threadIdx.x; i < end; i += blockDim.x) {
                d[i] = s[i];
            }
        }
    }
}

#define MOE_PF_MAX_IDS 256 // k19-g4pref: predicted ids per prefetch (see moe_pf_run)

bool ggml_cuda_moe_cache_lookup_supported(const ggml_tensor * op) {
    const int mode    = ggml_get_op_params_i32(op, 10);
    const int n_banks = ggml_get_op_params_i32(op, 1);
    if (mode == 4) {
        const int n_expert   = (int) op->src[1]->ne[0];
        const int n_seed_max = ggml_get_op_params_i32(op, 2);
        return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 && op->src[2]->type == GGML_TYPE_I32 &&
            (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_I32) &&
            (size_t) (n_expert + 3*n_seed_max + MOE_SEED_THREADS)*sizeof(int32_t) <= 48*1024;
    }
    if (mode == 5) {
        return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->op == GGML_OP_MUL_MAT_ID;
    }
    if (mode == 8) {
        const int n_expert = (int) op->src[1]->ne[0];
        return n_banks >= 1 && n_banks <= 4 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_I32 &&
            op->src[2]->type == GGML_TYPE_I32 && op->src[0]->ne[0] == n_expert && (size_t) n_expert*sizeof(float) <= 48*1024 &&
            ggml_get_op_params_i32(op, 11)*op->src[0]->ne[1] <= MOE_PF_MAX_IDS;
    }
    if (mode == 6 || mode == 7) {
#if defined(MOE_MAILBOX_OK)
        if (mode == 6) {
            return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[1] <= GGML_MOE_MAILBOX_MAX_TOKENS;
        }
        return op->src[0]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32;
#else
        return false;
#endif
    }
    if (mode == 1) {
        return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 && op->src[2]->type == GGML_TYPE_I32 &&
            op->src[3]->type == GGML_TYPE_I32 && ggml_nelements(op->src[0]) <= 4096 &&
            (op->src[4] == nullptr || op->src[4]->type == GGML_TYPE_I32);
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
    const ggml_tensor * scores = op->src[3 + n_banks];
    return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 && op->src[2]->type == GGML_TYPE_I32 &&
        n_ids <= 4096 && (scores == nullptr || scores->type == GGML_TYPE_I32);
}

// max_z > 0: at most max_z blocks per expert and bank (each copies several chunks), so that an overlapped fetch, which mostly waits
// for host memory, leaves the SMs to the kernels that run beside it
static void moe_cache_fetch_launch(cudaStream_t stream, const ggml_tensor * dst, const ggml_tensor * const * banks, const int n_banks, const int32_t * info, const int n_ids, const int ne0,
        const int max_z = 0, int32_t * admitted = nullptr) {
    moe_cache_fetch_args args = {};
    int64_t max_nb = 0;
    for (int b = 0; b < n_banks; ++b) {
        const ggml_tensor * bank = banks[b];
        memcpy(&args.src[b], (const int32_t *) dst->op_params + 2 + 2*b, sizeof(void *));
        args.dst[b] = (char *) bank->data;
        args.nb[b]  = bank->nb[2];
        max_nb = std::max(max_nb, (int64_t) bank->nb[2]);
    }

    // GGML_CUDA_MOE_FETCH_CHUNK: bytes per block of the copies (default 64 KiB); smaller chunks put more blocks, and so more host
    // memory reads, in flight at once (the copy is the same, only the split of the work changes)
    static const int64_t chunk_env = [] {
        const char * env = getenv("GGML_CUDA_MOE_FETCH_CHUNK");
        const int64_t v = env != nullptr ? atoll(env) : 0;
        return v >= 4096 ? (v/4096)*4096 : (int64_t) 64*1024;
    }();
    const int64_t chunk = max_z > 0 ? (int64_t) 64*1024 : chunk_env;
    int64_t n_z = (max_nb + chunk - 1)/chunk;
    if (max_z > 0) {
        n_z = std::min<int64_t>(n_z, max_z);
    }
    const dim3 grid(n_ids, n_banks, (unsigned) std::max<int64_t>(n_z, 1));
    moe_cache_fetch<<<grid, 256, 0, stream>>>(info, n_ids, ne0, args, chunk, admitted);
    CUDA_CHECK(cudaGetLastError());
}

// k19-g4pref: the n_top experts with the largest logits of each token (ties: the lower expert id, NaN read as -inf), rank-major:
// ids[r*n_tokens + t] is the expert of rank r of token t, so the assign meets every token's first choices before any second one
#define MOE_PF_THREADS 256
static __global__ void moe_pf_topn(const float * __restrict__ logits, const int n_expert, const int n_top, const int n_tokens,
        int32_t * __restrict__ ids) {
    extern __shared__ float pf_v[]; // [n_expert]
    __shared__ float bv[MOE_PF_THREADS/32];
    __shared__ int   bi[MOE_PF_THREADS/32];
    const int t = blockIdx.x;
    const float * x = logits + (int64_t) t*n_expert;
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) {
        const float v = x[i];
        pf_v[i] = v == v ? v : -INFINITY;
    }
    __syncthreads();
    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;
    for (int r = 0; r < n_top; ++r) {
        float best_v = -INFINITY;
        int   best_i = INT_MAX;
        for (int i = threadIdx.x; i < n_expert; i += blockDim.x) {
            const float v = pf_v[i];
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, best_v, o, 32);
            const int   i2 = __shfl_xor_sync(0xffffffffu, best_i, o, 32);
            if (v2 > best_v || (v2 == best_v && i2 < best_i)) {
                best_v = v2;
                best_i = i2;
            }
        }
        if (lane == 0) {
            bv[wid] = best_v;
            bi[wid] = best_i;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = bv[0];
            int   k = bi[0];
            for (int w = 1; w < (int) (blockDim.x/32); ++w) {
                if (bv[w] > v || (bv[w] == v && bi[w] < k)) {
                    v = bv[w];
                    k = bi[w];
                }
            }
            k = k < n_expert ? k : 0; // (n_top <= n_expert: some expert is left)
            ids[r*n_tokens + t] = k;
            pf_v[k] = NAN; // taken: NaN compares false, so it is never picked again (input NaN was read as -inf)
        }
        __syncthreads();
    }
}

// per backend context: the copy lists of each layer's prefetch and whether one is waiting for its join; per cache state (group): the
// slot marks, the eviction stamps and the counters
struct moe_pf_group {
    int                  device = -1;
    int32_t *            mark = nullptr;
    int32_t *            ev   = nullptr;
    unsigned long long * stat = nullptr;
    int64_t              n_slots = 0;
    unsigned long long   skipped = 0;
};
struct moe_pf_ctx {
    int32_t * buf     = nullptr; // [n_layer][MOE_PF_STRIDE]: ids, then the lookup result of a prefetch
    int       n_layer = 0;
    bool      pending = false;   // the copies of a prefetch were recorded in moe_join and nothing waited for them yet
    int       last_il = -1;      // the layer of the last real lookup / assign
};
#define MOE_PF_STRIDE 1024
static std::mutex moe_pf_mutex;
static std::unordered_map<const void *, moe_pf_group> moe_pf_groups; // key: the state's device pointer
static std::unordered_map<const ggml_backend_cuda_context *, moe_pf_ctx> moe_pf_ctxs;

// the counters at exit: registered with atexit at the first prefetch, after the CUDA runtime's own exit handlers, so it runs before them
static void moe_pf_report() {
    {
        for (auto & [key, g] : moe_pf_groups) {
            unsigned long long c[MOE_PF_STAT_N] = {};
            if (g.stat == nullptr || cudaSetDevice(g.device) != cudaSuccess ||
                    cudaMemcpy(c, g.stat, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) {
                continue;
            }
            c[MOE_PF_STAT_SKIPPED] = g.skipped;
            fprintf(stderr, "ggml_cuda: k19-g4pref: device %d (%lld slots): %llu prefetch layers, %llu predicted cached, %llu prefetch fills "
                "(%llu capped), %llu useful, %llu wasted (%llu by prefetches, %llu by lookups), %llu misses of prefetch-evicted experts, "
                "%llu lookups with %llu fills, %llu prefetch ops skipped\n", g.device, (long long) g.n_slots,
                c[MOE_PF_STAT_LAYERS], c[MOE_PF_STAT_CACHED], c[MOE_PF_STAT_FILLS], c[MOE_PF_STAT_CAPPED], c[MOE_PF_STAT_USEFUL],
                c[MOE_PF_STAT_WASTED_PF] + c[MOE_PF_STAT_WASTED_REAL], c[MOE_PF_STAT_WASTED_PF], c[MOE_PF_STAT_WASTED_REAL],
                c[MOE_PF_STAT_EVMISS], c[MOE_PF_STAT_REAL], c[MOE_PF_STAT_REAL_FILLS], c[MOE_PF_STAT_SKIPPED]);
        }
    }
}

// the prefetch arguments of a real lookup / assign of the group of state (all null without prefetches)
static moe_pf_args moe_pf_real_args(const ggml_tensor * state) {
    if (moe_pf_groups.empty()) {
        return moe_pf_args{nullptr, nullptr, nullptr, 0, 0}; // no prefetch op ever ran (LLAMA_MOE_PREFETCH off)
    }
    std::lock_guard<std::mutex> lock(moe_pf_mutex);
    const auto it = moe_pf_groups.find(state->data);
    if (it == moe_pf_groups.end()) {
        return moe_pf_args{nullptr, nullptr, nullptr, 0, 0};
    }
    return moe_pf_args{it->second.mark, it->second.ev, it->second.stat, 0, 0};
}

// asynchronous refill (see moe_rf_wait): the registered groups, key: the state's device pointer
struct moe_rf_reg {
    int       idx    = -1;
    int       device = -1;
    int32_t * h_need = nullptr; // pinned [MOE_RF_LAYERS]: the source of the need row
    int32_t * h_done = nullptr; // pinned [MOE_RF_LAYERS]: the source of each done flag
};
static std::mutex moe_rf_mutex;
static std::unordered_map<const void *, moe_rf_reg> moe_rf_regs;
static int moe_rf_count[GGML_CUDA_MAX_DEVICES] = {};

static int moe_rf_real(const ggml_tensor * state) {
    if (moe_rf_regs.empty()) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(moe_rf_mutex);
    const auto it = moe_rf_regs.find(state->data);
    return it == moe_rf_regs.end() ? -1 : it->second.idx;
}

int ggml_cuda_moe_refill_register(ggml_backend_cuda_context & ctx, const void * state_data) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    std::lock_guard<std::mutex> lock(moe_rf_mutex);
    const auto it = moe_rf_regs.find(state_data);
    if (it != moe_rf_regs.end()) {
        return it->second.idx;
    }
    if (ctx.device < 0 || ctx.device >= GGML_CUDA_MAX_DEVICES || moe_rf_count[ctx.device] >= MOE_RF_GROUPS) {
        return -1;
    }
    moe_rf_reg r;
    r.device = ctx.device;
    ggml_cuda_set_device(ctx.device);
    if (cudaMallocHost(&r.h_need, MOE_RF_LAYERS*sizeof(int32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    if (cudaMallocHost(&r.h_done, MOE_RF_LAYERS*sizeof(int32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        CUDA_CHECK(cudaFreeHost(r.h_need));
        return -1;
    }
    memset(r.h_need, 0, MOE_RF_LAYERS*sizeof(int32_t));
    memset(r.h_done, 0, MOE_RF_LAYERS*sizeof(int32_t));
    r.idx = moe_rf_count[ctx.device]++;
    // both rows start at 0 (nothing pending); synchronous, before any graph that reads them
    CUDA_CHECK(cudaMemcpyToSymbol(moe_rf_tab, r.h_need, MOE_RF_LAYERS*sizeof(int32_t), (size_t) (2*r.idx + 0)*MOE_RF_LAYERS*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(moe_rf_tab, r.h_done, MOE_RF_LAYERS*sizeof(int32_t), (size_t) (2*r.idx + 1)*MOE_RF_LAYERS*sizeof(int32_t), cudaMemcpyHostToDevice));
    const int idx = r.idx;
    moe_rf_regs[state_data] = r;
    return idx;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(state_data);
    return -1;
#endif
}

// the layers il[0..n) get a refill of this epoch: queued on the compute stream, before the graphs that read the slots
bool ggml_cuda_moe_refill_need(ggml_backend_cuda_context & ctx, const void * state_data, const int32_t * il, const int n, const int32_t epoch) {
    moe_rf_reg r;
    {
        std::lock_guard<std::mutex> lock(moe_rf_mutex);
        const auto it = moe_rf_regs.find(state_data);
        if (it == moe_rf_regs.end() || it->second.device != ctx.device) {
            return false;
        }
        r = it->second;
    }
    for (int i = 0; i < n; ++i) {
        if (il[i] < 0 || il[i] >= MOE_RF_LAYERS) {
            return false;
        }
    }
    // the previous copy of this row completed: the caller synchronized the compute stream since it was queued
    for (int i = 0; i < n; ++i) {
        r.h_need[il[i]] = epoch;
    }
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaMemcpyToSymbolAsync(moe_rf_tab, r.h_need, MOE_RF_LAYERS*sizeof(int32_t), (size_t) (2*r.idx + 0)*MOE_RF_LAYERS*sizeof(int32_t),
        cudaMemcpyHostToDevice, ctx.stream()));
    return true;
}

// the uploads of layer il queued before this on the stream of ctx (the upload stream) complete the refill of this epoch there
bool ggml_cuda_moe_refill_done(ggml_backend_cuda_context & ctx, const void * state_data, const int il, const int32_t epoch) {
    moe_rf_reg r;
    {
        std::lock_guard<std::mutex> lock(moe_rf_mutex);
        const auto it = moe_rf_regs.find(state_data);
        if (it == moe_rf_regs.end() || it->second.device != ctx.device || il < 0 || il >= MOE_RF_LAYERS) {
            return false;
        }
        r = it->second;
    }
    // the previous copy of this flag completed: the caller waited for the upload stream since it was queued
    r.h_done[il] = epoch;
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaMemcpyToSymbolAsync(moe_rf_tab, r.h_done + il, sizeof(int32_t), ((size_t) (2*r.idx + 1)*MOE_RF_LAYERS + il)*sizeof(int32_t),
        cudaMemcpyHostToDevice, ctx.stream()));
    return true;
}

// before a cache op other than a prefetch: the main stream waits for the copies of a prefetch (the next layer's lookup reads its
// slots as cached). moe_join_pending is cleared at the end of every graph, so a prefetch of another graph is never waited for here
static void moe_pf_join(ggml_backend_cuda_context & ctx, const int il) {
    if (moe_pf_ctxs.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(moe_pf_mutex);
    const auto it = moe_pf_ctxs.find(&ctx);
    if (it == moe_pf_ctxs.end()) {
        return;
    }
    if (it->second.pending && ctx.moe_join_pending) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), ctx.moe_join, 0));
        ctx.moe_join_pending = false;
    }
    it->second.pending = false;
    if (il >= 0) {
        it->second.last_il = il;
    }
}

static void moe_pf_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits = dst->src[0];
    ggml_tensor       * map    = dst->src[1];
    ggml_tensor       * state  = dst->src[2];
    const int il       = ggml_get_op_params_i32(dst, 0);
    const int n_banks  = ggml_get_op_params_i32(dst, 1);
    const int n_top    = ggml_get_op_params_i32(dst, 11);
    const int max_fill = ggml_get_op_params_i32(dst, 12);
    const int n_core   = ggml_get_op_params_i32(dst, 13);
    const ggml_tensor * scores = dst->src[4 + n_banks];
    const int n_expert = (int) map->ne[0];
    const int n_layer  = (int) map->ne[1];
    const int n_tokens = (int) logits->ne[1];
    const int n_slots  = (int) ((ggml_nelements(state) - MOE_CACHE_HDR)/3);
    const int n_ids    = n_top*n_tokens;
    if (n_tokens <= 0 || n_top > n_expert || n_ids > MOE_PF_MAX_IDS || 3*n_ids + n_tokens > MOE_PF_STRIDE - MOE_PF_MAX_IDS) {
        return;
    }

    std::lock_guard<std::mutex> lock(moe_pf_mutex);
    moe_pf_group & g = moe_pf_groups[state->data];
    if (g.stat == nullptr) {
        // first use (also inside a capture: the streams are non-blocking, the capture is relaxed)
        static const bool reg = std::atexit(moe_pf_report) == 0;
        GGML_UNUSED(reg);
        g.device  = ctx.device;
        g.n_slots = n_slots;
        CUDA_CHECK(cudaMalloc(&g.mark, (size_t) n_slots*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&g.ev, (size_t) ggml_nelements(map)*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&g.stat, MOE_PF_STAT_N*sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(g.mark, 0, (size_t) n_slots*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(g.ev, 0, (size_t) ggml_nelements(map)*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(g.stat, 0, MOE_PF_STAT_N*sizeof(unsigned long long)));
    }
    moe_pf_ctx & p = moe_pf_ctxs[&ctx];
    if (p.last_il >= il) {
        // the layer's own lookup (or a later one) ran before this op in the graph: its slots may be read now, nothing is admitted
        g.skipped++;
        return;
    }
    if (p.buf == nullptr || p.n_layer < n_layer) {
        if (p.buf != nullptr) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(p.buf));
        }
        CUDA_CHECK(cudaMalloc(&p.buf, (size_t) n_layer*MOE_PF_STRIDE*sizeof(int32_t)));
        p.n_layer = n_layer;
    }
    if (ctx.moe_fork == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.moe_fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.moe_join, cudaEventDisableTiming));
    }
    int32_t * ids = p.buf + (size_t) il*MOE_PF_STRIDE;
    int32_t * res = ids + MOE_PF_MAX_IDS;

    cudaStream_t stream = ctx.stream();
    cudaStream_t side   = ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1);
    CUDA_CHECK(cudaEventRecord(ctx.moe_fork, stream));
    CUDA_CHECK(cudaStreamWaitEvent(side, ctx.moe_fork, 0));
    moe_pf_topn<<<n_tokens, MOE_PF_THREADS, (size_t) n_expert*sizeof(float), side>>>((const float *) logits->data, n_expert, n_top,
        n_tokens, ids);
    CUDA_CHECK(cudaGetLastError());
    const moe_pf_args pa = { g.mark, g.ev, g.stat, 1, max_fill };
    const int n_scores = scores != nullptr ? (int) ggml_nelements(scores) : 0;
    moe_cache_assign<<<1, MOE_CACHE_LANES, (2*n_ids + (scores != nullptr ? 2*MOE_CACHE_WINDOW : 0))*sizeof(int32_t), side>>>(
        ids, (int32_t *) map->data, (int32_t *) state->data, res, n_ids, n_tokens, n_expert, il, n_slots, nullptr, 0, 0,
        scores != nullptr ? (int32_t *) scores->data : nullptr, 65536 + 1, 0, n_scores, 0, 0, nullptr, n_tokens, 0, 0, 0, 0, n_core, pa, moe_rf_real(state));
    CUDA_CHECK(cudaGetLastError());
    // GGML_CUDA_MOE_PREFETCH_Z: blocks per expert and bank of the copies (default: GGML_CUDA_MOE_FETCH_OVERLAP_Z's default, 2)
    static const int max_z = [] {
        const char * env = getenv("GGML_CUDA_MOE_PREFETCH_Z");
        return env != nullptr ? std::max(0, atoi(env)) : 2;
    }();
    moe_cache_fetch_launch(side, dst, dst->src + 3, n_banks, res, n_ids, n_tokens, max_z, nullptr);
    CUDA_CHECK(cudaEventRecord(ctx.moe_join, side));
    ctx.moe_join_pending = true;
    p.pending = true;
}

// single-graph host misses (k11, see ggml_moe_mailbox_publish): flags of the mapped mailbox at system scope
#if defined(MOE_MAILBOX_OK)
static __device__ __forceinline__ uint32_t moe_mailbox_load_acquire(const uint32_t * p) {
    uint32_t v;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
#else
    v = *(const volatile uint32_t *) p;
    __threadfence_system();
#endif
    return v;
}

static __device__ __forceinline__ void moe_mailbox_store_release(uint32_t * p, const uint32_t v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
#else
    __threadfence_system();
    *(volatile uint32_t *) p = v;
#endif
}

static __device__ __forceinline__ uint64_t moe_mailbox_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// k19-overlap GGML_MOE_MAILBOX_TIMING=1 (diagnostic, off by default): the publish notes the device time and the merge adds the device
// work beside the host and its wait to ggml_moe_mailbox_hdr::tm, which llama_moe_cache reports per layer at exit
static int ggml_moe_mailbox_timing() {
    static const int on = [] {
        const char * e = getenv("GGML_MOE_MAILBOX_TIMING");
        return e != nullptr && atoi(e) != 0 ? 1 : 0;
    }();
    return on;
}

// one block: copies the host ids and the activations when any id is on the host, then publishes the next sequence
static __global__ void moe_mailbox_publish(const int32_t * hid, const int n_ids, const int n_tokens, const float * x, const int n_x,
        char * mbox, const int n_used, const int n_embd, ggml_moe_mailbox_ctl * ctl, int32_t * dst, const int timing) {
    __shared__ int cnt;
    if (threadIdx.x == 0) {
        cnt = 0;
    }
    __syncthreads();
    int local = 0;
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        local += hid[i] >= 0;
    }
    if (local > 0) {
        atomicAdd(&cnt, local);
    }
    __syncthreads();
    if (cnt == 0) {
        if (threadIdx.x == 0) {
            ctl->pending = 0;
            dst[0] = 0;
        }
        return;
    }
    int32_t * m_ids = (int32_t *) (mbox + ggml_moe_mailbox_off_ids());
    float   * m_x   = (float   *) (mbox + ggml_moe_mailbox_off_x(n_used));
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        m_ids[i] = hid[i];
    }
    for (int i = threadIdx.x; i < n_x; i += blockDim.x) {
        m_x[i] = x[i];
    }
    ggml_moe_mailbox_hdr * hdr = (ggml_moe_mailbox_hdr *) mbox;
    if (threadIdx.x == 0) {
        hdr->n_tokens = n_tokens;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t seq = ctl->seq + 1;
        ctl->seq     = seq;
        ctl->pending = 1;
        dst[0]       = 1;
        if (timing) {
            ((volatile uint64_t *) hdr->tm)[6] = moe_mailbox_ns();
        }
        moe_mailbox_store_release(&hdr->request, seq);
        if (timing) {
            ctl->t_pub = moe_mailbox_ns();
        }
    }
    GGML_UNUSED(n_embd);
}

// one block per row [n_embd] of dev: rows on the host wait for the host and add its result
// k18 (GGML_MOE_MAILBOX_FAST, default on; =0 restores the loop below): with no timeout (what llama_moe_cache passes) only completed is
// polled, error every 16th poll and no sleep, so a wait costs one mapped read per poll instead of four plus 100 ns; the host rows are
// read 16 bytes at a time (a quarter of the mapped reads). The same sums of the same values, in the same order per element.
static __global__ void moe_mailbox_merge(const float * d, const int32_t * hid, const int n_embd, const char * mbox, const int n_used,
        ggml_moe_mailbox_ctl * ctl, float * dst, const uint64_t timeout_ns, const int fast, const int timing) {
    const int row = blockIdx.x;
    __shared__ int st;
    if (threadIdx.x == 0) {
        int s = 0;
        if (ctl->pending) {
            ggml_moe_mailbox_hdr * hdr = (ggml_moe_mailbox_hdr *) mbox;
            const uint32_t seq = ctl->seq;
            if (hid[row] >= 0) {
                // timeout_ns 0 (what llama_moe_cache passes): wait for the host as long as it takes, so the result is the split's; a
                // nonzero timeout runs only while the host thread neither computes this batch (started) nor another one (busy)
                uint64_t t0 = moe_mailbox_ns();
                const uint64_t t_in = t0;
                s = 1;
                if (fast && timeout_ns == 0) {
                    for (uint32_t it = 1; moe_mailbox_load_acquire(&hdr->completed) != seq; ++it) {
                        if ((it & 15) == 0 && moe_mailbox_load_acquire(&hdr->error) != 0) {
                            s = 2;
                            break;
                        }
                    }
                }
                while (s == 1 && moe_mailbox_load_acquire(&hdr->completed) != seq) {
                    if (moe_mailbox_load_acquire(&hdr->error) != 0) {
                        s = 2;
                        break;
                    }
                    if (moe_mailbox_load_acquire(&hdr->started) == seq || moe_mailbox_load_acquire(&hdr->busy) != 0) {
                        t0 = moe_mailbox_ns();
                    } else if (timeout_ns != 0 && moe_mailbox_ns() - t0 > timeout_ns) {
                        moe_mailbox_store_release(&hdr->error, 1);
                        ctl->failed = 1;
                        s = 2;
                        break;
                    }
                    __nanosleep(100);
                }
                // k19-overlap GGML_MOE_MAILBOX_TIMING: the block of the batch's first host row adds this batch to the sums
                bool first = timing != 0 && s == 1;
                for (int r = 0; first && r < row; ++r) {
                    first = hid[r] < 0;
                }
                if (first) {
                    const uint64_t t_out = moe_mailbox_ns();
                    const uint64_t t_pub = ctl->t_pub;
                    const uint64_t dev   = t_in > t_pub ? t_in - t_pub : 0;
                    const uint64_t wait  = t_out - t_in;
                    ctl->sums[0] += 1;
                    ctl->sums[1] += dev;
                    ctl->sums[2] += wait;
                    ctl->sums[3] += wait > 2000;
                    ctl->sums[4] += dev + wait;
                    volatile uint64_t * tm = hdr->tm;
                    // signed: the two clocks are apart by an offset (sums wrap as two's complement)
                    const int64_t d2 = (int64_t) (t_out - tm[7]);
                    ctl->sums[5] += (uint64_t) d2;
                    if (ctl->sums[0] == 1 || d2 < ctl->min_d2) {
                        ctl->min_d2 = d2;
                    }
                    for (int k = 0; k < 6; ++k) {
                        tm[k] = ctl->sums[k];
                    }
                    tm[8] = (uint64_t) ctl->min_d2;
                }
            }
        }
        if (ctl->failed) {
            s = 2;
        }
        st = s;
    }
    __syncthreads();
    const float * h = (const float *) (mbox + ggml_moe_mailbox_off_out(n_used, n_embd)) + (int64_t) row*n_embd;
    const float * dr = d + (int64_t) row*n_embd;
    float * o = dst + (int64_t) row*n_embd;
    if (fast && st == 1 && n_embd % 4 == 0 && ((((uintptr_t) h) | ((uintptr_t) dr) | ((uintptr_t) o)) & 15) == 0) {
        const float4 * h4 = (const float4 *) h;
        const float4 * d4 = (const float4 *) dr;
        float4       * o4 = (float4 *) o;
        for (int i = threadIdx.x; i < n_embd/4; i += blockDim.x) {
            const float4 hv = __ldcv(h4 + i);
            const float4 dv = d4[i];
            o4[i] = make_float4(dv.x + hv.x, dv.y + hv.y, dv.z + hv.z, dv.w + hv.w);
        }
        return;
    }
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        // a merge that gave up keeps the device rows only: finite values, since the ubatch is taken back or failed anyway (see
        // single_check) and NaN rows would stay in the KV cells it wrote, which masked attention of later requests still reads
        o[i] = st == 1 ? dr[i] + __ldcv(h + i) : dr[i];
    }
}
#endif

void ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 10);

#if defined(MOE_MAILBOX_OK)
    if (mode == 6 || mode == 7) {
        void * mbox = nullptr;
        void * ctl  = nullptr;
        memcpy(&mbox, (const int32_t *) dst->op_params + 2, sizeof(void *));
        memcpy(&ctl,  (const int32_t *) dst->op_params + 4, sizeof(void *));
        if (mode == 6) {
            const ggml_tensor * hid = dst->src[0];
            const ggml_tensor * x   = dst->src[1];
            const int n_used   = (int) hid->ne[0];
            const int n_tokens = (int) hid->ne[1];
            GGML_ASSERT(n_tokens <= GGML_MOE_MAILBOX_MAX_TOKENS && ggml_is_contiguous(hid) && ggml_is_contiguous(x));
            moe_mailbox_publish<<<1, 256, 0, ctx.stream()>>>((const int32_t *) hid->data, (int) ggml_nelements(hid), n_tokens,
                (const float *) x->data, (int) ggml_nelements(x), (char *) mbox, n_used, (int) (ggml_nelements(x)/n_tokens),
                (ggml_moe_mailbox_ctl *) ctl, (int32_t *) dst->data, ggml_moe_mailbox_timing());
        } else {
            const ggml_tensor * d   = dst->src[0];
            const ggml_tensor * hid = dst->src[2];
            const int n_embd = (int) d->ne[0];
            const int n_rows = (int) (ggml_nelements(d)/n_embd);
            GGML_ASSERT(ggml_is_contiguous(d) && ggml_is_contiguous(dst) && n_rows == (int) ggml_nelements(hid));
            const uint64_t timeout_ns = (uint64_t) std::max(0, ggml_get_op_params_i32(dst, 6))*1000000ull; // 0: no timeout
            static const int fast = [] {
                const char * e = getenv("GGML_MOE_MAILBOX_FAST");
                return e == nullptr || atoi(e) != 0 ? 1 : 0;
            }();
            moe_mailbox_merge<<<n_rows, 256, 0, ctx.stream()>>>((const float *) d->data, (const int32_t *) hid->data, n_embd,
                (const char *) mbox, (int) d->ne[1], (ggml_moe_mailbox_ctl *) ctl, (float *) dst->data, timeout_ns, fast,
                ggml_moe_mailbox_timing());
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif

    if (mode == 8) {
        moe_pf_run(ctx, dst); // k19-g4pref, see ggml_moe_cache_prefetch
        return;
    }
    if (mode != 3) {
        moe_pf_join(ctx, mode == 0 || mode == 1 ? ggml_get_op_params_i32(dst, 0) : -1);
    }

    if (mode == 4) {
        // device seed plan, see ggml_moe_cache_seed_plan
        const ggml_tensor * ids    = dst->src[0];
        ggml_tensor       * map    = dst->src[1];
        ggml_tensor       * state  = dst->src[2];
        ggml_tensor       * scores = dst->src[3];
        const int il         = ggml_get_op_params_i32(dst, 0);
        const int n_seed_max = ggml_get_op_params_i32(dst, 2);
        const int pos        = ggml_get_op_params_i32(dst, 3);
        const int n_expert   = (int) map->ne[0];
        const int n_slots    = (int) ((ggml_nelements(state) - MOE_CACHE_HDR)/3);
        const int n_ids      = (int) ggml_nelements(ids);
        const int n_seed     = n_ids > 0 ? std::min(ggml_get_op_params_i32(dst, 4), std::min(n_seed_max, n_expert)) : 0;
        const int n_victim   = std::min(ggml_get_op_params_i32(dst, 5), n_slots);
        GGML_ASSERT(ggml_is_contiguous(ids) && ggml_is_contiguous(dst) && ggml_nelements(dst) == 1 + 2*(int64_t) n_seed_max);
        const size_t smem = (size_t) (n_expert + 3*std::max(n_seed, 1) + MOE_SEED_THREADS)*sizeof(int32_t);
        moe_cache_seed_plan<<<1, MOE_SEED_THREADS, smem, ctx.stream()>>>(
            (const int32_t *) ids->data, n_ids, (int32_t *) map->data, (int32_t *) state->data,
            scores != nullptr ? (int32_t *) scores->data : nullptr, (int32_t *) dst->data, n_expert, il, n_slots,
            n_victim, n_seed, n_seed_max, pos, ggml_get_op_params_i32(dst, 7));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (mode == 5) {
        // device seed fill: from the experts mm read (src[0] of mm when it ran, the device copy of host experts) to the bank
        const ggml_tensor * plan = dst->src[0];
        const ggml_tensor * mm   = dst->src[1];
        ggml_tensor       * bank = dst->src[2];
        // the scheduler replaced the product by a copy (it runs on another backend): the host disabled the plan (see devseed_prepare)
        if (mm->op != GGML_OP_MUL_MAT_ID || mm->src[0] == nullptr || mm->src[0]->data == nullptr) {
            return;
        }
        const ggml_tensor * w = mm->src[0];
        const int n_seed_max = (int) ((ggml_nelements(plan) - 1)/2);
        GGML_ASSERT(w->nb[2] == bank->nb[2] && w->type == bank->type);
        const int64_t nb    = bank->nb[2];
        const int64_t chunk = 64*1024;
        const dim3 grid(n_seed_max, 1, (unsigned) std::max<int64_t>((nb + chunk - 1)/chunk, 1));
        moe_cache_seed_fill<<<grid, 256, 0, ctx.stream()>>>((const int32_t *) plan->data, n_seed_max, (const char *) w->data,
            (char *) bank->data, nb, chunk);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

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
        if (ggml_get_op_params_i32(dst, 12) == 2) {
            // k19-l4c deferred admission (ggml_moe_cache_set_defer): the host computes the admitted experts of this batch, the copies only
            // fill their slots for later batches. They run on the side stream without a join in the graph: the end of the graph waits
            // for them (moe_join_pending, see ggml_backend_cuda_graph_compute). The assign result is a graph tensor whose memory the
            // nodes after this one may reuse, so the copy list is first copied to a buffer of the context, one entry per layer (the
            // previous graph's copies, which read that entry, were waited for at its end); the result (dst) is neither read nor written
            const int il     = ggml_get_op_params_i32(dst, 0);
            const int n_info = 3*n_ids + ne0; // slots, fill count row, fill experts, fill slots
            cudaStream_t side = ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1);
            if (ctx.moe_defer_info == nullptr) {
                CUDA_CHECK(cudaMalloc(&ctx.moe_defer_info, (size_t) GGML_CUDA_MOE_DEFER_LAYERS*GGML_CUDA_MOE_DEFER_STRIDE*sizeof(int32_t)));
            }
            if (ctx.moe_fork == nullptr) {
                CUDA_CHECK(cudaEventCreateWithFlags(&ctx.moe_fork, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&ctx.moe_join, cudaEventDisableTiming));
            }
            if (il < 0 || il >= GGML_CUDA_MOE_DEFER_LAYERS || n_info > GGML_CUDA_MOE_DEFER_STRIDE) {
                // no entry for it: the copies run here in the stream (the host still computes the admitted experts: same output)
                moe_cache_fetch_launch(stream, dst, dst->src + 1, ggml_get_op_params_i32(dst, 1), (const int32_t *) res->data, n_ids, ne0);
                return;
            }
            int32_t * info = ctx.moe_defer_info + (size_t) il*GGML_CUDA_MOE_DEFER_STRIDE;
            CUDA_CHECK(cudaMemcpyAsync(info, res->data, n_info*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaEventRecord(ctx.moe_fork, stream));
            CUDA_CHECK(cudaStreamWaitEvent(side, ctx.moe_fork, 0));
            static const int max_z = [] {
                const char * env = getenv("GGML_CUDA_MOE_FETCH_OVERLAP_Z");
                return env != nullptr ? std::max(0, atoi(env)) : 2;
            }();
            moe_cache_fetch_launch(side, dst, dst->src + 1, ggml_get_op_params_i32(dst, 1), info, n_ids, ne0, max_z, nullptr);
            CUDA_CHECK(cudaEventRecord(ctx.moe_join, side));
            ctx.moe_join_pending = true;
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
        // GGML_CUDA_MOE_FETCH_OVERLAP_Z: blocks per expert and bank of the overlapped copies (0 = one per 64 KiB chunk); the copies
        // wait for host memory, a few blocks keep the link busy and leave the SMs to the experts computed beside them
        static const int max_z = [] {
            const char * env = getenv("GGML_CUDA_MOE_FETCH_OVERLAP_Z");
            return env != nullptr ? std::max(0, atoi(env)) : 2;
        }();
        moe_cache_fetch_launch(side, dst, dst->src + 1, ggml_get_op_params_i32(dst, 1), (const int32_t *) res->data, n_ids, ne0, max_z,
            (int32_t *) dst->data);
        CUDA_CHECK(cudaEventRecord(ctx.moe_join, side));
        ctx.moe_join_pending = true;
        return;
    }

    if (mode == 3) {
        // join of an overlapped fetch: the main stream waits for its copies, the result is a view of its admitted slots
        if (ggml_nelements(dst) == 0) {
            return; // nothing was fetched (see mode 2)
        }
        GGML_ASSERT(ctx.moe_join != nullptr && dst->view_src == dst->src[0] && dst->data == dst->src[0]->data);
        if (!ctx.moe_join_pending) {
            // the fetch was in another split: the end of its graph already made this stream wait for the copies (see
            // ggml_cuda_graph_evaluate_and_capture), and that wait is part of its CUDA graph, so a replay keeps it. Waiting on
            // moe_join again here would wait on an event last recorded in that other graph's capture, which CUDA rejects while
            // this split is captured (invalid argument)
            return;
        }
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), ctx.moe_join, 0));
        ctx.moe_join_pending = false;
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

    // decayed counts (ggml_moe_cache_set_scores): after the banks (lookup) or the counters (assign)
    const ggml_tensor * scores = dst->src[mode == 1 ? 4 : 3 + n_banks];
    const int grow     = ggml_get_op_params_i32(dst, 13);
    const int advance  = ggml_get_op_params_i32(dst, 14);
    const int n_scores = scores != nullptr ? (int) ggml_nelements(scores) : 0;
    GGML_ASSERT(scores == nullptr || (ggml_is_contiguous(scores) && ggml_nelements(scores) == ggml_nelements(map) && grow > 65536 && grow <= 2*65536));

    GGML_ASSERT(ids->nb[0] == sizeof(int32_t) && ggml_is_contiguous(dst));
    const int ids_s1 = (int) (ids->nb[1]/sizeof(int32_t)); // k15-fn: rows of ids may be strided

    if (n_ids == 0) {
        return; // for example the last layer of a ubatch without outputs
    }

    cudaStream_t stream = ctx.stream();

    // route tail skip (ggml_moe_cache_set_tail_skip): op param 12 = rank | default << 8. The default is taken on NVIDIA and on RDNA4
    // (gfx1200 / gfx1201); an explicit rank everywhere
    int tail_rank = ggml_get_op_params_i32(dst, 12) & 0xff;
    if (tail_rank > 0 && (ggml_get_op_params_i32(dst, 12) >> 8) & 1) {
#if defined(GGML_USE_HIP)
        if (!GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ctx.device].cc)) {
            tail_rank = 0;
        }
#endif // defined(GGML_USE_HIP)
    }

    // k19-ramevict: LLAMA_MOE_CACHE_ADMIT_MARGIN=M (e.g. 1.25; unset / 0 = off) and LLAMA_MOE_CACHE_ADMIT_ADD=A (share of one use's weight,
    // default 0): the admission margin of the hybrid assign with decayed counts (see moe_cache_assign)
    static const int admit_margin_q = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_ADMIT_MARGIN");
        const double m = e != nullptr ? atof(e) : 0.0;
        return m > 0.0 ? (int) lrint(std::min(m, 16.0)*65536.0) : 0;
    }();
    static const int admit_add_q = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_ADMIT_ADD");
        const double a = e != nullptr ? atof(e) : 0.0;
        return a > 0.0 ? (int) lrint(std::min(a, 16.0)*65536.0) : 0;
    }();

    if (mode == 1) {
        // hybrid assign: no copies here, see ggml_moe_cache_fetch
        ggml_tensor * counts = dst->src[3];
        float fetch;
        memcpy(&fetch, (const int32_t *) dst->op_params + 11, sizeof(float));
        const int fetch_q = (int) lrintf(fetch*65536.0f);
        GGML_ASSERT(ggml_is_contiguous(counts) && fetch_q >= 0 && fetch_q <= 65536);
        moe_cache_assign<<<1, MOE_CACHE_LANES, (3*n_ids + (scores != nullptr ? 2*MOE_CACHE_WINDOW : 0))*sizeof(int32_t), stream>>>(
            (const int32_t *) ids->data, (int32_t *) map->data, (int32_t *) state->data, (int32_t *) dst->data,
            n_ids, ne0, n_expert, il, n_slots, (int32_t *) counts->data, (int) map->ne[1], fetch_q,
            scores != nullptr ? (int32_t *) scores->data : nullptr, grow, advance, n_scores, ggml_get_op_params_i32(dst, 15),
            tail_rank, tail_rank > 0 ? (int32_t *) counts->data + (int64_t) map->ne[1]*n_expert + 4 : nullptr, ids_s1,
            ggml_get_op_params_i32(dst, 9), n_expert >= 8 ? admit_margin_q : 0, admit_add_q, ggml_get_op_params_i32(dst, 8),
            ggml_get_op_params_i32(dst, 7), moe_pf_real_args(state), moe_rf_real(state));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // the use counters of a lookup with the route tail skip (statistics only), after the banks and the scores
    const ggml_tensor * skip_counts = dst->src[4 + n_banks];
    GGML_ASSERT(skip_counts == nullptr || (ggml_is_contiguous(skip_counts) && skip_counts->ne[1] == map->ne[1] + 1));
    moe_cache_assign<<<1, MOE_CACHE_LANES, (2*n_ids + (scores != nullptr ? 2*MOE_CACHE_WINDOW : 0))*sizeof(int32_t), stream>>>(
        (const int32_t *) ids->data, (int32_t *) map->data, (int32_t *) state->data, (int32_t *) dst->data,
        n_ids, ne0, n_expert, il, n_slots, nullptr, 0, 0,
        scores != nullptr ? (int32_t *) scores->data : nullptr, grow, advance, n_scores, 0,
        tail_rank, tail_rank > 0 && skip_counts != nullptr ? (int32_t *) skip_counts->data + (int64_t) map->ne[1]*n_expert + 4 : nullptr, ids_s1,
        0, 0, 0, 0, n_banks <= 3 ? ggml_get_op_params_i32(dst, 9) : 0, // core: op param 9 (host pointers of a 4th bank otherwise)
        moe_pf_real_args(state), moe_rf_real(state));
    CUDA_CHECK(cudaGetLastError());

    moe_cache_fetch_launch(stream, dst, dst->src + 3, n_banks, (const int32_t *) dst->data, n_ids, ne0);
}
