// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "argsort.cuh"
#include "top-k.cuh"
#include "cpy.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered_legacy(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// -0.0 is mapped to +0.0 so that the two compare equal, as they do for the bitonic and CUB sort paths
static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = value == 0.0f ? 0u : __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// Original radix select (GGML_CUDA_TOPK_DETERMINISTIC=0): the gather places keys with atomics, so the output
// order, and for keys equal to the k-th largest the selected subset, depend on thread timing.
struct top_k_radix_state_legacy {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init_legacy(top_k_radix_state_legacy * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram_legacy(
        const float * __restrict__ src,
        const top_k_radix_state_legacy * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state_legacy state = states[row];
    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered_legacy(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select_legacy(
        const int * __restrict__ block_histograms,
        top_k_radix_state_legacy * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state_legacy state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters_legacy(top_k_radix_state_legacy * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather_legacy(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state_legacy * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state_legacy * state = &states[row];

    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered_legacy(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda_legacy(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state_legacy> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state_legacy * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init_legacy<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram_legacy<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, blocks_per_row, shift);
            top_k_radix_select_legacy<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_reset_counters_legacy
            <<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows);
        top_k_radix_gather_legacy<BLOCK_SIZE>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, ncols, k, blocks_per_row);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
};

// Each block of a row owns one contiguous range of columns, the same range in every pass, so the gather can
// emit the selected columns in column order: the result does not depend on thread timing, and of the keys
// equal to the k-th largest the ones with the lowest column are kept.
static __global__ void top_k_radix_init(top_k_radix_state * states, int * block_greater, int nrows, int blocks_per_row, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nrows) {
        states[i] = {0, 0, k};
    }
    if (i < nrows * blocks_per_row) {
        block_greater[i] = 0;
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int cols_per_block,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    const int col_end = min(ncols, (row_block + 1) * cols_per_block);
    for (int col = row_block * cols_per_block + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int * __restrict__ block_greater,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;
    static_assert(NBINS == BLOCK_SIZE, "one thread per bin");

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];
    __shared__ int selected_bin;

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
        selected_bin = bin;
    }
    __syncthreads();

    // the keys of each block above the selected bin are in the top k whatever the lower bits hold
    const int bin  = selected_bin;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    for (int row_block = warp; row_block < blocks_per_row; row_block += BLOCK_SIZE / WARP_SIZE) {
        const int * block_histogram = block_histograms + ((size_t) row * blocks_per_row + row_block) * NBINS;
        int greater = 0;
        for (int b = bin + 1 + lane; b < NBINS; b += WARP_SIZE) {
            greater += block_histogram[b];
        }
        greater = warp_reduce_sum(greater);
        if (lane == 0) {
            block_greater[(size_t) row * blocks_per_row + row_block] += greater;
        }
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int * __restrict__ block_histograms,
        const int * __restrict__ block_greater,
        int ncols,
        int k,
        int blocks_per_row,
        int cols_per_block) {
    constexpr int NBINS  = 1 << RADIX_BITS;
    constexpr int NWARPS = BLOCK_SIZE / WARP_SIZE;
    static_assert(BLOCK_SIZE < (1 << 15), "two 16 bit counters per int");

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;

    const top_k_radix_state state = states[row];
    const int n_equal   = state.rank; // keys equal to the threshold that are kept
    const int n_greater = k - n_equal;

    __shared__ int warp_counts[NWARPS];
    __shared__ int block_offsets[3];

    // the lower blocks place their keys first; after the last pass the histograms hold the
    // per block count of each low byte under the full prefix, so the selected bin counts the equal keys
    if (tid == 0) {
        const int * row_greater = block_greater + (size_t) row * blocks_per_row;
        const int   bin         = state.prefix & (NBINS - 1);
        int greater = 0;
        int equal   = 0;
        for (int i = 0; i < row_block; ++i) {
            greater += row_greater[i];
            equal   += block_histograms[((size_t) row * blocks_per_row + i) * NBINS + bin];
        }
        block_offsets[0] = greater;
        block_offsets[1] = equal;
        block_offsets[2] = greater + row_greater[row_block];
    }
    __syncthreads();

    int       pos_greater = block_offsets[0];
    int       pos_equal   = block_offsets[1];
    const int end_greater = block_offsets[2];

    const int col_end = min(ncols, (row_block + 1) * cols_per_block);
    for (int col0 = row_block * cols_per_block; col0 < col_end; col0 += BLOCK_SIZE) {
        if (pos_greater >= end_greater && pos_equal >= n_equal) {
            break; // uniform: every thread holds the same counters
        }
        const int col = col0 + tid;
        const uint32_t key = col < col_end ? top_k_float_to_ordered(row_src[col]) : 0;
        const bool greater = col < col_end && key >  state.prefix;
        const bool equal   = col < col_end && key == state.prefix;

        // exclusive scan of both flags in column order, the equal count in the high half
        const int flags     = (int) greater | ((int) equal << 16);
        const int inclusive = warp_prefix_inclusive_sum<int>(flags);
        if (lane == WARP_SIZE - 1) {
            warp_counts[warp] = inclusive;
        }
        __syncthreads();

        int before = 0;
        int total  = 0;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            const int c = warp_counts[w];
            before += w < warp ? c : 0;
            total  += c;
        }
        const int exclusive = before + inclusive - flags;

        if (greater) {
            row_dst[pos_greater + (exclusive & 0xFFFF)] = col;
        } else if (equal) {
            const int pos = pos_equal + (exclusive >> 16);
            if (pos < n_equal) {
                row_dst[n_greater + pos] = col;
            }
        }
        pos_greater += total & 0xFFFF;
        pos_equal   += total >> 16;
        __syncthreads();
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);
    const int cols_per_block = ((ncols + blocks_per_row - 1) / blocks_per_row + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    ggml_cuda_pool_alloc<int> greater_alloc(pool, (size_t) chunk_nrows * blocks_per_row);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();
    int * block_greater = greater_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        const int n_init = iter_nrows * blocks_per_row;
        top_k_radix_init<<<(n_init + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            states, block_greater, iter_nrows, blocks_per_row, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, blocks_per_row, cols_per_block, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, block_greater, blocks_per_row, shift);
        }

        top_k_radix_gather<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, histograms, block_greater, ncols, k, blocks_per_row, cols_per_block);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

// k19 (idea from ik_llama.cpp, author Joel Farthing, MIT): the radix select above in one launch per call, one block per
// row. The same four 8-bit histogram passes over the same keys pick the same threshold key and the same count of kept ties, and the
// gather writes the same order: the keys above the threshold in column order, then the first kept ties in column order. So the output
// is bitwise that of top_k_radix_cuda (init + 4 x (histogram + select) + gather = 10 launches per call). GGML_CUDA_TOPK_ONEBLOCK=0: off.
#ifndef GGML_CUDA_TOP_K_ONE_BLOCK_MAX_NCOLS
#    define GGML_CUDA_TOP_K_ONE_BLOCK_MAX_NCOLS 65536
#endif // GGML_CUDA_TOP_K_ONE_BLOCK_MAX_NCOLS

// adds 1 to hist[bin] for every lane with match; every lane of the warp must call it
static __device__ __forceinline__ void top_k_histogram_add(int * hist, const int bin, const bool match) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    // the top byte of most scores is the same: one atomic per distinct bin of the warp instead of up to 32 on one address
    const unsigned active = __ballot_sync(0xffffffff, match);
    if (match) {
        const unsigned peers = __match_any_sync(active, bin);
        if ((int) (threadIdx.x % WARP_SIZE) == __ffs(peers) - 1) {
            atomicAdd(&hist[bin], __popc(peers));
        }
    }
#else
    if (match) {
        atomicAdd(&hist[bin], 1);
    }
#endif
}

template<int BLOCK_SIZE>
static __global__ void __launch_bounds__(BLOCK_SIZE, 1) top_k_radix_one_block(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS      = 1 << RADIX_BITS;
    constexpr int NWARPS     = BLOCK_SIZE / WARP_SIZE;
    static_assert(BLOCK_SIZE >= NBINS && BLOCK_SIZE % WARP_SIZE == 0, "one thread per bin");
    static_assert(BLOCK_SIZE < (1 << 15), "two 16 bit counters per int");

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const float * row_src = src + (size_t) blockIdx.x * ncols;
    int *         row_dst = dst + (size_t) blockIdx.x * k;

    __shared__ int      histogram[NBINS];
    __shared__ int      warp_counts[NWARPS];
    __shared__ uint32_t s_prefix;
    __shared__ int      s_rank;

    uint32_t prefix      = 0;
    uint32_t prefix_mask = 0;
    int      rank        = k;
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        if (tid < NBINS) {
            histogram[tid] = 0;
        }
        __syncthreads();
        for (int col0 = 0; col0 < ncols; col0 += BLOCK_SIZE) {
            const int      col   = col0 + tid;
            const uint32_t key   = col < ncols ? top_k_float_to_ordered(row_src[col]) : 0;
            const bool     match = col < ncols && (key & prefix_mask) == prefix;
            top_k_histogram_add(histogram, (key >> shift) & (NBINS - 1), match);
        }
        __syncthreads();

        // top_k_radix_select's walk from the highest bin down, as a scan: with S(b) the count of keys in bins >= b, the selected bin is
        // the highest b with S(b) >= rank (bin 0 when there is none), and rank - S(bin + 1) of its keys are kept
        const int bin = NBINS - 1 - tid;
        const int h   = tid < NBINS ? histogram[bin] : 0;
        int incl = warp_prefix_inclusive_sum<int>(h);
        if (lane == WARP_SIZE - 1) {
            warp_counts[warp] = incl;
        }
        __syncthreads();
        if (tid < NBINS) {
            for (int w = 0; w < warp; ++w) {
                incl += warp_counts[w];
            }
            const int excl = incl - h; // S(bin + 1)
            if (excl < rank && (incl >= rank || bin == 0)) {
                s_prefix = prefix | ((uint32_t) bin << shift);
                s_rank   = rank - excl;
            }
        }
        __syncthreads();
        prefix       = s_prefix;
        rank         = s_rank;
        prefix_mask |= (uint32_t) (NBINS - 1) << shift;
    }

    // top_k_radix_gather's order: the keys above the threshold in column order, then the first n_equal ties in column order
    const int n_equal   = rank;
    const int n_greater = k - n_equal;
    int pos_greater = 0;
    int pos_equal   = 0;
    for (int col0 = 0; col0 < ncols; col0 += BLOCK_SIZE) {
        if (pos_greater >= n_greater && pos_equal >= n_equal) {
            break; // uniform: every thread holds the same counters
        }
        const int      col     = col0 + tid;
        const uint32_t key     = col < ncols ? top_k_float_to_ordered(row_src[col]) : 0;
        const bool     greater = col < ncols && key >  prefix;
        const bool     equal   = col < ncols && key == prefix;

        const int flags     = (int) greater | ((int) equal << 16);
        const int inclusive = warp_prefix_inclusive_sum<int>(flags);
        if (lane == WARP_SIZE - 1) {
            warp_counts[warp] = inclusive;
        }
        __syncthreads();

        int before = 0;
        int total  = 0;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            const int c = warp_counts[w];
            before += w < warp ? c : 0;
            total  += c;
        }
        const int exclusive = before + inclusive - flags;

        if (greater) {
            row_dst[pos_greater + (exclusive & 0xFFFF)] = col;
        } else if (equal) {
            const int pos = pos_equal + (exclusive >> 16);
            if (pos < n_equal) {
                row_dst[n_greater + pos] = col;
            }
        }
        pos_greater += total & 0xFFFF;
        pos_equal   += total >> 16;
        __syncthreads();
    }
}

// k19-dsv4long: top_k_radix_one_block with IPT contiguous columns per thread and pass (vector loads when aligned): the same histograms
// (counts do not depend on the visit order), the same threshold / kept-tie count, and the same gather order (keys above the threshold
// in column order, then the first kept ties in column order: one block scan of per-thread counts per BLOCK_SIZE*IPT columns), so the
// output is bitwise that of top_k_radix_one_block / top_k_radix_cuda. The old kernel issued one 4-byte load per thread and step, so a
// 60k-column decode row (DeepSeek-V4.1 indexer at long context) took ~200 us on one T4 SM. GGML_CUDA_TOPK_ONEBLOCK_V2=0: off (the old
// kernel up to 65536 columns, as before).
#ifndef GGML_CUDA_TOP_K_ONE_BLOCK_V2_MAX_NCOLS
#    define GGML_CUDA_TOP_K_ONE_BLOCK_V2_MAX_NCOLS (1 << 20)
#endif // GGML_CUDA_TOP_K_ONE_BLOCK_V2_MAX_NCOLS

template<int IPT>
static __device__ __forceinline__ void top_k_load_ipt(const float * __restrict__ row, const int c, const int ncols, const bool vec, float (&v)[IPT]) {
    if (vec && c + IPT <= ncols) {
#pragma unroll
        for (int j = 0; j < IPT; j += 4) {
            const float4 t = *(const float4 *) (row + c + j);
            v[j + 0] = t.x; v[j + 1] = t.y; v[j + 2] = t.z; v[j + 3] = t.w;
        }
    } else {
#pragma unroll
        for (int j = 0; j < IPT; ++j) {
            v[j] = c + j < ncols ? row[c + j] : 0.0f;
        }
    }
}

template<int BLOCK_SIZE, int IPT>
static __global__ void __launch_bounds__(BLOCK_SIZE, 1) top_k_radix_one_block_v2(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS      = 1 << RADIX_BITS;
    constexpr int NWARPS     = BLOCK_SIZE / WARP_SIZE;
    constexpr int CHUNK      = BLOCK_SIZE * IPT;
    static_assert(BLOCK_SIZE >= NBINS && BLOCK_SIZE % WARP_SIZE == 0, "one thread per bin");
    static_assert(CHUNK < (1 << 15), "two 16 bit counters per int");
    static_assert(IPT % 4 == 0, "float4 loads");

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const float * row_src = src + (size_t) blockIdx.x * ncols;
    int *         row_dst = dst + (size_t) blockIdx.x * k;
    const bool    vec     = ((uintptr_t) row_src % 16) == 0;

    __shared__ int      histogram[NBINS];
    __shared__ int      warp_counts[NWARPS];
    __shared__ uint32_t s_prefix;
    __shared__ int      s_rank;

    uint32_t prefix      = 0;
    uint32_t prefix_mask = 0;
    int      rank        = k;
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        if (tid < NBINS) {
            histogram[tid] = 0;
        }
        __syncthreads();
        for (int col0 = 0; col0 < ncols; col0 += CHUNK) {
            const int c = col0 + tid*IPT;
            float v[IPT];
            top_k_load_ipt<IPT>(row_src, c, ncols, vec, v);
#pragma unroll
            for (int j = 0; j < IPT; ++j) {
                const uint32_t key   = top_k_float_to_ordered(v[j]);
                const bool     match = c + j < ncols && (key & prefix_mask) == prefix;
                top_k_histogram_add(histogram, (key >> shift) & (NBINS - 1), match);
            }
        }
        __syncthreads();

        const int bin = NBINS - 1 - tid;
        const int h   = tid < NBINS ? histogram[bin] : 0;
        int incl = warp_prefix_inclusive_sum<int>(h);
        if (lane == WARP_SIZE - 1) {
            warp_counts[warp] = incl;
        }
        __syncthreads();
        if (tid < NBINS) {
            for (int w = 0; w < warp; ++w) {
                incl += warp_counts[w];
            }
            const int excl = incl - h; // S(bin + 1)
            if (excl < rank && (incl >= rank || bin == 0)) {
                s_prefix = prefix | ((uint32_t) bin << shift);
                s_rank   = rank - excl;
            }
        }
        __syncthreads();
        prefix       = s_prefix;
        rank         = s_rank;
        prefix_mask |= (uint32_t) (NBINS - 1) << shift;
    }

    const int n_equal   = rank;
    const int n_greater = k - n_equal;
    int pos_greater = 0;
    int pos_equal   = 0;
    for (int col0 = 0; col0 < ncols; col0 += CHUNK) {
        if (pos_greater >= n_greater && pos_equal >= n_equal) {
            break; // uniform: every thread holds the same counters
        }
        const int c = col0 + tid*IPT;
        float v[IPT];
        top_k_load_ipt<IPT>(row_src, c, ncols, vec, v);
        int n_g = 0;
        int n_e = 0;
#pragma unroll
        for (int j = 0; j < IPT; ++j) {
            const uint32_t key = top_k_float_to_ordered(v[j]);
            n_g += c + j < ncols && key >  prefix;
            n_e += c + j < ncols && key == prefix;
        }

        const int flags     = n_g | (n_e << 16);
        const int inclusive = warp_prefix_inclusive_sum<int>(flags);
        if (lane == WARP_SIZE - 1) {
            warp_counts[warp] = inclusive;
        }
        __syncthreads();

        int before = 0;
        int total  = 0;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            const int cw = warp_counts[w];
            before += w < warp ? cw : 0;
            total  += cw;
        }
        const int exclusive = before + inclusive - flags;

        if (n_g | n_e) {
            int og = pos_greater + (exclusive & 0xFFFF);
            int oe = pos_equal   + (exclusive >> 16);
#pragma unroll
            for (int j = 0; j < IPT; ++j) {
                const uint32_t key = top_k_float_to_ordered(v[j]);
                if (c + j < ncols && key > prefix) {
                    row_dst[og++] = c + j;
                } else if (c + j < ncols && key == prefix) {
                    if (oe < n_equal) {
                        row_dst[n_greater + oe] = c + j;
                    }
                    oe++;
                }
            }
        }
        pos_greater += total & 0xFFFF;
        pos_equal   += total >> 16;
        __syncthreads();
    }
}

// k19-dtopk: few rows of many columns and a small k (the MTP draft step's backend top_k(10) over 69632 / 248320 logits): the
// multi-block radix select is 10 dependent launches (init, 4 x histogram + select, gather), latency-bound at one row. Here one
// launch: each block picks its k best columns of its slice by (key desc, column asc), one per round (key = top_k_float_to_ordered:
// -0 == +0, the radix order of every value incl. NaN); the block that finishes last for a row merges the blocks' picks the same way
// and writes the radix select's output order: the picked keys above the k-th in column order, then the picked ties in column order.
// The top k under (key desc, column asc) are exactly the radix select's keys above its threshold plus its lowest-column ties, so the
// output is the same ids in the same order. GGML_CUDA_TOPK_TWO_STAGE=0: the radix select.
// k: one round per pick per block, so it pays for small k only (A100, test-topk-twostage-exact: k 1 / 10 at 0.14-0.94x the radix
// select's time over 32769-262144 columns, k 32 at 1.5-4.1x)
#define TOP_K_TWO_STAGE_MAX_K    10
#define TOP_K_TWO_STAGE_MAX_ROWS 8
#define TOP_K_TWO_STAGE_NONE     0x7FFFFFFF

// (ka, ca) before (kb, cb): higher key, then lower column; a column NONE is after everything
static __device__ __forceinline__ bool top_k_ts_before(uint32_t ka, int ca, uint32_t kb, int cb) {
    return ca != TOP_K_TWO_STAGE_NONE && (cb == TOP_K_TWO_STAGE_NONE || ka > kb || (ka == kb && ca < cb));
}

// the block's first (key, col) in the order; every thread gets it
template<int BLOCK_SIZE>
static __device__ __forceinline__ void top_k_ts_block_best(uint32_t & key, int & col, uint32_t * s_key, int * s_col) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        const uint32_t ko = __shfl_xor_sync(0xFFFFFFFF, key, off, WARP_SIZE);
        const int      co = __shfl_xor_sync(0xFFFFFFFF, col, off, WARP_SIZE);
        if (top_k_ts_before(ko, co, key, col)) {
            key = ko;
            col = co;
        }
    }
    if (tid % WARP_SIZE == 0) {
        s_key[tid / WARP_SIZE] = key;
        s_col[tid / WARP_SIZE] = col;
    }
    __syncthreads();
    if (tid < WARP_SIZE) {
        key = tid < BLOCK_SIZE / WARP_SIZE ? s_key[tid] : 0;
        col = tid < BLOCK_SIZE / WARP_SIZE ? s_col[tid] : TOP_K_TWO_STAGE_NONE;
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            const uint32_t ko = __shfl_xor_sync(0xFFFFFFFF, key, off, WARP_SIZE);
            const int      co = __shfl_xor_sync(0xFFFFFFFF, col, off, WARP_SIZE);
            if (top_k_ts_before(ko, co, key, col)) {
                key = ko;
                col = co;
            }
        }
        if (tid == 0) {
            s_key[0] = key;
            s_col[0] = col;
        }
    }
    __syncthreads();
    key = s_key[0];
    col = s_col[0];
    __syncthreads();
}

template<int BLOCK_SIZE>
static __global__ void top_k_two_stage(const float * __restrict__ src, int * __restrict__ dst, uint2 * __restrict__ part,
                                       unsigned int * __restrict__ counters, int ncols, int k, int blocks_per_row, int cols_per_block) {
    __shared__ uint32_t s_key[BLOCK_SIZE / WARP_SIZE];
    __shared__ int      s_col[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint32_t sel_key[TOP_K_TWO_STAGE_MAX_K];
    __shared__ int      sel_col[TOP_K_TWO_STAGE_MAX_K];
    __shared__ bool     s_last;

    const int row = blockIdx.x / blocks_per_row;
    const int rb  = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const int c0 = rb * cols_per_block;
    const int c1 = min(ncols, c0 + cols_per_block);
    uint2 * my_part = part + (size_t) blockIdx.x * k;

    // stage 1: this block's k first (key, col) of its slice, one per round: each round takes the first one after the previous pick
    uint32_t pk = 0;
    int      pc = -1;
    for (int r = 0; r < k; ++r) {
        uint32_t bk = 0;
        int      bc = TOP_K_TWO_STAGE_NONE;
        for (int c = c0 + tid; c < c1; c += BLOCK_SIZE) {
            const uint32_t key = top_k_float_to_ordered(row_src[c]);
            const bool after = r == 0 || key < pk || (key == pk && c > pc);
            if (after && top_k_ts_before(key, c, bk, bc)) {
                bk = key;
                bc = c;
            }
        }
        top_k_ts_block_best<BLOCK_SIZE>(bk, bc, s_key, s_col);
        if (tid == 0) {
            my_part[r] = make_uint2(bk, (unsigned int) bc);
        }
        pk = bk;
        pc = bc;
    }

    // the last block of the row to finish merges
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(&counters[row], 1u) == (unsigned int) (blocks_per_row - 1);
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    const uint2 * row_part = part + (size_t) row * blocks_per_row * k;
    const int n_cand = blocks_per_row * k;
    pk = 0;
    pc = -1;
    for (int r = 0; r < k; ++r) {
        uint32_t bk = 0;
        int      bc = TOP_K_TWO_STAGE_NONE;
        for (int i = tid; i < n_cand; i += BLOCK_SIZE) {
            const uint2 e = __ldcg(row_part + i);
            const uint32_t key = e.x;
            const int      c   = (int) e.y;
            if (c == TOP_K_TWO_STAGE_NONE) {
                continue;
            }
            const bool after = r == 0 || key < pk || (key == pk && c > pc);
            if (after && top_k_ts_before(key, c, bk, bc)) {
                bk = key;
                bc = c;
            }
        }
        top_k_ts_block_best<BLOCK_SIZE>(bk, bc, s_key, s_col);
        if (tid == 0) {
            sel_key[r] = bk;
            sel_col[r] = bc;
        }
        pk = bk;
        pc = bc;
    }
    __syncthreads();
    if (tid == 0) {
        // the radix select's order: the keys above the k-th in column order, then the ties of the k-th in column order
        const uint32_t thr = sel_key[k - 1];
        int * row_dst = dst + (size_t) row * k;
        int n = 0;
        for (int pass = 0; pass < 2; ++pass) {
            const int n0 = n;
            for (int i = 0; i < k; ++i) {
                if ((pass == 0) != (sel_key[i] > thr)) {
                    continue;
                }
                // insertion by column among this pass's entries
                int j = n;
                while (j > n0 && row_dst[j - 1] > sel_col[i]) {
                    row_dst[j] = row_dst[j - 1];
                    j--;
                }
                row_dst[j] = sel_col[i];
                n++;
            }
        }
    }
}

static bool ggml_cuda_top_k_two_stage_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TOPK_TWO_STAGE");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

static void top_k_two_stage_cuda(ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 128);
    const int cols_per_block = (ncols + blocks_per_row - 1) / blocks_per_row;
    ggml_cuda_pool_alloc<uint2>        part_alloc(pool, (size_t) nrows * blocks_per_row * k);
    ggml_cuda_pool_alloc<unsigned int> cnt_alloc(pool, (size_t) nrows);
    CUDA_CHECK(cudaMemsetAsync(cnt_alloc.get(), 0, nrows * sizeof(unsigned int), stream));
    top_k_two_stage<BLOCK_SIZE><<<(unsigned int) (nrows * blocks_per_row), BLOCK_SIZE, 0, stream>>>(
        src, dst, part_alloc.get(), cnt_alloc.get(), ncols, k, blocks_per_row, cols_per_block);
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: device top-k: top-k of %d in one two-stage launch (%lld rows of %d; GGML_CUDA_TOPK_TWO_STAGE=0: radix select)\n",
                __func__, k, (long long) nrows, ncols);
    }
}

static bool ggml_cuda_top_k_one_block_v2_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TOPK_ONEBLOCK_V2");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

static bool ggml_cuda_top_k_one_block_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TOPK_ONEBLOCK");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

static void top_k_radix_one_block_cuda(const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 1024;
    if (ggml_cuda_top_k_one_block_v2_enabled()) {
        top_k_radix_one_block_v2<BLOCK_SIZE, 8><<<(unsigned int) nrows, BLOCK_SIZE, 0, stream>>>(src, dst, ncols, k);
        static bool logged_v2 = false;
        if (!logged_v2) {
            logged_v2 = true;
            GGML_LOG_INFO("%s: k19 top-k radix select in one launch, 8 columns per thread (%lld rows of %d; GGML_CUDA_TOPK_ONEBLOCK_V2=0: off)\n",
                    __func__, (long long) nrows, ncols);
        }
        return;
    }
    top_k_radix_one_block<BLOCK_SIZE><<<(unsigned int) nrows, BLOCK_SIZE, 0, stream>>>(src, dst, ncols, k);
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: k19 top-k radix select in one launch (%lld rows of %d; GGML_CUDA_TOPK_ONEBLOCK=0: off)\n", __func__, (long long) nrows, ncols);
    }
}

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream,
        bool tie_index = false) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream, tie_index);
        }
        ggml_cuda_memcpy_d2d(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows, stream);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

// GGML_CUDA_TOPK_DETERMINISTIC (default 1): the same input gives the same output on every run. Order used to pick
// among equal values: descending value, -0.0 == +0.0, and of the values tied at the k-th position the lowest
// columns are kept (the CPU backend uses the same rule). Output order per path, all fixed for a given shape:
//   bitonic       (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC, or few rows): sorted by (value desc, column asc)
//   CUB radix sort (1 row, ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT): stable sort, same order as bitonic
//   radix select  (otherwise): the keys above the k-th value in column order, then the kept ties in column order
// cub::DeviceTopK is skipped: it only implements determinism::not_guaranteed (CCCL cub/device/device_topk.cuh).
// GGML_CUDA_TOPK_DETERMINISTIC=0 restores the previous selection (DeviceTopK when available, atomic radix gather).
static bool ggml_cuda_top_k_deterministic() {
    static const bool det = [] {
        const char * env = getenv("GGML_CUDA_TOPK_DETERMINISTIC");
        return env == nullptr || atoi(env) != 0;
    }();
    return det;
}

static void ggml_cuda_op_top_k_deterministic(ggml_cuda_pool & pool, const float * src0_d, int * dst_d,
                                             const int64_t ncols, const int64_t nrows, const int64_t k,
                                             const int device, cudaStream_t stream) {
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
    const bool bitonic_few_rows = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream, /*tie_index=*/true);
            return;
        }
    }

#ifdef GGML_CUDA_USE_CUB
    // DeviceRadixSort is stable; the segmented sort used for several rows is not, so only one row goes here
    if (nrows == 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
        return;
    }
#endif // GGML_CUDA_USE_CUB

    // k19: few rows (one wave of blocks): the same selection and order in one launch, see top_k_radix_one_block
    // k19-dsv4long (Kaggle T4, test-topk-oneblock-exact): one block per row wins up to ~32k columns (v2: 15360 cols 37 us vs 43
    // multi-block vs 58 old one-block; 30000: 54 / 58 / 105) and for many rows (40 x 60000: 190 / 355 / 308); for 1-2 rows of more
    // columns the 64-block-per-row multi-block path is faster (60000: 68 vs 96 v2 vs 194 old; 122880: 62 vs 167), so with v2 on, few
    // long rows go there (before, the old one-block kernel took them up to 65536 columns)
    const bool one_block_v2 = ggml_cuda_top_k_one_block_v2_enabled();
    if (ggml_cuda_top_k_one_block_enabled() && nrows <= ggml_cuda_info().devices[device].nsm &&
            (one_block_v2 ? ncols <= GGML_CUDA_TOP_K_ONE_BLOCK_V2_MAX_NCOLS && (ncols <= 32768 || nrows >= 16)
                          : ncols <= GGML_CUDA_TOP_K_ONE_BLOCK_MAX_NCOLS)) {
        top_k_radix_one_block_cuda(src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

    if (ggml_cuda_top_k_two_stage_enabled() && k <= TOP_K_TWO_STAGE_MAX_K && nrows <= TOP_K_TWO_STAGE_MAX_ROWS && ncols >= 2048) {
        top_k_two_stage_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int device = ggml_cuda_get_device();

    if (ggml_cuda_top_k_deterministic()) {
        if ((dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) && nrows > 1 && nrows <= 16) {
            // batch invariant: each row takes the path (and output order) of a single row
            for (int64_t i = 0; i < nrows; ++i) {
                ggml_cuda_op_top_k_deterministic(pool, src0_d + i*ncols, dst_d + i*k, ncols, 1, k, device, stream);
            }
            return;
        }
        ggml_cuda_op_top_k_deterministic(pool, src0_d, dst_d, ncols, nrows, k, device, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda_legacy(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda_legacy(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda_legacy(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}
