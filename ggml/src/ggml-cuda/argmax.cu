// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        // the lowest column on a tie, as the CPU backend (a greedy sampler on the device gives the host's token)
        if (val > maxval || (val == maxval && (unsigned) col < (unsigned) argmax)) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        // the lowest column on a tie, as the CPU backend (a greedy sampler on the device gives the host's token)
        if (val > maxval || (val == maxval && (unsigned) col < (unsigned) argmax)) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval || (val == maxval && (unsigned) col < (unsigned) argmax)) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

// a long row (an lm head's logits: 248320 columns) split over several blocks, one partial (max, lowest column) per block, then one warp per
// row merges the partials with the same comparison: the maximum with the lowest column on a tie does not depend on the order in which
// columns are compared, so the result is the single-block kernel's. One block read a 248320-column row in 48 us on an RTX PRO 6000
// (the greedy sampler of each row of a 7-row verify, 0.33 ms per window); GGML_CUDA_ARGMAX_SPLIT=0: one block per row
static __device__ __forceinline__ void argmax_merge(float & maxval, int & argmax, const float val, const int col) {
    if (val > maxval || (val == maxval && (unsigned) col < (unsigned) argmax)) {
        maxval = val;
        argmax = col;
    }
}

static __global__ void argmax_f32_part(const float * __restrict__ x, float * __restrict__ part_val, int32_t * __restrict__ part_col,
        const int64_t ncols, const int64_t chunk) {
    const int64_t row  = blockIdx.y;
    const int64_t c0   = (int64_t) blockIdx.x * chunk;
    const int64_t c1   = c0 + chunk < ncols ? c0 + chunk : ncols;
    const float * rowx = x + row * ncols;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    for (int64_t col = c0 + threadIdx.x; col < c1; col += blockDim.x) {
        argmax_merge(maxval, argmax, rowx[col], (int) col);
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        argmax_merge(maxval, argmax, val, col);
    }
    constexpr int max_warps = 1024 / WARP_SIZE;
    __shared__ float shared_maxval[max_warps];
    __shared__ int   shared_argmax[max_warps];
    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        shared_maxval[warp_id] = maxval;
        shared_argmax[warp_id] = argmax;
    }
    __syncthreads();
    if (warp_id == 0) {
        maxval = -FLT_MAX;
        argmax = -1;
        if (lane_id < n_warps) {
            maxval = shared_maxval[lane_id];
            argmax = shared_argmax[lane_id];
        }
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
            const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
            argmax_merge(maxval, argmax, val, col);
        }
        if (lane_id == 0) {
            part_val[row * gridDim.x + blockIdx.x] = maxval;
            part_col[row * gridDim.x + blockIdx.x] = argmax;
        }
    }
}

static __global__ void argmax_f32_join(const float * __restrict__ part_val, const int32_t * __restrict__ part_col,
        int32_t * __restrict__ dst, const int n_parts) {
    const int64_t row = blockIdx.x;
    float maxval = -FLT_MAX;
    int   argmax = -1;
    for (int i = threadIdx.x; i < n_parts; i += WARP_SIZE) {
        argmax_merge(maxval, argmax, part_val[row * n_parts + i], part_col[row * n_parts + i]);
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        argmax_merge(maxval, argmax, val, col);
    }
    if (threadIdx.x == 0) {
        dst[row] = argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    static const bool split_env = [] {
        const char * e = getenv("GGML_CUDA_ARGMAX_SPLIT");
        return e == nullptr || atoi(e) != 0;
    }();
    constexpr int64_t chunk = 8192;
    if (split_env && ne00 >= 4*chunk && nrows <= 64) {
        const int n_parts = (int) ((ne00 + chunk - 1) / chunk);
        ggml_cuda_pool_alloc<float>   part_val(ctx.pool(), nrows * n_parts);
        ggml_cuda_pool_alloc<int32_t> part_col(ctx.pool(), nrows * n_parts);
        argmax_f32_part<<<dim3(n_parts, nrows, 1), dim3(256, 1, 1), 0, stream>>>(src0_d, part_val.get(), part_col.get(), ne00, chunk);
        argmax_f32_join<<<dim3(nrows, 1, 1), dim3(WARP_SIZE, 1, 1), 0, stream>>>(part_val.get(), part_col.get(), dst_d, n_parts);
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
