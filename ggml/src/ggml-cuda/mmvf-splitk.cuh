// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

#include "common.cuh"

// k15-mm2: one block of the split-K F32 mat-vec (see mul_mat_vec_f_splitk_try in mmvf.cu): block (row, split) of nsplit sums the
// K range [split*chunk2, (split+1)*chunk2) of float2 pairs, writes its partial sums and the last block of the row to arrive adds
// them in split order. Used by mul_mat_vec_f_splitk and by the hyper-connection down + inject launch (mmvq.cu), which run the same
// code, so both write the same bytes. tid = the 1D thread index of a MMVF_SPLITK_BLOCK thread block (warp = tid / 32).
#define MMVF_SPLITK_MAX_ROWS  256
#define MMVF_SPLITK_MAX_SPLIT 64
#define MMVF_SPLITK_BLOCK     128

// what a split-K launch needs, see ggml_cuda_mul_mat_vec_f_splitk_plan
struct ggml_cuda_mmvf_splitk_args {
    const float  * x      = nullptr;
    const float  * y      = nullptr;
    float        * dst    = nullptr;
    float        * part   = nullptr;
    unsigned int * cnt    = nullptr;
    int nrows             = 0;
    int nsplit            = 0;
    int ncols2            = 0;
    int chunk2            = 0;
    int stride_row        = 0;
    int stride_col_y2     = 0;
    int stride_col_dst    = 0;
    int ncols_dst         = 0;
};

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <int ncols_dst>
static __device__ __forceinline__ void mul_mat_vec_f_splitk_block(
        const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst, float * __restrict__ part,
        unsigned int * __restrict__ cnt, const int ncols2, const int chunk2, const int stride_row, const int stride_col_y2,
        const int stride_col_dst, const int row, const int split, const int nsplit, const int tid) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = MMVF_SPLITK_BLOCK/warp_size;

    __shared__ float buf[nwarps][ncols_dst];
    __shared__ bool  last;

    ggml_cuda_pdl_sync();

    const float2 * x2 = (const float2 *) (x + int64_t(row)*stride_row);
    const float2 * y2 = (const float2 *) y;
    const int c0 = split*chunk2;
    const int c1 = min(ncols2, c0 + chunk2);

    float sumf[ncols_dst] = {0.0f};
    for (int col2 = c0 + tid; col2 < c1; col2 += MMVF_SPLITK_BLOCK) {
        const float2 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }
    ggml_cuda_pdl_lc();

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        if (tid % warp_size == 0) {
            buf[tid/warp_size][j] = sumf[j];
        }
    }
    __syncthreads();

    float * p = part + (int64_t(row)*MMVF_SPLITK_MAX_SPLIT)*ncols_dst;
    if (tid < ncols_dst) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            v += buf[w][tid];
        }
        p[split*ncols_dst + tid] = v;
        __threadfence();
    }
    __syncthreads();
    if (tid == 0) {
        last = atomicAdd(cnt + row, 1u) == (unsigned int) (nsplit - 1);
        if (last) {
            cnt[row] = 0;
        }
    }
    __syncthreads();
    if (!last || tid >= ncols_dst) {
        return;
    }
    __threadfence();
    float v = 0.0f;
    for (int sp = 0; sp < nsplit; ++sp) {
        v += __ldcg(p + sp*ncols_dst + tid);
    }
    dst[tid*stride_col_dst + row] = v;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
