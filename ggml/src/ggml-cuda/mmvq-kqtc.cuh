// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "common.cuh"

// k19-kqtc: dense quantized x q8_1 products of 2..8 columns (speculative verify widths) on int8 tensor cores. Each weight block is
// decoded once into int8 MMA fragments for all columns (mul_mat_vec_q decodes it once per pair of rows and re-reads the activations per
// pair of rows). Q4_K / Q5_K / Q6_K / IQ4_XS / Q8_0, no ids, no fusion, one channel / sample; single products and the multi-matrix
// launches. The integer sums are those of mul_mat_vec_q (same q8_1 activations, same int8 products, exact int32 sums); only the order of
// the f32 scale products and sums differs (a reassociation). A column's result does not depend on the number of columns of the launch.
// GGML_CUDA_MMVQ_KQTC: unset / 0 = off (default), 1 = every type above, or a comma list of type names (q4_K,q5_K,q6_K,iq4_xs,q8_0).
// GGML_CUDA_MMVQ_KQTC_MIN / _MAX: column range (2 / 8). GGML_CUDA_MMVQ_KQTC_CFG=NWxT: warps per block (4 / 8) x row tiles of 16 per
// warp (1 / 2) instead of the automatic choice. GGML_CUDA_MMVQ_KQTC_LOG=1: one log line per launch ("k19-kqtc:").

// the switch allows this type at this many columns on this device (NVIDIA, Turing MMA or newer)
bool ggml_cuda_mmvq_kqtc_ok(ggml_type type, int64_t ncols_x, int64_t ncols_dst, int cc);

// GGML_CUDA_VERIFY_REASSOC (default 1): batch invariant verify products may reassociate (0 = the bit-exact per-token verify)
bool ggml_cuda_mmvq_kqtc_reassoc_ok();

// n_mat (1..GGML_CUDA_MMVQ_MULTI_MAX) matrices of one type and K (ncols_x) with row stride stride_row_x bytes, over the q8_1 activation
// vy (column c at vy + c*stride_col_y blocks), dst[m] column c at dst[m] + c*stride_col_dst. Returns false (nothing launched) when the
// layout rules it out (alignment).
bool ggml_cuda_mmvq_kqtc_launch(ggml_type type, int n_mat, const void * const * vx, float * const * dst, const int64_t * nrows,
    int64_t stride_row_x, const void * vy, int64_t stride_col_y, int64_t stride_col_dst, int64_t ncols_x, int64_t ncols_dst,
    cudaStream_t stream);
