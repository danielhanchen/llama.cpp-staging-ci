// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "common.cuh"

// k19-ampmoe: multi-token MUL_MAT_ID (2..8 tokens of a speculative verify window) on int8 tensor cores for NVIDIA Ampere / Ada
// (cc 8.0 / 8.6 / 8.7 / 8.9). The block of an expert's first (token-major) entry takes every entry of that expert as one MMA column
// (a token picks an expert once, so <= 8 columns), decodes each weight block once into int8 m16n8k32 A fragments and multiplies it with
// the q8_1 activations of all of them; the warps split K in 256-value units (their number follows K only). The integer dot products are
// mul_mat_vec_q's (same q8_1 activations, exact int32 sums); the f32 scale / sum order differs (a reassociation, KLD-gated). A column's
// bytes do not depend on the other columns or tokens of the launch.
// The few-row int8-MMA structure (warps split K in units of 256 values, raw weight rows in a per-warp cp.async ring, per-type decoders
// into A fragments) follows llama.cpp's mmq-few.cu (llama.cpp authors, MIT), extended here with ids, gathered columns and a fused GLU.
// GGML_CUDA_MOE_TC: 0 = off (the mul_mat_vec_q MoE kernels, byte for byte), 1 = verify windows (default), 2 = every 2..8-token
// MUL_MAT_ID (bench); GGML_CUDA_MOE_TC_TYPES = comma list of type names or "all" (default q4_K,q5_K,q6_K; the others are opt-in); GGML_CUDA_MOE_TC_KW = warps
// splitting K (1 / 2 / 4, bench); GGML_CUDA_MOE_TC_LOG=1: one line per launch ("k19-ampmoe:").

// whether the kernel takes this MUL_MAT_ID (type, K, tokens, device, fusion, verify window)
bool ggml_cuda_moe_tc_ok(ggml_type type, int64_t ncols_x, int64_t ncols_dst, int cc, const ggml_cuda_mm_fusion_args_device & fusion);

// the MUL_MAT_ID column count up to which mmvq keeps this type (instead of MMQ) because this kernel takes it here: 8 inside a verify
// window on a covered device / type with the switch on, else 0 (ggml_cuda_mmvq_mmid_max_batch takes the larger of this and its table)
int ggml_cuda_moe_tc_max_batch(ggml_type type, int cc);

// strides as in mul_mat_vec_q_moe: stride_row_x / stride_channel_x in blocks of type, stride_col_y / stride_channel_y in block_q8_1,
// stride_col_dst / stride_channel_dst in floats. Returns false (nothing launched) when the layout rules it out.
bool ggml_cuda_moe_tc_launch(ggml_type type, const void * vx, const void * vy, const int32_t * ids,
    const ggml_cuda_mm_fusion_args_device & fusion, float * dst, int64_t ncols_x, int64_t nrows_x,
    int64_t stride_row_x, int64_t stride_col_y, int64_t stride_col_dst,
    int64_t stride_channel_x, int64_t stride_channel_y, int64_t stride_channel_dst,
    int64_t nchannels_y, int64_t nslots, int64_t ntok, int64_t ids_stride, cudaStream_t stream);
