// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// q8_1_out != nullptr: also write the q8_1 rows of the (contiguous) output for mul_mat_vec_q, see ggml_cuda_q8_1_cache
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, void * q8_1_out = nullptr,
                                 const ggml_tensor * add_tensor = nullptr);

// k15-fn: RMS_NORM -> MUL -> ROPE with a multi-section (non-vision) rope, bitwise the unfused ops
void ggml_cuda_op_rms_norm_mul_mrope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
