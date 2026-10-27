// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// k15-mm2: the DSV4_HC_POST (no comb) before the norm, computed in the norm kernel: x[i0, c, t] = bo[i0, t]*w(c, t) + res[i0, c, t]
// with w = chain(post[c + hc*t]), chain = SCALE (op 0: p0*w + p1) / SIGMOID (op 1) ops, as hc_post_chain_f32 computes it; x is
// stored to the norm source (the DSV4_HC_POST output)
#define GGML_CUDA_HC_POST_NORM_MAX_OPS 4
struct ggml_cuda_hc_post_args {
    const float * bo       = nullptr;
    const float * residual = nullptr;
    const float * post     = nullptr;
    int64_t sbo1 = 0; // token stride of bo
    int64_t sr1  = 0; // stream stride of residual
    int64_t sr2  = 0; // token stride of residual
    int     hc   = 0;
    int     n_ops = 0;
    int     op[GGML_CUDA_HC_POST_NORM_MAX_OPS] = {};
    float   p0[GGML_CUDA_HC_POST_NORM_MAX_OPS] = {};
    float   p1[GGML_CUDA_HC_POST_NORM_MAX_OPS] = {};
};

// q8_1_out != nullptr: also write the q8_1 rows of the (contiguous) output for mul_mat_vec_q, see ggml_cuda_q8_1_cache
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, void * q8_1_out = nullptr,
                                 const ggml_tensor * add_tensor = nullptr, const ggml_cuda_hc_post_args * hcp = nullptr);

// k15-fn: RMS_NORM -> MUL -> ROPE with a multi-section (non-vision) rope, bitwise the unfused ops
void ggml_cuda_op_rms_norm_mul_mrope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
