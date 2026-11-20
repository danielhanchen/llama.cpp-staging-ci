// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once
#include "common.cuh"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// split_src (k15-fn): the unary input is a CONT that did not run, read its 3D f32 source view instead
void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node, const ggml_tensor * split_src = nullptr);

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

// chain of single-input f32 elementwise ops (SCALE, CLAMP, SQR, SQRT, SIN, COS, LOG, UNARY), applied in registers
#define GGML_CUDA_EW_CHAIN_MAX 8

struct ggml_cuda_ew_chain {
    int   n_ops;
    int   op[GGML_CUDA_EW_CHAIN_MAX]; // ggml_cuda_ew_chain_op_code()
    float p0[GGML_CUDA_EW_CHAIN_MAX];
    float p1[GGML_CUDA_EW_CHAIN_MAX];
};

// -1 when the node cannot be part of a chain, else the code stored in ggml_cuda_ew_chain::op
int ggml_cuda_ew_chain_op_code(const ggml_tensor * node);

// dst = ops(src), src and dst contiguous f32 with the same number of elements
// q8_1_out != nullptr: also writes the q8_1 copy of dst (one row, or rows of a multiple of MATRIX_ROW_PADDING values) for the
// mul_mat_vec_q consumers, bit-identical to quantize_q8_1
// e[row] = chain(max|x[row,:]|), dst = x / e (row broadcast): the fused ABS -> POOL_1D -> chain -> DIV of build_amax_src1_f16
void ggml_cuda_op_amax_rescale(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * e, ggml_tensor * dst,
        const ggml_cuda_ew_chain & chain, void * q8_1_out = nullptr);

void ggml_cuda_op_ew_chain(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, const ggml_cuda_ew_chain & chain,
        void * q8_1_out = nullptr);

// MUL(other, UNARY(gate)) where gate has one element per row of other (ne0 == 1), e.g. a shared expert gate
bool ggml_cuda_unary_mul_gate_supported(const ggml_tensor * unary_node);
void ggml_cuda_op_unary_mul_gate(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);
void ggml_cuda_op_unary_mul_gate_add(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node, ggml_tensor * add_node);

// [chain ->] DSV4_HC_POST(x, residual, post) without comb: post = chain(post_src), applied by the hc_post kernel itself
void ggml_cuda_op_hc_post_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_ew_chain & chain, const ggml_tensor * post_src,
        ggml_tensor * hc_post);

// MUL(UNARY(ADD(a, bias)), scale) with bias and scale one ne0-row each, e.g. the delta-net alpha gate
void ggml_cuda_op_add_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add_node, ggml_tensor * unary_node, ggml_tensor * mul_node);

__device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ggml_cuda_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_clamp_single(float gate, float up, float limit) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);

    return ggml_cuda_op_silu_single(gate) * up;
}
