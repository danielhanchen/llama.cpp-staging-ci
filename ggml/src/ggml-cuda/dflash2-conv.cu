#include "dflash2-conv.cuh"

// the DFlash2 drafter's dynamic grouped convolution (ggml_dflash2_conv) in one kernel. Each product and sum is rounded on
// its own (__fmul_rn / __fadd_rn: nvcc never contracts them into a fused multiply-add), in the order of the add / mul / pad chain it
// replaces, so the result has its bits: w_k = dyn + base, r = w_0 * v_0, r = r + w_k * v_k, v_k = 0.0f before the block's start.
// HIP: clang's __fmul_rn / __fadd_rn are plain operators there, which its default -ffp-contract=fast fuses into v_fmac_f32, so the
// kernel uses plain operators with contraction off in its body (test-dflash2-conv-exact on gfx1151; the pragma does not reach the
// header's __fmul_rn / __fadd_rn bodies)
#if defined(GGML_USE_HIP)
#define D2C_MUL(a, b) ((a) * (b))
#define D2C_ADD(a, b) ((a) + (b))
#else
#define D2C_MUL(a, b) __fmul_rn(a, b)
#define D2C_ADD(a, b) __fadd_rn(a, b)
#endif // defined(GGML_USE_HIP)
static __global__ void k_dflash2_conv(const float * __restrict__ x, const float * __restrict__ dyn, const float * __restrict__ base,
        float * __restrict__ dst, const int64_t H, const int64_t T, const int64_t dr, const int ng, const int gs, const int K,
        const int bs, const int nt, const int side) {
#if defined(GGML_USE_HIP)
#pragma clang fp contract(off)
#endif // defined(GGML_USE_HIP)
    const int64_t h = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.x;
    if (h >= H || t >= T) {
        return;
    }
    const int     j = (int) (t % bs);
    const int     g = (int) (h / gs);
    const float * c = dyn + t * dr + (int64_t) ng * K * side;
    float r = D2C_MUL(D2C_ADD(c[g], base[h]), x[t * H + h]);
    for (int k = 1; k < nt; ++k) {
        const float w = D2C_ADD(c[g + ng * k], base[h + H * k]);
        const float v = j >= k ? x[(t - k) * H + h] : 0.0f;
        r = D2C_ADD(r, D2C_MUL(w, v));
    }
    dst[t * H + h] = r;
}

void ggml_cuda_op_dflash2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * dyn  = dst->src[1];
    const ggml_tensor * base = dst->src[2];
    const int K    = ggml_get_op_params_i32(dst, 0);
    const int gs   = ggml_get_op_params_i32(dst, 1);
    const int nblk = ggml_get_op_params_i32(dst, 2);
    const int side = ggml_get_op_params_i32(dst, 3);
    const int64_t H = x->ne[0];
    const int64_t T = x->ne[1];
    const int bs = (int) (T / nblk);
    const int nt = K < bs ? K : bs;
    const int nth = 256;
    GGML_ASSERT((H + nth - 1) / nth <= 65535);
    const dim3 grid((unsigned) T, (unsigned) ((H + nth - 1) / nth), 1);
    k_dflash2_conv<<<grid, nth, 0, ctx.stream()>>>((const float *) x->data, (const float *) dyn->data, (const float *) base->data,
            (float *) dst->data, H, T, dyn->ne[0], (int) (H / gs), gs, K, bs, nt, side);
}
