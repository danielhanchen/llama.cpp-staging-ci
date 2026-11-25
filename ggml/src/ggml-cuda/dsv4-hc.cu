// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

// dsv4_hc_pre_f32 for a contiguous dst, plus the q8_1 copy of dst (rows of n_embd values padded to n_embd_padded) for the
// mul_mat_vec_q consumers: value ir of the padded rows is computed and quantized by thread ir, as quantize_q8_1 does it
// (one value per lane, the same 32-lane reductions; n_embd_padded % QK8_1 == 0), so both outputs are bit-identical to the
// unfused dsv4_hc_pre + quantize_q8_1 sequence
template <bool gated>
static __global__ void dsv4_hc_pre_q8_1_f32(
        const float * x,
        const float * weights,
        float * dst,
        void * vq,
        int64_t n_embd,
        int64_t n_embd_padded,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        float   scale,
        bool    exact) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd_padded * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd_padded;
    const int64_t it = ir / n_embd_padded;

    float xi = 0.0f;
    if (i0 < n_embd) {
        float sum = 0.0f;
        for (int64_t ih = 0; ih < hc; ++ih) {
            const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
            float wv;
            if constexpr (gated) {
                wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
            } else {
                wv = weights[ih*sw0 + it*sw1];
            }
            sum += xv * wv;
        }
        xi = scale * sum;
        dst[it*n_embd + i0] = xi;
    }

    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    float d;
    float qsum;
    const int8_t q = ggml_cuda_q8_1_lane(xi, amax, exact, d, qsum);

    block_q8_1 * y = (block_q8_1 *) vq;
    const int64_t ib  = ir / QK8_1;
    const int     iqs = ir % QK8_1;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(d, qsum);
    }
}

// gated: post = gate_scale_out*sigmoid(gate_scale_in*post), the fused SCALE -> SIGMOID -> SCALE chain of qwen4exp
template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float gate_scale_in,
        float gate_scale_out) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float p = post[idst*sp0 + it*sp1];
    if constexpr (gated) {
        p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));
    }

    float sum = x[i0*sx0 + it*sx1] * p;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    // a mul_mat_vec_q reads dst: write its q8_1 copy in the same kernel (see ggml_cuda_q8_1_cache)
    ggml_cuda_q8_1_cache & cache = ctx.q8_1_cache;
    if (cache.active() && ctx.curr_stream_no == 0 && cache.norm_targets.count(dst) && ggml_is_contiguous(dst) &&
        (n_tokens == 1 || n_embd % MATRIX_ROW_PADDING == 0)) {
        const int64_t n_embd_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);
        int64_t key[ggml_cuda_q8_1_cache::n_key];
        ggml_cuda_q8_1_cache::key_contiguous(n_embd, n_tokens, key);
        void * q8_1_out = cache.reserve(dst->data, ggml_nbytes(dst), key, n_tokens*n_embd_padded/QK8_1*sizeof(block_q8_1));
        if (q8_1_out != nullptr) {
            static_assert(256 % QK8_1 == 0, "a warp must cover whole q8_1 blocks");
            const int64_t nrq = n_embd_padded * n_tokens;
            const dim3 grid_q((nrq + block_size - 1) / block_size, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params_q = ggml_cuda_kernel_launch_params(grid_q, block_dims, 0, ctx.stream());
            auto kernel_q = gated ? dsv4_hc_pre_q8_1_f32<true> : dsv4_hc_pre_q8_1_f32<false>;
            ggml_cuda_kernel_launch(kernel_q, launch_params_q,
                    (const float *) x->data, (const float *) weights->data, (float *) dst->data, q8_1_out,
                    n_embd, n_embd_padded, hc, n_tokens,
                    nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
                    nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
                    scale, ggml_cuda_q8_1_exact());
            return;
        }
    }

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    // when fused, read the raw post weights from before the SCALE -> SIGMOID -> SCALE chain
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];

    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? (gated ? dsv4_hc_post_f32<true,  true> : dsv4_hc_post_f32<true,  false>)
                       : (gated ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
}
