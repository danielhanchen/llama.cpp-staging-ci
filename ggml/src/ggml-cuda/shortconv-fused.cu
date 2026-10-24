// k11-decode: LFM2 short-conv block fusions for small batches (decode and speculative verify windows, at most 32 tokens per sequence).
//
// The short-conv block of LFM2 / LFM2-MoE (src/models/lfm2.cpp build_shortconv_block) is, per layer:
//   bx   = b * x                        MUL      (b, x: views of the in_proj output)
//   bxT  = transpose(bx)                TRANSPOSE (view)
//   cat  = concat(state, bxT, dim 0)    CONCAT   (the conv window: the d_conv state columns, then the new tokens)
//   (state snapshots: CPY of views of cat into the recurrent cache)
//   conv = ssm_conv(cat, w)             SSM_CONV
//   y    = c * conv                     MUL      (c: a view of the in_proj output)
// Two fusions, each bit exact against the unfused kernels (the same float operations in the same order):
//   1. MUL + TRANSPOSE + CONCAT -> one kernel that writes the conv window, b * x formed in place;
//   2. SSM_CONV + MUL -> the ssm_conv kernel of ssm-conv.cu with the product by c applied to its result.
// GGML_CUDA_SHORTCONV_FUSE=0 turns both off (GGML_CUDA_DISABLE_FUSION=1 too).

#include "shortconv-fused.cuh"

static bool shortconv_fuse_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_SHORTCONV_FUSE");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

static bool shortconv_f32(const ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32;
}

// ---------------------------------------------------------------------------------------------------------- 1. b * x into the window
// out[t, ch, s] = t < d ? state[t, ch, s] : b[ch, t - d, s] * x[ch, t - d, s]
// k15: + the conv state snapshots (the CPYs of views of the window into the recurrent cache that follow it, lfm2.cpp): snapshot k
// writes dst[k] + s*dst_nb1[k] + (ch*d + c)*4 = out[off[k] + c, ch, s], the values this thread just wrote (GGML_CUDA_SHORTCONV_STATE=0: off)
struct shortconv_snaps {
    static constexpr int n_max = 8;
    char *  dst[n_max];
    int64_t dst_nb1[n_max];
    int64_t off[n_max];
    int     n;
};

static __global__ void shortconv_window_f32(const char * state, const char * b, const char * x, char * out,
                                            const int64_t d, const int64_t n_t, const int64_t nr,
                                            const int64_t st_nb0, const int64_t st_nb1, const int64_t st_nb2,
                                            const int64_t b_nb0, const int64_t b_nb1, const int64_t b_nb2,
                                            const int64_t x_nb0, const int64_t x_nb1, const int64_t x_nb2,
                                            const int64_t o_nb0, const int64_t o_nb1, const int64_t o_nb2,
                                            const shortconv_snaps snaps) {
    const int64_t ch = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t s  = blockIdx.y;
    if (ch >= nr) {
        return;
    }
    for (int64_t t = 0; t < d; ++t) {
        *(float *) (out + t*o_nb0 + ch*o_nb1 + s*o_nb2) = *(const float *) (state + t*st_nb0 + ch*st_nb1 + s*st_nb2);
    }
    for (int64_t t = 0; t < n_t; ++t) {
        const float bv = *(const float *) (b + ch*b_nb0 + t*b_nb1 + s*b_nb2);
        const float xv = *(const float *) (x + ch*x_nb0 + t*x_nb1 + s*x_nb2);
        *(float *) (out + (d + t)*o_nb0 + ch*o_nb1 + s*o_nb2) = bv * xv;
    }
    for (int k = 0; k < snaps.n; ++k) {
        for (int64_t c = 0; c < d; ++c) {
            *(float *) (snaps.dst[k] + s*snaps.dst_nb1[k] + (ch*d + c)*sizeof(float)) =
                *(const float *) (out + (snaps.off[k] + c)*o_nb0 + ch*o_nb1 + s*o_nb2);
        }
    }
}

int ggml_cuda_try_fuse_shortconv_window(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!shortconv_fuse_enabled() || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mul = cgraph->nodes[i];
    ggml_tensor * tr  = cgraph->nodes[i + 1];
    ggml_tensor * cat = cgraph->nodes[i + 2];
    if (mul->op != GGML_OP_MUL || tr->op != GGML_OP_TRANSPOSE || cat->op != GGML_OP_CONCAT) {
        return 0;
    }
    if (tr->src[0] != mul || cat->src[1] != tr || ggml_get_op_params_i32(cat, 0) != 0) {
        return 0;
    }
    // the product is read only through the transposed view, and that view only by the window (a view: ggml_node_has_n_uses refuses it)
    if (!ggml_node_has_n_uses(cgraph, i, 1) || ggml_node_get_use_count(cgraph, i + 1) != 1) {
        return 0;
    }
    const ggml_tensor * b     = mul->src[0];
    const ggml_tensor * x     = mul->src[1];
    const ggml_tensor * state = cat->src[0];
    if (!shortconv_f32(b) || !shortconv_f32(x) || !shortconv_f32(state) || !shortconv_f32(mul) || !shortconv_f32(cat)) {
        return 0;
    }
    // no broadcast in the product, 3-d tensors, and a window of at most 32 new tokens
    if (!ggml_are_same_shape(b, x) || !ggml_are_same_shape(b, mul) || mul->ne[3] != 1 || cat->ne[3] != 1 || state->ne[3] != 1) {
        return 0;
    }
    const int64_t nr  = mul->ne[0];
    const int64_t n_t = mul->ne[1];
    const int64_t n_s = mul->ne[2];
    const int64_t d   = state->ne[0];
    if (n_t > 32 || state->ne[1] != nr || state->ne[2] != n_s || cat->ne[0] != d + n_t || cat->ne[1] != nr || cat->ne[2] != n_s) {
        return 0;
    }
    // the window must not alias what it reads
    const char * o0 = (const char *) cat->data;
    const char * o1 = o0 + ggml_nbytes(cat);
    for (const ggml_tensor * t : { b, x, state }) {
        const char * a0 = (const char *) t->data;
        const char * a1 = a0 + ggml_nbytes(t);
        if (a0 < o1 && o0 < a1) {
            return 0;
        }
    }
    // k15: the state snapshots right after the window (views and CPYs; any other node ends the scan)
    static const bool state_on = [] {
        const char * e = getenv("GGML_CUDA_SHORTCONV_STATE");
        return e == nullptr || atoi(e) != 0;
    }();
    shortconv_snaps snaps = {};
    int n_extra = 0;
    if (state_on) {
        for (int j = i + 3; j < cgraph->n_nodes && snaps.n < shortconv_snaps::n_max; ++j) {
            const ggml_tensor * node = cgraph->nodes[j];
            if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE) {
                continue;
            }
            const ggml_tensor * src = node->src[0];
            const ggml_tensor * dsn = node->src[1];
            if (node->op != GGML_OP_CPY || src == nullptr || dsn == nullptr || src->view_src != cat || src->type != GGML_TYPE_F32 ||
                    dsn->type != GGML_TYPE_F32 || src->ne[0] != d || src->ne[1] != nr || src->ne[2] != n_s || src->ne[3] != 1 ||
                    src->nb[0] != (size_t) cat->nb[0] || src->nb[1] != cat->nb[1] || src->nb[2] != cat->nb[2] || src->view_offs % cat->nb[0] != 0 ||
                    (int64_t) (src->view_offs / cat->nb[0]) + d > cat->ne[0] || dsn->ne[0] != d*nr || dsn->ne[1] != n_s || dsn->ne[2] != 1 ||
                    dsn->ne[3] != 1 || dsn->nb[0] != sizeof(float) || node->data != dsn->data) {
                break;
            }
            // the snapshot must not overlap what the kernel reads or the window it copies from
            const char * d0 = (const char *) dsn->data;
            const char * d1 = d0 + (n_s - 1)*dsn->nb[1] + d*nr*sizeof(float);
            bool clash = false;
            for (const ggml_tensor * t : { b, x, state, (const ggml_tensor *) cat }) {
                const char * a0 = (const char *) t->data;
                const char * a1 = a0 + ggml_nbytes(t);
                clash = clash || (a0 < d1 && d0 < a1);
            }
            if (clash) {
                break;
            }
            snaps.dst[snaps.n]     = (char *) dsn->data;
            snaps.dst_nb1[snaps.n] = dsn->nb[1];
            snaps.off[snaps.n]     = src->view_offs / cat->nb[0];
            snaps.n++;
            n_extra = j - (i + 2);
        }
    }
    const int threads = 128;
    const dim3 blocks((unsigned) ((nr + threads - 1)/threads), (unsigned) n_s, 1);
    shortconv_window_f32<<<blocks, threads, 0, ctx.stream()>>>((const char *) state->data, (const char *) b->data, (const char *) x->data,
        (char *) cat->data, d, n_t, nr, state->nb[0], state->nb[1], state->nb[2], b->nb[0], b->nb[1], b->nb[2],
        x->nb[0], x->nb[1], x->nb[2], cat->nb[0], cat->nb[1], cat->nb[2], snaps);
    CUDA_CHECK(cudaGetLastError());
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: LFM2 short-conv window fused (b * x into the conv window, %lld tokens, %d state snapshots; GGML_CUDA_SHORTCONV_FUSE=0: off)\n",
            __func__, (long long) n_t, snaps.n);
    }
    return 2 + n_extra;
}

// ---------------------------------------------------------------------------------------------------------- 2. ssm_conv * c
// ssm_conv_f32 of ssm-conv.cu (no bias, no silu), the result multiplied by c
template <size_t split_d_inner, size_t d_conv>
static __global__ void shortconv_conv_mul_f32(const float * __restrict__ src0, const float * __restrict__ src1, const char * cmul,
                                              const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                              const int64_t c_nb0, const int64_t c_nb1, const int64_t c_nb2,
                                              float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                              const int64_t n_t, void * __restrict__ vq = nullptr, const int ncols_padded = 0) {
    // k15: vq (nullable): the q8_1 copy of the output for the out_proj mul_mat_vec_q (ggml_cuda_q8_1_cache), rows = tokens; a warp holds
    // 32 consecutive channels of a token, lane = element of its q8_1 block, the same 32-lane reductions as quantize_q8_1, so the copy is
    // bit-identical to quantizing the f32 output
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);
    const char *  c_block = cmul + bidx * c_nb2 + (bidy * split_d_inner + tid) * c_nb0;

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    const float b = 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        const float cv = *(const float *) (c_block + i * c_nb1);
        const float yv = cv * sumf;
        y_block[i * stride_y + tid] = yv;
        if (vq != nullptr) {
            float amax = fabsf(yv);
            float sum  = yv;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(yv / d);
            const int64_t ch  = (int64_t) bidy * split_d_inner + tid;
            block_q8_1 * yq = (block_q8_1 *) vq + ((int64_t) bidx * n_t + i) * (ncols_padded / QK8_1) + ch / QK8_1;
            const int iqs = ch % QK8_1;
            yq->qs[iqs] = q;
            if (iqs == 0) {
                yq->ds = make_half2(d, sum);
            }
        }
    }
}

int ggml_cuda_try_fuse_shortconv_conv_mul(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!shortconv_fuse_enabled() || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * conv = cgraph->nodes[i];
    ggml_tensor * mul  = cgraph->nodes[i + 1];
    if (conv->op != GGML_OP_SSM_CONV || mul->op != GGML_OP_MUL || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return 0;
    }
    if (mul->src[0] != conv && mul->src[1] != conv) {
        return 0;
    }
    const ggml_tensor * c    = mul->src[0] == conv ? mul->src[1] : mul->src[0];
    const ggml_tensor * src0 = conv->src[0];
    const ggml_tensor * src1 = conv->src[1];
    if (!shortconv_f32(c) || !shortconv_f32(conv) || !shortconv_f32(mul) || !shortconv_f32(src0) || !shortconv_f32(src1)) {
        return 0;
    }
    if (!ggml_are_same_shape(c, conv) || !ggml_are_same_shape(mul, conv) || conv->ne[3] != 1) {
        return 0;
    }
    const int64_t nc  = src1->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = conv->ne[1];
    const int64_t n_s = conv->ne[2];
    const int threads = 128;
    if (n_t > 32 || nr % threads != 0 || nc != 3 || conv->ne[0] != nr || src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) ||
            src0->nb[1] != src0->ne[0]*sizeof(float)) {
        return 0;
    }
    // the product may be in place on the conv result (gallocr): it only must not overlap the window or the kernel
    const char * o0 = (const char *) mul->data;
    const char * o1 = o0 + ggml_nbytes(mul);
    for (const ggml_tensor * t : { src0, src1, c }) {
        const char * a0 = (const char *) t->data;
        const char * a1 = a0 + ggml_nbytes(t);
        if (a0 < o1 && o0 < a1 && t != c) {
            return 0;
        }
    }
    {   // c may alias the output only element for element (in place): the same offsets
        const char * a0 = (const char *) c->data;
        const char * a1 = a0 + ggml_nbytes(c);
        if (a0 < o1 && o0 < a1 && (a0 != o0 || c->nb[1] != mul->nb[1] || c->nb[2] != mul->nb[2])) {
            return 0;
        }
    }
    // k15: the q8_1 copy of the product for the out_proj reader (GGML_CUDA_SHORTCONV_Q8_1=0: off)
    static const bool q8_on = [] {
        const char * e = getenv("GGML_CUDA_SHORTCONV_Q8_1");
        return e == nullptr || atoi(e) != 0;
    }();
    void * q8_1_out = nullptr;
    const int ncols_padded = (int) GGML_PAD(mul->ne[0], MATRIX_ROW_PADDING);
    ggml_cuda_q8_1_cache & cache = ctx.q8_1_cache;
    // (rows without padding only: the padding blocks would need zeros the kernel does not write)
    if (q8_on && cache.active() && ctx.curr_stream_no == 0 && cache.norm_targets.count(mul) && ggml_is_contiguous(mul) && ncols_padded == mul->ne[0] &&
            mul->nb[0] == sizeof(float)) {
        int64_t key[ggml_cuda_q8_1_cache::n_key];
        ggml_cuda_q8_1_cache::key_contiguous(mul->ne[0], ggml_nrows(mul), key);
        q8_1_out = cache.reserve(mul->data, ggml_nbytes(mul), key, ggml_nrows(mul)*ncols_padded/QK8_1*sizeof(block_q8_1));
    }
    const dim3 blocks((unsigned) n_s, (unsigned) ((nr + threads - 1)/threads), 1);
    shortconv_conv_mul_f32<threads, 3><<<blocks, threads, 0, ctx.stream()>>>((const float *) src0->data, (const float *) src1->data,
        (const char *) c->data, (int) src0->nb[1], (int) src0->nb[2], (int) src1->nb[1], c->nb[0], c->nb[1], c->nb[2],
        (float *) mul->data, (int) mul->nb[0], (int) mul->nb[1], (int) mul->nb[2], n_t, q8_1_out, ncols_padded);
    CUDA_CHECK(cudaGetLastError());
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: LFM2 short conv fused with its output gate (%lld tokens; GGML_CUDA_SHORTCONV_FUSE=0: off)\n", __func__, (long long) n_t);
    }
    return 1;
}
