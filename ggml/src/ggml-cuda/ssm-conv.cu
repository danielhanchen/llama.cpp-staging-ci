// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"
#include "cpy.cuh"
#include "concat.cuh"

// k15-fn: conv-state copies the short-token kernel also writes (see ggml_cuda_try_conv_state_defer): copy c writes, for every channel
// and sequence, the cols floats of the channel's input row starting at column s_idx[c] to dst[c] + seq * seq_stride[c] + ch * ch_stride[c]
struct ssm_conv_state_copies {
    static constexpr int max_copies = 4;
    float * dst[max_copies];
    int64_t ch_stride[max_copies];
    int64_t seq_stride[max_copies];
    int     s_idx[max_copies];
    int     n;
    int     cols;
};

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t, const ssm_conv_state_copies cs) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
    // the deferred conv-state copies: the same floats a CPY of the input's tail writes (they read the input, which this kernel only reads)
    for (int c = 0; c < cs.n; ++c) {
        const float * s = x_block + tid * stride_x + cs.s_idx[c];
        float * d = cs.dst[c] + bidx * cs.seq_stride[c] + (int64_t) (bidy * split_d_inner + tid) * cs.ch_stride[c];
        for (int k = 0; k < cs.cols; ++k) {
            d[k] = s[k];
        }
    }
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

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
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// k15-fn: the conv input is the CONCAT (dim 0) of d_conv - 1 state columns a and the n_t <= max_t new columns b, not run
// (ggml_cuda_try_conv_cat_elide): a thread loads all of its channel's inputs into registers first, then writes the deferred conv-state
// copies (which may write the very state row it read: each channel's row is private to its thread, checked by the caller), then the
// convolution, with the same float operations in the same order as ssm_conv_f32
struct ssm_conv_cat {
    const float * a;
    const float * b;
    int64_t a_nb1, a_nb2;           // floats
    int64_t b_nb0, b_nb1, b_nb2;    // floats
};

static constexpr int ssm_conv_cat_max_t = 8;

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_cat_f32(const ssm_conv_cat cat, const float * src1_ptr, const float * bias_ptr, const int src1_nb1,
                                        float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                        const int64_t n_t, const ssm_conv_state_copies cs) {
    constexpr int sc = d_conv - 1;
    constexpr int nx = sc + ssm_conv_cat_max_t;
    ggml_cuda_pdl_lc();
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int64_t ch = (int64_t) bidy * split_d_inner + tid;

    const float * ap = cat.a + bidx * cat.a_nb2 + ch * cat.a_nb1;
    const float * bp = cat.b + bidx * cat.b_nb2 + ch * cat.b_nb1;
    const float * w_block = (const float *) ((const char *) src1_ptr + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst_ptr + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[nx];
    float w[d_conv];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int t = 0; t < sc; ++t) {
        x[t] = ap[t];
    }
#pragma unroll
    for (int t = 0; t < ssm_conv_cat_max_t; ++t) {
        x[sc + t] = t < n_t ? bp[t * cat.b_nb0] : 0.0f;
    }
    for (int c = 0; c < cs.n; ++c) {
        float * d = cs.dst[c] + bidx * cs.seq_stride[c] + ch * cs.ch_stride[c];
        const int s = cs.s_idx[c];
#pragma unroll
        for (int p = 0; p < nx; ++p) {
            if (p >= s && p < s + cs.cols) {
                d[p - s] = x[p];
            }
        }
    }
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias_ptr != nullptr ? bias_ptr[ch] : 0.0f;

#pragma unroll
    for (int i = 0; i < ssm_conv_cat_max_t; i++) {
        if (i >= n_t) {
            break;
        }
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream, const ssm_conv_state_copies & cs,
                              const ssm_conv_cat * cat) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (cat != nullptr) {
            GGML_ASSERT(n_t <= ssm_conv_cat_max_t);
            const dim3 blocks(n_s, nr / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_cat_f32<apply_silu, threads, kNC>, launch_params, *cat, src1, bias, src1_nb1, dst, dst_nb0,
                                    dst_nb1, dst_nb2, n_t, cs);
        } else if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t, cs);
        } else {
            GGML_ASSERT(cs.n == 0);
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    // k15-fn: the conv-state copies deferred to this convolution (ggml_cuda_try_conv_state_defer); a long input runs them as copies first
    ssm_conv_state_copies cs = {};
    std::vector<ggml_tensor *> cpys;
    auto pend = ctx.conv_state_pending.find(dst);
    if (pend != ctx.conv_state_pending.end()) {
        cpys = std::move(pend->second);
        ctx.conv_state_pending.erase(pend);
    }

    // k15-fn: the conv input is a CONCAT that did not run (ggml_cuda_try_conv_cat_elide): read its two parts, or run it now when this
    // launch cannot (too many tokens or copies, a copy or the output over a part other than the thread-private state row)
    ssm_conv_cat cat = {};
    bool use_cat = false;
    if (!ctx.conv_cat_elided.empty() && ctx.conv_cat_elided.count(src0)) {
        ctx.conv_cat_elided.erase(src0);
        const ggml_tensor * a = src0->src[0];
        const ggml_tensor * b = src0->src[1];
        auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
            const char * x0 = (const char *) x->data;
            const char * y0 = (const char *) y->data;
            return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
        };
        use_cat = n_t <= ssm_conv_cat_max_t && (int) cpys.size() <= ssm_conv_state_copies::max_copies && a->ne[0] == nc - 1 &&
                  b->ne[0] == n_t && !overlap(out, a) && !overlap(out, b);
        for (const ggml_tensor * c : cpys) {
            const ggml_tensor * d = c->src[1];
            if (overlap(d, b) || (overlap(d, a) && !(d->data == a->data && d->nb[1] == a->nb[1] && d->nb[2] == a->nb[2] &&
                                                     ggml_are_same_shape(d, a)))) {
                use_cat = false;
            }
        }
        if (use_cat) {
            cat.a     = (const float *) a->data;
            cat.b     = (const float *) b->data;
            cat.a_nb1 = a->nb[1] / sizeof(float);
            cat.a_nb2 = a->nb[2] / sizeof(float);
            cat.b_nb0 = b->nb[0] / sizeof(float);
            cat.b_nb1 = b->nb[1] / sizeof(float);
            cat.b_nb2 = b->nb[2] / sizeof(float);
        } else {
            static const bool trace = getenv("GGML_CUDA_FUSION_TRACE") != nullptr && atoi(getenv("GGML_CUDA_FUSION_TRACE")) != 0;
            if (trace) {
                GGML_LOG_INFO("conv cat elide: %s runs at its convolution (n_t %d, %d copies)\n", src0->name, (int) n_t, (int) cpys.size());
            }
            ggml_cuda_op_concat(ctx, const_cast<ggml_tensor *>(src0));
        }
    }

    if (!cpys.empty()) {
        if (n_t > 32 || (int) cpys.size() > ssm_conv_state_copies::max_copies) {
            for (ggml_tensor * c : cpys) {
                ggml_cuda_cpy(ctx, c->src[0], c->src[1]);
            }
        } else {
            for (ggml_tensor * c : cpys) {
                const ggml_tensor * t = c->src[0];
                cs.dst[cs.n]        = (float *) c->src[1]->data;
                cs.ch_stride[cs.n]  = c->src[1]->nb[1] / sizeof(float);
                cs.seq_stride[cs.n] = c->src[1]->nb[2] / sizeof(float);
                cs.s_idx[cs.n]      = (int) (((const char *) t->data - (const char *) src0->data) / sizeof(float));
                cs.cols             = (int) t->ne[0];
                cs.n++;
            }
        }
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream, cs, use_cat ? &cat : nullptr);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream, cs, use_cat ? &cat : nullptr);
    }
}
