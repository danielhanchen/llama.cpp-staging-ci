#include "unary.cuh"
#include "convert.cuh"

#include <type_traits>

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
    return ggml_cuda_op_gelu_single(x);
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_relu_sqr(float x) {
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    return roundf(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[i])));
}

template <float (*op)(float), typename T>
static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel<op, T>, launch_params, x, dst, k);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    int64_t rem = i;
    const int64_t i0 = rem % ne00; rem /= ne00;
    const int64_t i1 = rem % ne01; rem /= ne01;
    const int64_t i2 = rem % ne02;
    const int64_t i3 = rem / ne02;
    const size_t src_byte_offset = i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    const T * src_ptr = (const T *)((const char *)x + src_byte_offset);

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(*src_ptr)));
}

template <float (*op)(float), typename T>
static void unary_cuda_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel_strided<op, T>, launch_params, x, dst, k, ne00,ne01,ne02,nb00,nb01,nb02,nb03);
}

template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    if (ggml_is_contiguous(src0)) {
        if (src0->type == GGML_TYPE_F16) {
            unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            unary_cuda<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), stream);
        } else {
            unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
        }
    } else {
        if (src0->type == GGML_TYPE_F16) {
            unary_cuda_strided<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            unary_cuda_strided<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else {
            unary_cuda_strided<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        }
    }
}

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_abs>(ctx, dst);
}

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sgn>(ctx, dst);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_neg>(ctx, dst);
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_step>(ctx, dst);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu>(ctx, dst);
}

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_quick>(ctx, dst);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_silu>(ctx, dst);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_tanh>(ctx, dst);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_relu>(ctx, dst);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sigmoid>(ctx, dst);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardsigmoid>(ctx, dst);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardswish>(ctx, dst);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_exp>(ctx, dst);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqr>(ctx, dst);
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqrt>(ctx, dst);
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sin>(ctx, dst);
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_cos>(ctx, dst);
}

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_log>(ctx, dst);
}

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_elu>(ctx, dst);
}

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_floor>(ctx, dst);
}

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_ceil>(ctx, dst);
}

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_round>(ctx, dst);
}

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_trunc>(ctx, dst);
}

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_expm1>(ctx, dst);
}

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_softplus>(ctx, dst);
}
/* gated ops */

// k15-fn: xn0 > 0: the rows of x are split into blocks of xn0 values xs1 apart (the strided view a CONT that did not run would have
// copied, see ggml_cuda_try_unary_src_cont_elide); the value read is the one that copy would hold
template <float (*op)(float), typename T, bool x_split = false>
static __global__ void unary_gated_op_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1,
        const int64_t xn0 = 0, const int64_t xs1 = 0) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = x_split ? (i / n) * o0 + ((i % n) / xn0) * xs1 + (i % n) % xn0 : (i / n) * o0 + (i % n);
    const int64_t j1 = (o0 == o1 && !x_split) ? j0 : (i / n) * o1 + (i % n);

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[j0])) * ggml_cuda_cast<float>(g[j1]));
}

// k15-mm2: unary_gated_op_kernel (F32) plus the q8_1 copy of dst for the mul_mat_vec_q that reads it (see ggml_cuda_q8_1_cache).
// Thread i holds q8_1 element i of rows of n values padded to nq (nq == n: no padding, the rows are one flat run), computes it with
// the expression of unary_gated_op_kernel (0 in the padding) and quantizes it as quantize_q8_1 does (one value per lane, the same
// 32-lane reductions), so dst and the copy have the bytes of the unfused unary_gated_op_kernel + quantize_q8_1 pair.
template <float (*op)(float), bool x_split = false>
static __global__ void unary_gated_q8_1_kernel(const float * x, const float * g, float * dst, void * vq, const int64_t kq, const int64_t n,
        const int64_t nq, const int64_t o0, const int64_t o1, const int64_t xn0 = 0, const int64_t xs1 = 0) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= kq) {
        return;
    }
    const int64_t r = i / nq;
    const int64_t c = i % nq;

    ggml_cuda_pdl_sync();
    float v = 0.0f;
    if (c < n) {
        const int64_t j0 = x_split ? r * o0 + (c / xn0) * xs1 + c % xn0 : r * o0 + c;
        const int64_t j1 = (o0 == o1 && !x_split) ? j0 : r * o1 + c;
        v = op(x[j0]) * g[j1];
        dst[r*n + c] = v;
    }

    float amax = fabsf(v);
    float sum  = v;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(v / d);

    block_q8_1 * y = (block_q8_1 *) vq;
    const int64_t ib  = i / QK8_1;
    const int     iqs = i % QK8_1;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(d, sum);
    }
}

// k15-mm2: GGML_CUDA_GATED_Q8_1=0 turns the q8_1 copy of gated / unary-mul outputs off
static bool ggml_cuda_gated_q8_1_disabled() {
    // GGML_CUDA_DISABLE_GLU_Q8_1=1 (k15-fn 414e7c2d4a5, which this replaces) is an alias
    static const bool off = [] {
        const char * e = getenv("GGML_CUDA_GATED_Q8_1");
        const char * d = getenv("GGML_CUDA_DISABLE_GLU_Q8_1");
        return (e != nullptr && atoi(e) == 0) || (d != nullptr && atoi(d) != 0);
    }();
    return off;
}

// the q8_1 copy for a contiguous F32 dst of nrows rows of n values that a mul_mat_vec_q reads (a q8_1 cache producer target), or nullptr.
// The reader's rows are either these rows (nq = n padded to MATRIX_ROW_PADDING) or rows of a multiple of MATRIX_ROW_PADDING (nq = n:
// the copy is the flat run of all values, see ggml_cuda_q8_1_cache::key_contiguous)
static void * unary_gated_q8_1_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const int64_t n, int64_t & nq) {
    ggml_cuda_q8_1_cache & cache = ctx.q8_1_cache;
    if (ggml_cuda_gated_q8_1_disabled() || !cache.active() || ctx.curr_stream_no != 0 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(dst) || !cache.norm_targets.count(dst)) {
        return nullptr;
    }
    const auto it = cache.norm_target_ne10.find(dst);
    const int64_t ne10 = it != cache.norm_target_ne10.end() ? it->second : n;
    const int64_t nelem = ggml_nelements(dst);
    int64_t key[ggml_cuda_q8_1_cache::n_key];
    if (ne10 == n) {
        nq = GGML_PAD(n, MATRIX_ROW_PADDING);
        ggml_cuda_q8_1_cache::key_contiguous(n, nelem / n, key);
    } else {
        // the reader's rows carry no padding: one flat run
        if (ne10 % MATRIX_ROW_PADDING != 0 || nelem % ne10 != 0 || ggml_cuda_q8_1_cache::reshape_disabled()) {
            return nullptr;
        }
        nq = n;
        ggml_cuda_q8_1_cache::key_contiguous(ne10, nelem / ne10, key);
    }
    const int64_t kq = (nelem / n) * nq;
    return cache.reserve(dst->data, ggml_nbytes(dst), key, kq/QK8_1*sizeof(block_q8_1));
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream,
        void * q8_1_out = nullptr, const int64_t nq = 0, const int64_t xn0 = 0, const int64_t xs1 = 0) {
    if constexpr (std::is_same_v<T, float>) {
        if (q8_1_out != nullptr) {
            static_assert(CUDA_GLU_BLOCK_SIZE % QK8_1 == 0, "a warp must cover whole q8_1 blocks");
            const int64_t kq = (k / n) * nq;
            const int64_t num_blocks = (kq + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
            if (xn0 > 0) {
                ggml_cuda_kernel_launch(unary_gated_q8_1_kernel<op, true>, launch_params, x, g, dst, q8_1_out, kq, n, nq, o0, o1, xn0, xs1);
            } else {
                ggml_cuda_kernel_launch(unary_gated_q8_1_kernel<op>, launch_params, x, g, dst, q8_1_out, kq, n, nq, o0, o1, (int64_t) 0, (int64_t) 0);
            }
            return;
        }
    }
    GGML_ASSERT(q8_1_out == nullptr);
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    if (xn0 > 0) {
        ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T, true>, launch_params, x, g, dst, k, n, o0, o1, xn0, xs1);
    } else {
        ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T>, launch_params, x, g, dst, k, n, o0, o1, (int64_t) 0, (int64_t) 0);
    }
}

template <float (*op)(float)>
void ggml_cuda_op_unary_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (half *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        nv_bfloat16 * src0_p = (nv_bfloat16 *) src0_d;
        nv_bfloat16 * src1_p = (nv_bfloat16 *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (nv_bfloat16 *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(nv_bfloat16), src1_o / sizeof(nv_bfloat16), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        // k15-mm2: a mul_mat_vec_q reads dst: its q8_1 copy from this kernel
        int64_t nq = 0;
        void * q8_1_out = unary_gated_q8_1_reserve(ctx, dst, nc, nq);
        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream,
            q8_1_out, nq);
    }
    GGML_UNUSED(dst_d);
    GGML_UNUSED(stream);
}

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_relu>(ctx, dst);
}

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu>(ctx, dst);
}

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_silu>(ctx, dst);
}

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst);
}

// swiglu_oai

template <typename T>
static __global__ void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float alpha, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = ggml_cuda_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_oai_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1, alpha, limit);
}

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_cuda(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

// swiglu_clamp

template <typename T>
static __global__ void swiglu_clamp_kernel(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    dst[i] = (T) ggml_cuda_op_swiglu_clamp_single((float) gate[j0], (float) up[j1], limit);
}

template <typename T>
static void swiglu_clamp_cuda(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_clamp_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(gate, up, dst, k, n, o0, o1, limit);
}

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float limit = ggml_get_op_params_f32(dst, 3);

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (half *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), limit, stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (float *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), limit, stream);
    }
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xi = ggml_cuda_cast<float>(x[i]);

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fminf(xi, eps);
    const float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    dst[i] = ggml_cuda_cast<T>(out);
}

template <typename T>
static void xielu_cuda(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE;
    xielu_kernel<<<num_blocks, CUDA_XIELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, alpha_n, alpha_p, beta, eps);
}

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16 ||  dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    if (src0->type == GGML_TYPE_F16) {
        xielu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        xielu_cuda((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}



/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return grad * s * (1.0f + x * (1.0f - s));
}

template <class T>
static __global__ void silu_back_kernel(const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_silu_back((float)grad[i], (float)xf[i]);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
    return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template <class T>
static __global__ void leaky_relu_kernel(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_leaky_relu((float)x[i], negative_slope);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_kernel<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

/* fused unary + mul */

template <float (*op)(float)>
static void ggml_cuda_op_unary_mul_impl(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node,
        const ggml_tensor * split_src = nullptr) {
    // unary_node: UNARY op applied to unary_node->src[0]
    // mul_node:   MUL(a, b) where one of a/b is unary_node
    // Output goes to mul_node->data

    const ggml_tensor * unary_src = unary_node->src[0];  // input to the unary op
    const ggml_tensor * other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    // k15-fn: the unary input is a CONT that did not run: read its source, a 3D f32 view [n0, n1, rows] (ggml_cuda_try_unary_src_cont_elide)
    if (split_src != nullptr) {
        GGML_ASSERT(unary_src->type == GGML_TYPE_F32 && other_src->type == GGML_TYPE_F32 && mul_node->type == GGML_TYPE_F32);
        GGML_ASSERT(split_src->type == GGML_TYPE_F32 && split_src->nb[0] == sizeof(float) && split_src->ne[3] == 1);
        GGML_ASSERT(ggml_is_contiguous_1(other_src) && other_src->nb[0] == sizeof(float) && ggml_are_same_shape(unary_src, other_src));
        GGML_ASSERT(unary_src->ne[0] == split_src->ne[0] * split_src->ne[1] && ggml_nrows(unary_src) == split_src->ne[2]);
        const int64_t k  = ggml_nelements(mul_node);
        const int64_t nc = unary_src->ne[0];
        int64_t nq = 0;
        void * q8_1_out = unary_gated_q8_1_reserve(ctx, mul_node, nc, nq);
        unary_gated_cuda<op>((const float *) split_src->data, (const float *) other_src->data, (float *) mul_node->data, k, nc,
                             split_src->nb[2] / sizeof(float), other_src->nb[1] / sizeof(float), ctx.stream(), q8_1_out, nq,
                             split_src->ne[0], split_src->nb[1] / sizeof(float));
        return;
    }

    GGML_ASSERT(ggml_is_contiguous_1(unary_src));
    GGML_ASSERT(unary_src->nb[0] == ggml_element_size(unary_src));
    GGML_ASSERT(ggml_is_contiguous_1(other_src));
    GGML_ASSERT(other_src->nb[0] == ggml_element_size(other_src));
    GGML_ASSERT(ggml_are_same_shape(unary_src, other_src));

    GGML_ASSERT(unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16 || unary_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(unary_src->type == other_src->type);
    GGML_ASSERT(unary_src->type == mul_node->type);

    cudaStream_t stream = ctx.stream();

    const int64_t k  = ggml_nelements(mul_node);
    const int64_t nc = unary_src->ne[0];
    const int64_t unary_stride = unary_src->nb[1];
    const int64_t other_stride = other_src->nb[1];

    if (unary_src->type == GGML_TYPE_F16) {
        unary_gated_cuda<op>((const half *) unary_src->data, (const half *) other_src->data,
                             (half *) mul_node->data, k, nc,
                             unary_stride / sizeof(half), other_stride / sizeof(half), stream);
    } else if (unary_src->type == GGML_TYPE_BF16) {
        unary_gated_cuda<op>((const nv_bfloat16 *) unary_src->data, (const nv_bfloat16 *) other_src->data,
                             (nv_bfloat16 *) mul_node->data, k, nc,
                             unary_stride / sizeof(nv_bfloat16), other_stride / sizeof(nv_bfloat16), stream);
    } else {
        // k15-mm2: a mul_mat_vec_q reads the product: its q8_1 copy from this kernel
        int64_t nq = 0;
        void * q8_1_out = unary_gated_q8_1_reserve(ctx, mul_node, nc, nq);
        unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                             (float *) mul_node->data, k, nc,
                             unary_stride / sizeof(float), other_stride / sizeof(float), stream, q8_1_out, nq);
    }
}

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node, const ggml_tensor * split_src) {
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:
            ggml_cuda_op_unary_mul_impl<op_silu>(ctx, unary_node, mul_node, split_src);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ggml_cuda_op_unary_mul_impl<op_sigmoid>(ctx, unary_node, mul_node, split_src);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            ggml_cuda_op_unary_mul_impl<op_softplus>(ctx, unary_node, mul_node, split_src);
            break;
        default:
            GGML_ABORT("Unsupported unary op for fused unary+mul");
    }
}

/* fused relu + sqr */

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node) {
    const ggml_tensor * src = relu_node->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}

/* fused chains of single-input elementwise ops */

enum ggml_cuda_ew_code {
    GGML_CUDA_EW_SCALE = 0,
    GGML_CUDA_EW_CLAMP,
    GGML_CUDA_EW_SQR,
    GGML_CUDA_EW_SQRT,
    GGML_CUDA_EW_SIN,
    GGML_CUDA_EW_COS,
    GGML_CUDA_EW_LOG,
    GGML_CUDA_EW_UNARY, // + ggml_unary_op
};

int ggml_cuda_ew_chain_op_code(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_SCALE: return GGML_CUDA_EW_SCALE;
        case GGML_OP_CLAMP: return GGML_CUDA_EW_CLAMP;
        case GGML_OP_SQR:   return GGML_CUDA_EW_SQR;
        case GGML_OP_SQRT:  return GGML_CUDA_EW_SQRT;
        case GGML_OP_SIN:   return GGML_CUDA_EW_SIN;
        case GGML_OP_COS:   return GGML_CUDA_EW_COS;
        case GGML_OP_LOG:   return GGML_CUDA_EW_LOG;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return GGML_CUDA_EW_UNARY + (int) ggml_get_unary_op(node);
                default:
                    return -1;
            }
        default:
            return -1;
    }
}

// same expressions as the single-op kernels (scale_f32, op_clamp_kernel, unary_op_kernel), so the result is bit-identical
static __device__ __forceinline__ float ew_chain_apply(const int code, const float p0, const float p1, const float x) {
    switch (code) {
        case GGML_CUDA_EW_SCALE: return p0 * x + p1;
        case GGML_CUDA_EW_CLAMP: return fminf(fmaxf(x, p0), p1);
        case GGML_CUDA_EW_SQR:   return op_sqr(x);
        case GGML_CUDA_EW_SQRT:  return op_sqrt(x);
        case GGML_CUDA_EW_SIN:   return op_sin(x);
        case GGML_CUDA_EW_COS:   return op_cos(x);
        case GGML_CUDA_EW_LOG:   return op_log(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_ABS:         return op_abs(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_SGN:         return op_sgn(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_NEG:         return op_neg(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_STEP:        return op_step(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_TANH:        return op_tanh(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_ELU:         return op_elu(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_RELU:        return op_relu(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_SIGMOID:     return op_sigmoid(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_GELU:        return op_gelu(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_GELU_QUICK:  return op_gelu_quick(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_SILU:        return op_silu(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_HARDSWISH:   return op_hardswish(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_HARDSIGMOID: return op_hardsigmoid(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_EXP:         return op_exp(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_EXPM1:       return op_expm1(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_SOFTPLUS:    return op_softplus(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_GELU_ERF:    return op_gelu_erf(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_FLOOR:       return op_floor(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_CEIL:        return op_ceil(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_ROUND:       return op_round(x);
        case GGML_CUDA_EW_UNARY + GGML_UNARY_OP_TRUNC:       return op_trunc(x);
        default: return x; // unreachable: the host only builds chains of supported codes
    }
}

static __global__ void ew_chain_f32_kernel(const float * x, float * dst, const int64_t k, const ggml_cuda_ew_chain chain) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    float v = x[i];
#pragma unroll 1
    for (int j = 0; j < chain.n_ops; ++j) {
        v = ew_chain_apply(chain.op[j], chain.p0[j], chain.p1[j], v);
    }
    dst[i] = v;
}

// ew_chain_f32_kernel plus the q8_1 copy of the output: element i is computed and quantized by thread i, as quantize_q8_1
// does it (one value per lane, the same 32-lane reductions; CUDA_NEG_BLOCK_SIZE % QK8_1 == 0), padding values are 0
// k15-mm2: rows of ne0 values, their q8_1 copy padded to ne0q per row (ne0q == ne0 or a single row: the layout as before)
static __global__ void ew_chain_q8_1_f32_kernel(const float * x, float * dst, void * vq, const int64_t ne0, const int64_t ne0q, const int64_t kq,
        const ggml_cuda_ew_chain chain) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= kq) {
        return;
    }

    ggml_cuda_pdl_sync();
    const int64_t row = i / ne0q;
    const int64_t col = i - row*ne0q;
    float v = 0.0f;
    if (col < ne0) {
        const int64_t ix = row*ne0 + col;
        v = x[ix];
#pragma unroll 1
        for (int j = 0; j < chain.n_ops; ++j) {
            v = ew_chain_apply(chain.op[j], chain.p0[j], chain.p1[j], v);
        }
        dst[ix] = v;
    }

    float amax = fabsf(v);
    float sum  = v;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(v / d);

    block_q8_1 * y = (block_q8_1 *) vq;
    const int64_t ib  = i / QK8_1;
    const int     iqs = i % QK8_1;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(d, sum);
    }
}

void ggml_cuda_op_ew_chain(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, const ggml_cuda_ew_chain & chain,
        void * q8_1_out) {
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    GGML_ASSERT(chain.n_ops > 0 && chain.n_ops <= GGML_CUDA_EW_CHAIN_MAX);

    const int64_t k = ggml_nelements(dst);
    if (q8_1_out != nullptr) {
        static_assert(CUDA_NEG_BLOCK_SIZE % QK8_1 == 0, "a warp must cover whole q8_1 blocks");
        GGML_ASSERT(ggml_nrows(dst) == 1 || dst->ne[0] % QK8_1 == 0);
        const int64_t ne0  = ggml_nrows(dst) == 1 ? k : dst->ne[0];
        const int64_t ne0q = GGML_PAD(ne0, MATRIX_ROW_PADDING);
        const int64_t kq   = ggml_nrows(dst) * ne0q;
        const int64_t num_blocks = (kq + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
        ggml_cuda_kernel_launch(ew_chain_q8_1_f32_kernel, launch_params, (const float *) src->data, (float *) dst->data, q8_1_out, ne0, ne0q, kq, chain);
        return;
    }
    const int64_t num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(ew_chain_f32_kernel, launch_params, (const float *) src->data, (float *) dst->data, k, chain);
}

// ABS -> POOL_1D(MAX over a whole row) -> elementwise chain -> DIV(x, chain): the per-row power-of-two rescale of
// build_amax_src1_f16 (Flint MoE down projection) in one kernel. One warp per row. Bit-identical to the unfused ops:
// the row max is order-independent (fmaxf over |x| from -FLT_MAX, as pool1d_nchw_kernel), the chain uses ew_chain_apply
// (the single-op expressions) and the quotient is op_div's a / b.
static __global__ void amax_rescale_f32_kernel(const float * x, float * e_out, float * dst, void * vq, const int ne0, const int ne0q,
        const int64_t nrows, const ggml_cuda_ew_chain chain) {
    ggml_cuda_pdl_lc();
    const int64_t row = int64_t(blockIdx.x)*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (row >= nrows) {
        return;
    }
    ggml_cuda_pdl_sync();
    const float * xr = x + row*ne0;
    float m = -FLT_MAX;
    for (int i = lane; i < ne0; i += WARP_SIZE) {
        m = fmaxf(m, fabsf(xr[i]));
    }
    m = warp_reduce_max(m);
    float e = m;
#pragma unroll 1
    for (int j = 0; j < chain.n_ops; ++j) {
        e = ew_chain_apply(chain.op[j], chain.p0[j], chain.p1[j], e);
    }
    if (lane == 0) {
        e_out[row] = e;
    }
    float * dr = dst + row*ne0;
    if (vq == nullptr) {
        for (int i = lane; i < ne0; i += WARP_SIZE) {
            dr[i] = xr[i] / e;
        }
        return;
    }
    // with the q8_1 copy of dst (rows padded to ne0q, padding 0): lane i of block b holds element 32*b + i, as one warp of
    // quantize_q8_1 does, with its expressions, so the copy is bit-identical to quantizing dst afterwards
    block_q8_1 * y = (block_q8_1 *) vq + row*(ne0q/QK8_1);
    for (int i = lane; i < ne0q; i += WARP_SIZE) {
        float v = 0.0f;
        if (i < ne0) {
            v = xr[i] / e;
            dr[i] = v;
        }
        float amax = fabsf(v);
        float sum  = v;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
        y[i/QK8_1].qs[lane] = q;
        if (lane == 0) {
            y[i/QK8_1].ds = make_half2(d, sum);
        }
    }
}

void ggml_cuda_op_amax_rescale(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * e, ggml_tensor * dst,
        const ggml_cuda_ew_chain & chain, void * q8_1_out) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && e->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(e) && ggml_is_contiguous(dst));
    const int     ne0   = (int) x->ne[0];
    const int64_t nrows = ggml_nrows(x);
    GGML_ASSERT(ggml_nelements(e) == nrows && ggml_nelements(dst) == ggml_nelements(x));
    constexpr int rows_per_block = 4;
    const int64_t num_blocks = (nrows + rows_per_block - 1) / rows_per_block;
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params((dim3) num_blocks, rows_per_block*WARP_SIZE, 0, ctx.stream());
    static_assert(QK8_1 == WARP_SIZE, "one q8_1 block per warp iteration");
    GGML_ASSERT(q8_1_out == nullptr || nrows == 1 || ne0 % MATRIX_ROW_PADDING == 0);
    const int ne0q = q8_1_out ? (int) GGML_PAD(ne0, MATRIX_ROW_PADDING) : ne0;
    ggml_cuda_kernel_launch(amax_rescale_f32_kernel, launch_params, (const float *) x->data, (float *) e->data, (float *) dst->data,
        q8_1_out, ne0, ne0q, nrows, chain);
}

/* elementwise chain -> DSV4_HC_POST (no comb): the chain that turns the injection logits into the post weights (2*sigmoid(x/hc)
   in qwen4exp) is applied once per (stream, token) row, with the expressions of ew_chain_f32_kernel, and the update itself is
   the expression of dsv4_hc_post_f32<false>, so the output is bit-identical to the unfused sequence. One block row (y) per
   (stream, token) row, so the chain runs once per block and the per-element work is that of dsv4_hc_post_f32. */
static __global__ void hc_post_chain_f32(
        const float * x, const float * residual, const float * post, const ggml_cuda_ew_chain chain, float * dst,
        const int64_t n_embd, const int64_t hc, const int64_t n_tokens,
        const int64_t sx0, const int64_t sx1, const int64_t sr0, const int64_t sr1, const int64_t sr2,
        const int64_t sd0, const int64_t sd1, const int64_t sd2) {
    ggml_cuda_pdl_lc();
    __shared__ float wv_s;
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();

    for (int64_t r = blockIdx.y; r < hc*n_tokens; r += gridDim.y) {
        const int64_t idst = r % hc;
        const int64_t it   = r / hc;

        if (threadIdx.x == 0) {
            // the chain source is contiguous [hc, n_tokens], as ew_chain_f32_kernel reads it
            float wv = post[r];
            // constant indices: a dynamically indexed parameter struct is copied to the stack by every thread
#pragma unroll
            for (int j = 0; j < GGML_CUDA_EW_CHAIN_MAX; ++j) {
                if (j < chain.n_ops) {
                    wv = ew_chain_apply(chain.op[j], chain.p0[j], chain.p1[j], wv);
                }
            }
            wv_s = wv;
        }
        __syncthreads();
        const float wv = wv_s;
        if (i0 < n_embd) {
            float sum = x[i0*sx0 + it*sx1] * wv;
            sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        }
        __syncthreads(); // wv_s is rewritten for the next row
    }
}

void ggml_cuda_op_hc_post_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_ew_chain & chain, const ggml_tensor * post_src,
        ggml_tensor * hc_post) {
    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];
    GGML_ASSERT(ggml_is_contiguous(post_src) && ggml_nelements(post_src) == hc*n_tokens);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, (unsigned) std::min<int64_t>(hc*n_tokens, 65535), 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(hc_post_chain_f32, launch_params,
        (const float *) x->data, (const float *) residual->data, (const float *) post_src->data, chain, (float *) hc_post->data,
        n_embd, hc, n_tokens,
        (int64_t) (x->nb[0] / sizeof(float)), (int64_t) (x->nb[1] / sizeof(float)),
        (int64_t) (residual->nb[0] / sizeof(float)), (int64_t) (residual->nb[1] / sizeof(float)), (int64_t) (residual->nb[2] / sizeof(float)),
        (int64_t) (hc_post->nb[0] / sizeof(float)), (int64_t) (hc_post->nb[1] / sizeof(float)), (int64_t) (hc_post->nb[2] / sizeof(float)));
}

/* fused unary + mul with a one-element-per-row gate */

template <float (*op)(float)>
static __global__ void unary_mul_gate_f32_kernel(const float * g, const float * x, float * dst, const int64_t k, const int64_t n,
                                                 const int64_t sg, const int64_t sx) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t row = i / n;
    ggml_cuda_pdl_sync();
    // MUL(x, UNARY(g)) is x * op(g); float multiplication commutes, so this equals the unfused result
    dst[i] = op(g[row*sg]) * x[row*sx + i % n];
}

// k15-fn: the gated product added to a (ADD(a, MUL(x, UNARY(g))), the shared expert added to the routed experts): one kernel for the
// three ops. a + v equals v + a (float addition commutes), so this is bit-identical to the unfused ops
template <float (*op)(float)>
static __global__ void unary_mul_gate_add_f32_kernel(const float * g, const float * x, const float * a, float * dst, const int64_t k,
                                                     const int64_t n, const int64_t sg, const int64_t sx) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t row = i / n;
    ggml_cuda_pdl_sync();
    // no contraction into an fma: the unfused ops round the product before the add
    const float v = __fmul_rn(op(g[row*sg]), x[row*sx + i % n]);
    dst[i] = __fadd_rn(a[i], v);
}

template <float (*op)(float)>
static void unary_mul_gate_f32_cuda(const float * g, const float * x, float * dst, const int64_t k, const int64_t n,
                                    const int64_t sg, const int64_t sx, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_mul_gate_f32_kernel<op>, launch_params, g, x, dst, k, n, sg, sx);
}

bool ggml_cuda_unary_mul_gate_supported(const ggml_tensor * unary_node) {
    if (unary_node->op != GGML_OP_UNARY) {
        return false;
    }
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_EXP:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_op_unary_mul_gate(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    const ggml_tensor * g = unary_node->src[0];
    const ggml_tensor * x = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(g->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && mul_node->type == GGML_TYPE_F32);
    GGML_ASSERT(g->ne[0] == 1 && ggml_are_same_shape(x, mul_node));
    GGML_ASSERT(g->ne[1] == x->ne[1] && g->ne[2] == x->ne[2] && g->ne[3] == x->ne[3]);
    GGML_ASSERT(ggml_is_contiguous_1(g) && ggml_is_contiguous_1(x) && x->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(mul_node));

    const int64_t k  = ggml_nelements(mul_node);
    const int64_t n  = x->ne[0];
    const int64_t sg = g->nb[1] / sizeof(float);
    const int64_t sx = x->nb[1] / sizeof(float);
    const float * g_d = (const float *) g->data;
    const float * x_d = (const float *) x->data;
    float * dst_d     = (float *) mul_node->data;
    cudaStream_t stream = ctx.stream();

    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SIGMOID:  unary_mul_gate_f32_cuda<op_sigmoid> (g_d, x_d, dst_d, k, n, sg, sx, stream); break;
        case GGML_UNARY_OP_SILU:     unary_mul_gate_f32_cuda<op_silu>    (g_d, x_d, dst_d, k, n, sg, sx, stream); break;
        case GGML_UNARY_OP_SOFTPLUS: unary_mul_gate_f32_cuda<op_softplus>(g_d, x_d, dst_d, k, n, sg, sx, stream); break;
        case GGML_UNARY_OP_TANH:     unary_mul_gate_f32_cuda<op_tanh>    (g_d, x_d, dst_d, k, n, sg, sx, stream); break;
        case GGML_UNARY_OP_EXP:      unary_mul_gate_f32_cuda<op_exp>     (g_d, x_d, dst_d, k, n, sg, sx, stream); break;
        default: GGML_ABORT("unsupported unary op for the fused gate mul");
    }
}

void ggml_cuda_op_unary_mul_gate_add(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node, ggml_tensor * add_node) {
    const ggml_tensor * g = unary_node->src[0];
    const ggml_tensor * x = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];
    const ggml_tensor * a = (add_node->src[0] == mul_node) ? add_node->src[1] : add_node->src[0];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && add_node->type == GGML_TYPE_F32 && ggml_are_same_shape(a, add_node) &&
                ggml_are_same_shape(mul_node, add_node) && ggml_is_contiguous(a) && ggml_is_contiguous(add_node));
    GGML_ASSERT(g->ne[0] == 1 && ggml_are_same_shape(x, mul_node) && ggml_is_contiguous_1(g) && ggml_is_contiguous_1(x) &&
                x->nb[0] == sizeof(float));

    const int64_t k  = ggml_nelements(add_node);
    const int64_t n  = x->ne[0];
    const int64_t sg = g->nb[1] / sizeof(float);
    const int64_t sx = x->nb[1] / sizeof(float);
    const float * g_d = (const float *) g->data;
    const float * x_d = (const float *) x->data;
    const float * a_d = (const float *) a->data;
    float * dst_d     = (float *) add_node->data;

    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, ctx.stream());
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SIGMOID:  ggml_cuda_kernel_launch(unary_mul_gate_add_f32_kernel<op_sigmoid>,  lp, g_d, x_d, a_d, dst_d, k, n, sg, sx); break;
        case GGML_UNARY_OP_SILU:     ggml_cuda_kernel_launch(unary_mul_gate_add_f32_kernel<op_silu>,     lp, g_d, x_d, a_d, dst_d, k, n, sg, sx); break;
        case GGML_UNARY_OP_SOFTPLUS: ggml_cuda_kernel_launch(unary_mul_gate_add_f32_kernel<op_softplus>, lp, g_d, x_d, a_d, dst_d, k, n, sg, sx); break;
        case GGML_UNARY_OP_TANH:     ggml_cuda_kernel_launch(unary_mul_gate_add_f32_kernel<op_tanh>,     lp, g_d, x_d, a_d, dst_d, k, n, sg, sx); break;
        case GGML_UNARY_OP_EXP:      ggml_cuda_kernel_launch(unary_mul_gate_add_f32_kernel<op_exp>,      lp, g_d, x_d, a_d, dst_d, k, n, sg, sx); break;
        default: GGML_ABORT("unsupported unary op for the fused gate mul add");
    }
}

/* fused add (bias) + unary + mul (scale), bias and scale broadcast over the rows */

template <float (*op)(float)>
static __global__ void add_unary_mul_f32_kernel(const float * a, const float * bias, const float * scale, float * dst,
                                                const int64_t k, const int64_t n) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t h = i % n;
    ggml_cuda_pdl_sync();
    // float addition and multiplication commute: equal to ADD, UNARY, MUL run one by one in either operand order
    dst[i] = op(a[i] + bias[h]) * scale[h];
}

template <float (*op)(float)>
static void add_unary_mul_f32_cuda(const float * a, const float * bias, const float * scale, float * dst, const int64_t k,
                                   const int64_t n, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(add_unary_mul_f32_kernel<op>, launch_params, a, bias, scale, dst, k, n);
}

void ggml_cuda_op_add_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add_node, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    const ggml_tensor * a     = ggml_are_same_shape(add_node->src[0], add_node) ? add_node->src[0] : add_node->src[1];
    const ggml_tensor * bias  = a == add_node->src[0] ? add_node->src[1] : add_node->src[0];
    const ggml_tensor * scale = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 && mul_node->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_node) && ggml_is_contiguous(a) && ggml_is_contiguous(mul_node));
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0] && ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    const int64_t k = ggml_nelements(mul_node);
    const int64_t n = a->ne[0];
    const float * a_d = (const float *) a->data;
    const float * b_d = (const float *) bias->data;
    const float * s_d = (const float *) scale->data;
    float * dst_d     = (float *) mul_node->data;
    cudaStream_t stream = ctx.stream();

    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SIGMOID:  add_unary_mul_f32_cuda<op_sigmoid> (a_d, b_d, s_d, dst_d, k, n, stream); break;
        case GGML_UNARY_OP_SILU:     add_unary_mul_f32_cuda<op_silu>    (a_d, b_d, s_d, dst_d, k, n, stream); break;
        case GGML_UNARY_OP_SOFTPLUS: add_unary_mul_f32_cuda<op_softplus>(a_d, b_d, s_d, dst_d, k, n, stream); break;
        case GGML_UNARY_OP_TANH:     add_unary_mul_f32_cuda<op_tanh>    (a_d, b_d, s_d, dst_d, k, n, stream); break;
        case GGML_UNARY_OP_EXP:      add_unary_mul_f32_cuda<op_exp>     (a_d, b_d, s_d, dst_d, k, n, stream); break;
        default: GGML_ABORT("unsupported unary op for the fused add + unary + mul");
    }
}
