#include "moe-weighted-reduction.cuh"

// k19-g4fuse: two_src = the experts are the sum of two products (experts + experts2, the ADD of the MoE cache's overlapped verify: hits
// and fetched experts, each slot computed by one of them), extra = an addend of the result (the ADD that follows the reduction). Both
// as the separate nodes compute them: (a + b) per element first, the reduction's own order, then sum + extra (float addition commutes)
template <bool two_src, bool with_extra>
static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used,
                                                  const float * __restrict__ experts2 = nullptr,
                                                  const float * __restrict__ extra = nullptr) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          e0          = experts[first_row * n_embd + col];
    if constexpr (two_src) {
        e0 = e0 + experts2[first_row * n_embd + col];
    }
    float          sum         = (e0 * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        float         e     = experts[row * n_embd + col];
        if constexpr (two_src) {
            e = e + experts2[row * n_embd + col];
        }
        sum += (e * scale) * weights[row];
    }
    if constexpr (with_extra) {
        sum = sum + extra[token * n_embd + col];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32<false, false>
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_weighted_reduction_add2(ggml_backend_cuda_context & ctx,
                                              const ggml_tensor *         experts,
                                              const ggml_tensor *         experts2,
                                              const ggml_tensor *         expert_scale,
                                              const ggml_tensor *         weights,
                                              const ggml_tensor *         extra,
                                              ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32 && experts2->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(extra == nullptr || (extra->type == GGML_TYPE_F32 && ggml_is_contiguous(extra) && ggml_nelements(extra) == ggml_nelements(dst)));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts) && ggml_is_contiguous(experts2) && ggml_are_same_shape(experts, experts2));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    const float * e2 = (const float *) experts2->data;
    const float * sc = expert_scale ? (const float *) expert_scale->data : nullptr;
    if (extra != nullptr) {
        moe_weighted_reduction_f32<true, true><<<blocks, threads, 0, stream>>>((const float *) experts->data, sc,
            (const float *) weights->data, (float *) dst->data, n_embd, (int) n_expert_used, e2, (const float *) extra->data);
    } else {
        moe_weighted_reduction_f32<true, false><<<blocks, threads, 0, stream>>>((const float *) experts->data, sc,
            (const float *) weights->data, (float *) dst->data, n_embd, (int) n_expert_used, e2, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
}
