#include "kq-mask.cuh"

// the values of llama_kv_cache::set_input_kq_mask (no ALiBi, sequence ids < 64): 0 = attend, -INFINITY = masked
static __device__ __forceinline__ bool kq_mask_keep(const int32_t * c, const int32_t * t) {
    const int32_t p0 = c[0];
    const int32_t p1 = t[0], seq = t[1], causal = t[2], swa_type = t[3], n_swa = t[4], is_2d = t[5], x1 = t[6], y1 = t[7];
    if (p0 == INT32_MIN || (((uint32_t) c[seq < 32 ? 1 : 2]) & (1u << (seq & 31))) == 0) {
        return false;
    }
    if (causal && p0 > p1) {
        return false;
    }
    if (causal && is_2d && p0 == p1 && (c[4] > y1 || (c[4] == y1 && c[3] > x1))) {
        return false;
    }
    switch (swa_type) {
        case 1: return !(p1 - p0 >= n_swa);
        case 2: return !(p0 < (p1 / n_swa) * n_swa);
        case 3: { const int32_t h = n_swa / 2; const int32_t d = p1 - p0; return !(d < -h || d > h); }
        default: return true;
    }
}

template <typename T>
static __global__ void kq_mask_kernel(const int32_t * __restrict__ cells, const int32_t * __restrict__ toks, T * __restrict__ dst,
        const int64_t n_kv, const int64_t n_tps, const size_t nb1, const size_t nb3) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i = blockIdx.y;
    const int64_t s = blockIdx.z;
    if (j >= n_kv) {
        return;
    }
    const bool keep = kq_mask_keep(cells + 5*(s*n_kv + j), toks + 8*(s*n_tps + i));
    T * row = (T *) ((char *) dst + s*nb3 + i*nb1);
    if constexpr (std::is_same_v<T, half>) {
        row[j] = keep ? __float2half(0.0f) : __float2half(-INFINITY);
    } else {
        row[j] = keep ? 0.0f : -INFINITY;
    }
}

void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cells = dst->src[0];
    const ggml_tensor * toks  = dst->src[1];
    GGML_ASSERT(ggml_is_contiguous(cells) && ggml_is_contiguous(toks));
    const int64_t n_kv = dst->ne[0], n_tps = dst->ne[1], n_stream = dst->ne[3];
    if (n_kv == 0 || n_tps == 0) {
        return;
    }
    const dim3 block(256, 1, 1);
    const dim3 grid((unsigned) ((n_kv + 255)/256), (unsigned) n_tps, (unsigned) n_stream);
    cudaStream_t stream = ctx.stream();
    if (dst->type == GGML_TYPE_F16) {
        kq_mask_kernel<half><<<grid, block, 0, stream>>>((const int32_t *) cells->data, (const int32_t *) toks->data, (half *) dst->data,
            n_kv, n_tps, dst->nb[1], dst->nb[3]);
    } else {
        kq_mask_kernel<float><<<grid, block, 0, stream>>>((const int32_t *) cells->data, (const int32_t *) toks->data, (float *) dst->data,
            n_kv, n_tps, dst->nb[1], dst->nb[3]);
    }
    CUDA_CHECK(cudaGetLastError());
}
