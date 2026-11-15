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
    const int64_t s = blockIdx.z;
    if (j >= n_kv) {
        return;
    }
    // grid.y is at most 65535: a stream of more tokens takes several rows per block
    for (int64_t i = blockIdx.y; i < n_tps; i += gridDim.y) {
        const bool keep = kq_mask_keep(cells + 5*(s*n_kv + j), toks + 8*(s*n_tps + i));
        T * row = (T *) ((char *) dst + s*nb3 + i*nb1);
        if constexpr (std::is_same_v<T, half>) {
            row[j] = keep ? __float2half(0.0f) : __float2half(-INFINITY);
        } else {
            row[j] = keep ? 0.0f : -INFINITY;
        }
    }
}

// k19-kvlong: GGML_OP_KQ_MASK mode 1 (ggml_sel_mask). Every cell kq - INFINITY, then the selected cells kq + 0 (one stream, in order),
// each rounded as the f16 add of the chain add(view(set_rows(fill(-inf), fill(0), idx)), kq) it replaces
static __global__ void sel_mask_fill_kernel(const half * __restrict__ kq, half * __restrict__ dst, const int64_t n_kv,
        const size_t nb1_kq, const size_t nb1_dst) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t r = blockIdx.y;
    if (j >= n_kv) {
        return;
    }
    const half k = ((const half *) ((const char *) kq + r*nb1_kq))[j];
    ((half *) ((char *) dst + r*nb1_dst))[j] = __float2half(-INFINITY + __half2float(k));
}

static __global__ void sel_mask_scatter_kernel(const int32_t * __restrict__ idx, const half * __restrict__ kq, half * __restrict__ dst,
        const int64_t n_kv, const int64_t n_sel, const size_t nb1_idx, const size_t nb1_kq, const size_t nb1_dst) {
    const int64_t s = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t r = blockIdx.y;
    if (s >= n_sel) {
        return;
    }
    const int32_t j = ((const int32_t *) ((const char *) idx + r*nb1_idx))[s];
    if (j < 0 || j >= n_kv) {
        return; // a dead slot
    }
    const half k = ((const half *) ((const char *) kq + r*nb1_kq))[j];
    ((half *) ((char *) dst + r*nb1_dst))[j] = __float2half(0.0f + __half2float(k));
}

static void ggml_cuda_op_sel_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * idx = dst->src[0];
    const ggml_tensor * kq  = dst->src[1];
    GGML_ASSERT(dst->type == GGML_TYPE_F16 && kq->type == GGML_TYPE_F16 && idx->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(kq) && idx->nb[0] == sizeof(int32_t));
    const int64_t n_kv   = dst->ne[0];
    const int64_t n_rows = dst->ne[1]*dst->ne[2]*dst->ne[3];
    const int64_t n_sel  = idx->ne[0];
    if (n_kv == 0 || n_rows == 0) {
        return;
    }
    cudaStream_t stream = ctx.stream();
    const dim3 block(256, 1, 1);
    for (int64_t r0 = 0; r0 < n_rows; r0 += 65535) {
        const int64_t nr = std::min<int64_t>(n_rows - r0, 65535);
        const half * kq_r  = (const half *) ((const char *) kq->data + r0*kq->nb[1]);
        half       * dst_r = (half *) ((char *) dst->data + r0*dst->nb[1]);
        const dim3 grid_f((unsigned) ((n_kv + 255)/256), (unsigned) nr, 1);
        sel_mask_fill_kernel<<<grid_f, block, 0, stream>>>(kq_r, dst_r, n_kv, kq->nb[1], dst->nb[1]);
        if (n_sel > 0) {
            const int32_t * idx_r = (const int32_t *) ((const char *) idx->data + r0*idx->nb[1]);
            const dim3 grid_s((unsigned) ((n_sel + 255)/256), (unsigned) nr, 1);
            sel_mask_scatter_kernel<<<grid_s, block, 0, stream>>>(idx_r, kq_r, dst_r, n_kv, n_sel, idx->nb[1], kq->nb[1], dst->nb[1]);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 0) == 1) {
        ggml_cuda_op_sel_mask(ctx, dst);
        return;
    }
    const ggml_tensor * cells = dst->src[0];
    const ggml_tensor * toks  = dst->src[1];
    GGML_ASSERT(ggml_is_contiguous(cells) && ggml_is_contiguous(toks));
    const int64_t n_kv = dst->ne[0], n_tps = dst->ne[1], n_stream = dst->ne[3];
    if (n_kv == 0 || n_tps == 0) {
        return;
    }
    const dim3 block(256, 1, 1);
    GGML_ASSERT(n_stream <= 65535);
    const dim3 grid((unsigned) ((n_kv + 255)/256), (unsigned) std::min<int64_t>(n_tps, 65535), (unsigned) n_stream);
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
