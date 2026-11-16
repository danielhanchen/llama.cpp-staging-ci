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

// k19-kvwin (ggml_kq_mask_win): win_mode 1 = without the window cells that have a ring copy (flags), 2 = the ring's window cells only
template <typename T>
static __global__ void kq_mask_kernel(const int32_t * __restrict__ cells, const int32_t * __restrict__ toks, T * __restrict__ dst,
        const int64_t n_kv, const int64_t n_tps, const size_t nb1, const size_t nb3,
        const int32_t * __restrict__ flags, const int win_mode, const int32_t n_win, const int32_t n_sinks) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t s = blockIdx.z;
    if (j >= n_kv) {
        return;
    }
    // grid.y is at most 65535: a stream of more tokens takes several rows per block
    for (int64_t i = blockIdx.y; i < n_tps; i += gridDim.y) {
        const int32_t * cj = cells + 5*(s*n_kv + j);
        const int32_t * ti = toks + 8*(s*n_tps + i);
        bool keep = kq_mask_keep(cj, ti);
        if (keep && win_mode != 0) {
            const int32_t p0 = cj[0], p1 = ti[0];
            const bool in_win = (p0 <= p1 && (int64_t) p0 > (int64_t) p1 - n_win) || p0 < n_sinks;
            keep = win_mode == 1 ? !(in_win && flags[s*n_kv + j] != 0) : in_win;
        }
        T * row = (T *) ((char *) dst + s*nb3 + i*nb1);
        if constexpr (std::is_same_v<T, half>) {
            row[j] = keep ? __float2half(0.0f) : __float2half(-INFINITY);
        } else {
            row[j] = keep ? 0.0f : -INFINITY;
        }
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
    GGML_ASSERT(n_stream <= 65535);
    const dim3 grid((unsigned) ((n_kv + 255)/256), (unsigned) std::min<int64_t>(n_tps, 65535), (unsigned) n_stream);
    cudaStream_t stream = ctx.stream();
    const int win_mode = ggml_get_op_params_i32(dst, 8);
    const int32_t n_win = ggml_get_op_params_i32(dst, 9), n_sinks = ggml_get_op_params_i32(dst, 10);
    const int32_t * flags = win_mode == 1 ? (const int32_t *) dst->src[2]->data : nullptr;
    GGML_ASSERT(win_mode != 1 || (dst->src[2] != nullptr && ggml_is_contiguous(dst->src[2])));
    if (dst->type == GGML_TYPE_F16) {
        kq_mask_kernel<half><<<grid, block, 0, stream>>>((const int32_t *) cells->data, (const int32_t *) toks->data, (half *) dst->data,
            n_kv, n_tps, dst->nb[1], dst->nb[3], flags, win_mode, n_win, n_sinks);
    } else {
        kq_mask_kernel<float><<<grid, block, 0, stream>>>((const int32_t *) cells->data, (const int32_t *) toks->data, (float *) dst->data,
            n_kv, n_tps, dst->nb[1], dst->nb[3], flags, win_mode, n_win, n_sinks);
    }
    CUDA_CHECK(cudaGetLastError());
}
