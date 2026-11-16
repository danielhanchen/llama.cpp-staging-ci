// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

// k19-graph: the live KV length of a padded flash attention node (ggml_flash_attn_ext_set_kv_len), read on the device by the dense
// stream-k MMA kernel and its fixup. They take the loop bound and the stream-k block count that the launch of a node of kv_len cells
// takes (ggml_cuda_fattn_stream_k_blocks, the same function), so the padded node computes the bits of the unpadded one while its
// launch (the grid of the padded length) and so its CUDA graph stay fixed as the length grows. kv_len == nullptr: off
struct ggml_cuda_fattn_dev_len {
    const int * kv_len;
    int nbatch_fa;      // the host's KV tile, which the block count is planned with
    int ntiles_dst;
    int max_blocks;
    int ntiles_dst_ref; // k19-tpp: the output tiles of the reference window (0: none)
    int stream_k;       // 0: one block per output tile
    // k19-fattnq4: the second KV source of a two-source node (ggml_flash_attn_ext_set_src2; it rides in this argument so the kernel
    // signature stays the one of every FA kernel). n_kv2 == 0: none. Its cells continue the first source's: KV tiles it1 ..
    // it1 + n_kv2/nbatch_fa - 1 of the kernel's loop (it1 = the first source's tiles), K2 / V2 / mask2 with their own byte strides
    // (V2 == K2 for a K-only cache whose V is a view of K). Read by the MMA kernel only (the others never get a two-source node)
    const char * K2;
    const char * V2;
    const char * mask2;
    int32_t n_kv2;
    int32_t it1;
    int32_t nb11_2, nb12_2;
    int64_t nb13_2;
    int32_t nb21_2, nb22_2;
    int64_t nb23_2;
    int32_t nb31_2;
    int64_t nb33_2;
    // per (mask2 stream, query tile): ring_nw words of bits, bit t = the second source's KV tile t holds a finite mask value for some
    // row of the tile (flash_attn_src2_ring_bits); nullptr: every tile. The other tiles are fully masked and skipped (a fully masked
    // tile leaves the softmax state exactly as it is, so the result is the same bits)
    const uint32_t * ring_bits;
    int32_t          ring_nw;
    // the first source's KV tiles [x, y) that hold a finite mask value for some row of the tile (flash_attn_src2_body_range),
    // nullptr: every tile
    const int2 * body_range;
};

// k19-fattnq4: the first bit >= k (set: a set bit, else a clear bit) of bits, stop if there is none before stop
template <bool set>
static __device__ __forceinline__ int flash_attn_src2_next_bit(const uint32_t * bits, int k, const int stop) {
    int w = k >> 5;
    uint32_t b = (set ? bits[w] : ~bits[w]) & (0xFFFFFFFFu << (k & 31));
    while (b == 0) {
        ++w;
        if (32*w >= stop) {
            return stop;
        }
        b = set ? bits[w] : ~bits[w];
    }
    return min(stop, 32*w + __ffs(b) - 1);
}

// the stream-k blocks of a launch over ntiles_KV KV tiles (launch_fattn); on the device for a length read there
static __host__ __device__ __forceinline__ int ggml_cuda_fattn_stream_k_blocks(
        const int ntiles_KV, const int ntiles_dst, const int max_blocks, const int ntiles_dst_ref) {
    const int max_efficiency_loss_percent = 5;
    // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
    // Only do this if the occupancy loss from rounding is acceptable.
    const int raw     = max_blocks < ntiles_KV*ntiles_dst ? max_blocks : ntiles_KV*ntiles_dst;
    const int rounded = (raw / ntiles_dst) * ntiles_dst;
    const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    int nblocks = loss <= max_efficiency_loss_percent ? rounded : raw;
    // k19-tpp: one-query tiles of a verify window wider than the reference: the blocks per output tile of the reference window,
    // so every query's KV range is split at the same cells as there (each tile's split depends only on its block count when the
    // count divides evenly). Only when the reference launch itself rounds to whole tiles; otherwise as before
    if (ntiles_dst_ref > 0) {
        const int raw_ref     = max_blocks < ntiles_KV*ntiles_dst_ref ? max_blocks : ntiles_KV*ntiles_dst_ref;
        const int rounded_ref = (raw_ref / ntiles_dst_ref) * ntiles_dst_ref;
        const int loss_ref    = rounded_ref > 0 ? 100 * (raw_ref - rounded_ref) / raw_ref : 100;
        if (loss_ref <= max_efficiency_loss_percent) {
            nblocks = (rounded_ref / ntiles_dst_ref) * ntiles_dst;
        }
    }
    return nblocks;
}

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const ggml_cuda_fattn_dev_len dev_len);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
};

static inline ggml_cuda_flash_attn_ext_f16_extra_data ggml_cuda_flash_attn_ext_get_f16_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ggml_cuda_flash_attn_ext_f16_extra_data data = {};
    data.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        data.end += ggml_nelements(K)*ggml_type_size(GGML_TYPE_F16);
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            data.end += ggml_nelements(V)*ggml_type_size(GGML_TYPE_F16);
        }
    }

    return data;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
#if defined(GGML_USE_HIP)
    // Keep this VMEM read close to its packed-byte dequantization. Hoisting it too far
    // increases VGPR pressure substantially in some FlashAttention vector kernels.
    __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
#endif // defined(GGML_USE_HIP)
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsub4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsub4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

// k19-fattnq4: the ring tiles a query tile of a two-source node can see: per (mask2 stream blockIdx.y, query tile blockIdx.x of ncols1
// rows < n_queries) the bits of the KV tiles of nbatch_fa cells with a finite mask2 value in any row of the tile, set with atomicOr into
// words the host zeroed; blockIdx.z: a group of tpb tiles (many blocks: the scan reads the whole ring mask)
__launch_bounds__(256, 1)
static __global__ void flash_attn_src2_ring_bits(
        const half2 * __restrict__ mask2, uint32_t * __restrict__ bits, const int n_kv2, const int ncols1, const int n_queries,
        const int nbatch_fa, const int tpb, const int nw, const int64_t s21, const int64_t s23) {
    ggml_cuda_pdl_sync();
    __shared__ int flag_s[32];
    if (threadIdx.x < 32) {
        flag_s[threadIdx.x] = 0;
    }
    __syncthreads();
    const int q0 = blockIdx.x*ncols1;
    const int nq = min(q0 + ncols1, n_queries) - q0;
    const int c0 = blockIdx.z*tpb*nbatch_fa;              // first cell (even: nbatch_fa is)
    const int nc = (min(c0 + tpb*nbatch_fa, n_kv2) - c0) / 2; // cell pairs
    const half2 * m = mask2 + blockIdx.y*s23 + (int64_t) q0*s21 + c0/2;
    for (int i = threadIdx.x; i < nq*nc; i += blockDim.x) {
        const int q = i / nc;
        const int c = i - q*nc;
        const float2 v = __half22float2(m[q*s21 + c]);
        if (!isinf(v.x) || !isinf(v.y)) {
            flag_s[(2*c) / nbatch_fa] = 1;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t f = 0;
        for (int t = 0; t < tpb; ++t) {
            f |= flag_s[t] ? 1u << t : 0u;
        }
        if (f != 0) {
            const int t0 = blockIdx.z*tpb; // tpb divides 32: the group lies in one word
            atomicOr(bits + (int64_t) (blockIdx.y*gridDim.x + blockIdx.x)*nw + t0/32, f << (t0 % 32));
        }
    }
    ggml_cuda_pdl_lc();
}

// k19-fattnq4: the body tiles a query tile of a two-source node can see, [first, last + 1) of the KV tiles with a finite mask value
// in a row of the tile, (0, 0) for none; scanned from both ends with an early exit (the body's visible cells of a prompt tile are a
// prefix: the scan reads the masked tail of about W + ubatch cells, not the whole mask)
__launch_bounds__(256, 1)
static __global__ void flash_attn_src2_body_range(
        const half2 * __restrict__ mask, int2 * __restrict__ ranges, const int n_kv, const int ncols1, const int n_queries,
        const int nbatch_fa, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();
    const int q0 = blockIdx.x*ncols1;
    const int nq = min(q0 + ncols1, n_queries) - q0;
    const half2 * m = mask + blockIdx.y*s33 + (int64_t) q0*s31;
    const int nt = (n_kv + nbatch_fa - 1) / nbatch_fa;
    auto visible = [&](const int t) {
        const int c0 = t*nbatch_fa/2;
        const int nc = (min((t + 1)*nbatch_fa, n_kv) - t*nbatch_fa) / 2;
        int f = 0;
        for (int i = threadIdx.x; i < nq*nc; i += blockDim.x) {
            const int q = i / nc;
            const float2 v = __half22float2(m[q*s31 + c0 + (i - q*nc)]);
            f |= !isinf(v.x) || !isinf(v.y);
        }
        return __syncthreads_or(f) != 0;
    };
    int lo = 0;
    while (lo < nt && !visible(lo)) {
        ++lo;
    }
    int hi = nt;
    while (hi > lo && !visible(hi - 1)) {
        --hi;
    }
    if (threadIdx.x == 0) {
        ranges[blockIdx.y*gridDim.x + blockIdx.x] = lo < hi ? make_int2(lo, hi) : make_int2(0, 0);
    }
    ggml_cuda_pdl_lc();
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream);

bool ggml_cuda_qsa_maskless_gather_ok(const ggml_cuda_qsa_maskless & qsa);
// GGML_CUDA_QSA_MASKLESS: 0 builds the dense mask in the graph (kill switch), 1 (default) gathers from the selection,
// 2 skips the graph's mask nodes but always rebuilds the dense mask (tests the fallback)
int  ggml_cuda_qsa_maskless_mode();
void ggml_cuda_qsa_maskless_compact(
        const ggml_cuda_qsa_maskless & qsa, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max,
        cudaStream_t stream);
void ggml_cuda_qsa_maskless_materialize(const ggml_cuda_qsa_maskless & qsa, half * mask, cudaStream_t stream);

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile_host,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j,
        const ggml_cuda_fattn_dev_len dev_len) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();

    // k19-graph: the blocks per tile of the launch at the live length (see ggml_cuda_fattn_dev_len); the buffer layout keeps the
    // launched grid (nblocks_stream_k), as the kernel's blocks write it
    int blocks_per_tile = blocks_per_tile_host;
    if (dev_len.kv_len != nullptr) {
        blocks_per_tile = ggml_cuda_fattn_stream_k_blocks((*dev_len.kv_len + dev_len.nbatch_fa - 1) / dev_len.nbatch_fa,
            dev_len.ntiles_dst, dev_len.max_blocks, dev_len.ntiles_dst_ref) / dev_len.ntiles_dst;
    }

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    // a block with nothing to fix up still waits for the flash attention kernel before it leaves: a grid whose blocks all left
    // before their griddepcontrol.wait would complete before that kernel, and the PDL wait of the next kernel (only on this grid)
    // would then not order it after the flash attention output
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        ggml_cuda_pdl_sync();
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        ggml_cuda_pdl_sync();
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// k19-tpp: a verify window (GGML_TENSOR_FLAG_VERIFY, not batch invariant) of more rows than GGML_CUDA_VERIFY_REF_ROWS (default 3: the
// window of 2 drafts that every MTP verify forms) takes the launch choices of that reference window where they follow the row count:
// the sparse / dense attention pick (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse) and the stream-k blocks per output tile of the
// one-query sparse kernel. Its rows then sum in the order of a 3-row window's (0 = off)
static int ggml_cuda_fattn_verify_ref_rows(const ggml_tensor * KQV) {
    if (!(KQV->flags & GGML_TENSOR_FLAG_VERIFY) || (KQV->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT)) {
        return 0;
    }
    static const int ref = getenv("GGML_CUDA_VERIFY_REF_ROWS") ? atoi(getenv("GGML_CUDA_VERIFY_REF_ROWS")) : 3;
    return ref > 0 && KQV->src[0]->ne[1] > ref ? ref : 0;
}

// k19-graph (ggml_cuda_flash_attn_ext): a node whose live KV length (ggml_flash_attn_ext_set_kv_len) is below its padded one is
// launched as the node of the live length (K / V / mask views of kv_len cells, the launch choices of that node). These tell
// launch_fattn the graph node behind that copy and, for the device length, the padded length the grid is planned with
struct ggml_cuda_fattn_kv_len_ctx {
    const ggml_tensor * node;     // the graph node (its QSA maskless plan, its mask)
    const int *         kv_len;   // device copy of the live length; nullptr: the launch keeps the live length as a constant
    int64_t             n_kv_pad; // the node's K / V cells
};
// a dry run of the launch choices (no allocation, no launch), see ggml_cuda_flash_attn_ext_kv_len_sig
struct ggml_cuda_fattn_probe {
    bool         done       = false;
    bool         stream_k   = false;
    bool         use_sparse = false;
    bool         scan_mask  = false;
    bool         dev_ok     = false; // the launch can read the length on the device (dense, f16 K / V, uniform or no stream-k fixup)
    const void * kernel     = nullptr;
    int          nblocks    = 0;
    int          fixup      = 0;     // stream-k fixup at this length: 0 none, 1 uniform, 2 general (-1: not stream-k)
};
extern thread_local const ggml_cuda_fattn_kv_len_ctx * ggml_cuda_fattn_kv_len_cur;
extern thread_local ggml_cuda_fattn_probe *            ggml_cuda_fattn_probe_cur;

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const bool use_sparse,
    const int warp_size = WARP_SIZE, const bool async_kv_preload = false,
    // k15: the number of parallel KV blocks is the one a launch of pb_kernel with pb_ncols2 query heads per block would pick (its
    // occupancy and output tiles), so a kernel that packs the query heads of a KV head (ncols2 > 1) splits the KV exactly as the
    // per-head kernel does and its outputs are bit-identical (fattn-vec.cuh, GGML_CUDA_FA_VEC_GQA)
    fattn_kernel_t pb_kernel = nullptr, const int pb_ncols2 = 1,
    // k15: thread groups per block (blockDim.z; fattn-vec.cuh grouped heads), the occupancy then is that of pb_kernel's group size
    const int block_z = 1,
    // k15-mm2: the vector kernel's KV split may be capped (fattn_vec_kv_split_cap): it is a split-K over the KV cells with an f32
    // combine, and the vector kernel accumulates VKQ in f32 on NVIDIA, so a cap only reorders f32 sums
    const bool vec_kv_cap = false,
    // k19-fattnq8: the occupancy (so the parallel / stream-k blocks) of this kernel at this shared memory instead of fattn_kernel's: a
    // kernel reading quantized K / V in place splits the KV as the f16 kernel of the staged launch does
    fattn_kernel_t occ_kernel = nullptr, const size_t occ_nbytes_shared = 0
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();

    // k19-graph: the launch of a padded node at its live length (see ggml_cuda_fattn_kv_len_ctx), or a dry run of the choices
    const ggml_cuda_fattn_kv_len_ctx * kvl = ggml_cuda_fattn_kv_len_cur;
    ggml_cuda_fattn_probe * probe = ggml_cuda_fattn_probe_cur;
    const ggml_tensor * node      = kvl ? kvl->node : dst;
    const ggml_tensor * node_mask = node->src[3];

    // k15-mm2 QSA maskless: the graph did not build this node's dense selection mask. A gather takes its lists from the selected
    // cells and reads the causal mask, which equals the dense one at every listed cell; any other kernel gets the dense mask now
    const ggml_cuda_qsa_maskless * qsa = nullptr;
    if (!ctx.qsa_maskless.empty()) {
        const auto it = ctx.qsa_maskless.find(node);
        if (it != ctx.qsa_maskless.end()) {
            qsa = &it->second;
        }
    }
    const bool qsa_gather = qsa && !qsa->prompt && use_sparse && ggml_cuda_qsa_maskless_gather_ok(*qsa) && ggml_cuda_qsa_maskless_mode() == 1;
    // the mask node's buffer may hold the selection or the causal mask (the allocator reused it), then a pool copy takes the mask;
    // allocated before the kernel's pool buffers, freed after them
    // (k19-graph: the dense mask of the graph node, all of its padded cells; the launch reads the live ones through the same strides)
    ggml_cuda_pool_alloc<half> qsa_mask(pool);
    const char * mask_data = mask ? (const char *) mask->data : nullptr;
    if (probe) {
        // a dry run: no copy, no conversion, no allocation
    } else if (qsa_gather) {
        mask_data = (const char *) qsa->kq->data;
    } else if (qsa) {
        half * m = qsa->alias ? qsa_mask.alloc(ggml_nelements(node_mask)) : (half *) node_mask->data;
        ggml_cuda_qsa_maskless_materialize(*qsa, m, main_stream);
        mask_data = (const char *) m;
    }
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V);

    // k19-fattnq4: the f16 copies of a two-source node's second source, allocated before the buffers below (the pool frees in reverse order)
    ggml_cuda_pool_alloc<half> K2_f16(pool);
    ggml_cuda_pool_alloc<half> V2_f16(pool);
    ggml_cuda_pool_alloc<int2> ring_range(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16 && !probe) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(f16_extra.K != 0);
        half * K_f16 = (half *) f16_extra.K;
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16 && !probe) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(f16_extra.V != 0);
            half * V_f16 = (half *) f16_extra.V;
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    // k19-fattnq4: the second source of a two-source node (ggml_flash_attn_ext_set_src2, MMA kernel only): read in place by the
    // kernel when the K / V of the first are (need_f16_K false), else staged to f16 from the pool. Rides in dev_len (see there)
    const ggml_tensor * K2 = dst->src[6];
    const ggml_tensor * V2 = dst->src[7];
    const ggml_tensor * M2 = dst->src[8];
    ggml_cuda_fattn_dev_len src2 = {};
    if (K2 != nullptr) {
        GGML_ASSERT(V2 != nullptr && M2 != nullptr && M2->type == GGML_TYPE_F16 && !use_sparse);
        GGML_ASSERT(K2->ne[1] % nbatch_fa == 0 && "two-source flash attention: the second source's cells must fill whole KV tiles");
        GGML_ASSERT(M2->ne[0] == K2->ne[1] && M2->ne[2] == 1 && (!mask || (M2->ne[1] == mask->ne[1] && M2->ne[3] == mask->ne[3])));
        const bool V2_is_K2_view = V2 == K2 || (V2->view_src && (V2->view_src == K2 || (V2->view_src == K2->view_src && V2->view_offs == K2->view_offs)));
        const char * k2d = (const char *) K2->data;
        const char * v2d = (const char *) V2->data;
        size_t k2nb[3] = { K2->nb[1], K2->nb[2], K2->nb[3] };
        size_t v2nb[3] = { V2->nb[1], V2->nb[2], V2->nb[3] };
        const auto stage = [&](const ggml_tensor * t, ggml_cuda_pool_alloc<half> & buf, const char *& d, size_t * nb) {
            if (t->type == GGML_TYPE_F16 || probe) {
                return;
            }
            const size_t ts = ggml_type_size(t->type);
            GGML_ASSERT(t->nb[0] == ts);
            half * f = buf.alloc(ggml_nelements(t));
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(t->type);
            to_fp16(d, f, t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[1] / ts, t->nb[2] / ts, t->nb[3] / ts, main_stream);
            d     = (const char *) f;
            nb[0] = t->ne[0] * sizeof(half);
            nb[1] = t->ne[1] * nb[0];
            nb[2] = t->ne[2] * nb[1];
        };
        if (need_f16_K) {
            stage(K2, K2_f16, k2d, k2nb);
            if (V2_is_K2_view) {
                v2d = k2d;
                v2nb[0] = k2nb[0]; v2nb[1] = k2nb[1]; v2nb[2] = k2nb[2];
            } else {
                stage(V2, V2_f16, v2d, v2nb);
            }
        }
        src2.K2     = k2d;
        src2.V2     = v2d;
        src2.mask2  = (const char *) M2->data;
        src2.n_kv2  = (int32_t) K2->ne[1];
        src2.it1    = (int32_t) ((K->ne[1] + nbatch_fa - 1) / nbatch_fa);
        src2.nb11_2 = (int32_t) k2nb[0]; src2.nb12_2 = (int32_t) k2nb[1]; src2.nb13_2 = (int64_t) k2nb[2];
        src2.nb21_2 = (int32_t) v2nb[0]; src2.nb22_2 = (int32_t) v2nb[1]; src2.nb23_2 = (int64_t) v2nb[2];
        src2.nb31_2 = (int32_t) M2->nb[1]; src2.nb33_2 = (int64_t) M2->nb[3];
        // k19-fattnq4: skip the ring tiles a query tile cannot see (GGML_CUDA_FATTN_SRC2_SKIP=0: off), from
        // GGML_CUDA_FATTN_SRC2_SKIP_MIN_ROWS query rows (default 17: prompt ubatches, where most of the ring is masked per tile)
        static const bool skip_on = getenv("GGML_CUDA_FATTN_SRC2_SKIP") == nullptr || atoi(getenv("GGML_CUDA_FATTN_SRC2_SKIP")) != 0;
        static const int64_t skip_min_rows = getenv("GGML_CUDA_FATTN_SRC2_SKIP_MIN_ROWS") ? atoll(getenv("GGML_CUDA_FATTN_SRC2_SKIP_MIN_ROWS")) : 17;
        if (skip_on && Q->ne[1] >= skip_min_rows && !probe) {
            // ring tiles from mask2, body tiles from mask (the window cells are masked in the body: its last ~W + ubatch cells are
            // fully masked for a query tile of a prompt)
            const int ntx = (int) ((Q->ne[1] + ncols1 - 1) / ncols1);
            const size_t nr = size_t(ntx) * M2->ne[3];
            const int nt2 = (int) (K2->ne[1] / nbatch_fa);
            const int nw  = (nt2 + 31) / 32;
            const int tpb = 8; // ring tiles per block of the bits scan (divides 32)
            GGML_ASSERT(nbatch_fa % 2 == 0 && M2->nb[1] % sizeof(half2) == 0 && mask->nb[1] % sizeof(half2) == 0);
            ring_range.alloc(nr + (nr*nw + 1)/2); // the body ranges, then the ring bits
            uint32_t * ring_bits = (uint32_t *) (ring_range.ptr + nr);
            CUDA_CHECK(cudaMemsetAsync(ring_bits, 0, nr*nw*sizeof(uint32_t), main_stream));
            const ggml_cuda_kernel_launch_params lp_rb(dim3(ntx, M2->ne[3], (nt2 + tpb - 1) / tpb), dim3(256, 1, 1), 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_src2_ring_bits, lp_rb, (const half2 *) M2->data, ring_bits, (int) (nt2*nbatch_fa), ncols1,
                (int) Q->ne[1], nbatch_fa, tpb, nw, (int64_t) (M2->nb[1] / sizeof(half2)), (int64_t) (M2->nb[3] / sizeof(half2)));
            CUDA_CHECK(cudaGetLastError());
            const ggml_cuda_kernel_launch_params lp_br(dim3(ntx, M2->ne[3], 1), dim3(256, 1, 1), 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_src2_body_range, lp_br, (const half2 *) mask_data, ring_range.ptr, (int) K->ne[1], ncols1,
                (int) Q->ne[1], nbatch_fa, (int64_t) (mask->nb[1] / sizeof(half2)), (int64_t) (mask->nb[3] / sizeof(half2)));
            CUDA_CHECK(cudaGetLastError());
            src2.ring_bits  = ring_bits;
            src2.ring_nw    = nw;
            src2.body_range = ring_range.ptr;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // sparse: a query tile of ncols1 queries shares one index list, the union of the queries' visible columns
    int32_t n_kv_max = 0;
    if (use_sparse) {
        GGML_ASSERT(mask != nullptr);
        const int32_t n_kv_max_query = ggml_get_op_params_i32(KQV, 4);
        GGML_ASSERT(n_kv_max_query > 0);
        n_kv_max = std::min<int64_t>(K->ne[1], int64_t(ncols1)*n_kv_max_query);

        const size_t n_lists = size_t(ntiles_x) * mask->ne[3];

        if (probe) {
            // a dry run
        } else if (KV_max.alloc(size_t(n_kv_max)*n_lists + n_lists), qsa_gather) {
            ggml_cuda_qsa_maskless_compact(*qsa, KV_max.ptr, KV_max.ptr + size_t(n_kv_max)*n_lists, Q->ne[1], ncols1, n_kv_max, main_stream);
        } else {
            ggml_tensor mask_m = *mask; // the QSA maskless rebuild may live in a pool copy
            mask_m.data = (void *) mask_data;
            ggml_cuda_flash_attn_ext_compact_mask(&mask_m, KV_max.ptr, KV_max.ptr + size_t(n_kv_max)*n_lists, Q->ne[1], ncols1, n_kv_max, main_stream);
        }
    }

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    // (k19-fattnq4: not for a two-source node, whose kernel loop bound covers both sources)
    const bool scan_mask = !use_sparse && mask && K2 == nullptr && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1);
    if (scan_mask && !probe) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half2 *) mask_data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, block_z);
    GGML_ASSERT(block_z == 1 || pb_kernel != nullptr);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    GGML_ASSERT(occ_kernel == nullptr || pb_kernel == nullptr);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, pb_kernel ? pb_kernel : (occ_kernel ? occ_kernel : fattn_kernel),
        pb_kernel ? block_dim.x * block_dim.y : block_dim.x * block_dim.y * block_dim.z, occ_kernel ? occ_nbytes_shared : nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    GGML_ASSERT(pb_kernel == nullptr || !stream_k);
    const int ntiles_dst_pb = pb_kernel ? ntiles_x * ((gqa_ratio + pb_ncols2 - 1) / pb_ncols2) * K->ne[2] * Q->ne[3] : ntiles_dst;
    int parallel_blocks = max_blocks_per_sm;

    const int64_t n_kv = use_sparse ? n_kv_max : K->ne[1];
    // (k19-fattnq4: + the second source's tiles of a two-source node)
    const int ntiles_KV = (n_kv + nbatch_fa - 1) / nbatch_fa + (K2 ? src2.n_kv2 / nbatch_fa : 0); // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    // k19-graph: see ggml_cuda_fattn_dev_len / ggml_cuda_fattn_probe
    ggml_cuda_fattn_dev_len dev_len = {}; // all members zero (no device length, no second source)
    int  ntiles_dst_ref = 0;
    int  fixup_kind     = 0; // stream-k fixup: 0 none, 1 uniform, 2 general
    bool dev_ok         = false;
    bool use_stream_k   = false;
    if (stream_k) {
        // Stream-K splits the work before the mask scan is applied, so skipped KV tiles make the blocks uneven.
        const bool prefer_whole_tiles = GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK && async_kv_preload && scan_mask;

        auto should_use_stream_k = [prefer_whole_tiles](const int cc, const int ntiles_dst, const int max_blocks, const int DKQ) {
            const int tiles_nwaves             = (ntiles_dst + max_blocks - 1) / max_blocks;
            const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

            if (prefer_whole_tiles && tiles_efficiency_percent >= 75) {
                return false;
            }
            if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return true;
            }
            if (amd_wmma_available(cc) && DKQ == 64) {
                return true; // TODO better configuration
            }
            return tiles_efficiency_percent < 75;
        };

        const int  max_blocks   = max_blocks_per_sm*nsm;
        use_stream_k = should_use_stream_k(cc, ntiles_dst, max_blocks, Q->ne[0]);

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            // k19-tpp: one-query tiles of a verify window wider than the reference (see ggml_cuda_fattn_stream_k_blocks)
            const int ref_rows = ncols1 == 1 ? ggml_cuda_fattn_verify_ref_rows(KQV) : 0;
            ntiles_dst_ref = ref_rows > 0 ? ntiles_dst / ntiles_x * ref_rows : 0;
            blocks_num.x = ggml_cuda_fattn_stream_k_blocks(ntiles_KV, ntiles_dst, max_blocks, ntiles_dst_ref);
        }

        // k19-graph: the fixup of the launch at this length; the device length takes the dense kernel with a uniform fixup (each
        // tile the same number of blocks) or none, read again by the kernel and the fixup from the live length
        const int nblocks_live = blocks_num.x;
        fixup_kind = (nblocks_live % ntiles_dst == 0 && nblocks_live > ntiles_dst) ? 1 : (ntiles_dst % nblocks_live != 0 ? 2 : 0);
        // (k19-fattnq8: or K / V read in place by the kernel, need_f16_* false: no conversion that follows the length)
        dev_ok = !use_sparse && !scan_mask && K2 == nullptr && (K->type == GGML_TYPE_F16 || !need_f16_K) && (V->type == GGML_TYPE_F16 || !need_f16_V) &&
            fixup_kind != 2;
        if (kvl && kvl->kv_len && !probe) {
            GGML_ASSERT(dev_ok && "a device KV length needs the launch choices the probe found");
            // the grid of the padded length (the block count only grows with the length), the blocks past the live count exit
            const int ntiles_KV_pad = int((kvl->n_kv_pad + nbatch_fa - 1) / nbatch_fa);
            const int nblocks_pad   = use_stream_k ? ggml_cuda_fattn_stream_k_blocks(ntiles_KV_pad, ntiles_dst, max_blocks, ntiles_dst_ref) : ntiles_dst;
            blocks_num.x = std::max(nblocks_live, nblocks_pad);
            dev_len                = {};
            dev_len.kv_len         = kvl->kv_len;
            dev_len.nbatch_fa      = nbatch_fa;
            dev_len.ntiles_dst     = ntiles_dst;
            dev_len.max_blocks     = max_blocks;
            dev_len.ntiles_dst_ref = ntiles_dst_ref;
            dev_len.stream_k       = use_stream_k ? 1 : 0;
        }

        if (!probe && (ntiles_dst % blocks_num.x != 0 || (dev_len.kv_len && fixup_kind == 1))) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // batch invariant one column tiles (k15-fv): the KV split of a launch with one query, so every query is split as it alone is.
        // From the count above, which for a GQA-packed launch (GGML_CUDA_FA_VEC_GQA, pb_kernel, ncols2 > 1) is the per-head kernel's:
        // a count of its own here (ntiles_dst) dropped that, so the packed kernel no longer split as the per-head one, and verify
        // (ncols == 4 there, so not divided) split by the batch size
        const bool bi_one_query = (KQV->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) && ncols1 == 1 && (ncols2 == 1 || pb_kernel != nullptr);
        const int ntiles_dst_split = bi_one_query ? ntiles_dst_pb / ntiles_x : ntiles_dst_pb;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst_split * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

#ifndef GGML_USE_HIP
        // k15-mm2: fewer, longer KV blocks for the vector kernel, whose blocks otherwise get as little as one KV tile each, so that
        // their fixed cost (Q load, the final reduction, the combine reads) sets the time. GGML_CUDA_FA_PB_MIN_TILES = KV tiles each
        // block keeps at least (0 = no minimum), GGML_CUDA_FA_PB_MAX = at most that many parallel blocks (0 = no cap). Not on HIP:
        // there the vector kernel accumulates VKQ in half2 (V_DOT2_F32_F16_AVAILABLE) and longer blocks would add more half sums.
        if (vec_kv_cap && parallel_blocks > 1) {
            static const int pb_max = [] {
                const char * e = getenv("GGML_CUDA_FA_PB_MAX");
                return e != nullptr ? atoi(e) : 0;
            }();
            static const int min_tiles = [] {
                const char * e = getenv("GGML_CUDA_FA_PB_MIN_TILES");
                return e != nullptr ? atoi(e) : 0;
            }();
            int cap = parallel_blocks;
            if (pb_max > 0) {
                cap = std::min(cap, pb_max);
            }
            if (min_tiles > 0) {
                cap = std::min(cap, std::max(1, ntiles_KV / min_tiles));
            }
            parallel_blocks = std::max(1, cap);
        }
#else
        GGML_UNUSED(vec_kv_cap);
#endif // GGML_USE_HIP

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1 && !probe) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    if (probe) {
        probe->done       = true;
        probe->stream_k   = stream_k && use_stream_k;
        probe->use_sparse = use_sparse;
        probe->scan_mask  = scan_mask;
        probe->dev_ok     = stream_k && dev_ok;
        probe->kernel     = (const void *) fattn_kernel;
        probe->nblocks    = stream_k ? (int) blocks_num.x : parallel_blocks;
        probe->fixup      = stream_k ? fixup_kind : -1;
        return;
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    // m1 is read only for ALiBi (max_bias > 0); without ALiBi a negative m1 tells the vector kernel not to skip KV tiles whose
    // mask is -inf for every column (GGML_CUDA_FA_VEC_SKIP_MASKED=0, the kill switch of that skip)
    static const bool vec_skip_masked = !getenv("GGML_CUDA_FA_VEC_SKIP_MASKED") || atoi(getenv("GGML_CUDA_FA_VEC_SKIP_MASKED")) != 0;
    const float m1 = max_bias <= 0.0f && !vec_skip_masked ? -1.0f : powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    if (K2 != nullptr) {
        GGML_ASSERT(dev_len.kv_len == nullptr);
        src2.nbatch_fa = dev_len.nbatch_fa; src2.ntiles_dst = dev_len.ntiles_dst; src2.max_blocks = dev_len.max_blocks;
        src2.ntiles_dst_ref = dev_len.ntiles_dst_ref; src2.stream_k = dev_len.stream_k;
        dev_len = src2;
    }

    ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
    ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask_data,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], n_kv, K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        dev_len
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if (dev_len.kv_len ? fixup_kind == 1 : (int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2, dev_len);
        } else if (dev_len.kv_len) {
            // no fixup at the live length (fixup_kind 0)
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
