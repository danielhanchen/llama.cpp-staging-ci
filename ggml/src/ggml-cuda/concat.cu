// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "concat.cuh"
#include "cpy.cuh"

#include <stdint.h>
#include <cstdlib>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// dim 0 concat of a row-contiguous src0 with a transposed src1 (src1 = ggml_transpose of a contiguous tensor, e.g. the conv
// state of GDN / Mamba prefill: concat(state, transpose(x), 0)). The generic non-contiguous kernel reads src1 with a stride of
// ne11 elements per thread; here a 32x32 tile goes through shared memory so both the src1 reads and the dst writes coalesce.
// Pure copy, bit-identical to the generic kernels. GGML_CUDA_CONCAT_TRANSPOSED=0 disables it.
template <typename T>
static __global__ void concat_transposed_src1_dim0(
        const char * __restrict__ src0, const char * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const int64_t ne10, const int64_t ne11,
        const uint64_t nb01, const uint64_t nb02,
        const uint64_t nb10, const uint64_t nb12,
        const uint64_t nb1, const uint64_t nb2) {
    constexpr int tile = 32;
    __shared__ T values[tile][tile + 1];

    const int     tx       = threadIdx.x;
    const int     ty       = threadIdx.y;
    const int64_t channel0 = (int64_t) blockIdx.x * tile;
    const int64_t token0   = (int64_t) blockIdx.y * tile;
    const int64_t i2       = blockIdx.z;

    ggml_cuda_pdl_sync();
    // src1 element (token, channel) sits at channel*sizeof(T) + token*nb10: consecutive tx = consecutive channels
    for (int j = 0; j < tile; j += blockDim.y) {
        const int64_t token   = token0 + ty + j;
        const int64_t channel = channel0 + tx;
        if (token < ne10 && channel < ne11) {
            values[ty + j][tx] = *(const T *) (src1 + i2*nb12 + channel*sizeof(T) + token*nb10);
        }
    }
    __syncthreads();

    // dst row = channel, columns ne00 + token: consecutive tx = consecutive tokens
    for (int j = 0; j < tile; j += blockDim.y) {
        const int64_t token   = token0 + tx;
        const int64_t channel = channel0 + ty + j;
        if (token < ne10 && channel < ne11) {
            *(T *) (dst + i2*nb2 + channel*nb1 + (ne00 + token)*sizeof(T)) = values[tx][ty + j];
        }
    }

    // the src0 part of the rows of this block, once per channel tile
    if (blockIdx.y == 0) {
        for (int j = 0; j < tile; j += blockDim.y) {
            const int64_t channel = channel0 + ty + j;
            if (channel < ne11) {
                for (int64_t i0 = tx; i0 < ne00; i0 += tile) {
                    *(T *) (dst + i2*nb2 + channel*nb1 + i0*sizeof(T)) = *(const T *) (src0 + i2*nb02 + channel*nb01 + i0*sizeof(T));
                }
            }
        }
    }
}

static bool concat_transposed_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_CONCAT_TRANSPOSED");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// k19-q4xt (GGML_CUDA_CONCAT_TRANSPOSED_MIN, default 2; 32 = as before): the fewest src1 columns (tokens) the tile kernel takes. A
// speculative verify window (2..8 tokens) of a GDN layer concatenated its conv state through the generic kernel: one 256-thread block
// per channel (8192 for Qwen3.6-35B-A3B) for 6 values each, ~40 us per layer on a T4, ~1.2 ms per 3-token verify pass. The tile
// kernel masks the tokens past ne10 (its loops already guard token < ne10); a pure copy either way, so the bytes are the same.
static int64_t concat_transposed_min_tokens() {
    static const int64_t v = []() -> int64_t {
        const char * env = getenv("GGML_CUDA_CONCAT_TRANSPOSED_MIN");
        const int n = env != nullptr ? atoi(env) : 2;
        return n >= 1 ? n : 2;
    }();
    return v;
}

// shape gate (any arch): dim 0, every tensor 3D (ne3 == 1), src0 and dst with contiguous rows, src1 the transpose of a
// tensor with contiguous rows (channels contiguous, tokens strided), at least concat_transposed_min_tokens() src1 columns (tokens)
template <typename T>
static bool concat_transposed_src1_dim0_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int dim) {
    return dim == 0 && concat_transposed_enabled() &&
        src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1 &&
        src1->ne[0] >= concat_transposed_min_tokens() &&
        src0->ne[1] == src1->ne[1] && src0->ne[2] == src1->ne[2] && src1->ne[2] <= 65535 &&
        src0->nb[0] == sizeof(T) && dst->nb[0] == sizeof(T) &&
        src1->nb[1] == sizeof(T) && src1->nb[0] >= (size_t) src1->ne[1]*sizeof(T) &&
        (src1->ne[0] + 31)/32 <= 65535;
}

template <typename T>
static void concat_transposed_src1_dim0_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream) {
    constexpr int tile = 32;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int tile_y = GGML_CUDA_CC_IS_RDNA3_5(cc) ? 16 : 8;
    const dim3 block_dims(tile, tile_y, 1);
    const dim3 block_nums((src1->ne[1] + tile - 1) / tile, (src1->ne[0] + tile - 1) / tile, src1->ne[2]);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(concat_transposed_src1_dim0<T>, launch_params,
        (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
        src0->ne[0], src1->ne[0], src1->ne[1],
        (uint64_t) src0->nb[1], (uint64_t) src0->nb[2],
        (uint64_t) src1->nb[0], (uint64_t) src1->nb[2],
        (uint64_t) dst->nb[1], (uint64_t) dst->nb[2]);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    if (!ggml_is_quantized(src0->type) && concat_transposed_src1_dim0_ok<T>(src0, src1, dst, dim)) {
        concat_transposed_src1_dim0_cuda<T>(src0, src1, dst, stream);
        return;
    }
    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        ggml_cuda_memcpy_d2d((char *) dst->data,         size0, src0->data, size0, size0, 1, stream);
        ggml_cuda_memcpy_d2d((char *) dst->data + size0, size1, src1->data, size1, size1, 1, stream);
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
        }
    }
}
