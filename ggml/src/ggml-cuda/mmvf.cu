// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "mmvf-splitk.cuh"
#include "mmf.cuh"
#include "convert.cuh"

// k15-mm2: is_multi = up to GGML_CUDA_MMVQ_MULTI_MAX matrices of one type and K over the same activation in one launch (alpha + beta of
// a gated delta net layer, the router + the shared expert gate, ...): blocks [multi_block0[m], multi_block0[m + 1]) compute matrix m's
// rows exactly as its own launch would (same block size, same loop, same reduction) into its contiguous output multi_dst[m]
template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false, bool is_multi_token_id = false, bool is_multi = false>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst_arg,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride) {
    int m_sel = 0;
    if constexpr (is_multi) {
#pragma unroll
        for (int k = 1; k < GGML_CUDA_MMVQ_MULTI_MAX; ++k) {
            if (blockIdx.x >= fusion.multi_block0[k]) {
                m_sel = k;
            }
        }
    }
    const T       * GGML_CUDA_RESTRICT x   = is_multi ? (const T *) fusion.multi_x[m_sel] : x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = is_multi ? fusion.multi_dst[m_sel] : dst_ptr;
    const int row            = is_multi ? int(blockIdx.x - fusion.multi_block0[m_sel]) : int(blockIdx.x);
    const int stride_col_dst = is_multi ? int(fusion.multi_nrows[m_sel]) : stride_col_dst_arg;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
    const int channel_dst = blockIdx.y;
    const int tid         = threadIdx.x;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    ggml_cuda_pdl_sync();
    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? ids[blockIdx.y + token_idx * ids_stride]            : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        dst += token_idx*stride_col_dst;
    }

    // channel_x is uniform in the block, so the whole block leaves on a skipped slot
    if (ids && channel_x == -1) {
        if (tid < ncols_dst) {
            dst[tid*stride_col_dst + row] = 0.0f;
        }
        return;
    }

    x   += int64_t(sample_x)  *stride_sample_x   + int64_t(channel_x)*stride_channel_x + int64_t(row)*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    float glu_limit = 0.0f;
    const T * gate_x = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        glu_op = fusion.glu_op;
        glu_limit = fusion.glu_limit;

        if (use_gate) {
            gate_x = static_cast<const T *>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float *>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float *>(fusion.gate_bias);
            use_gate_bias = use_gate;
        } else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        gate_x += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    }

    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;
    [[maybe_unused]] float * buf_iw_gate = nullptr;
    if constexpr (has_fusion) {
        buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
    }

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};
    float sumf_gate[ncols_dst];
    if constexpr (has_fusion) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf_gate[j] = 0.0f;
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
        [[maybe_unused]] const float2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const float2 *) gate_x;
            }
        }

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            float2 tmpx_gate = make_float2(0.0f, 0.0f);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
        [[maybe_unused]] const half2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const half2 *) gate_x;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                float2 tmpx_gate = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = __half22float2(gate_x2[col2]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};
            half2 sumh2_gate[ncols_dst] = {{0.0f, 0.0f}};

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];
                half2 tmpx_gate = make_half2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
//TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        const int * x2 = (const int *) x;
        const int * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const int *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
            int tmpx_gate = 0;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[0]);
                        const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[1]);
                        ggml_cuda_mad(sumf_gate[j], tmpx0_gate, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx1_gate, tmpy.y);
                    }
                }
            }
        }
#else
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        [[maybe_unused]] const nv_bfloat162 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162 *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
            [[maybe_unused]] nv_bfloat162 tmpx_gate;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    ggml_cuda_pdl_lc();
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if constexpr (has_fusion) {
            if (use_gate) {
                sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
            }
        }

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid/warp_size] = sumf_gate[j];
                }
            }
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[j] = buf_iw_gate[tid];
                        sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                    }
                }
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    float value = sumf[tid];

    if constexpr (has_fusion) {
        if (use_bias) {
            value += x_bias[tid*stride_col_dst + row];
        }

        if (use_gate) {
            float gate_value = sumf_gate[tid];
            if (use_gate_bias) {
                gate_value += gate_bias[tid*stride_col_dst + row];
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    value *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    value *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI: {
                    value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                    break;
                }
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    value = ggml_cuda_op_swiglu_clamp_single(gate_value, value, glu_limit);
                    break;
                default:
                    break;
            }
        }
    }

    dst[tid*stride_col_dst + row] = value;

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, glu_op, glu_limit, gate_x, x_bias, gate_bias, sumf_gate);
    }
}

template<typename T, typename type_acc, int ncols_dst, int block_size, bool is_multi_token_id = false>
static void mul_mat_vec_f_switch_fusion(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const uint3 nchannels_y,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const dim3 & block_dims, const dim3 & block_nums, const int nbytes_shared, const int ids_stride, const cudaStream_t stream) {

    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, nbytes_shared, stream};

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>, launch_params,
                x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
       }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>, launch_params,
        x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);

}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, cudaStream_t stream) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);
    const dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   64: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   96: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  128: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  160: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  192: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  224: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, cudaStream_t stream) {

    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst, true>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_cuda<T, type_acc, 1>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 2:
            launch_mul_mat_vec_f_cuda<T, type_acc, 2>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 3:
            launch_mul_mat_vec_f_cuda<T, type_acc, 3>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 4:
            launch_mul_mat_vec_f_cuda<T, type_acc, 4>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 5:
            launch_mul_mat_vec_f_cuda<T, type_acc, 5>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 6:
            launch_mul_mat_vec_f_cuda<T, type_acc, 6>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 7:
            launch_mul_mat_vec_f_cuda<T, type_acc, 7>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 8:
            launch_mul_mat_vec_f_cuda<T, type_acc, 8>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream) {

    if constexpr(std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
}

// k15-mm2: split-K for F32 mat-vecs with few rows and a long K (Flash-Next: hc inject 10240 -> 4, ssm alpha / beta 2560 -> 48, the
// shared expert gate 2560 -> 1), whose one-block-per-row grid leaves most SMs idle. Block (row, split) sums K range
// [split*chunk, (split+1)*chunk) in f32 like the one-block kernel, writes its partial sums, and the last block of a row to arrive
// (counted in cnt[row], which it resets) adds the partials in split order. Same f32 inputs and f32 accumulation, a different
// summation order (reassociation); deterministic, and every column is summed as a single column launch would sum it.
// It runs for K >= 4096 only: at K 2560 (48 rows, 1 row) the arrival count and the second pass cost what the split saves (B200,
// Flash-Next decode, PDL off: hc inject 11.1 -> 4.2 us at 512 per block, ssm alpha / beta 4.6 -> 4.7, shared expert gate 4.3 -> 4.0).
// GGML_CUDA_MMVF_SPLITK = K elements per block (default 512), 0 = off.
// k15-mm2: the block size launch_mul_mat_vec_f_cuda picks for K = ncols
static int64_t mul_mat_vec_f_block_size(const int64_t ncols, const int device) {
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if (ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return block_size_best;
}

bool ggml_cuda_mul_mat_vec_f_multi_ok(const ggml_type type, const int64_t ncols, const int64_t ncols_dst, const int device) {
    return (type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) && ncols % 2 == 0 && ncols_dst >= 1 && ncols_dst <= 4 &&
        ggml_cuda_info().devices[device].warp_size == 32 && mul_mat_vec_f_block_size(ncols, device) == 256;
}

template <typename T>
static void mul_mat_vec_f_multi_launch(const ggml_cuda_mm_fusion_args_device & fusion, const float * y, const int64_t ncols,
        const int64_t nblocks, const int64_t ncols_dst, const int64_t stride_row, const int64_t stride_col_y, cudaStream_t stream) {
    const uint3 one = init_fastdiv_values(1);
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, (int) (32*sizeof(float)), stream};
#define MMVF_MULTI_LAUNCH(NC) case NC: ggml_cuda_kernel_launch(mul_mat_vec_f<T, float, NC, 256, false, false, true>, launch_params, \
        (const T *) nullptr, y, (const int32_t *) nullptr, fusion, (float *) nullptr, (int) (ncols/2), make_uint3(0, 0, 0), (int) stride_row, \
        (int) (stride_col_y/2), 0, one, 0, 0, 0, one, 0, 0, 0, 0); break;
    switch (ncols_dst) {
        MMVF_MULTI_LAUNCH(1)
        MMVF_MULTI_LAUNCH(2)
        MMVF_MULTI_LAUNCH(3)
        MMVF_MULTI_LAUNCH(4)
        default: GGML_ABORT("fatal error");
    }
#undef MMVF_MULTI_LAUNCH
}

void ggml_cuda_mul_mat_vec_f_multi(ggml_backend_cuda_context & ctx, const int n, const ggml_tensor * const * src0, const ggml_tensor * src1,
        ggml_tensor * const * dst) {
    GGML_ASSERT(n >= 2 && n <= GGML_CUDA_MMVQ_MULTI_MAX);
    ggml_cuda_mm_fusion_args_device fusion{};
    int64_t nblocks = 0;
    for (int m = 0; m < GGML_CUDA_MMVQ_MULTI_MAX; ++m) {
        if (m < n) {
            GGML_ASSERT(src0[m]->type == src0[0]->type && src0[m]->ne[0] == src0[0]->ne[0] && src0[m]->nb[1] == src0[0]->nb[1]);
            GGML_ASSERT(ggml_is_contiguous(dst[m]) && dst[m]->ne[0] == src0[m]->ne[1] && dst[m]->ne[1] == src1->ne[1]);
            fusion.multi_x[m]      = src0[m]->data;
            fusion.multi_dst[m]    = (float *) dst[m]->data;
            fusion.multi_nrows[m]  = (uint32_t) src0[m]->ne[1];
            fusion.multi_block0[m] = (uint32_t) nblocks;
            nblocks += src0[m]->ne[1];
        } else {
            fusion.multi_block0[m] = UINT32_MAX;
        }
    }
    const int64_t ncols        = src0[0]->ne[0];
    const int64_t stride_row   = src0[0]->nb[1] / ggml_type_size(src0[0]->type);
    const int64_t stride_col_y = src1->nb[1] / sizeof(float);
    GGML_ASSERT(ggml_cuda_mul_mat_vec_f_multi_ok(src0[0]->type, ncols, src1->ne[1], ctx.device) && stride_row % 2 == 0 && stride_col_y % 2 == 0);
    const float * y = (const float *) src1->data;
    if (src0[0]->type == GGML_TYPE_F32) {
        mul_mat_vec_f_multi_launch<float>(fusion, y, ncols, nblocks, src1->ne[1], stride_row, stride_col_y, ctx.stream());
    } else {
        mul_mat_vec_f_multi_launch<nv_bfloat16>(fusion, y, ncols, nblocks, src1->ne[1], stride_row, stride_col_y, ctx.stream());
    }
}

// CUDA only: HIP / MUSA keep the one-block-per-row kernel (the arrival count uses __ldcg and the stream capture query)
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <int ncols_dst>
static __global__ void __launch_bounds__(MMVF_SPLITK_BLOCK) mul_mat_vec_f_splitk(
        const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst, float * __restrict__ part,
        unsigned int * __restrict__ cnt, const int ncols2, const int chunk2, const int stride_row, const int stride_col_y2,
        const int stride_col_dst) {
    mul_mat_vec_f_splitk_block<ncols_dst>(x, y, dst, part, cnt, ncols2, chunk2, stride_row, stride_col_y2, stride_col_dst,
        blockIdx.x, blockIdx.y, gridDim.y, threadIdx.x);
}

// the split-K launch of this product, if it applies (false: the caller runs the one-block-per-row kernel). alloc = false: never
// allocates the partial sum buffers (a planner asking whether the launch would be taken; they are allocated outside a capture)
static bool mul_mat_vec_f_splitk_plan(ggml_backend_cuda_context & ctx, const float * x, const float * y, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst, const int64_t stride_row, const int64_t stride_col_y,
        const int64_t stride_col_dst, const bool alloc, ggml_cuda_mmvf_splitk_args & a) {
    static const int chunk = [] {
        const char * e = getenv("GGML_CUDA_MMVF_SPLITK");
        return e != nullptr ? atoi(e) : 512;
    }();
    const auto & info = ggml_cuda_info().devices[ctx.device];
    if (chunk < 2*MMVF_SPLITK_BLOCK || chunk % 2 != 0 || info.warp_size != 32 || ncols_dst < 1 || ncols_dst > MMVF_MAX_BATCH_SIZE ||
            nrows > MMVF_SPLITK_MAX_ROWS || nrows >= info.nsm || ncols < std::max(2*chunk, 4096) || ncols % 2 != 0 || stride_row % 2 != 0 ||
            stride_col_y % 2 != 0 || ncols > INT_MAX || stride_row > INT_MAX || stride_col_y*ncols_dst > INT_MAX) {
        return false;
    }
    const int64_t nsplit = (ncols + chunk - 1) / chunk;
    if (nsplit > MMVF_SPLITK_MAX_SPLIT) {
        return false;
    }
    const size_t part_per_stream = (size_t) MMVF_SPLITK_MAX_ROWS*MMVF_SPLITK_MAX_SPLIT*MMVF_MAX_BATCH_SIZE;
    if (ctx.mmvf_splitk_part == nullptr) {
        if (!alloc) {
            return false;
        }
        cudaStream_t stream = ctx.stream();
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMalloc((void **) &ctx.mmvf_splitk_part, GGML_CUDA_MAX_STREAMS*part_per_stream*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **) &ctx.mmvf_splitk_cnt,  GGML_CUDA_MAX_STREAMS*MMVF_SPLITK_MAX_ROWS*sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(ctx.mmvf_splitk_cnt, 0, GGML_CUDA_MAX_STREAMS*MMVF_SPLITK_MAX_ROWS*sizeof(unsigned int)));
    }
    // one region per stream: products on the side / concurrent streams may run at the same time as those on stream 0
    a.x              = x;
    a.y              = y;
    a.dst            = dst;
    a.part           = ctx.mmvf_splitk_part + ctx.curr_stream_no*part_per_stream;
    a.cnt            = ctx.mmvf_splitk_cnt  + ctx.curr_stream_no*MMVF_SPLITK_MAX_ROWS;
    a.nrows          = (int) nrows;
    a.nsplit         = (int) nsplit;
    a.ncols2         = (int) (ncols/2);
    a.chunk2         = chunk/2;
    a.stride_row     = (int) stride_row;
    a.stride_col_y2  = (int) (stride_col_y/2);
    a.stride_col_dst = (int) stride_col_dst;
    a.ncols_dst      = (int) ncols_dst;
    return true;
}

static bool mul_mat_vec_f_splitk_try(ggml_backend_cuda_context & ctx, const float * x, const float * y, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst, const int64_t stride_row, const int64_t stride_col_y,
        const int64_t stride_col_dst) {
    ggml_cuda_mmvf_splitk_args a;
    if (!mul_mat_vec_f_splitk_plan(ctx, x, y, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst, true, a)) {
        return false;
    }
    const dim3 block_nums(a.nrows, a.nsplit, 1);
    const dim3 block_dims(MMVF_SPLITK_BLOCK, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, ctx.stream()};
#define MMVF_SPLITK_LAUNCH(NC) case NC: ggml_cuda_kernel_launch(mul_mat_vec_f_splitk<NC>, launch_params, a.x, a.y, a.dst, a.part, a.cnt, \
        a.ncols2, a.chunk2, a.stride_row, a.stride_col_y2, a.stride_col_dst); break;
    switch (ncols_dst) {
        MMVF_SPLITK_LAUNCH(1)
        MMVF_SPLITK_LAUNCH(2)
        MMVF_SPLITK_LAUNCH(3)
        MMVF_SPLITK_LAUNCH(4)
        MMVF_SPLITK_LAUNCH(5)
        MMVF_SPLITK_LAUNCH(6)
        MMVF_SPLITK_LAUNCH(7)
        MMVF_SPLITK_LAUNCH(8)
        default: GGML_ABORT("fatal error");
    }
#undef MMVF_SPLITK_LAUNCH
    return true;
}
#else
static bool mul_mat_vec_f_splitk_plan(ggml_backend_cuda_context & ctx, const float * x, const float * y, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst, const int64_t stride_row, const int64_t stride_col_y,
        const int64_t stride_col_dst, const bool alloc, ggml_cuda_mmvf_splitk_args & a) {
    GGML_UNUSED_VARS(ctx, x, y, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst, alloc, a);
    return false;
}

static bool mul_mat_vec_f_splitk_try(ggml_backend_cuda_context & ctx, const float * x, const float * y, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst, const int64_t stride_row, const int64_t stride_col_y,
        const int64_t stride_col_dst) {
    GGML_UNUSED_VARS(ctx, x, y, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst);
    return false;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// k15-mm2: the split-K launch ggml_cuda_mul_mat would make for this dense F32 product (ggml_cuda_mul_mat_vec_f, no fusion), without
// allocating; false if it would not take that launch (see the hyper-connection down + inject launch in mmvq.cu)
bool ggml_cuda_mul_mat_vec_f_splitk_plan(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, ggml_cuda_mmvf_splitk_args & a) {
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src0->nb[0] != sizeof(float) ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 ||
            src1->ne[3] != 1 || src0->data == nullptr || src1->data == nullptr || dst->data == nullptr) {
        return false;
    }
    return mul_mat_vec_f_splitk_plan(ctx, (const float *) src0->data, (const float *) src1->data, (float *) dst->data, src0->ne[0],
        src0->ne[1], src1->ne[1], src0->nb[1] / sizeof(float), src1->nb[1] / sizeof(float), dst->nb[1] / sizeof(float), false, a);
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 <= MMVF_MAX_BATCH_SIZE);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            const bool has_fusion = fusion_local.gate != nullptr || fusion_local.x_bias != nullptr || fusion_local.gate_bias != nullptr;
            if (!ids && !has_fusion && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
                    mul_mat_vec_f_splitk_try(ctx, src0_d, src1_d, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst)) {
                break;
            }
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  =  dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t stride_col_y       = ne10;
    const int64_t stride_col_dst     = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    ggml_cuda_mm_fusion_args_device empty{};
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED_VARS(ctx, src1, dst, src1_ddq_i, src1_ncols, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                // MMF needs full row tiles, for other row counts MMVF still beats cuBLAS at small batch size
                if (src0_small && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (GGML_CUDA_CC_IS_RDNA(cc) && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                // k15-mm2: GGML_CUDA_MMVF_BF16_MAX_N=N (opt-in, 0 = off): small BF16 weights at up to N columns take the vector kernel
                // (f32 activations and sums) instead of MMF (BF16 activations): more precise, not bit-identical (KLD-gated)
                static const int bf16_max_n = getenv("GGML_CUDA_MMVF_BF16_MAX_N") ? atoi(getenv("GGML_CUDA_MMVF_BF16_MAX_N")) : 0;
                if (src0_small && ne11 <= std::min(bf16_max_n, MMVF_MAX_BATCH_SIZE)) {
                    return true;
                }
                // MMF needs full row tiles, for other row counts MMVF still beats cuBLAS at small batch size
                if (src0_small && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (GGML_CUDA_CC_IS_RDNA(cc) && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        default:
            return false;
    }
}
