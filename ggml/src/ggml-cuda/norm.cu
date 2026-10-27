#include "norm.cuh"
#include "convert.cuh"
#include "rope-common.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float2 s_sum2[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float2 mean_var = make_float2(0.0f, 0.0f);

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                mean_var.x += xi;
                mean_var.y += xi * xi;
            }

            // sum up partial sums
            mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

            const float mean = mean_var.x / ncols;
            const float var = mean_var.y / ncols - mean * mean;
            const float inv_std = rsqrtf(var + eps);

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = (xc[col] - mean) * inv_std;
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum2 across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// k15-fn: the multi-section rope applied to each normed row by the norm kernel itself (RMS_NORM -> MUL -> ROPE, the q / k norms of
// qwen4exp attention): row = head, channel = token, sample = sequence of the rope, see ggml_cuda_op_rms_norm_rope_fused
struct ggml_cuda_rms_rope_args {
    float *         dst          = nullptr;
    int64_t         s1           = 0; // dst strides in floats: head, token, sequence
    int64_t         s2           = 0;
    int64_t         s3           = 0;
    int             n_dims       = 0;
    int             n_offs       = 0;
    const int32_t * pos          = nullptr;
    float           freq_scale   = 0.0f;
    float           ext_factor   = 0.0f;
    float           attn_factor  = 0.0f;
    rope_corr_dims  corr_dims    = {};
    float           theta_scale  = 0.0f;
    const float *   freq_factors = nullptr;
    mrope_sections  sections     = {};
    bool            is_imrope    = false;
    bool            inplace      = false;
};

template <int block_size, int vpt, bool do_vec, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool do_rope = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int     nchannels,
                                    const int     nsamples,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f,
                                    const ggml_cuda_rms_rope_args rope = {}) {
    static_assert(!do_rope || (do_multiply && !do_add && !do_scale), "the rope epilogue follows rms_norm + mul only");
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            [[maybe_unused]] const float * mulc = nullptr;
            if constexpr (do_multiply) {
                const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
                const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
                const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
                mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
            }

            [[maybe_unused]] const float * addc = nullptr;
            if constexpr (do_add) {
                const int add_row     = fastmodulo(row, add_nrows_packed);
                const int add_channel = fastmodulo(channel, add_nchannels_packed);
                const int add_sample  = fastmodulo(sample, add_nsamples_packed);
                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
            }

            if constexpr (do_vec) {
                const int      ncols4 = ncols / 4;
                const float4 * x4     = (const float4 *) xc;
                const float4 * mul4   = (const float4 *) mulc;
                const float4 * add4   = (const float4 *) addc;
                float4 *       dst4   = (float4 *) dstc;

                float4 xs[vpt];
                float  tmp = 0.0f;

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        xs[k] = x4[i];
                        tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
                    }
                }
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
                        if constexpr (do_multiply) {
                            const float4 m = mul4[i];
                            r.x *= m.x;
                            r.y *= m.y;
                            r.z *= m.z;
                            r.w *= m.w;
                        }
                        if constexpr (do_add) {
                            const float4 a = add4[i];
                            r.x += a.x;
                            r.y += a.y;
                            r.z += a.z;
                            r.w += a.w;
                        }
                        if constexpr (do_scale) {
                            r.x *= scale_out;
                            r.y *= scale_out;
                            r.z *= scale_out;
                            r.w *= scale_out;
                        }
                        dst4[i] = r;
                    }
                }
            } else {
                float tmp = 0.0f; // partial sum for thread in warp

                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = xc[col];
                    tmp += xi * xi;
                }

                // sum up partial sums
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

                const float mean = tmp / ncols;
                const float scale = rsqrtf(mean + eps);

                for (int col = tid; col < ncols; col += block_size) {
                    if constexpr (do_multiply && do_add) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        const int add_col = fastmodulo(col, add_ncols_packed);
                        dstc[col]         = scale * xc[col] * mulc[mul_col] + addc[add_col];
                    } else if constexpr (do_multiply) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        dstc[col]         = scale * xc[col] * mulc[mul_col];
                    } else if constexpr (do_scale) {
                        dstc[col] = scale_out * (scale * xc[col]);
                    } else {
                        dstc[col] = scale * xc[col];
                    }
                }
            }

            if constexpr (do_rope) {
                // the normed row is read by other threads of the block
                __syncthreads();
                float * rd = rope.dst + sample*rope.s3 + channel*rope.s2 + row*rope.s1;
                for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
                    if (i0 < rope.n_offs || i0 >= rope.n_offs + rope.n_dims) {
                        if (!rope.inplace) {
                            rd[i0 + 0] = dstc[i0 + 0];
                            rd[i0 + 1] = dstc[i0 + 1];
                        }
                        continue;
                    }
                    if (rope.freq_factors != nullptr) {
                        rope_multi_rotate<true, true>((const float *) dstc, rd, i0, (uint32_t) channel, nchannels, rope.n_dims, rope.n_offs,
                            rope.pos, rope.freq_scale, rope.ext_factor, rope.attn_factor, rope.corr_dims, rope.theta_scale, rope.freq_factors,
                            rope.sections, rope.is_imrope);
                    } else {
                        rope_multi_rotate<true, false>((const float *) dstc, rd, i0, (uint32_t) channel, nchannels, rope.n_dims, rope.n_offs,
                            rope.pos, rope.freq_scale, rope.ext_factor, rope.attn_factor, rope.corr_dims, rope.theta_scale, rope.freq_factors,
                            rope.sections, rope.is_imrope);
                    }
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float tmp = 0.0f; // partial sum for thread in warp

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            // sum up partial sums
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
            const float scale = rsqrtf(fmaxf(tmp, eps * eps));

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = scale * xc[col];
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

// rms_norm(x) * mul that also writes the q8_1 copy of its output for the mul_mat_vec_q consumers (see ggml_cuda_q8_1_cache).
// The f32 output uses the same expressions as rms_norm_f32<block_size, vpt, do_vec, true> (the sum of squares in the same
// order: float4 loads when do_vec, launched with the block size rms_norm_f32_dispatch picks); the q8_1 rows are produced
// from the same f32 values exactly as quantize_q8_1 does (one value per lane, the same 32-lane reductions), so both
// outputs are bit-identical to the unfused rms_norm + mul + quantize_q8_1 sequence.
template <int block_size, int vpt, bool do_vec, int batch = 1, bool c_hcpost = false>
static __global__ void rms_norm_mul_q8_1_f32(const float * x,
                                             float *       dst,
                                             void *        vq,
                                             const int     ncols,
                                             const int     ncols_padded,
                                             const int     nchannels,
                                             const int     nsamples,
                                             const int64_t stride_row,
                                             const int64_t stride_channel,
                                             const int64_t stride_sample,
                                             const float   eps,
                                             const float * mul,
                                             const int64_t mul_stride_row,
                                             const int64_t mul_stride_channel,
                                             const int64_t mul_stride_sample,
                                             const uint3   mul_ncols_packed,
                                             const uint3   mul_nrows_packed,
                                             const uint3   mul_nchannels_packed,
                                             const uint3   mul_nsamples_packed,
                                             const float * ra = nullptr,
                                             const float * rb = nullptr,
                                             const ggml_cuda_hc_post_args hcp = {}) {
    // k15: ra / rb (both or neither): x = ra + rb is computed here (the ADD before the norm, a residual add) and stored to x, with the
    // same float add as k_bin_bcast's op_add; the norm reads the same values, so every output is bit-identical to ADD + RMS_NORM + MUL
    static_assert(block_size % QK8_1 == 0, "a warp must cover whole q8_1 blocks");
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const int64_t irow = ((int64_t) sample*nchannels + channel)*nrows + row;
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + irow*ncols;

            const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
            const float * mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

            float tmp = 0.0f;
            const int64_t xoff = sample*stride_sample + channel*stride_channel + row*stride_row;
            // c_hcpost: the row is stream `row` of token `channel` of the DSV4_HC_POST output; its weight once per block
            [[maybe_unused]] float hc_w = 0.0f;
            [[maybe_unused]] const float * hc_bo = nullptr;
            [[maybe_unused]] const float * hc_r  = nullptr;
            if constexpr (c_hcpost) {
                __shared__ float hc_w_s;
                __syncthreads();
                if (tid == 0) {
                    float w = hcp.post[row + (int64_t) hcp.hc*channel];
#pragma unroll
                    for (int j = 0; j < GGML_CUDA_HC_POST_NORM_MAX_OPS; ++j) {
                        if (j < hcp.n_ops) {
                            // the expressions of ew_chain_apply (SCALE) and op_sigmoid
                            w = hcp.op[j] == 0 ? hcp.p0[j] * w + hcp.p1[j] : 1.0f / (1.0f + expf(-w));
                        }
                    }
                    hc_w_s = w;
                }
                __syncthreads();
                hc_w  = hc_w_s;
                hc_bo = hcp.bo + channel*hcp.sbo1;
                hc_r  = hcp.residual + row*hcp.sr1 + channel*hcp.sr2;
            }
            if constexpr (do_vec) {
                const int      ncols4 = ncols / 4;
                const float4 * x4     = (const float4 *) xc;
#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        float4 xs;
                        if constexpr (c_hcpost) {
                            // hc_post_chain_f32: sum = x*w; sum += residual
                            const float4 b = ((const float4 *) hc_bo)[i];
                            const float4 r = ((const float4 *) hc_r)[i];
                            xs.x = b.x * hc_w; xs.x += r.x;
                            xs.y = b.y * hc_w; xs.y += r.y;
                            xs.z = b.z * hc_w; xs.z += r.z;
                            xs.w = b.w * hc_w; xs.w += r.w;
                            ((float4 *) const_cast<float *>(xc))[i] = xs;
                        } else if (ra != nullptr) {
                            const float4 a = ((const float4 *) (ra + xoff))[i];
                            const float4 b = ((const float4 *) (rb + xoff))[i];
                            xs = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
                            ((float4 *) const_cast<float *>(xc))[i] = xs;
                        } else {
                            xs = x4[i];
                        }
                        tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
                    }
                }
            } else {
                for (int col = tid; col < ncols; col += block_size) {
                    float xi;
                    if constexpr (c_hcpost) {
                        xi = hc_bo[col] * hc_w;
                        xi += hc_r[col];
                        const_cast<float *>(xc)[col] = xi;
                    } else if (ra != nullptr) {
                        xi = ra[xoff + col] + rb[xoff + col];
                        const_cast<float *>(xc)[col] = xi;
                    } else {
                        xi = xc[col];
                    }
                    tmp += xi * xi;
                }
            }
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
            if (c_hcpost || ra != nullptr) {
                __syncthreads(); // the second pass reads elements other threads stored
            }

            const float mean  = tmp / ncols;
            const float scale = rsqrtf(mean + eps);

            // one pass: element i0 is computed, stored and quantized by the same thread, and lane i0 % QK8_1 of its warp is
            // the q8_1 lane quantize_q8_1 uses for it (block_size % QK8_1 == 0), so no read-back is needed
            block_q8_1 * y = (block_q8_1 *) vq + irow*(ncols_padded/QK8_1);
            // k15-mm2: a batch of RMS_Q8_1_BATCH elements per thread has its loads issued before its math (GGML_CUDA_RMS_Q8_1_BATCH=0:
            // one element at a time); every element is computed by the same expressions and lanes, so the outputs do not change
            for (int i00 = tid; i00 < ncols_padded; i00 += batch*block_size) { // ncols_padded % WARP_SIZE == 0: whole warps
                float xv[batch];
                float mv[batch];
#pragma unroll
                for (int u = 0; u < batch; ++u) {
                    const int i0 = i00 + u*block_size;
                    xv[u] = i0 < ncols ? xc[i0] : 0.0f;
                    mv[u] = i0 < ncols ? mulc[fastmodulo(i0, mul_ncols_packed)] : 0.0f;
                }
#pragma unroll
                for (int u = 0; u < batch; ++u) {
                    const int i0 = i00 + u*block_size;
                    if (i0 >= ncols_padded) {
                        break; // uniform per warp
                    }
                    float xi = 0.0f;
                    if (i0 < ncols) {
                        xi       = scale * xv[u] * mv[u];
                        dstc[i0] = xi;
                    }
                    float amax = fabsf(xi);
                    float sum  = xi;

                    amax = warp_reduce_max<QK8_1>(amax);
                    sum  = warp_reduce_sum<QK8_1>(sum);

                    const float  d = amax / 127.0f;
                    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

                    const int64_t ib  = i0 / QK8_1;
                    const int     iqs = i0 % QK8_1;
                    y[ib].qs[iqs] = q;
                    if (iqs == 0) {
                        y[ib].ds = make_half2(d, sum);
                    }
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                __syncthreads();
            }
        }
    }
}

#define RMS_NORM_LAUNCH(BLOCK_SIZE, VPT, VEC) \
    do { \
        const dim3 block_dims(BLOCK_SIZE, 1, 1); \
        const ggml_cuda_kernel_launch_params launch_params{ blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream }; \
        ggml_cuda_kernel_launch(rms_norm_f32<BLOCK_SIZE, VPT, VEC, do_multiply, do_add, do_scale, do_rope>, launch_params, \
            x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, \
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \
            add, add_stride_row, add_stride_channel, add_stride_sample, \
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out, rope); \
    } while (0)

template <bool do_multiply, bool do_add, bool do_scale, bool do_rope = false>
static void rms_norm_f32_dispatch(
        const dim3     blocks_num,
        const ggml_tensor * x_t,   float * dst,   const int ncols, const int nchannels, const int nsamples,
        const int64_t  stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const ggml_tensor * mul_t, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3    mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const ggml_tensor * add_t, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3    add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
        cudaStream_t   stream, const float scale_out = 1.0f, const ggml_cuda_rms_rope_args & rope = {}) {
    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    const float * x   = (const float *) x_t->data;
    const float * mul = mul_t ? (const float *) mul_t->data : nullptr;
    const float * add = add_t ? (const float *) add_t->data : nullptr;

    // dst is written at row * ncols, i.e. always at a multiple of ncols, so only its base needs checking here
    bool vec = ncols % 4 == 0 && ncols > 0 && ((uintptr_t) dst & 15) == 0;
    vec = vec && ggml_cuda_is_aligned(x_t, 16);
    if constexpr (do_multiply) {
        vec = vec && mul_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(mul_t, 16);
    }
    if constexpr (do_add) {
        vec = vec && add_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(add_t, 16);
    }

    const int n4 = ncols / 4;
    if (vec) {
        if (n4 <= 32) {
            RMS_NORM_LAUNCH(32,   1, true);
        } else if (n4 <= 128) {
            RMS_NORM_LAUNCH(128,  1, true);
        } else if (n4 <= 256) {
            RMS_NORM_LAUNCH(256,  1, true);
        } else if (n4 <= 512) {
            RMS_NORM_LAUNCH(512,  1, true);
        } else if (n4 <= 1024) {
            RMS_NORM_LAUNCH(1024, 1, true);
        } else if (n4 <= 4096) {
            RMS_NORM_LAUNCH(1024, 4, true);
        } else {
            RMS_NORM_LAUNCH(1024, 1, false);
        }
    } else if (ncols < 1024) {
        RMS_NORM_LAUNCH(256,  1, false);
    } else {
        RMS_NORM_LAUNCH(1024, 1, false);
    }
}
#undef RMS_NORM_LAUNCH

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const ggml_tensor * x_t, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    rms_norm_f32_dispatch</*do_multiply*/false, /*do_add*/false, do_scale>(
        blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        stream, scale_out);
}

static void rms_norm_mul_f32_cuda(const ggml_tensor * x_t,
                                  const ggml_tensor * mul_t,
                                  const ggml_tensor * add_t,
                                  float             * dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (mul_t == nullptr) {
        rms_norm_f32_cuda(x_t, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
    const uint3 zero_packed          = make_uint3(0, 0, 0);

    if (add_t == nullptr) {
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream);
    } else {
        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/true, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            add_t, add_stride_row, add_stride_channel, add_stride_sample,
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, stream);
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

// k11: rms_norm(x) * mul that also writes the copy of its output that the next matrix products convert it to (prompt batches,
// see ggml_cuda_src1_memo): mode 0 / 1 = the q8_1 MMQ rows (block_q8_1_mmq, D4 / DS4 scales) of the products' [K_out, N_out]
// view of the output, mode 2 = its BF16 copy. Only the launch rms_norm_f32_dispatch picks for float4 rows of at most 1024
// float4 (vpt 1), so the f32 output is computed by the same expressions in the same order; each thread holds the same 4
// consecutive values quantize_mmq_q8_1 / convert_unary_cont_vec4 give one thread, with the same lane reductions, so every
// output is bit-identical to the unfused rms_norm + mul + conversion.
#define K11_Q8_1_MMQ_BLOCK 144 // sizeof(block_q8_1_mmq): 16 bytes of scales, then 128 int8 (static_assert in mmq.cu)
template <int block_size, int mode>
static __global__ void rms_norm_mul_conv_f32(const float * x,
                                             float *       dst,
                                             void *        vout,
                                             const int     ncols,
                                             const int     nchannels,
                                             const int     nsamples,
                                             const int64_t stride_row,
                                             const int64_t stride_channel,
                                             const int64_t stride_sample,
                                             const float   eps,
                                             const float * mul,
                                             const int64_t mul_stride_row,
                                             const int64_t mul_stride_channel,
                                             const int64_t mul_stride_sample,
                                             const uint3   mul_nrows_packed,
                                             const uint3   mul_nchannels_packed,
                                             const uint3   mul_nsamples_packed,
                                             const int64_t K_out,
                                             const int64_t N_out) {
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const int64_t irow = ((int64_t) sample*nchannels + channel)*nrows + row;
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + irow*ncols;

            const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
            const float * mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

            const int      ncols4 = ncols / 4;
            const float4 * x4     = (const float4 *) xc;
            const float4 * mul4   = (const float4 *) mulc;
            float4 *       dst4   = (float4 *) dstc;

            const int i = tid;
            float4 xs = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float  tmp = 0.0f;
            if (i < ncols4) {
                xs = x4[i];
                tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
            }
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
            const float scale = rsqrtf(tmp / ncols + eps);

            if (i < ncols4) { // ncols4 % WARP_SIZE == 0: whole warps
                float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
                const float4 m = mul4[i];
                r.x *= m.x;
                r.y *= m.y;
                r.z *= m.z;
                r.w *= m.w;
                dst4[i] = r;

                const int64_t e   = irow*ncols + 4*i; // flat index of the first value = its place in the [K_out, N_out] view
                const int64_t tok = e / K_out;
                const int64_t k   = e % K_out;
                if constexpr (mode == 2) {
                    nv_bfloat16 * yb = (nv_bfloat16 *) vout + e;
                    yb[0] = ggml_cuda_cast<nv_bfloat16>(r.x);
                    yb[1] = ggml_cuda_cast<nv_bfloat16>(r.y);
                    yb[2] = ggml_cuda_cast<nv_bfloat16>(r.z);
                    yb[3] = ggml_cuda_cast<nv_bfloat16>(r.w);
                } else {
                    float amax = fabsf(r.x);
                    amax = fmaxf(amax, fabsf(r.y));
                    amax = fmaxf(amax, fabsf(r.z));
                    amax = fmaxf(amax, fabsf(r.w));
#pragma unroll
                    for (int offset = 4; offset > 0; offset >>= 1) {
                        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
                    }
                    float sum = 0.0f;
                    if constexpr (mode == 1) {
                        sum = r.x + r.y + r.z + r.w;
#pragma unroll
                        for (int offset = 4; offset > 0; offset >>= 1) {
                            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
                        }
                    }
                    const float d_inv = 127.0f / amax;
                    char4 q;
                    q.x = roundf(r.x*d_inv);
                    q.y = roundf(r.y*d_inv);
                    q.z = roundf(r.z*d_inv);
                    q.w = roundf(r.w*d_inv);
                    const float d = 1.0f / d_inv;

                    const int64_t ib  = (k / 128)*N_out + tok;
                    const int     iqs = k % 128;
                    char * yb = (char *) vout + ib*K11_Q8_1_MMQ_BLOCK;
                    ((char4 *) (yb + 16))[iqs/4] = q;
                    if (iqs % 32 == 0) {
                        if constexpr (mode == 1) {
                            ((half2 *) yb)[iqs/32] = make_half2(d, sum);
                        } else {
                            ((float *) yb)[iqs/32] = d;
                        }
                    }
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                __syncthreads();
            }
        }
    }
}

// k11: launch rms_norm_mul_conv_f32 when rms_norm_f32_dispatch would pick the float4, vpt 1 kernel for this norm; false if not
static bool rms_norm_mul_conv_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * x_t, const ggml_tensor * mul_t, float * dst_d,
        int mode, void * out, int64_t K_out, int64_t N_out, float eps) {
    const int64_t ne00 = x_t->ne[0], ne01 = x_t->ne[1], ne02 = x_t->ne[2], ne03 = x_t->ne[3];
    const size_t ts0 = ggml_type_size(x_t->type);
    const size_t tsm = ggml_type_size(mul_t->type);
    bool vec = ne00 % 4 == 0 && ne00 > 0 && ((uintptr_t) dst_d & 15) == 0 && ggml_cuda_is_aligned(x_t, 16) &&
        mul_t->ne[0] == ne00 && ggml_cuda_is_aligned(mul_t, 16) && x_t->nb[0] == ts0 && mul_t->nb[0] == tsm;
    const int64_t n4 = ne00 / 4;
    if (!vec || n4 > 1024 || n4 % WARP_SIZE != 0 || ne00 % 128 != 0 || K_out % 128 != 0 || K_out*N_out != ggml_nelements(x_t) ||
            ne02 > UINT16_MAX || ne03 > UINT16_MAX) {
        return false;
    }
    const dim3 blocks_num(ne01, ne02, ne03);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_t->ne[1]);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_t->ne[2]);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_t->ne[3]);
    const float * x   = (const float *) x_t->data;
    const float * mul = (const float *) mul_t->data;
    cudaStream_t stream = ctx.stream();
#define K11_CONV_LAUNCH(BS) \
    do { \
        const ggml_cuda_kernel_launch_params lp = {blocks_num, dim3(BS, 1, 1), 32 * sizeof(float), stream}; \
        if (mode == 0) { \
            ggml_cuda_kernel_launch(rms_norm_mul_conv_f32<BS, 0>, lp, x, dst_d, out, (int) ne00, (int) ne02, (int) ne03, \
                (int64_t) (x_t->nb[1]/ts0), (int64_t) (x_t->nb[2]/ts0), (int64_t) (x_t->nb[3]/ts0), eps, mul, \
                (int64_t) (mul_t->nb[1]/tsm), (int64_t) (mul_t->nb[2]/tsm), (int64_t) (mul_t->nb[3]/tsm), \
                mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, K_out, N_out); \
        } else if (mode == 1) { \
            ggml_cuda_kernel_launch(rms_norm_mul_conv_f32<BS, 1>, lp, x, dst_d, out, (int) ne00, (int) ne02, (int) ne03, \
                (int64_t) (x_t->nb[1]/ts0), (int64_t) (x_t->nb[2]/ts0), (int64_t) (x_t->nb[3]/ts0), eps, mul, \
                (int64_t) (mul_t->nb[1]/tsm), (int64_t) (mul_t->nb[2]/tsm), (int64_t) (mul_t->nb[3]/tsm), \
                mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, K_out, N_out); \
        } else { \
            ggml_cuda_kernel_launch(rms_norm_mul_conv_f32<BS, 2>, lp, x, dst_d, out, (int) ne00, (int) ne02, (int) ne03, \
                (int64_t) (x_t->nb[1]/ts0), (int64_t) (x_t->nb[2]/ts0), (int64_t) (x_t->nb[3]/ts0), eps, mul, \
                (int64_t) (mul_t->nb[1]/tsm), (int64_t) (mul_t->nb[2]/tsm), (int64_t) (mul_t->nb[3]/tsm), \
                mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, K_out, N_out); \
        } \
    } while (0)
    if (n4 <= 32) {
        K11_CONV_LAUNCH(32);
    } else if (n4 <= 128) {
        K11_CONV_LAUNCH(128);
    } else if (n4 <= 256) {
        K11_CONV_LAUNCH(256);
    } else if (n4 <= 512) {
        K11_CONV_LAUNCH(512);
    } else {
        K11_CONV_LAUNCH(1024);
    }
#undef K11_CONV_LAUNCH
    return true;
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, void * q8_1_out,
                                 const ggml_tensor * add_tensor, const ggml_cuda_hc_post_args * hcp) {
    GGML_ASSERT(hcp == nullptr || (q8_1_out != nullptr && add_tensor == nullptr));
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    // k15: add_tensor (the ADD producing rms_norm_src, fused before it): only with the q8_1 output (see ggml_cuda_try_fuse_add_rms_norm)
    GGML_ASSERT(add_tensor == nullptr || (add_tensor == rms_norm_src && q8_1_out != nullptr));
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    // k11: the next products' conversion of this output, written by the norm (see ggml_cuda_src1_memo)
    ggml_cuda_src1_memo & memo = ctx.src1_memo;
    if (q8_1_out == nullptr && add_tensor == nullptr && memo.producer == mul_tensor && memo.held && ggml_is_contiguous(mul_tensor) &&
            ggml_are_same_shape(rms_norm_src, mul_tensor) && ggml_is_contiguous(rms_norm_src)) {
        if (rms_norm_mul_conv_launch(ctx, rms_norm_src, mul_src, dst_d, memo.producer_mode, memo.ptr, memo.K_out, memo.N_out, eps)) {
            memo.produced = true;
            return;
        }
    }

    if (q8_1_out != nullptr) {
        GGML_ASSERT(ggml_is_contiguous(mul_tensor));
        const int ncols_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
        const dim3 blocks_num(ne01, MIN(ne02, UINT16_MAX), MIN(ne03, UINT16_MAX));
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        // same launch choice as rms_norm_f32_dispatch<true, false, false> (block size, vpt, float4 loads), so the reduction order
        // (and the f32 output) is unchanged
        const float * src0_d = (const float *) rms_norm_src->data;
        const float * mul_d  = (const float *) mul_src->data;
        static_assert(MATRIX_ROW_PADDING % WARP_SIZE == 0, "q8_1 pass must cover whole warps");
        const float * ra_d = add_tensor ? (const float *) add_tensor->src[0]->data : nullptr;
        const float * rb_d = add_tensor ? (const float *) add_tensor->src[1]->data : nullptr;
        bool vec = ne00 % 4 == 0 && ne00 > 0 && ((uintptr_t) dst_d & 15) == 0;
        vec = vec && ggml_cuda_is_aligned(rms_norm_src, 16) && mul_ncols == ne00 && ggml_cuda_is_aligned(mul_src, 16);
        const ggml_cuda_hc_post_args hc_args = hcp ? *hcp : ggml_cuda_hc_post_args{};
        // the vector loads of the hc post inputs: 16-byte aligned rows (the fusion checks the tensors, these the row strides)
        GGML_ASSERT(!hcp || !vec || (((uintptr_t) hc_args.bo & 15) == 0 && ((uintptr_t) hc_args.residual & 15) == 0 &&
            hc_args.sbo1 % 4 == 0 && hc_args.sr1 % 4 == 0 && hc_args.sr2 % 4 == 0));
        const int64_t n4 = ne00 / 4;
        static const bool rms_q8_1_batch = !getenv("GGML_CUDA_RMS_Q8_1_BATCH") || atoi(getenv("GGML_CUDA_RMS_Q8_1_BATCH")) != 0;
#define RMS_NORM_Q8_1_LAUNCH(BLOCK_SIZE, VPT, VEC) \
        do { \
            const dim3 block_dims(BLOCK_SIZE, 1, 1); \
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32 * sizeof(float), stream}; \
            const auto kernel = hcp ? (rms_q8_1_batch ? rms_norm_mul_q8_1_f32<BLOCK_SIZE, VPT, VEC, 4, true> : rms_norm_mul_q8_1_f32<BLOCK_SIZE, VPT, VEC, 1, true>) : \
                                      (rms_q8_1_batch ? rms_norm_mul_q8_1_f32<BLOCK_SIZE, VPT, VEC, 4> : rms_norm_mul_q8_1_f32<BLOCK_SIZE, VPT, VEC, 1>); \
            ggml_cuda_kernel_launch(kernel, launch_params, src0_d, dst_d, q8_1_out, (int) ne00, ncols_padded, \
                (int) ne02, (int) ne03, s01, s02, s03, eps, mul_d, mul_s01, mul_s02, mul_s03, \
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, ra_d, rb_d, hc_args); \
        } while (0)
        if (vec) {
            if (n4 <= 32) {
                RMS_NORM_Q8_1_LAUNCH(32,   1, true);
            } else if (n4 <= 128) {
                RMS_NORM_Q8_1_LAUNCH(128,  1, true);
            } else if (n4 <= 256) {
                RMS_NORM_Q8_1_LAUNCH(256,  1, true);
            } else if (n4 <= 512) {
                RMS_NORM_Q8_1_LAUNCH(512,  1, true);
            } else if (n4 <= 1024) {
                RMS_NORM_Q8_1_LAUNCH(1024, 1, true);
            } else if (n4 <= 4096) {
                RMS_NORM_Q8_1_LAUNCH(1024, 4, true);
            } else {
                RMS_NORM_Q8_1_LAUNCH(1024, 1, false);
            }
        } else if (ne00 < 1024) {
            RMS_NORM_Q8_1_LAUNCH(256,  1, false);
        } else {
            RMS_NORM_Q8_1_LAUNCH(1024, 1, false);
        }
#undef RMS_NORM_Q8_1_LAUNCH
        return;
    }

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

// k15-fn: RMS_NORM -> MUL -> ROPE (multi-section, not vision) in the rms_norm + mul kernel: the norm launch and the rope arithmetic
// are the unfused ones (rms_norm_f32_dispatch, rope_multi_rotate), so the output is bitwise the unfused one; the rope reads the
// mul output row back from global memory after a block sync. See ggml_cuda_should_fuse_rms_norm_mul_mrope for the conditions
void ggml_cuda_op_rms_norm_mul_mrope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope) {
    const ggml_tensor * x       = rms_norm->src[0];
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && mul_src->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && rope->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->src[0] == mul && ggml_is_contiguous(mul));

    const int64_t ne00 = x->ne[0];
    const int64_t ne01 = x->ne[1];
    const int64_t ne02 = x->ne[2];
    const int64_t ne03 = x->ne[3];

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);

    ggml_cuda_rms_rope_args a;
    const size_t ts_dst = ggml_type_size(rope->type);
    a.dst = (float *) rope->data;
    a.s1  = rope->nb[1] / ts_dst;
    a.s2  = rope->nb[2] / ts_dst;
    a.s3  = rope->nb[3] / ts_dst;

    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];
    a.n_dims = ((const int32_t *) rope->op_params)[1];
    a.n_offs = ((const int32_t *) rope->op_params)[15];
    const int mode = ((const int32_t *) rope->op_params)[2];
    float freq_base;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base,     (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&a.freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&a.ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&a.attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,     (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,     (const int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(&a.sections.v,  (const int32_t *) rope->op_params + 11, sizeof(int)*4);
    ggml_rope_yarn_corr_dims(a.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, a.corr_dims.v);
    // same expression as rope_multi_cuda
    a.theta_scale  = powf(freq_base, -2.0f / a.n_dims);
    a.pos          = (const int32_t *) rope->src[1]->data;
    a.freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;
    a.is_imrope    = mode == GGML_ROPE_TYPE_IMROPE;
    a.inplace      = rope->data == mul->data;

    const dim3 blocks_num(ne01, MIN(ne02, UINT16_MAX), MIN(ne03, UINT16_MAX));
    const uint3 zero_packed = make_uint3(0, 0, 0);
    rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false, /*do_rope*/true>(
        blocks_num, x, (float *) mul->data, ne00, ne02, ne03, x->nb[1] / ts0, x->nb[2] / ts0, x->nb[3] / ts0, eps,
        mul_src, mul_src->nb[1] / ts_mul, mul_src->nb[2] / ts_mul, mul_src->nb[3] / ts_mul,
        init_fastdiv_values(mul_src->ne[0]), init_fastdiv_values(mul_src->ne[1]),
        init_fastdiv_values(mul_src->ne[2]), init_fastdiv_values(mul_src->ne[3]),
        nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, ctx.stream(), 1.0f, a);
}

// k15-fn: a QSA layer's new pooled keys in one warp per pool (qwen4exp build_qsa_sel: GET_ROWS of the kpool member keys -> CONT + ADDs
// (their sum) -> SCALE (the mean) -> RMS_NORM -> MUL -> ROPE (multi-section) -> SET_ROWS into the pooled half of the rep row). Every
// step is the unfused kernel's float arithmetic in its order: the member keys converted as get_rows does, summed member 0 first, the
// scale kernel's scale * x + bias, the norm of rms_norm_f32<32, 1, true, true> (the launch rms_norm_f32_dispatch picks for these rows,
// checked by the caller), rope_multi_rotate, and set_rows' conversion. See ggml_cuda_qsa_pool_match
template <bool keys_f16, bool dst_f16, bool has_ff>
static __global__ void k_qsa_pool(const ggml_cuda_qsa_pool_args a, const ggml_cuda_rms_rope_args rope) {
    constexpr int block_size = 32;
    ggml_cuda_pdl_lc();
    const int p   = blockIdx.x;
    const int tid = threadIdx.x;
    const int ncols  = a.ncols;
    const int ncols4 = ncols / 4;

    __shared__ float s_sum[32];
    __shared__ float s_row[GGML_CUDA_QSA_POOL_MAX_COLS];
    __shared__ float s_out[GGML_CUDA_QSA_POOL_MAX_COLS];

    ggml_cuda_pdl_sync();

    float4 xs[1];
    float  tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < 1; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            float v[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int d = 4*i + c;
                float acc = 0.0f;
                for (int m = 0; m < a.kpool; ++m) {
                    const int32_t row = a.ids[(int64_t) (p*a.kpool + m)*a.ids_s0];
                    const char * kr = (const char *) a.keys + (int64_t) row*a.keys_nb1;
                    float r;
                    if constexpr (keys_f16) {
                        r = ggml_cuda_cast<float>(((const half *) kr)[d]);
                    } else {
                        r = ((const float *) kr)[d];
                    }
                    acc = m == 0 ? r : acc + r;
                }
                v[c] = a.scale * acc + a.bias;
            }
            xs[k] = make_float4(v[0], v[1], v[2], v[3]);
            tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + a.eps);

#pragma unroll
    for (int k = 0; k < 1; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
            const float4 m = ((const float4 *) a.w)[i];
            r.x *= m.x;
            r.y *= m.y;
            r.z *= m.z;
            r.w *= m.w;
            s_row[4*i + 0] = r.x;
            s_row[4*i + 1] = r.y;
            s_row[4*i + 2] = r.z;
            s_row[4*i + 3] = r.w;
        }
    }
    __syncthreads();

    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
        if (i0 < rope.n_offs || i0 >= rope.n_offs + rope.n_dims) {
            s_out[i0 + 0] = s_row[i0 + 0];
            s_out[i0 + 1] = s_row[i0 + 1];
            continue;
        }
        rope_multi_rotate<true, has_ff>((const float *) s_row, s_out, i0, (uint32_t) p, a.n_new, rope.n_dims, rope.n_offs, rope.pos,
            rope.freq_scale, rope.ext_factor, rope.attn_factor, rope.corr_dims, rope.theta_scale, rope.freq_factors, rope.sections,
            rope.is_imrope);
    }
    __syncthreads();

    const int64_t dst_row = a.rep_i64 ? ((const int64_t *) a.rep)[(int64_t) p*a.rep_s0] : (int64_t) ((const int32_t *) a.rep)[(int64_t) p*a.rep_s0];
    char * dr = (char *) a.dst + dst_row*a.dst_nb1;
    for (int d = tid; d < ncols; d += block_size) {
        if constexpr (dst_f16) {
            ((half *) dr)[d] = ggml_cuda_cast<half>(s_out[d]);
        } else {
            ((float *) dr)[d] = s_out[d];
        }
    }
}

void ggml_cuda_op_qsa_pool_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, const ggml_tensor * scale_node,
                                 const ggml_tensor * rms_norm, const ggml_tensor * mul, const ggml_tensor * rope, const ggml_tensor * set_rows,
                                 const int kpool) {
    const ggml_tensor * keys = get_rows->src[0];
    const ggml_tensor * w    = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    ggml_cuda_qsa_pool_args a;
    a.keys     = keys->data;
    a.keys_nb1 = keys->nb[1];
    a.ids      = (const int32_t *) get_rows->src[1]->data;
    a.ids_s0   = get_rows->src[1]->nb[0] / sizeof(int32_t);
    a.kpool    = kpool;
    a.ncols    = (int) get_rows->ne[0];
    a.n_new    = (int) (get_rows->ne[1] / kpool);
    memcpy(&a.scale, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&a.bias,  (const float *) scale_node->op_params + 1, sizeof(float));
    memcpy(&a.eps, rms_norm->op_params, sizeof(float));
    a.w        = (const float *) w->data;
    a.dst      = set_rows->data;
    a.dst_nb1  = set_rows->nb[1];
    a.rep      = set_rows->src[1]->data;
    a.rep_i64  = set_rows->src[1]->type == GGML_TYPE_I64;
    a.rep_s0   = set_rows->src[1]->nb[0] / ggml_type_size(set_rows->src[1]->type);

    // the rope parameters as ggml_cuda_op_rope reads them
    ggml_cuda_rms_rope_args r;
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];
    r.n_dims = ((const int32_t *) rope->op_params)[1];
    r.n_offs = ((const int32_t *) rope->op_params)[15];
    const int mode = ((const int32_t *) rope->op_params)[2];
    float freq_base;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base,     (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&r.freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&r.ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&r.attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,     (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,     (const int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(&r.sections.v,  (const int32_t *) rope->op_params + 11, sizeof(int)*4);
    ggml_rope_yarn_corr_dims(r.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, r.corr_dims.v);
    r.theta_scale  = powf(freq_base, -2.0f / r.n_dims);
    r.pos          = (const int32_t *) rope->src[1]->data;
    r.freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;
    r.is_imrope    = mode == GGML_ROPE_TYPE_IMROPE;

    GGML_ASSERT(a.ncols % 4 == 0 && a.ncols / 4 <= 32 && a.ncols <= GGML_CUDA_QSA_POOL_MAX_COLS);
    const bool keys_f16 = keys->type == GGML_TYPE_F16;
    const bool dst_f16  = set_rows->type == GGML_TYPE_F16;
    const bool has_ff   = r.freq_factors != nullptr;

    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(a.n_new, 1, 1), dim3(32, 1, 1), 0, ctx.stream());
#define QSA_POOL_LAUNCH(KF, DF, FF) ggml_cuda_kernel_launch(k_qsa_pool<KF, DF, FF>, lp, a, r)
    if (keys_f16) {
        if (dst_f16) { if (has_ff) { QSA_POOL_LAUNCH(true, true, true); } else { QSA_POOL_LAUNCH(true, true, false); } }
        else         { if (has_ff) { QSA_POOL_LAUNCH(true, false, true); } else { QSA_POOL_LAUNCH(true, false, false); } }
    } else {
        if (dst_f16) { if (has_ff) { QSA_POOL_LAUNCH(false, true, true); } else { QSA_POOL_LAUNCH(false, true, false); } }
        else         { if (has_ff) { QSA_POOL_LAUNCH(false, false, true); } else { QSA_POOL_LAUNCH(false, false, false); } }
    }
#undef QSA_POOL_LAUNCH
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src,add_src,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
