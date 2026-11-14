// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "gated_delta_net.cuh"
#include "p1fuse-stats.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <atomic>
#include <cstdint>
#include <cstdlib>

constexpr int gdn_cols_per_warp = 4;

// k15-fn: the gates the gated_delta_net applies itself (ggml_cuda_try_gdn_gates), the formulas of unary.cu's op_sigmoid / op_softplus and
// of add_unary_mul_f32_kernel, so the values are those the skipped kernels wrote
static __device__ __forceinline__ float gdn_gate_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static __device__ __forceinline__ float gdn_gate_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// One token of the (non-KDA) recurrence for the state column a segment of lanes_per_col lanes owns. The AR kernel and the
// token-tiled kernel both call this, so the arithmetic of the two is the same source: the tiled kernel only changes where
// q / k / v / g / beta are read from, which keeps its results bit-identical to the AR kernel (batch invariance).
template <int rows_per_lane, int lanes_per_col>
static __device__ __forceinline__ float gdn_token_step(
        float (&s_shard)[rows_per_lane], const float (&k_reg)[rows_per_lane], const float (&q_reg)[rows_per_lane],
        const float g_log, const float beta_val, const float v_col) {
    const float g_val = expf(g_log);

    // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        kv_shard += s_shard[r] * k_reg[r];
    }
    float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

    // delta[col] = (v[col] - g * kv[col]) * beta
    float delta_col = (v_col - g_val * kv_col) * beta_val;

    // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
    // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }

    return warp_reduce_sum<lanes_per_col>(attn_partial);
}

// ggml_gated_delta_net_replay (k16-draft): the stored window of each sequence and where this window's inputs go; strides in floats
struct gdn_win_args {
    float *         wk;  // [S_k, H_k, 2 * n_win, n_seqs]
    float *         wv;  // [S_v, H_v, 2 * n_win, n_seqs]
    float *         wgb; // [2, H_v, 2 * n_win, n_seqs] (g, beta: the gates as applied, also where the kernel computes them itself)
    const int32_t * wp;  // [2, n_seqs]: kept tokens of the stored window, its half
    int64_t         sk1, sk2, sk3, sv1, sv2, sv3, sg1, sg2, sg3, sp1;
    int             n_win;
    int             H_k;
};

template <int S_v, bool KDA, bool keep_rs_t, bool WIN = false, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * s_ids,
                                     int64_t       s_row_stride,
                                     const float * beta_raw,
                                     const float * ga,
                                     const float * ga_bias,
                                     const float * ga_scale,
                                     const gdn_win_args wa) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    if (s_ids == nullptr) {
        curr_state += state_in_offset + col * S_v;
    }
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    if (s_ids != nullptr) {
        // k15-fn: the initial state read in place from its cache row (the gather into a contiguous copy was skipped)
        curr_state += (int64_t) s_ids[sequence] * s_row_stride + h_idx * S_v * S_v + col * S_v;
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    int w_par = 0;
    if constexpr (WIN) {
        static_assert(!KDA && !keep_rs_t, "the replay window takes the scalar gate and one state");
        // the commit of the stored window: the token step of the loop below without the attention scores (same expressions)
        const int32_t * wp = wa.wp + sequence * wa.sp1;
        const int keep = wp[0];
        w_par = wp[1];
        for (int t = 0; t < keep; t++) {
            const int64_t ts  = (int64_t) w_par * wa.n_win + t;
            const float * k_t = wa.wk  + sequence * wa.sk3 + ts * wa.sk2 + iq1   * wa.sk1;
            const float * v_t = wa.wv  + sequence * wa.sv3 + ts * wa.sv2 + h_idx * wa.sv1;
            const float * gb  = wa.wgb + sequence * wa.sg3 + ts * wa.sg2 + h_idx * wa.sg1;

            const float beta_val = gb[1];

            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                k_reg[r] = k_t[i];
            }

            const float g_val = expf(gb[0]);

            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
            }
        }
        // the committed state is the output state
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = beta_raw != nullptr ? gdn_gate_sigmoid(beta_raw[gb_offset]) : *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (WIN) {
            // this window's inputs into the other half of the stores (the key head once, by one column segment); the gates as applied
            const int64_t ts = (int64_t) (1 - w_par) * wa.n_win + t;
            if ((int) h_idx < wa.H_k && blockIdx.z == 0 && threadIdx.y == 0 && col_in_warp == 0) {
                float * ko = wa.wk + sequence * wa.sk3 + ts * wa.sk2 + h_idx * wa.sk1;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    ko[r * lanes_per_col + lane_in_col] = k_reg[r];
                }
            }
            if (lane_in_col == 0) {
                wa.wv[sequence * wa.sv3 + ts * wa.sv2 + h_idx * wa.sv1 + col] = v_t[col];
            }
            if (col == 0 && lane_in_col == 0) {
                float * gbo = wa.wgb + sequence * wa.sg3 + ts * wa.sg2 + h_idx * wa.sg1;
                gbo[0] = ga != nullptr ? gdn_gate_softplus(ga[gb_offset] + ga_bias[h_idx]) * ga_scale[h_idx] : *g_t;
                gbo[1] = beta_val;
            }
        }

        if constexpr (!KDA) {
            const float g_log = ga != nullptr ? gdn_gate_softplus(ga[gb_offset] + ga_bias[h_idx]) * ga_scale[h_idx] : *g_t;
            const float attn_col = gdn_token_step<rows_per_lane, lanes_per_col>(s_shard, k_reg, q_reg, g_log, beta_val, v_t[col]);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t && !WIN) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Token-tiled variant of gated_delta_net_cuda (non-KDA): the same lanes own the same state columns and run the same
// gdn_token_step, but q / k / v / g / beta of TOKEN_TILE tokens are first staged into shared memory by the whole block
// with coalesced loads, so the per-token critical path no longer waits on scattered global loads. Bit-identical to the
// AR kernel by construction (same arithmetic, same reduction order); only the load source differs.
template <int S_v, bool keep_rs_t, int num_warps, int TOKEN_TILE>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * num_warps, 1)
gated_delta_net_tiled_cuda(const float * q,
                           const float * k,
                           const float * v,
                           const float * g,
                           const float * beta,
                           const float * curr_state,
                           float *       dst,
                           float *       state,
                           int64_t       H,
                           int64_t       n_tokens,
                           int64_t       sq1,
                           int64_t       sq2,
                           int64_t       sq3,
                           int64_t       sv1,
                           int64_t       sv2,
                           int64_t       sv3,
                           int64_t       sb1,
                           int64_t       sb2,
                           int64_t       sb3,
                           const uint3   neqk1_magic,
                           const uint3   rq3_magic,
                           float         scale,
                           int64_t       state_slot_stride,
                           int           K) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int cols_per_warp = gdn_cols_per_warp;
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    constexpr int block_cols    = cols_per_warp * num_warps;
    constexpr int nthreads      = warp_size * num_warps;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");
    static_assert(S_v % block_cols == 0, "the block columns must divide S_v");

    __shared__ float q_sh[TOKEN_TILE][S_v];
    __shared__ float k_sh[TOKEN_TILE][S_v];
    __shared__ float v_sh[TOKEN_TILE][block_cols];
    __shared__ float g_sh[TOKEN_TILE];
    __shared__ float b_sh[TOKEN_TILE];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    const int lane        = threadIdx.x;
    const int tid         = threadIdx.y * warp_size + lane;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col0        = blockIdx.z * block_cols;
    const int col_blk     = threadIdx.y * cols_per_warp + col_in_warp; // column within the block
    const int col         = col0 + col_blk;                            // = the AR kernel's column of this lane

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    const float * q_s = q + iq3 * sq3 + iq1 * sq1;
    const float * k_s = k + iq3 * sq3 + iq1 * sq1;
    const float * v_s = v + sequence * sv3 + h_idx * sv1 + col0;
    const int64_t gb0 = sequence * sb3 + h_idx * sb1;

    float s_shard[rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int64_t t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile = n_tokens - t0 < TOKEN_TILE ? (int) (n_tokens - t0) : TOKEN_TILE;

        for (int idx = tid; idx < tile * S_v; idx += nthreads) {
            const int     tt = idx / S_v;
            const int     i  = idx - tt * S_v;
            const int64_t t  = t0 + tt;
            q_sh[tt][i] = q_s[t * sq2 + i];
            k_sh[tt][i] = k_s[t * sq2 + i];
        }
        for (int idx = tid; idx < tile * block_cols; idx += nthreads) {
            const int     tt = idx / block_cols;
            const int     c  = idx - tt * block_cols;
            v_sh[tt][c] = v_s[(t0 + tt) * sv2 + c];
        }
        if (tid < tile) {
            const int64_t gb_offset = gb0 + (t0 + tid) * sb2;
            g_sh[tid] = g[gb_offset];
            b_sh[tid] = beta[gb_offset];
        }
        __syncthreads();

        for (int tt = 0; tt < tile; ++tt) {
            const int64_t t = t0 + tt;

            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                k_reg[r] = k_sh[tt][i];
                q_reg[r] = q_sh[tt][i];
            }

            const float attn_col = gdn_token_step<rows_per_lane, lanes_per_col>(s_shard, k_reg, q_reg, g_sh[tt], b_sh[tt], v_sh[tt][col_blk]);

            if (lane_in_col == 0) {
                attn_data[t * (S_v * H) + col] = attn_col * scale;
            }

            if constexpr (keep_rs_t) {
                // the AR kernel's snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back
                const int target_slot = (int) (n_tokens - 1 - t);
                if (target_slot >= 0 && target_slot < K) {
                    float * snap = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * lanes_per_col + lane_in_col;
                        snap[col * S_v + i] = s_shard[r];
                    }
                }
            }
        }
        __syncthreads(); // the next tile overwrites the staged inputs
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// GGML_CUDA_GDN_TILED: 0 = never, 1 = on every arch (tests / timing), unset = the per-arch default (RDNA3.5 at 1024+ tokens and 16+ heads)
static int ggml_cuda_gdn_tiled_env() {
    static const int value = []() {
        const char * env = getenv("GGML_CUDA_GDN_TILED");
        return env ? atoi(env) : -1;
    }();
    return value;
}

static constexpr int gdn_tiled_min_tokens = 16;
static constexpr int gdn_tiled_token_tile = 16;

// RDNA3.5 default (gfx1151 test-backend-ops, Linux + Windows): the tiled kernel wins from about 1024 tokens with 16+
// value heads per launch, loses ~10% at 64 tokens, mixed at 256 - 512, and loses 35-40% with 4 heads (too few blocks)
static constexpr int gdn_tiled_rdna3_5_min_tokens = 1024;
static constexpr int gdn_tiled_rdna3_5_min_heads  = 16;

// whether the token-tiled kernel runs a non-KDA op of this shape (the caller already let the chunked MMA kernel decline);
// n_heads = value heads * sequences, the blocks per column slice
static bool gdn_use_tiled(int device, int64_t S_v, int64_t n_heads, int64_t n_tokens) {
    if (n_tokens < gdn_tiled_min_tokens || (S_v != 64 && S_v != 128)) {
        return false;
    }
    const auto & info = ggml_cuda_info().devices[device];
    if (info.warp_size != 32) {
        return false; // only the wave32 instantiation is tested
    }
    const int env = ggml_cuda_gdn_tiled_env();
    if (env >= 0) {
        return env != 0;
    }
    return GGML_CUDA_CC_IS_RDNA3_5(info.cc) && n_tokens >= gdn_tiled_rdna3_5_min_tokens &&
        n_heads >= gdn_tiled_rdna3_5_min_heads;
}

template <int S_v, bool keep_rs_t, int num_warps>
static void launch_gated_delta_net_tiled_nw(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    constexpr int block_cols = gdn_cols_per_warp * num_warps;
    const dim3 grid_dims(H, n_seqs, S_v / block_cols);
    const dim3 block_dims(32, num_warps, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_tiled_cuda<S_v, keep_rs_t, num_warps, gdn_tiled_token_tile>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
}

template <int S_v, bool keep_rs_t>
static void launch_gated_delta_net_tiled(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    // wide blocks amortise the staging of q / k (every block of a head stages them); halve while the grid would leave
    // SMs / CUs without a block
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    int num_warps = S_v / gdn_cols_per_warp < 16 ? S_v / gdn_cols_per_warp : 16;
    while (num_warps > 4 && H*n_seqs*(S_v / (gdn_cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
#define GDN_TILED_LAUNCH(nw) launch_gated_delta_net_tiled_nw<S_v, keep_rs_t, nw>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, stream)
    switch (num_warps) {
        case 16: GDN_TILED_LAUNCH(16); break;
        case  8: GDN_TILED_LAUNCH( 8); break;
        default: GDN_TILED_LAUNCH( 4); break;
    }
#undef GDN_TILED_LAUNCH
}

template <bool KDA, bool keep_rs_t, bool WIN = false>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * s_ids, int64_t s_row_stride,
        const float * beta_raw, const float * ga, const float * ga_bias, const float * ga_scale, cudaStream_t stream,
        const gdn_win_args wa = {}) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    if constexpr (!KDA) {
        // the tiled kernel reads neither an in-place state (s_ids, k15-fn 24dbe0fcacd) nor raw gates (k15-fn 153a4696ebb): those run the
        // AR kernel, as they skip the chunked MMA kernel
        if (!WIN && s_ids == nullptr && beta_raw == nullptr && ga == nullptr && gdn_use_tiled(ggml_cuda_get_device(), S_v, H*n_seqs, n_tokens)) {
            // GGML_CUDA_GDN_TILED_LOG=1: every launch (test-gdn-tiled-exact counts them), else the first one
            static const bool log_all = getenv("GGML_CUDA_GDN_TILED_LOG") != nullptr && atoi(getenv("GGML_CUDA_GDN_TILED_LOG")) != 0;
            static std::atomic<bool> logged {false};
            if (log_all || !logged.exchange(true)) {
                GGML_LOG_DEBUG("%s: token-tiled gated delta-net kernel (S_v %d, %d tokens)\n", __func__, (int) S_v, (int) n_tokens);
            }
            if (S_v == 128) {
                launch_gated_delta_net_tiled<128, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                    sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, stream);
            } else {
                launch_gated_delta_net_tiled<64, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                    sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, stream);
            }
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, WIN>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, wa);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, WIN>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, wa);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, WIN>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, wa);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, WIN>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, wa);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst);

// k15-fn: whether ggml_cuda_op_gated_delta_net may take the mma kernel for dst (the predicate of ggml_cuda_gdn_mma_args_from_tensor).
// k19-p1fuse (GGML_CUDA_GDN_FUSE_PRED=0: the shape-only test as before): the dispatch itself, so that the fusions which make the op
// take the AR kernel (the in-place state read, the gates) are not refused where the AR kernel runs anyway: a replay window (the per-token
// kernel, before the mma launch), fewer than 64 tokens, a GPU or head layout the mma kernel does not take. Conservative: a launch whose
// kernel set-up would fail still counts as mma (fusion refused, as before)
bool ggml_cuda_gated_delta_net_may_use_mma(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    if (!(g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && v->ne[0] == 128 && q->ne[0] == 128)) {
        return false;
    }
    static const bool exact_pred = [] {
        const char * e = getenv("GGML_CUDA_GDN_FUSE_PRED");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!exact_pred) {
        return true;
    }
    if (ggml_get_op_params_i32(dst, 1) == 1) {
        return false; // replay window: always the per-token kernel
    }
    return ggml_cuda_gdn_mma_candidate(ggml_cuda_get_device(), ggml_cuda_gdn_mma_args_from_tensor(dst));
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data; // replaced by the cache rows below when read in place
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // k15-fn: the initial state read from the cache rows (ggml_cuda_try_gdn_state_in_place skipped the gather), only the simple kernel
    const int32_t * s_ids        = nullptr;
    int64_t         s_row_stride = 0;
    {
        auto it = ctx.gdn_state_in_place.find(dst);
        if (it != ctx.gdn_state_in_place.end()) {
            s_d          = it->second.base;
            s_row_stride = it->second.row_stride;
            s_ids        = it->second.ids;
            ctx.gdn_state_in_place.erase(it);
        }
    }

    const float * beta_raw = nullptr, * ga = nullptr, * ga_bias = nullptr, * ga_scale = nullptr;
    {
        auto it = ctx.gdn_gates_in.find(dst);
        if (it != ctx.gdn_gates_in.end()) {
            beta_raw = it->second.beta_raw;
            ga       = it->second.a;
            ga_bias  = it->second.bias;
            ga_scale = it->second.scale;
            ctx.gdn_gates_in.erase(it);
        }
    }

    if (s_ids != nullptr || beta_raw != nullptr || ga != nullptr) {
        ggml_cuda_p1_stat(GGML_CUDA_P1_GDN_FUSED);
        if (src_g->ne[0] == 1 && K < n_tokens && S_v == 128 && neq0 == 128) {
            ggml_cuda_p1_stat(GGML_CUDA_P1_GDN_FUSED_NEW); // refused by the shape-only test (k19-p1fuse)
        }
    }

    if (ggml_get_op_params_i32(dst, 1) == 1) {
        // ggml_gated_delta_net_replay: the per-token kernel, which the commit of the stored window shares with the window
        const ggml_tensor * wk = dst->src[6];
        const ggml_tensor * wv = dst->src[7];
        const ggml_tensor * wg = dst->src[8];
        const ggml_tensor * wp = dst->src[9];
        GGML_ASSERT(!kda && K == 1);
        gdn_win_args wa;
        wa.wk    = (float *) wk->data;
        wa.wv    = (float *) wv->data;
        wa.wgb   = (float *) wg->data;
        wa.wp    = (const int32_t *) wp->data;
        wa.sk1   = wk->nb[1] / sizeof(float);
        wa.sk2   = wk->nb[2] / sizeof(float);
        wa.sk3   = wk->nb[3] / sizeof(float);
        wa.sv1   = wv->nb[1] / sizeof(float);
        wa.sv2   = wv->nb[2] / sizeof(float);
        wa.sv3   = wv->nb[3] / sizeof(float);
        wa.sg1   = wg->nb[1] / sizeof(float);
        wa.sg2   = wg->nb[2] / sizeof(float);
        wa.sg3   = wg->nb[3] / sizeof(float);
        wa.sp1   = wp->nb[1] / sizeof(int32_t);
        wa.n_win = (int) (wk->ne[2] / 2);
        wa.H_k   = (int) wk->ne[1];
        launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, stream, wa);
        return;
    }

    if (s_ids == nullptr && beta_raw == nullptr && ga == nullptr) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, beta_raw, ga, ga_bias, ga_scale, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
