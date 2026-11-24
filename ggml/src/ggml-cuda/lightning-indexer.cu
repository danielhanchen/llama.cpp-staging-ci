// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"
#include "preload.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#include <climits>

// candidate blocks (DeepSeek-V4.1 candidate scoring, dst->src[4]): the key slots of a block are mapped to the rows of the token's
// candidate blocks (slot -> block id -> row), so only those rows are scored; every other row of dst is -inf (filled beforehand)
struct li_cand {
    const char * C;       // I32 [n_cand, n_batch, 1, n_stream]
    int          n_slots; // n_cand * bs
    int          bs;      // rows per block
    size_t       nbc1;
    size_t       nbc3;
    // the live key count on the device (ggml_lightning_indexer_set_live_len, nullptr: every key): the keys at and past it are padding
    // whose mask is -inf in every row, so a block of keys all past it writes -inf (the padded score) without reading K / Q / W
    const int *  live;
};

// true when the keys [start, start + n) are all past the live count: the block writes the padded score -inf instead
static __device__ __forceinline__ bool li_past_live(const li_cand & cand, const int start) {
    return cand.live != nullptr && start >= *cand.live;
}

// key row of a slot, -1 when it has none
template <bool CAND>
static __device__ __forceinline__ int li_key_row(const li_cand & cand, const int32_t * c_row, const int slot, const int64_t n_kv) {
    if constexpr (!CAND) {
        GGML_UNUSED_VARS(cand, c_row);
        return slot < n_kv ? slot : -1;
    } else {
        if (slot >= cand.n_slots) {
            return -1;
        }
        const int64_t row = (int64_t) c_row[slot / cand.bs]*cand.bs + slot % cand.bs;
        return row >= 0 && row < n_kv ? (int) row : -1;
    }
}

template <bool CAND>
static __device__ __forceinline__ const int32_t * li_cand_row(const li_cand & cand, const int i_batch, const int i_stream) {
    if constexpr (!CAND) {
        GGML_UNUSED_VARS(cand, i_batch, i_stream);
        return nullptr;
    } else {
        return (const int32_t *) (cand.C + i_batch*cand.nbc1 + i_stream*cand.nbc3);
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K, bool CAND>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3, const li_cand cand
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors (key slots: rows, or candidate rows)
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;
    const int32_t * c_row = li_cand_row<CAND>(cand, i_batch, i_stream);

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = li_key_row<CAND>(cand, c_row, start_kv + i_k_vec, n_kv);
            if (i_kv >= 0) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = li_key_row<CAND>(cand, c_row, start_kv + i_k_vec, n_kv);
            if (i_kv >= 0) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = li_key_row<CAND>(cand, c_row, start_kv + tid, n_kv);
        if (i_kv >= 0) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else // defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K, bool CAND>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3, const li_cand cand
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3, cand);
    NO_DEVICE_CODE;
}

#endif // defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// tokens scored per block by the tile kernel
#define LIGHTNING_INDEXER_TILE_TOKENS 8

// heads whose queries the tile kernel stages per pass, MUSA arch 21 caps static shared memory
// at 28 KB and the queries of four heads do not fit there next to the key tile
#if defined(GGML_USE_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ < 220
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 2
#else
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 4
#endif

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K, bool CAND>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3, const li_cand cand
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;
    const int32_t * c_row = li_cand_row<CAND>(cand, i_batch, i_stream);

    if (!CAND && li_past_live(cand, start_kv_block)) {
        if (tid < K_VECS_PER_BLOCK && start_kv_block + tid < n_kv) {
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[start_kv_block + tid] = -INFINITY;
        }
        return;
    }

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            const int i_kv = li_key_row<CAND>(cand, c_row, start_kv + k, n_kv);
            if (i_kv >= 0) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            const int i_kv = li_key_row<CAND>(cand, c_row, start_kv + k, n_kv);
            if (i_kv >= 0) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = li_key_row<CAND>(cand, c_row, start_kv_block + tid, n_kv);
        if (i_kv >= 0) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// lightning_indexer_kernel_vec for up to ROWS tokens per block (n_rows <= ROWS, a 2..7-row verify batch or one token): the block
// loads its keys into registers once and stages the queries and weights of every row and every head (N_HEAD <= 8) behind one barrier;
// each (token, key) score runs lightning_indexer_kernel_vec's instruction sequence (the same per-head float dot, warp reduction,
// ReLU and weighted sum in head order, then the mask), so the scores are byte-identical to it. GGML_CUDA_LI_VEC_MULTI=0: off
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K, int ROWS>
static __global__ void lightning_indexer_kernel_vec_rows(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3, const li_cand cand
    ) {
    GGML_UNUSED_VARS(nb2, nbk1, nbw2, nbm2);
    static_assert(N_HEAD <= 8, "every head's query is staged at once");

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int n_rows   = (int) n_batch;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    if (li_past_live(cand, start_kv_block)) {
        for (int i = tid; i < ROWS*K_VECS_PER_BLOCK; i += THREADS_PER_BLOCK) {
            const int r = i / K_VECS_PER_BLOCK;
            const int j = i % K_VECS_PER_BLOCK;
            if (r < n_rows && start_kv_block + j < n_kv) {
                ((float *) ((char *) dst + r*nb1 + i_stream*nb3))[start_kv_block + j] = -INFINITY;
            }
        }
        return;
    }

    // phase 1 - load (and dequantize if needed) K to registers, as lightning_indexer_kernel_vec

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            const int i_kv = li_key_row<false>(cand, nullptr, start_kv + k, n_kv);
            if (i_kv >= 0) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            const int i_kv = li_key_row<false>(cand, nullptr, start_kv + k, n_kv);
            if (i_kv >= 0) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    // phase 2 - the weights and queries of every row and head to shared memory, one barrier

    __shared__ float  w_shared[ROWS][N_HEAD];
    __shared__ float4 q_shared_f[ROWS][N_HEAD][N_EMBD / 4];

    if (tid < ROWS*N_HEAD) {
        const int r = tid / N_HEAD;
        const int h = tid % N_HEAD;
        w_shared[r][h] = r < n_rows ? ((const float *) ((const char *) W + r*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    constexpr int n_q = ROWS * N_HEAD * (N_EMBD / 4);
#pragma unroll
    for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
        const int r      = i_q / (N_HEAD * (N_EMBD / 4));
        const int i_head = i_q / (N_EMBD / 4) % N_HEAD;
        const int i_embd = i_q % (N_EMBD / 4);
        q_shared_f[r][i_head][i_embd] = r < n_rows ?
            *(const float4 *) ((const char *) Q + r*nbq2 + i_stream*nbq3 + i_head*nbq1 + i_embd*sizeof(float4)) : make_float4(0, 0, 0, 0);
    }

    __syncthreads();

    // phase 3 - per row, lightning_indexer_kernel_vec's scores

    __shared__ float dst_shared[ROWS][K_VECS_PER_BLOCK];

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        if (r >= n_rows) {
            break;
        }
        float score_k[K_VECS_PER_WARP] = { 0.0f };

        for (int i_head = 0; i_head < N_HEAD; ++i_head) {
            const float w_val = w_shared[r][i_head];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            const float4 q_vec = q_shared_f[r][i_head][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        if (i_lane == 0) {
#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                dst_shared[r][i_warp * K_VECS_PER_WARP + k] = score_k[k];
            }
        }
    }

    __syncthreads();

    // phase 4 - write from shared memory to VRAM in a coalesced manner

    for (int i = tid; i < ROWS*K_VECS_PER_BLOCK; i += THREADS_PER_BLOCK) {
        const int r = i / K_VECS_PER_BLOCK;
        const int j = i % K_VECS_PER_BLOCK;
        const int i_kv = li_key_row<false>(cand, nullptr, start_kv_block + j, n_kv);
        if (r < n_rows && i_kv >= 0) {
            const half * m_base = (const half *) ((const char *) M + r*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + r*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[r][j] + __half2float(m_base[i_kv]);
        }
    }
}

// one block scores a tile of K_VECS_PER_BLOCK keys against TOKENS_PER_BLOCK tokens: the keys are
// staged in half precision and the queries of every head in float, each thread owns KEYS_PER_THREAD
// keys for one token, a warp shares its token so the query reads are broadcasts, and every key
// element is widened once for all heads, so no dot product needs a cross thread reduction
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K, bool CAND>
static __global__ void lightning_indexer_kernel_tile(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3, const li_cand cand
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TOKENS_PER_BLOCK  = LIGHTNING_INDEXER_TILE_TOKENS;
    constexpr int KEY_LANES         = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;
    constexpr int KEYS_PER_THREAD   = K_VECS_PER_BLOCK / KEY_LANES;
    constexpr int N_EMBD_H2         = N_EMBD / 2;
    constexpr int HEADS_PER_PASS    = N_HEAD < LIGHTNING_INDEXER_TILE_HEADS_PER_PASS ? N_HEAD : LIGHTNING_INDEXER_TILE_HEADS_PER_PASS;

    static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");
    static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");
    static_assert(N_HEAD % HEADS_PER_PASS == 0, "head passes must cover the heads");
    static_assert(!CAND, "the tile kernel shares keys across tokens: candidates are applied after it");

    const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_batch = blockIdx.y * TOKENS_PER_BLOCK;
    const int i_stream    = blockIdx.z;

    if (li_past_live(cand, start_kv)) {
        for (int i = tid; i < TOKENS_PER_BLOCK*K_VECS_PER_BLOCK; i += THREADS_PER_BLOCK) {
            const int r = i / K_VECS_PER_BLOCK;
            const int j = i % K_VECS_PER_BLOCK;
            if (start_batch + r < n_batch && start_kv + j < n_kv) {
                ((float *) ((char *) dst + (start_batch + r)*nb1 + i_stream*nb3))[start_kv + j] = -INFINITY;
            }
        }
        return;
    }

    // the row padding keeps the keys of consecutive threads in distinct banks
    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];
    __shared__ float2 q_shared[HEADS_PER_PASS][TOKENS_PER_BLOCK][N_EMBD_H2];
    __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];

    // phase 1 - stage the key tile four elements at a time, rows past n_kv are zero

#pragma unroll
    for (int i = tid; i < K_VECS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
        const int r  = i / (N_EMBD / 4);
        const int c4 = i % (N_EMBD / 4);

        half2 lo = make_half2(0.0f, 0.0f);
        half2 hi = lo;
        if (start_kv + r < n_kv) {
            const char * k_row = K + (start_kv + r)*nbk2 + i_stream*nbk3;
            if constexpr (TYPE_K == GGML_TYPE_F16) {
                lo = ((const half2 *) k_row)[2*c4 + 0];
                hi = ((const half2 *) k_row)[2*c4 + 1];
            } else {
                float4 v;
                if constexpr (TYPE_K == GGML_TYPE_F32) {
                    v = ((const float4 *) k_row)[c4];
                } else {
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                    dequantize_k(k_row, &v, c4 * 4);
                }
                lo = make_half2(v.x, v.y);
                hi = make_half2(v.z, v.w);
            }
        }

        k_shared[r][2*c4 + 0] = lo;
        k_shared[r][2*c4 + 1] = hi;
    }

    // phase 2 - stage the weights of every head, tokens past n_batch are zero

    if (tid < N_HEAD * TOKENS_PER_BLOCK) {
        const int h = tid / TOKENS_PER_BLOCK;
        const int r = tid % TOKENS_PER_BLOCK;
        w_shared[h][r] = start_batch + r < n_batch ?
            ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    const int kl = tid % KEY_LANES;
    const int tl = tid / KEY_LANES;

    float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };

#pragma unroll
    for (int h0 = 0; h0 < N_HEAD; h0 += HEADS_PER_PASS) {
        // the previous pass is fully consumed before its queries are replaced
        if (h0 > 0) {
            __syncthreads();
        }

        // phase 3 - stage the queries of the heads of this pass, tokens past n_batch are zero

#pragma unroll
        for (int i = tid; i < HEADS_PER_PASS * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
            const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
            const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
            const int c4 = i % (N_EMBD / 4);

            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (start_batch + r < n_batch) {
                v = *(const float4 *) ((const char *) Q + (h0 + h)*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
            }

            q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
            q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
        }

        __syncthreads();

        // phase 4 - float products of the widened keys for the heads of this pass

#pragma unroll 8
        for (int c = 0; c < N_EMBD_H2; ++c) {
            float2 k_val[KEYS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
            }
#pragma unroll
            for (int h = 0; h < HEADS_PER_PASS; ++h) {
                const float2 q_val = q_shared[h][tl][c];
#pragma unroll
                for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                    qk[h0 + h][j] = fmaf(k_val[j].x, q_val.x, qk[h0 + h][j]);
                    qk[h0 + h][j] = fmaf(k_val[j].y, q_val.y, qk[h0 + h][j]);
                }
            }
        }
    }

    // phase 5 - ReLU, weight, add the mask and write, consecutive threads write consecutive keys

    float score[KEYS_PER_THREAD] = { 0.0f };

#pragma unroll
    for (int h = 0; h < N_HEAD; ++h) {
#pragma unroll
        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
            score[j] += fmaxf(qk[h][j], 0.0f) * w_shared[h][tl];
        }
    }

    const int i_batch = start_batch + tl;
    if (i_batch >= n_batch) {
        return;
    }

    const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);

#pragma unroll
    for (int j = 0; j < KEYS_PER_THREAD; ++j) {
        const int i_kv = start_kv + kl + j*KEY_LANES;
        if (i_kv < n_kv) {
            dst_base[i_kv] = score[j] + __half2float(m_base[i_kv]);
        }
    }
}

#define LIGHTNING_INDEXER_LAUNCH(lightning_indexer_kernel, n_embd, n_head, type_K, cand_t, grid_l) \
    if (ggml_cuda_preload_on()) {                                                                 \
        ggml_cuda_preload_func((const void *)                                                     \
            lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K, cand_t>); \
    } else                                                                                        \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K, cand_t>  \
            <<<grid_l, block, 0, ctx.stream()>>>(                                                 \
            q_d, k_d, w_d, m_d, dst_d,                                                            \
            n_stream, n_batch, n_kv,                                                              \
            nb1, nb2, nb3,                                                                        \
            nbq1, nbq2, nbq3,                                                                     \
            nbk1, nbk2, nbk3,                                                                     \
            nbw1, nbw2, nbw3,                                                                     \
            nbm1, nbm2, nbm3,                                                                     \
            nem3, cand                                                                            \
        );

// per token kernels: with candidates the grid covers only the candidate slots
#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)                                       \
    if (K->type == (type_K)) {                                                                                            \
        if (cand.C) {                                                                                                     \
            const dim3 grid_c((cand.n_slots + K_VECS_PER_BLOCK - 1) / K_VECS_PER_BLOCK, n_batch, n_stream);               \
            LIGHTNING_INDEXER_LAUNCH(lightning_indexer_kernel, n_embd, n_head, type_K, true, grid_c)                     \
        } else {                                                                                                          \
            LIGHTNING_INDEXER_LAUNCH(lightning_indexer_kernel, n_embd, n_head, type_K, false, grid)                      \
        }                                                                                                                 \
    } else

// the tile kernel always scores every row, candidates are applied after it
#define LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel, n_embd, n_head, K, type_K)                                   \
    if (K->type == (type_K)) {                                                                                            \
        LIGHTNING_INDEXER_LAUNCH(lightning_indexer_kernel, n_embd, n_head, type_K, false, grid)                          \
    } else

// the multi-row vec kernel (n_head 4) for ROWS and every K type
#define LIGHTNING_INDEXER_ROWS_LAUNCH(type_K, rows)                                                                     \
    if (k->type == (type_K)) {                                                                                          \
        if (ggml_cuda_preload_on()) {                                                                                   \
            ggml_cuda_preload_func((const void *)                                                                       \
                lightning_indexer_kernel_vec_rows<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 4, type_K, rows>);            \
        } else                                                                                                          \
            lightning_indexer_kernel_vec_rows<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 4, type_K, rows>                  \
                <<<grid_r, block, 0, ctx.stream()>>>(                                                                   \
                q_d, k_d, w_d, m_d, dst_d,                                                                              \
                n_stream, n_batch, n_kv,                                                                                \
                nb1, nb2, nb3,                                                                                          \
                nbq1, nbq2, nbq3,                                                                                       \
                nbk1, nbk2, nbk3,                                                                                       \
                nbw1, nbw2, nbw3,                                                                                       \
                nbm1, nbm2, nbm3,                                                                                       \
                nem3, cand                                                                                              \
            );                                                                                                          \
    } else

#define LIGHTNING_INDEXER_ROWS_TYPES(rows)                \
    LIGHTNING_INDEXER_ROWS_LAUNCH(GGML_TYPE_F16,  rows)   \
    LIGHTNING_INDEXER_ROWS_LAUNCH(GGML_TYPE_Q4_0, rows)   \
    LIGHTNING_INDEXER_ROWS_LAUNCH(GGML_TYPE_Q4_1, rows)   \
    LIGHTNING_INDEXER_ROWS_LAUNCH(GGML_TYPE_Q5_0, rows)   \
    LIGHTNING_INDEXER_ROWS_LAUNCH(GGML_TYPE_Q5_1, rows)   \
    LIGHTNING_INDEXER_ROWS_LAUNCH(GGML_TYPE_Q8_0, rows)   \
    LIGHTNING_INDEXER_ROWS_LAUNCH(GGML_TYPE_BF16, rows)   \
    LIGHTNING_INDEXER_ROWS_LAUNCH(GGML_TYPE_F32,  rows)   \
    GGML_ABORT("fatal error");

// every row of dst -> -inf (the candidate kernels write only the candidate rows)
static __global__ void lightning_indexer_fill_ninf(float * dst, const int64_t n_kv, const size_t nb1, const size_t nb3) {
    float * row = (float *) ((char *) dst + blockIdx.y*nb1 + blockIdx.z*nb3);
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n_kv; i += (int64_t) gridDim.x*blockDim.x) {
        row[i] = -INFINITY;
    }
}

// flags[token][block] = 1 for the token's candidate blocks
static __global__ void lightning_indexer_cand_flags(const li_cand cand, uint8_t * flags, const int64_t n_blk, const int n_batch) {
    const int i_batch  = blockIdx.x;
    const int i_stream = blockIdx.y;
    const int32_t * c_row = (const int32_t *) (cand.C + i_batch*cand.nbc1 + i_stream*cand.nbc3);
    uint8_t * f = flags + ((int64_t) i_stream*n_batch + i_batch)*n_blk;
    for (int i = threadIdx.x; i < cand.n_slots / cand.bs; i += blockDim.x) {
        const int32_t b = c_row[i];
        if (b >= 0 && b < n_blk) {
            f[b] = 1;
        }
    }
}

// rows outside the token's candidate blocks -> -inf
static __global__ void lightning_indexer_cand_apply(const uint8_t * flags, float * dst, const int64_t n_kv, const int64_t n_blk,
        const int bs, const int n_batch, const size_t nb1, const size_t nb3) {
    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const uint8_t * f = flags + ((int64_t) i_stream*n_batch + i_batch)*n_blk;
    float * row = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n_kv; i += (int64_t) gridDim.x*blockDim.x) {
        if (!f[i / bs]) {
            row[i] = -INFINITY;
        }
    }
}

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const half  *   m_d = (const half  *)   m->data;
    float       * dst_d = (      float *) dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    // candidate blocks (optional): I32 [n_cand, n_batch, 1, n_stream], rows per block in op_params[0]
    const ggml_tensor * c = dst->src[4];
    li_cand cand = { nullptr, 0, 1, 0, 0, nullptr };
    if (c) {
        GGML_ASSERT(c->type == GGML_TYPE_I32 && c->nb[0] == sizeof(int32_t));
        GGML_ASSERT(c->ne[1] == n_batch && c->ne[2] == 1 && c->ne[3] == n_stream);
        const int bs = ggml_get_op_params_i32(dst, 0);
        GGML_ASSERT(bs > 0 && c->ne[0]*bs <= INT_MAX);
        cand = { (const char *) ggml_cuda_preload_ptr(c), (int) (c->ne[0]*bs), bs, c->nb[1], c->nb[3], nullptr };
    }
    // the live key count (src[5], ggml_lightning_indexer_set_live_len): the vec and tile kernels write -inf for the blocks of keys past
    // it (their padded score) without scoring them; the wmma kernels and the candidate path score every key
    if (const ggml_tensor * lv = dst->src[5]; lv != nullptr && !cand.C) {
        GGML_ASSERT(lv->type == GGML_TYPE_I32 && ggml_nelements(lv) == 1);
        cand.live = (const int *) ggml_cuda_preload_ptr(lv);
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: kpool live: the indexer scores the live keys only (%d keys, %d tokens)\n", __func__, n_kv, n_batch);
        }
    }
    const bool tile = n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS &&
        !(dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT);
    if (cand.C && !tile) {
        // the per token kernels write only the candidate rows
        const dim3 grid_f(std::min<int64_t>((n_kv + 255)/256, 64), n_batch, n_stream);
        if (ggml_cuda_preload_on()) {
            ggml_cuda_preload_func((const void *) lightning_indexer_fill_ninf);
        } else {
            lightning_indexer_fill_ninf<<<grid_f, 256, 0, ctx.stream()>>>(dst_d, n_kv, nb1, nb3);
        }
    }

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (tile) { // batch invariant: per token, as for one token
        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = 64;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        int num_batch_blocks = (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1) / LIGHTNING_INDEXER_TILE_TOKENS;
        dim3 grid(num_kv_blocks, num_batch_blocks, n_stream);

        LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE_ALL(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");

        if (cand.C && ggml_cuda_preload_on()) {
            ggml_cuda_preload_func((const void *) lightning_indexer_cand_flags);
            ggml_cuda_preload_func((const void *) lightning_indexer_cand_apply);
        } else if (cand.C) {
            // every row was scored: rows outside the token's candidate blocks -> -inf
            const int64_t n_blk = (n_kv + cand.bs - 1)/cand.bs;
            ggml_cuda_pool_alloc<uint8_t> flags(ctx.pool(), n_blk*n_batch*n_stream);
            CUDA_CHECK(cudaMemsetAsync(flags.get(), 0, n_blk*n_batch*n_stream, ctx.stream()));
            lightning_indexer_cand_flags<<<dim3(n_batch, n_stream), 256, 0, ctx.stream()>>>(cand, flags.get(), n_blk, n_batch);
            const dim3 grid_a(std::min<int64_t>((n_kv + 255)/256, 64), n_batch, n_stream);
            lightning_indexer_cand_apply<<<grid_a, 256, 0, ctx.stream()>>>(flags.get(), dst_d, n_kv, n_blk, cand.bs, n_batch, nb1, nb3);
        }
    } else if (n_embd == 128 && n_head == 4) {
        // a batch smaller than a token tile, use vector kernel
        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        dim3 grid(num_kv_blocks, n_batch, n_stream);

        // up to 7 tokens in one block per key tile, the same scores (lightning_indexer_kernel_vec_rows); not with candidates, nor for
        // a batch-invariant node (each token keeps its own launch, as for one token)
        static const bool vec_multi = [] {
            const char * e = getenv("GGML_CUDA_LI_VEC_MULTI");
            return e == nullptr || atoi(e) != 0;
        }();
        if (vec_multi && !cand.C && !(dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) && n_batch >= 1 && n_batch <= 7) {
            static bool logged = false;
            if (!logged && !ggml_cuda_preload_on()) {
                logged = true;
                GGML_LOG_INFO("%s: indexer vec multi: %d tokens per key tile in one block (GGML_CUDA_LI_VEC_MULTI=0: off)\n", __func__, n_batch);
            }
            const dim3 grid_r(num_kv_blocks, 1, n_stream);
            switch (n_batch <= 1 ? 1 : n_batch <= 2 ? 2 : n_batch <= 4 ? 4 : 7) {
                case 1: { LIGHTNING_INDEXER_ROWS_TYPES(1) } break;
                case 2: { LIGHTNING_INDEXER_ROWS_TYPES(2) } break;
                case 4: { LIGHTNING_INDEXER_ROWS_TYPES(4) } break;
                default: { LIGHTNING_INDEXER_ROWS_TYPES(7) } break;
            }
            return;
        }

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else {
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32 && neq1 != 4) {
        return false;
    }

    // candidate blocks (optional)
    const ggml_tensor * c = dst->src[4];
    if (c && (c->type != GGML_TYPE_I32 || c->nb[0] != sizeof(int32_t) || c->ne[1] != neq2 || c->ne[2] != 1 || c->ne[3] != neq3 ||
              ggml_get_op_params_i32(dst, 0) <= 0)) {
        return false;
    }

    // alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}
