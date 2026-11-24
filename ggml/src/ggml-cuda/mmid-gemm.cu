#include "mmid-gemm.cuh"
#include "mmid.cuh"
#include "mma.cuh"
#include "getrows.cuh"
#include "cp-async.cuh"
#include "mmf.cuh"

#include <cstring>

using namespace ggml_cuda_mma;

// Block tile: MMID_GEMM_BM rows of src0 (output features) x MMID_GEMM_BN compact src1 columns (token slots),
// stepping over MMID_GEMM_BK values of the shared dimension per stage. 8 warps in a 4 x 2 grid, 32 x 32 per warp.
#define MMID_GEMM_BM     128
#define MMID_GEMM_BN      64
#define MMID_GEMM_BK      64
#define MMID_GEMM_NWARPS   8
#define MMID_GEMM_MIN_TOKENS 16
#define MMID_GEMM_LDS     (MMID_GEMM_BK/2 + 4) // row stride of the shared tiles in 2-element units, padded against bank conflicts
#define MMID_GEMM_LDC     (MMID_GEMM_BM + 4)   // row stride of the staged output tile in floats

// The GEMM uses mma.sync m16n8k16 with F16/BF16 inputs, ldmatrix and cp.async, so it needs NVIDIA Ampere or newer.
// The device guard and the host check below must agree: the host check selects the kernel at run time.
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
#define MMID_GEMM_AVAILABLE
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)

static bool mmid_gemm_available(const int cc) {
    return ampere_mma_available(cc) && cp_async_available(cc);
}

static constexpr size_t mmid_gemm_stage_elems = (MMID_GEMM_BM + MMID_GEMM_BN) * MMID_GEMM_LDS; // per stage, 2-element units
static constexpr size_t mmid_gemm_nbytes_shared = 2*mmid_gemm_stage_elems*4;

static_assert(MMID_GEMM_BN*MMID_GEMM_LDC*sizeof(float) <= mmid_gemm_nbytes_shared, "output staging does not fit");

#ifdef MMID_GEMM_AVAILABLE
static __device__ __forceinline__ void mmid_gemm_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int n>
static __device__ __forceinline__ void mmid_gemm_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
}
#endif // MMID_GEMM_AVAILABLE

// tile_start[e] = number of BN-wide column tiles of all experts before e, tile_start[n_experts] = total.
// One warp, each lane scans a contiguous range of experts.
static __global__ void mmid_gemm_tile_schedule(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_start, const int n_experts) {
#ifdef MMID_GEMM_AVAILABLE
    const int lane  = threadIdx.x;
    const int chunk = (n_experts + WARP_SIZE - 1) / WARP_SIZE;
    const int e0    = min(lane*chunk, n_experts);
    const int e1    = min(e0 + chunk, n_experts);

    int local = 0;
    for (int e = e0; e < e1; ++e) {
        local += (expert_bounds[e + 1] - expert_bounds[e] + MMID_GEMM_BN - 1) / MMID_GEMM_BN;
    }

    int incl = local;
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        const int tmp = __shfl_up_sync(0xFFFFFFFF, incl, offset, WARP_SIZE);
        if (lane >= offset) {
            incl += tmp;
        }
    }

    int running = incl - local;
    for (int e = e0; e < e1; ++e) {
        tile_start[e] = running;
        running += (expert_bounds[e + 1] - expert_bounds[e] + MMID_GEMM_BN - 1) / MMID_GEMM_BN;
    }

    const int total = __shfl_sync(0xFFFFFFFF, incl, WARP_SIZE - 1, WARP_SIZE);
    if (lane == 0) {
        tile_start[n_experts] = total;
    }
#else
    GGML_UNUSED_VARS(expert_bounds, tile_start, n_experts);
    NO_DEVICE_CODE;
#endif // MMID_GEMM_AVAILABLE
}

// x: src0 [n_experts][M][K], y: compact src1 [n_compact][K] in the type of src0, sorted by expert.
// blockIdx.x: BM row block, blockIdx.y: column tile in the device-built schedule (excess CTAs exit).
template <typename T2>
__launch_bounds__(WARP_SIZE*MMID_GEMM_NWARPS, 1)
static __global__ void mmid_gemm_kernel(
        const T2 * __restrict__ x, const T2 * __restrict__ y, float * __restrict__ dst,
        const int32_t * __restrict__ ids, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_start,
        const int n_experts, const int M, const int K, const int64_t stride_row_x, const int64_t stride_expert_x,
        const int n_expert_used, const int si1, const int64_t stride_slot_dst, const int64_t stride_token_dst,
        const bool broadcast_y) {
#ifdef MMID_GEMM_AVAILABLE
    typedef tile<16, 8, T2>    tile_A;
    typedef tile< 8, 8, T2>    tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int ntA = 32 / tile_A::I;
    constexpr int ntB = 32 / tile_B::I;

    const int t = blockIdx.y;
    const int n_tiles = tile_start[n_experts];
    if (t >= n_tiles) {
        return;
    }

    // largest e with tile_start[e] <= t; experts without columns share their start with the next one
    int lo = 0;
    int hi = n_experts - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (tile_start[mid] <= t) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int expert = lo;

    const int col0  = expert_bounds[expert] + (t - tile_start[expert])*MMID_GEMM_BN; // first compact column
    const int ncols = min(MMID_GEMM_BN, expert_bounds[expert + 1] - col0);
    const int row0  = blockIdx.x*MMID_GEMM_BM;

    // strides in 2-element units
    const int64_t sx = stride_row_x/2;
    const T2 * x_blk = x + expert*(stride_expert_x/2);
    const T2 * y_blk = y + int64_t(col0)*(K/2);

    extern __shared__ char data_mmid_gemm[];
    T2 * smem = (T2 *) data_mmid_gemm;

    const int tid  = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int warp = threadIdx.y;
    const int wm   = warp / 2;
    const int wn   = warp % 2;

    const int4 zero4 = make_int4(0, 0, 0, 0);

    auto load_stage = [&](const int stage, const int k0) {
        T2 * sA = smem + stage*mmid_gemm_stage_elems;
        T2 * sB = sA + MMID_GEMM_BM*MMID_GEMM_LDS;
        // 16-byte chunks: 8 per BK row
#pragma unroll
        for (int c0 = 0; c0 < MMID_GEMM_BM*MMID_GEMM_BK/8; c0 += WARP_SIZE*MMID_GEMM_NWARPS) {
            const int c   = c0 + tid;
            const int row = c / (MMID_GEMM_BK/8);
            const int cc  = c % (MMID_GEMM_BK/8);
            T2 * dst_s = sA + row*MMID_GEMM_LDS + cc*4;
            if (row0 + row < M) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dst_s), x_blk + (row0 + row)*sx + k0/2 + cc*4);
            } else {
                *(int4 *) dst_s = zero4;
            }
        }
#pragma unroll
        for (int c0 = 0; c0 < MMID_GEMM_BN*MMID_GEMM_BK/8; c0 += WARP_SIZE*MMID_GEMM_NWARPS) {
            const int c   = c0 + tid;
            const int col = c / (MMID_GEMM_BK/8);
            const int cc  = c % (MMID_GEMM_BK/8);
            T2 * dst_s = sB + col*MMID_GEMM_LDS + cc*4;
            if (col < ncols) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dst_s), y_blk + int64_t(col)*(K/2) + k0/2 + cc*4);
            } else {
                *(int4 *) dst_s = zero4;
            }
        }
    };

    tile_C C[ntA][ntB];

    const int nk = K / MMID_GEMM_BK;
    load_stage(0, 0);
    mmid_gemm_cp_async_commit();

    for (int ik = 0; ik < nk; ++ik) {
        const int stage = ik % 2;
        if (ik + 1 < nk) {
            load_stage(stage ^ 1, (ik + 1)*MMID_GEMM_BK);
        }
        mmid_gemm_cp_async_commit();
        mmid_gemm_cp_async_wait<1>();
        __syncthreads();

        const T2 * sA = smem + stage*mmid_gemm_stage_elems;
        const T2 * sB = sA + MMID_GEMM_BM*MMID_GEMM_LDS;
#pragma unroll
        for (int kk = 0; kk < MMID_GEMM_BK/2; kk += tile_A::J) {
            tile_A A[ntA];
            tile_B B[ntB];
#pragma unroll
            for (int i = 0; i < ntA; ++i) {
                load_ldmatrix(A[i], sA + (wm*32 + i*tile_A::I)*MMID_GEMM_LDS + kk, MMID_GEMM_LDS);
            }
#pragma unroll
            for (int j = 0; j < ntB; ++j) {
                load_ldmatrix(B[j], sB + (wn*32 + j*tile_B::I)*MMID_GEMM_LDS + kk, MMID_GEMM_LDS);
            }
#pragma unroll
            for (int i = 0; i < ntA; ++i) {
#pragma unroll
                for (int j = 0; j < ntB; ++j) {
                    mma(C[i][j], A[i], B[j]);
                }
            }
        }
        __syncthreads();
    }
    mmid_gemm_cp_async_wait<0>();

    // Stage the output tile in shared memory as [column][row] so that rows of dst are written contiguously.
    float * sC = (float *) data_mmid_gemm;
#pragma unroll
    for (int i = 0; i < ntA; ++i) {
#pragma unroll
        for (int j = 0; j < ntB; ++j) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int r = wm*32 + i*tile_C::I + tile_C::get_i(l);
                const int c = wn*32 + j*tile_C::J + tile_C::get_j(l);
                sC[c*MMID_GEMM_LDC + r] = C[i][j].x[l];
            }
        }
    }
    __syncthreads();

    for (int c = warp; c < ncols; c += MMID_GEMM_NWARPS) {
        const int v   = ids_dst[col0 + c];
        if (v < 0) {
            continue;
        }
        const int it  = v / n_expert_used;
        const int iex = v % n_expert_used;
        const float * src_row = sC + c*MMID_GEMM_LDC;

        for (int jex = 0; jex < n_expert_used; ++jex) {
            // with one src1 row per token every slot that selected this expert gets the same row,
            // otherwise only the slot that the compact column was built from
            const bool write = broadcast_y ? ids[it*si1 + jex] == expert : jex == iex;
            if (!write) {
                continue;
            }
            float * dst_row = dst + it*stride_token_dst + jex*stride_slot_dst;
#pragma unroll
            for (int r0 = 0; r0 < MMID_GEMM_BM; r0 += WARP_SIZE) {
                const int r = r0 + threadIdx.x;
                if (row0 + r < M) {
                    dst_row[row0 + r] = src_row[r];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, ids, ids_dst, expert_bounds, tile_start, n_experts, M, K, stride_row_x, stride_expert_x,
        n_expert_used, si1, stride_slot_dst, stride_token_dst, broadcast_y);
    NO_DEVICE_CODE;
#endif // MMID_GEMM_AVAILABLE
}

bool ggml_cuda_should_use_mmid_gemm(const ggml_tensor * dst, const int cc) {
    // GGML_CUDA_MMID_LEGACY=1 (or GGML_CUDA_DISABLE_MMID_GEMM=1) restores the previous paths for A/B and rollback
    static const bool disabled = [] {
        const char * legacy = getenv("GGML_CUDA_MMID_LEGACY");
        return getenv("GGML_CUDA_DISABLE_MMID_GEMM") != nullptr || (legacy != nullptr && strcmp(legacy, "0") != 0);
    }();
    if (disabled) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (!mmid_gemm_available(cc)) {
        return false;
    }

    const size_t ts = ggml_type_size(src0->type);
    if (src0->ne[0] % MMID_GEMM_BK != 0 || src0->nb[0] != ts || src0->nb[1] % 16 != 0 || src0->nb[2] % 16 != 0 ||
            (uintptr_t) src0->data % 16 != 0) {
        return false;
    }
    if (src0->ne[3] != 1 || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) || src1->nb[2] % src1->nb[1] != 0) {
        return false;
    }
    if (dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[2] % sizeof(float) != 0) {
        return false;
    }

    const int64_t n_tokens = src1->ne[2];
    const int64_t n_slots  = n_tokens*ids->ne[0];
    // Below this many tokens mmf is kept where it applies; from 16 tokens up this kernel was faster on B200
    // for the shapes measured (512 experts x 640 rows with 10 used, 128 experts x 768 rows with 8 used).
    if (n_tokens < MMID_GEMM_MIN_TOKENS && ggml_cuda_should_use_mmf(src0->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size,
            src0->ne, src0->nb, n_tokens, /*mul_mat_id=*/true)) {
        return false;
    }
    if (n_tokens >= (1 << 22) || ids->ne[0] >= (1 << 10)) {
        return false;
    }
    // the expert sort runs on the device (the single-warp helper or the counting sort, whichever fits the shared memory)
    if (!ggml_cuda_mm_ids_helper_supported(src0->ne[2], n_tokens, ids->ne[0])) {
        return false;
    }
    if ((n_slots + MMID_GEMM_BN - 1)/MMID_GEMM_BN + src0->ne[2] > 65535) { // gridDim.y
        return false;
    }
    return true;
}

template <typename T2>
static void launch_mmid_gemm(
        const T2 * x, const T2 * y, float * dst, const int32_t * ids, const int32_t * ids_dst,
        const int32_t * expert_bounds, const int32_t * tile_start, const int n_experts, const int M, const int K,
        const int64_t stride_row_x, const int64_t stride_expert_x, const int n_expert_used, const int si1,
        const int64_t stride_slot_dst, const int64_t stride_token_dst, const bool broadcast_y, const int64_t n_slots,
        cudaStream_t stream) {
    CUDA_SET_SHARED_MEMORY_LIMIT(mmid_gemm_kernel<T2>, mmid_gemm_nbytes_shared);

    const int max_tiles = (n_slots + MMID_GEMM_BN - 1)/MMID_GEMM_BN + n_experts;
    const dim3 grid((M + MMID_GEMM_BM - 1)/MMID_GEMM_BM, max_tiles, 1);
    const dim3 block(WARP_SIZE, MMID_GEMM_NWARPS, 1);
    mmid_gemm_kernel<T2><<<grid, block, mmid_gemm_nbytes_shared, stream>>>(
        x, y, dst, ids, ids_dst, expert_bounds, tile_start, n_experts, M, K, stride_row_x, stride_expert_x,
        n_expert_used, si1, stride_slot_dst, stride_token_dst, broadcast_y);
}

void ggml_cuda_mul_mat_id_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ne12;
    const int64_t n_slots       = n_tokens*n_expert_used;
    const int     n_experts     = ne02;

    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: grouped %s MUL_MAT_ID GEMM (%lld tokens, %d experts, %lld used; GGML_CUDA_MMID_LEGACY=1: previous paths)\n",
            __func__, ggml_type_name(src0->type), (long long) n_tokens, n_experts, (long long) n_expert_used);
    }

    const int si1  = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = nb12 / nb11;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> tile_start(ctx.pool(), n_experts + 1);

    // With repeated expert ids for a token, mm_ids_helper writes one compact column per (token, expert) but sizes
    // expert_bounds by slots, which leaves unwritten columns inside an expert's range. Those must gather a valid
    // src1 row (index 0) and are skipped when writing dst (ids_dst == -1).
    CUDA_CHECK(cudaMemsetAsync(ids_src1.get(), 0,    n_slots*sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(ids_dst.get(),  0xFF, n_slots*sizeof(int32_t), stream));

    ggml_cuda_launch_mm_ids_helper(ctx.pool(), (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());

    // the GEMM only writes the rows an expert owns, the rest are skipped slots (id -1): zero them as MMQ / MMF do
    if (ggml_cuda_mm_ids_may_skip(ids)) {
        ggml_cuda_launch_mm_ids_zero_dst((const int32_t *) ids->data, (float *) dst->data,
            n_tokens, n_expert_used, ne0, si1, nb1/sizeof(float), nb2/sizeof(float), stream);
        CUDA_CHECK(cudaGetLastError());
    }

    mmid_gemm_tile_schedule<<<1, WARP_SIZE, 0, stream>>>(expert_bounds.get(), tile_start.get(), n_experts);
    CUDA_CHECK(cudaGetLastError());

    // Compact src1 sorted by expert, converted to the type of src0 (the per-expert cuBLAS path rounds the same way).
    const size_t ts = ggml_type_size(src0->type);
    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), n_slots*ne10*ts);
    get_rows_cuda(src1->data, src1->type, ids_src1.get(), src1_sorted.ptr, src0->type,
        ne10, nb11, nb12, nb13,
        n_slots, 1, 1, sizeof(int32_t), n_slots*sizeof(int32_t), n_slots*sizeof(int32_t),
        ne10*ts, n_slots*ne10*ts, n_slots*ne10*ts, stream);
    CUDA_CHECK(cudaGetLastError());

    const bool broadcast_y = ne11 == 1;
    const int64_t stride_row_x    = nb01 / ts;
    const int64_t stride_expert_x = nb02 / ts;
    const int64_t stride_slot_dst  = nb1 / sizeof(float);
    const int64_t stride_token_dst = nb2 / sizeof(float);

    if (src0->type == GGML_TYPE_BF16) {
        launch_mmid_gemm<nv_bfloat162>((const nv_bfloat162 *) src0->data, (const nv_bfloat162 *) src1_sorted.ptr,
            (float *) dst->data, (const int32_t *) ids->data, ids_dst.get(), expert_bounds.get(), tile_start.get(),
            n_experts, ne01, ne00, stride_row_x, stride_expert_x, n_expert_used, si1, stride_slot_dst, stride_token_dst,
            broadcast_y, n_slots, stream);
    } else {
        GGML_ASSERT(src0->type == GGML_TYPE_F16);
        launch_mmid_gemm<half2>((const half2 *) src0->data, (const half2 *) src1_sorted.ptr,
            (float *) dst->data, (const int32_t *) ids->data, ids_dst.get(), expert_bounds.get(), tile_start.get(),
            n_experts, ne01, ne00, stride_row_x, stride_expert_x, n_expert_used, si1, stride_slot_dst, stride_token_dst,
            broadcast_y, n_slots, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
