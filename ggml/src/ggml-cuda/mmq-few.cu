// few-row MMQ ported from upstream llama.cpp (the llama.cpp authors, MIT), routing in ggml-cuda.cu behind GGML_CUDA_MMQ_FEW.
#include "mmq-few.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "mmid.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <type_traits>

bool ggml_cuda_verify_reassoc_ok() {
    static const bool reassoc = [] {
        const char * e = getenv("GGML_CUDA_VERIFY_REASSOC");
        return e == nullptr || atoi(e) != 0;
    }();
    return reassoc;
}

// NVIDIA only: the kernel uses the NVIDIA int8 MMA fragment layout and cp.async.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// helpers that upstream keeps in common.cuh / cp-async.cuh / quantize.cuh, kept in this file in the fork so the port
// recompiles only this file (those headers are included by every CUDA object)
static __device__ __forceinline__ void ggml_cuda_prefetch_l2(const void * p) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
    GGML_UNUSED(p);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// Commits the copies issued since the last commit as one group.
static __device__ __forceinline__ void cp_async_commit_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Makes each thread wait until at most n of its copy groups are pending.
// Like cp_async_wait_all, this does NOT provide any additional synchronization.
template <int n>
static __device__ __forceinline__ void cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// The 6-bit scales 4*ksc .. 4*ksc + 3 of a q3_K block, minus 32, as int8 (upstream mmq-load-tiles.cuh).
static __device__ __forceinline__ int unpack_scales_q3_K(const uint8_t * scales, const int ksc) {
    const int sc_low  = (get_int_b2(scales, ksc % 2) >> (4 * (ksc / 2))) & 0x0F0F0F0F;
    const int sc_high = ((get_int_b2(scales, 2) >> (2 * ksc)) << 4) & 0x30303030;
    return __vsub4(sc_low | sc_high, 0x20202020);
}

// src1 of the few-row MMQ in the layout of the MMA fragments. Per channel, 256 values and group of MMQ_FEW_NCOLS columns:
//   qs: int4[4][32], [j][lane] holds the B fragments of sub-blocks 2j and 2j+1 of lane (g, t): the values [4t, 4t + 4) and [16 + 4t, 16 + 4t + 4) of column g.
//   ds: int4[4][4], [t][j] holds d (D4, float) or (d, sum) (DS4, half2) of the columns 2t and 2t+1 for sub-block 2j, then 2j+1.
// The column groups of a unit are adjacent, columns beyond ne1 are zero.
static constexpr int MMQ_FEW_NCOLS     = 8; // src1 columns per column group
static constexpr int MMQ_FEW_Q8_1_INT4 = 4*WARP_SIZE + 4*4; // int4 per 256 values and column group

void quantize_mmq_few_q8_1_cuda(
        const float * x, void * vy, mmq_q8_1_ds_layout ds_layout, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// MUL_MAT_ID: a pair is an expert and one column group of up to MMQ_FEW_NCOLS of its (token, slot) entries. The src1 of pair p is
// channel p of the layout above, its columns past the entries are not written.
// A pair is MMQ_FEW_PAIR_INTS ints: expert, number of entries (0 for an unused pair), then the entries token*n_expert_used + slot.
static constexpr int MMQ_FEW_PAIR_INTS = 2 + MMQ_FEW_NCOLS;

void quantize_mmq_few_id_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, int * pairs, mmq_q8_1_ds_layout ds_layout, int64_t ne00, int64_t s11, int64_t s12,
        int64_t ne11, int64_t n_experts, int64_t n_expert_used, int64_t n_tokens, int64_t si1, int64_t npairs_max, cudaStream_t stream);

using namespace ggml_cuda_mma;

// Few-row MMQ: int8 MMA mat-mul for 2-24 src1 columns. A CUDA block has 16 src0 rows (32 for 2 column groups on large matrices).
// Its warps split K in units of 256 values and copy raw src0 rows into their own cp.async ring, src1 is quantized to the MMA fragment layout.
// MUL_MAT_ID: one column group per expert and CUDA block; for short rows the warps split the rows instead of K (64 rows per block).

static constexpr int MMQ_FEW_MAX_NCG = 3;  // max. number of column groups
static constexpr int MMQ_FEW_NROWS   = 16; // src0 rows of one MMA tile
static constexpr int MMQ_FEW_NWARPS  = 4;

// GGML_CUDA_MMQ_FEW: unset = AUTO (the per-architecture table in ggml_cuda_mmq_few_auto_min_ncols: A100 / A30 and Ada
// measured, off elsewhere, also on cc 12.0 since a later fix), 0 = off, 1 = the upstream per-type minimum columns (ggml_cuda_mmq_few_min_ncols) on every Ampere+ device,
// N >= 2 = from N columns for every type it decodes; GGML_CUDA_MMQ_FEW_MAX (default 8, the verify widths) = the most columns taken
// (upstream: 24; wider products keep MMQ)
struct mmq_few_env_t { int mode; int max_cols; };
static const mmq_few_env_t & mmq_few_env() {
    static const mmq_few_env_t e = [] {
        mmq_few_env_t r = { -1, 8 };
        if (const char * v = getenv("GGML_CUDA_MMQ_FEW")) {
            r.mode = atoi(v);
        }
        if (const char * v = getenv("GGML_CUDA_MMQ_FEW_MAX")) {
            r.max_cols = std::min(MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS, std::max(2, atoi(v)));
        }
        return r;
    }();
    return e;
}


template <ggml_type type>
static constexpr __host__ __device__ int mmq_few_unit_size_of() {
    return QK_K/ggml_cuda_type_traits<type>::qk*ggml_cuda_type_traits<type>::bs;
}

// src0 bytes per 256 values, 0 for the types without a few-row decoder.
static constexpr __host__ __device__ int mmq_few_unit_size(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return mmq_few_unit_size_of<GGML_TYPE_Q8_0>();
        case GGML_TYPE_Q3_K:   return mmq_few_unit_size_of<GGML_TYPE_Q3_K>();
        case GGML_TYPE_Q4_K:   return mmq_few_unit_size_of<GGML_TYPE_Q4_K>();
        case GGML_TYPE_Q5_K:   return mmq_few_unit_size_of<GGML_TYPE_Q5_K>();
        case GGML_TYPE_Q6_K:   return mmq_few_unit_size_of<GGML_TYPE_Q6_K>();
        case GGML_TYPE_IQ3_S:  return mmq_few_unit_size_of<GGML_TYPE_IQ3_S>();
        case GGML_TYPE_IQ4_NL: return mmq_few_unit_size_of<GGML_TYPE_IQ4_NL>();
        case GGML_TYPE_IQ4_XS: return mmq_few_unit_size_of<GGML_TYPE_IQ4_XS>();
        default:               return 0;
    }
}

// Q4_K and Q5_K also need the src1 sums for their mins.
static constexpr __host__ __device__ mmq_q8_1_ds_layout mmq_few_ds_layout(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
}

// Units of 256 values per ring stage: 1 for Q6_K, 3 column groups (fewer free registers) and 32-row tiles (2 blocks per SM), else 2.
static constexpr __host__ __device__ int mmq_few_nunits_stage(const ggml_type type, const int ncg, const int nmt) {
    return type == GGML_TYPE_Q6_K || ncg > 2 || nmt > 1 ? 1 : 2;
}

// A third ring stage only makes Q6_K faster, the other types get slower on large matrices.
static constexpr __host__ __device__ int mmq_few_nstages(const ggml_type type) {
    return type == GGML_TYPE_Q6_K ? 3 : 2;
}

// The src0 data of a ring stage does not fill whole 16 byte chunks, it is copied from the preceding 16 byte boundary.
static constexpr __host__ __device__ bool mmq_few_unaligned(const ggml_type type, const int nunits_stage) {
    return (nunits_stage*mmq_few_unit_size(type)) % 16 != 0;
}

// for unaligned data the max. over the 8 possible start offsets
static constexpr __host__ __device__ int mmq_few_nchunks(const ggml_type type, const int nunits_stage) {
    const int nbytes = nunits_stage*mmq_few_unit_size(type);
    return mmq_few_unaligned(type, nunits_stage) ? (14 + nbytes + 15)/16 : nbytes/16;
}

// Unaligned rows are padded to an odd number of 16 byte chunks against bank conflicts, for aligned rows the padding costs more than it saves.
static constexpr __host__ __device__ int mmq_few_sram_stride(const ggml_type type, const int nunits_stage) {
    const int nchunks = mmq_few_nchunks(type, nunits_stage);
    return mmq_few_unaligned(type, nunits_stage) && nchunks % 2 == 0 ? 16*(nchunks + 1) : 16*nchunks;
}

static constexpr __host__ __device__ int mmq_few_nbytes_shared(const ggml_type type, const int ncg, const int nmt) {
    return MMQ_FEW_NWARPS*mmq_few_nstages(type)*nmt*MMQ_FEW_NROWS*mmq_few_sram_stride(type, mmq_few_nunits_stage(type, ncg, nmt));
}

// With Q6_K and Q8_0 two CUDA blocks with 32-row tiles would not fit into an SM.
// MUL_MAT_ID with fewer than 2 units of 256 values per warp: the warps split the rows, each warp has its own 16 rows.
static constexpr __host__ __device__ bool mmq_few_ksplit(const bool has_ids, const int nunits) {
    return !has_ids || nunits >= 2*MMQ_FEW_NWARPS;
}

static constexpr __host__ __device__ bool mmq_few_tall_supported(const ggml_type type) {
    return type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS;
}

#if defined(CP_ASYNC_AVAILABLE)

// The src1 fragments of a lane for one unit of 256 values, see quantize.cuh.
struct mmq_few_y_frag {
    int4 qs[4];
    int4 ds[4];
};

static __device__ __forceinline__ void mmq_few_load_y(mmq_few_y_frag & yf, const int4 * y) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.qs[j] = y[j*WARP_SIZE + threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.ds[j] = y[4*WARP_SIZE + 4*(threadIdx.x % 4) + j];
    }
}

static __device__ __forceinline__ int2 mmq_few_get_b(const mmq_few_y_frag & yf, const int s) {
    const int4 q = yf.qs[s/2];
    return s % 2 == 0 ? make_int2(q.x, q.y) : make_int2(q.z, q.w);
}

template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ float2 mmq_few_get_d(const mmq_few_y_frag & yf, const int s) {
    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        const float * d = (const float *) &yf.ds[s/2];
        return make_float2(d[2*(s % 2) + 0], d[2*(s % 2) + 1]);
    } else {
        const half2 * ds = (const half2 *) &yf.ds[s/2];
        return make_float2(__low2float(ds[2*(s % 2) + 0]), __low2float(ds[2*(s % 2) + 1]));
    }
}

static __device__ __forceinline__ float2 mmq_few_get_sum(const mmq_few_y_frag & yf, const int s) {
    const half2 * ds = (const half2 *) &yf.ds[s/2];
    return make_float2(__high2float(ds[2*(s % 2) + 0]), __high2float(ds[2*(s % 2) + 1]));
}

// C.x[0], C.x[1]: row g, columns 2t and 2t+1. C.x[2], C.x[3]: row g+8.
static __device__ __forceinline__ tile<16, 8, int> mmq_few_mma(const int (&lo)[2], const int (&hi)[2], const int2 b) {
    tile<16, 8, int> A;
    A.x[0] = lo[0];
    A.x[1] = lo[1];
    A.x[2] = hi[0];
    A.x[3] = hi[1];
    tile<8, 8, int> B;
    B.x[0] = b.x;
    B.x[1] = b.y;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

static __device__ __forceinline__ tile<16, 8, int> mmq_few_mma_k16(const int (&a)[2], const int b) {
    tile<16, 4, int> A;
    A.x[0] = a[0];
    A.x[1] = a[1];
    tile<8, 4, int> B;
    B.x[0] = b;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

// sum_d += d * (C * ls) of sub-block s for all column groups, ls are the src0 row scales
template <ggml_type type, int ncg, typename scale_t>
static __device__ __forceinline__ void mmq_few_mma_add(
        const int (&lo)[2], const int (&hi)[2], const scale_t (&ls)[2], const mmq_few_y_frag (&yf)[ncg], const int s, float (&sum_d)[ncg][4]) {
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
        const tile<16, 8, int> C = mmq_few_mma(lo, hi, mmq_few_get_b(yf[c], s));

        const float2 d = mmq_few_get_d<mmq_few_ds_layout(type)>(yf[c], s);
        sum_d[c][0] += d.x * float(C.x[0]*ls[0]);
        sum_d[c][1] += d.y * float(C.x[1]*ls[0]);
        sum_d[c][2] += d.x * float(C.x[2]*ls[1]);
        sum_d[c][3] += d.y * float(C.x[3]*ls[1]);
    }
}

// Same with one k=16 MMA per half of sub-block s, the int8 scales of the halves are bytes 0 and 1 of sc[r].
template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_mma_k16_add(
        const int (&lo)[2], const int (&hi)[2], const int (&sc)[2], const mmq_few_y_frag (&yf)[ncg], const int s, float (&sum_d)[ncg][4]) {
    const int sc_lo[2] = {(int8_t) sc[0], (int8_t) sc[1]};
    const int sc_hi[2] = {(int8_t) (sc[0] >> 8), (int8_t) (sc[1] >> 8)};
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
        const int2 b = mmq_few_get_b(yf[c], s);
        const tile<16, 8, int> C_lo = mmq_few_mma_k16(lo, b.x);
        const tile<16, 8, int> C_hi = mmq_few_mma_k16(hi, b.y);

        const float2 d = mmq_few_get_d<mmq_few_ds_layout(type)>(yf[c], s);
        sum_d[c][0] += d.x * float(C_lo.x[0]*sc_lo[0] + C_hi.x[0]*sc_hi[0]);
        sum_d[c][1] += d.y * float(C_lo.x[1]*sc_lo[0] + C_hi.x[1]*sc_hi[0]);
        sum_d[c][2] += d.x * float(C_lo.x[2]*sc_lo[1] + C_hi.x[2]*sc_hi[1]);
        sum_d[c][3] += d.y * float(C_lo.x[3]*sc_lo[1] + C_hi.x[3]*sc_hi[1]);
    }
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_add_scaled(float (&sum)[ncg][4], const float (&d)[2], const float (&sum_d)[ncg][4]) {
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[c][l] += d[l/2]*sum_d[c][l];
        }
    }
}

// vec_dot: one unit of 256 values of the 16 rows of a warp, x[0] and x[1] are the rows g and g+8 of lane (g, t).
// The lane decodes the values [4t, 4t + 4) and [16 + 4t, 16 + 4t + 4) of each sub-block, its A fragments.

template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q45_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
    const int t = threadIdx.x % 4;

    int   qs[2][4][2];
    [[maybe_unused]] int qh[2][2]; // Q5_K: bit s holds the high bit of sub-block s
    int   sc[2][2];
    int   mn[2][2];
    float d[2];
    float dmin[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_t * bx = (const block_t *) x[r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qs[r][j][0] = get_int_b4(bx->qs + 32*j,      t);
            qs[r][j][1] = get_int_b4(bx->qs + 32*j + 16, t);
        }
        if constexpr (type == GGML_TYPE_Q5_K) {
            qh[r][0] = get_int_b4(bx->qh,      t);
            qh[r][1] = get_int_b4(bx->qh + 16, t);
        }
        const int4 hdr = *(const int4 *) bx; // dm and the 12 scale bytes
        const int scales[3] = {hdr.y, hdr.z, hdr.w};
        sc[r][0] = unpack_scales_q45_K(scales, 0);
        sc[r][1] = unpack_scales_q45_K(scales, 1);
        mn[r][0] = unpack_scales_q45_K(scales, 2);
        mn[r][1] = unpack_scales_q45_K(scales, 3);
        const float2 dm = __half22float2(bx->dm);
        d[r]    = dm.x;
        dmin[r] = dm.y;
    }

    float sum_d[ncg][4] = {{0.0f}};
    float sum_m[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int j     = s/2;
        const int shift = 4*(s % 2);
        int lo[2];
        int hi[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            lo[r] = (qs[r][j][0] >> shift) & 0x0F0F0F0F;
            hi[r] = (qs[r][j][1] >> shift) & 0x0F0F0F0F;
            if constexpr (type == GGML_TYPE_Q5_K) {
                lo[r] |= ((qh[r][0] >> s) & 0x01010101) << 4;
                hi[r] |= ((qh[r][1] >> s) & 0x01010101) << 4;
            }
        }
        const int   ls[2] = {(sc[0][s/4] >> (8*(s % 4))) & 0xFF, (sc[1][s/4] >> (8*(s % 4))) & 0xFF};
        const float m0    = (mn[0][s/4] >> (8*(s % 4))) & 0xFF;
        const float m1    = (mn[1][s/4] >> (8*(s % 4))) & 0xFF;
        mmq_few_mma_add<type>(lo, hi, ls, yf, s, sum_d);
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
            const float2 sm = mmq_few_get_sum(yf[c], s);
            sum_m[c][0] += m0 * sm.x;
            sum_m[c][1] += m0 * sm.y;
            sum_m[c][2] += m1 * sm.x;
            sum_m[c][3] += m1 * sm.y;
        }
    }
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[c][l] += d[l/2]*sum_d[c][l] - dmin[l/2]*sum_m[c][l];
        }
    }
}

// Q6_K and Q3_K have a scale per 16 values: one k=16 MMA per half of a sub-block.
template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q6_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        d[r] = __half2float(((const block_q6_K *) x[r])->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int ql[2][2][2]; // [row][ql offset 0 or 32][lo or hi]
        int qh[2][2];
        int sc[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q6_K * bx = (const block_q6_K *) x[r];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                ql[r][k][0] = get_int_b2(bx->ql + 64*n + 32*k,      t);
                ql[r][k][1] = get_int_b2(bx->ql + 64*n + 32*k + 16, t);
            }
            qh[r][0] = get_int_b2(bx->qh + 32*n,      t);
            qh[r][1] = get_int_b2(bx->qh + 32*n + 16, t);
            sc[r][0] = get_int_b2(bx->scales + 8*n, 0);
            sc[r][1] = get_int_b2(bx->scales + 8*n, 1);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4*n + j;
            int lo[2];
            int hi[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                lo[r] = __vsub4(((ql[r][j % 2][0] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][0] >> (2*j)) & 0x03030303) << 4), 0x20202020);
                hi[r] = __vsub4(((ql[r][j % 2][1] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][1] >> (2*j)) & 0x03030303) << 4), 0x20202020);
            }
            const int sc_s[2] = {sc[0][j/2] >> (16*(j % 2)), sc[1][j/2] >> (16*(j % 2))}; // scales 2s and 2s+1
            mmq_few_mma_k16_add<GGML_TYPE_Q6_K>(lo, hi, sc_s, yf, s, sum_d);
        }
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q3_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    int   hm[2][2];
    int   sc[2][4]; // the 16 scales of a row as int8, minus 32
    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_q3_K * bx = (const block_q3_K *) x[r];
        hm[r][0] = ~get_int_b2(bx->hmask,      t); // a set bit means that 4 is subtracted
        hm[r][1] = ~get_int_b2(bx->hmask + 16, t);
#pragma unroll
        for (int ksc = 0; ksc < 4; ++ksc) {
            sc[r][ksc] = unpack_scales_q3_K(bx->scales, ksc);
        }
        d[r] = __half2float(bx->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int qs[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q3_K * bx = (const block_q3_K *) x[r];
            qs[r][0] = get_int_b2(bx->qs + 32*n,      t);
            qs[r][1] = get_int_b2(bx->qs + 32*n + 16, t);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4*n + j;
            int lo[2];
            int hi[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                lo[r] = __vsub4((qs[r][0] >> (2*j)) & 0x03030303, ((hm[r][0] >> s) & 0x01010101) << 2);
                hi[r] = __vsub4((qs[r][1] >> (2*j)) & 0x03030303, ((hm[r][1] >> s) & 0x01010101) << 2);
            }
            const int sc_s[2] = {sc[0][s/2] >> (16*(s % 2)), sc[1][s/2] >> (16*(s % 2))}; // scales 2s and 2s+1
            mmq_few_mma_k16_add<GGML_TYPE_Q3_K>(lo, hi, sc_s, yf, s, sum_d);
        }
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_iq4_xs(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
    int   scales_h[2];
    int   scales_l[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_iq4_xs * bx = (const block_iq4_xs *) x[r];
        d[r]        = __half2float(bx->d);
        scales_h[r] = bx->scales_h;
        scales_l[r] = get_int_b4(bx->scales_l, 0);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int lo[2];
        int hi[2];
        int ls[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_iq4_xs * bx = (const block_iq4_xs *) x[r];
            const int2 v = get_int_from_table_16(get_int_b4(bx->qs + 16*s, t), kvalues_iq4nl);
            lo[r] = v.x;
            hi[r] = v.y;
            ls[r] = (((scales_l[r] >> (4*s)) & 0x0F) | (((scales_h[r] >> (2*s)) & 0x03) << 4)) - 32;
        }
        mmq_few_mma_add<GGML_TYPE_IQ4_XS>(lo, hi, ls, yf, s, sum_d);
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

// Q8_0 and IQ4_NL have a block with its own scale per 32 values.
template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q8_0_iq4_nl(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int   lo[2];
        int   hi[2];
        float d[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if constexpr (type == GGML_TYPE_Q8_0) {
                const block_q8_0 * bx = (const block_q8_0 *) x[r] + s;
                lo[r] = get_int_b2(bx->qs,      t);
                hi[r] = get_int_b2(bx->qs + 16, t);
                d[r]  = __half2float(bx->d);
            } else {
                const block_iq4_nl * bx = (const block_iq4_nl *) x[r] + s;
                const int2 v = get_int_from_table_16(get_int_b2(bx->qs, t), kvalues_iq4nl);
                lo[r] = v.x;
                hi[r] = v.y;
                d[r]  = __half2float(bx->d);
            }
        }
        mmq_few_mma_add<type>(lo, hi, d, yf, s, sum);
    }
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_iq3_s(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        d[r] = __half2float(((const block_iq3_s *) x[r])->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int lo[2];
        int hi[2];
        int ls[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_iq3_s * bx = (const block_iq3_s *) x[r];
            const int qh = bx->qh[s];
            // the values [4k, 4k + 4) of a sub-block come from grid entry k, with the signs in bits 4*(k%2) of byte k/2
            const int k_lo = t;
            const int k_hi = 4 + t;
            const int grid_lo = iq3s_grid[bx->qs[8*s + k_lo] | (((qh >> k_lo) & 1) << 8)];
            const int grid_hi = iq3s_grid[bx->qs[8*s + k_hi] | (((qh >> k_hi) & 1) << 8)];
            const int sign_lo = (bx->signs[4*s + k_lo/2] >> (4*(k_lo % 2))) & 0x0F;
            const int sign_hi = (bx->signs[4*s + k_hi/2] >> (4*(k_hi % 2))) & 0x0F;
            const int mask_lo = __vcmpne4(((sign_lo & 0x03) << 7) | ((sign_lo & 0x0C) << 21), 0x00000000);
            const int mask_hi = __vcmpne4(((sign_hi & 0x03) << 7) | ((sign_hi & 0x0C) << 21), 0x00000000);
            lo[r] = __vsub4(grid_lo ^ mask_lo, mask_lo);
            hi[r] = __vsub4(grid_hi ^ mask_hi, mask_hi);
            ls[r] = 1 + 2*((bx->scales[s/2] >> (4*(s % 2))) & 0x0F);
        }
        mmq_few_mma_add<GGML_TYPE_IQ3_S>(lo, hi, ls, yf, s, sum_d);
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL) {
        mmq_few_vec_dot_q8_0_iq4_nl<type, ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        mmq_few_vec_dot_q3_K<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        mmq_few_vec_dot_q45_K<type, ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        mmq_few_vec_dot_q6_K<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        mmq_few_vec_dot_iq3_s<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        mmq_few_vec_dot_iq4_xs<ncg>(x, yf, sum);
    } else {
        static_assert(type == GGML_TYPE_COUNT, "unsupported type");
    }
}

#endif // defined(CP_ASYNC_AVAILABLE)

// has_ids: MUL_MAT_ID, blockIdx.y is a pair of quantize_mmq_few_id_q8_1, its expert is the src0 channel and dst dims 1 and 2 are slot
// and token. Without ids every index below is the one of the dense kernel.
template <ggml_type type, int ncg, int nmt, bool has_ids = false>
__launch_bounds__(MMQ_FEW_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_q_few(
        const char * x, const int4 * y_ptr, float * dst,
        const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x, const int stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int stride_sample_dst, const bool x_is_constant, const int * pairs) {
#if defined(CP_ASYNC_AVAILABLE)
    constexpr int nwarps    = MMQ_FEW_NWARPS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == WARP_SIZE, "unsupported warp size");
    static_assert(!has_ids || (ncg == 1 && nmt == 1), "unsupported MUL_MAT_ID config");

    constexpr int  unit_size    = mmq_few_unit_size(type);
    constexpr int  nrows        = nmt*MMQ_FEW_NROWS;
    constexpr int  nunits_stage = mmq_few_nunits_stage(type, ncg, nmt);
    constexpr int  nstages      = mmq_few_nstages(type);
    constexpr bool unaligned    = mmq_few_unaligned(type, nunits_stage);
    constexpr int  nchunks      = mmq_few_nchunks(type, nunits_stage);
    constexpr int  sram_stride  = mmq_few_sram_stride(type, nunits_stage);
    constexpr int  nbytes_stage = nrows*sram_stride;
    constexpr int  ncopies      = nrows*nchunks;
    static_assert(unit_size > 0, "unsupported type");

    extern __shared__ char data_mmq_few[];
    char * ring = data_mmq_few + threadIdx.y*nstages*nbytes_stage;

    const bool ksplit   = mmq_few_ksplit(has_ids, nunits);
    const int  row0_blk = blockIdx.x*(ksplit ? nrows : nwarps*nrows);
    const int  row0     = ksplit ? row0_blk : row0_blk + threadIdx.y*nrows;
    const int  iu_begin = ksplit ? nunits_stage*threadIdx.y : 0;
    const int  iu_step  = ksplit ? nunits_stage*nwarps : nunits_stage;

    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int nchannels   = gridDim.y;

    int ncols = ncols_dst;
    [[maybe_unused]] const int * pair = nullptr;
    if constexpr (has_ids) {
        ggml_cuda_pdl_sync(); // the pairs come from the preceding kernel
        pair  = pairs + channel_dst*MMQ_FEW_PAIR_INTS;
        ncols = pair[1];
        if (ncols == 0) {
            return;
        }
        x += int64_t(pair[0])*stride_channel_x;
    } else {
        x   += int64_t(sample_dst/sample_ratio)*stride_sample_x + int64_t(channel_dst/channel_ratio)*stride_channel_x;
        dst += int64_t(sample_dst)*stride_sample_dst + int64_t(channel_dst)*stride_channel_dst;
    }
    const int4 * GGML_CUDA_RESTRICT y = y_ptr + (int64_t(sample_dst)*nchannels + channel_dst)*nunits*ncg*MMQ_FEW_Q8_1_INT4;

    // No bounds check for the units, the gate makes nunits a multiple of nunits_stage.
    // The extra bytes of unaligned data are inside the allocation, which ggml aligns and pads to 128 bytes.
    auto load_stage = [&](const int stage, const int iu0) {
        const unsigned int dst_stage = ggml_cuda_cvta_generic_to_shared(ring + stage*nbytes_stage);
#pragma unroll
        for (int c0 = 0; c0 < ncopies; c0 += warp_size) {
            const int c  = c0 + threadIdx.x;
            const int i  = c / nchunks;
            const int ic = c % nchunks;
            if (ncopies % warp_size == 0 || c < ncopies) {
                const int row = min(row0 + i, nrows_x - 1);
                const char * src = x + int64_t(row)*stride_row_x + int64_t(iu0)*unit_size;
                if constexpr (unaligned) {
                    const int off = (uintptr_t) src % 16;
                    if (16*ic < off + nunits_stage*unit_size) {
                        cp_async_cg_16<0>(dst_stage + i*sram_stride + 16*ic, src - off + 16*ic);
                    }
                } else {
                    cp_async_cg_16<0>(dst_stage + i*sram_stride + 16*ic, src + 16*ic);
                }
            }
        }
    };

    // offsets of the rows of this lane from their 16 byte boundary
    int off_row[2*nmt] = {0};
    if constexpr (unaligned) {
#pragma unroll
        for (int r = 0; r < 2*nmt; ++r) {
            const int row = min(row0 + int(threadIdx.x)/4 + 8*r, nrows_x - 1);
            off_row[r] = (uintptr_t) (x + int64_t(row)*stride_row_x) % 16;
        }
    }

    // Only weights, which no preceding kernel writes, are read before the PDL wait.
    if (!x_is_constant) {
        ggml_cuda_pdl_sync();
    }

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    // DGX Spark DRAM serves the short row pieces of the stage copies slowly, so the rows of the block are first prefetched into L2 in address order.
    {
        const int     tid       = threadIdx.y*warp_size + threadIdx.x;
        const int     nrows_blk = min(ksplit ? nrows : nwarps*nrows, nrows_x - row0_blk);
        const int64_t row_bytes = int64_t(nunits)*unit_size;
        const bool    contig    = stride_row_x == row_bytes;
        for (int ir = 0; ir < (contig ? 1 : nrows_blk); ++ir) {
            const uintptr_t p0 = (uintptr_t) (x + int64_t(row0_blk + ir)*stride_row_x);
            const uintptr_t p1 = p0 + (contig ? nrows_blk*row_bytes : row_bytes);
            for (uintptr_t p = (p0 & ~uintptr_t(127)) + 128*tid; p < p1; p += 128*nwarps*warp_size) {
                ggml_cuda_prefetch_l2((const void *) p);
            }
        }
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK

#pragma unroll
    for (int stage = 0; stage < nstages - 1; ++stage) {
        const int iu0 = iu_begin + stage*iu_step;
        if (iu0 < nunits) {
            load_stage(stage, iu0);
        }
        cp_async_commit_group();
    }
    if (x_is_constant) {
        ggml_cuda_pdl_sync();
    }

    auto load_y = [&](mmq_few_y_frag (&yf)[ncg], const int iu) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
            mmq_few_load_y(yf[c], y + (int64_t(iu)*ncg + c)*MMQ_FEW_Q8_1_INT4);
        }
    };

    // The next src1 fragments are loaded right after the MMAs that use the current ones, an earlier load is slower.
    mmq_few_y_frag yf[nunits_stage][ncg];
    if (iu_begin < nunits) {
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
            load_y(yf[k], iu_begin + k);
        }
    }

    float sum[nmt][ncg][4] = {{{0.0f}}};

    for (int iter = 0, iu0 = iu_begin; iu0 < nunits; ++iter, iu0 += iu_step) {
        const int iu_load = iu0 + (nstages - 1)*iu_step;
        if (iu_load < nunits) {
            load_stage((iter + nstages - 1) % nstages, iu_load);
        }
        cp_async_commit_group();

        const bool has_next = iu0 + iu_step < nunits;

        cp_async_wait_group<nstages - 1>();
        ggml_cuda_syncwarp();

        const char * xs = ring + (iter % nstages)*nbytes_stage;
        const int off_stage = unaligned ? (iu0*unit_size) % 16 : 0;
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
#pragma unroll
            for (int m = 0; m < nmt; ++m) {
                const char * const xr[2] = {
                    xs + (MMQ_FEW_NROWS*m + threadIdx.x/4 + 0)*sram_stride + (off_row[2*m + 0] + off_stage) % 16 + k*unit_size,
                    xs + (MMQ_FEW_NROWS*m + threadIdx.x/4 + 8)*sram_stride + (off_row[2*m + 1] + off_stage) % 16 + k*unit_size,
                };
                mmq_few_vec_dot<type, ncg>(xr, yf[k], sum[m]);
            }
            if (has_next) {
                load_y(yf[k], iu0 + iu_step + k);
            }
        }

        ggml_cuda_syncwarp(); // the stage is overwritten in the next iteration
    }
    cp_async_wait_all();

    ggml_cuda_pdl_lc();

    // reduce the sums over the warps as one flat array (when the warps split K)
    if (ksplit) {
        constexpr int nsum = nmt*ncg*4;
        float * sum_flat = &sum[0][0][0];

        float * buf_sum = (float *) data_mmq_few;
        __syncthreads();
        if (threadIdx.y > 0) {
#pragma unroll
            for (int i = 0; i < nsum; ++i) {
                buf_sum[((threadIdx.y - 1)*nsum + i)*warp_size + threadIdx.x] = sum_flat[i];
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
#pragma unroll
        for (int w = 0; w < nwarps - 1; ++w) {
#pragma unroll
            for (int i = 0; i < nsum; ++i) {
                sum_flat[i] += buf_sum[(w*nsum + i)*warp_size + threadIdx.x];
            }
        }
    }

#pragma unroll
    for (int m = 0; m < nmt; ++m) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + MMQ_FEW_NROWS*m + tile<16, 8, int>::get_i(l);
                const int j = c*MMQ_FEW_NCOLS + tile<16, 8, int>::get_j(l);
                if (i < nrows_x && j < ncols) {
                    if constexpr (has_ids) {
                        const int e = pair[2 + j]; // token*n_expert_used + slot
                        dst[int64_t(e / ncols_dst)*stride_channel_dst + int64_t(e % ncols_dst)*stride_col_dst + i] = sum[m][c][l];
                    } else {
                        dst[int64_t(j)*stride_col_dst + i] = sum[m][c][l];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y_ptr, dst, nunits, nrows_x, ncols_dst, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_dst, x_is_constant, pairs);
    NO_DEVICE_CODE;
#endif // defined(CP_ASYNC_AVAILABLE)
}

// ggml_is_constant (static in ggml.c) on the tensor that a view points to
static bool mmq_few_is_constant(const ggml_tensor * src0) {
    const ggml_tensor * x_root = src0->view_src ? src0->view_src : src0;
    return x_root->buffer && ggml_backend_buffer_get_usage(x_root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        !(x_root->flags & GGML_TENSOR_FLAG_PARAM);
}

// MUL_MAT_ID: grid y is the pairs, the kernel takes the expert from the pair and does not use the channel and sample ratios.
template <ggml_type type, int ncg, int nmt, bool has_ids = false>
static void launch_mul_mat_q_few(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream, const int * pairs = nullptr, const int npairs_max = 0) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const int nunits = ne00 / QK_K;
    GGML_ASSERT(nunits % mmq_few_nunits_stage(type, ncg, nmt) == 0);

    const int warp_size     = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nbytes_shared = mmq_few_nbytes_shared(type, ncg, nmt);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_few<type, ncg, nmt, has_ids>), nbytes_shared);

    const bool x_is_constant = mmq_few_is_constant(src0);

    const int  nrows_blk = (mmq_few_ksplit(has_ids, nunits) ? 1 : MMQ_FEW_NWARPS)*nmt*MMQ_FEW_NROWS;
    const dim3 block_nums((ne01 + nrows_blk - 1) / nrows_blk, has_ids ? npairs_max : ne12, has_ids ? 1 : ne13);
    const dim3 block_dims(warp_size, MMQ_FEW_NWARPS, 1);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_q_few<type, ncg, nmt, has_ids>, params,
        (const char *) src0->data, src1_q, (float *) dst->data,
        nunits, (int) ne01, (int) ne1, (int64_t) nb01, (int) (nb1/sizeof(float)),
        (int) (ne12/ne02), (int64_t) nb02, (int) (nb2/sizeof(float)),
        (int) (ne13/ne03), (int64_t) nb03, (int) (nb3/sizeof(float)), x_is_constant, pairs);
}

static int mmq_few_ncg(const int64_t ncols) {
    return (ncols + MMQ_FEW_NCOLS - 1) / MMQ_FEW_NCOLS;
}

// With 2 column groups, 32-row tiles halve the src1 loads per src0 byte. With 3 column groups they spill registers.
static int mmq_few_nmt(const ggml_type type, const int ncg, const int64_t nrows_x) {
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    return mmq_few_tall_supported(type) && ncg == 2 && nrows_x >= int64_t(nsm)*2*MMQ_FEW_NROWS ? 2 : 1;
}

template <ggml_type type>
static void launch_mul_mat_q_few_tall(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream) {
    if constexpr (mmq_few_tall_supported(type)) {
        if (mmq_few_nmt(type, 2, src0->ne[1]) == 2) {
            launch_mul_mat_q_few<type, 2, 2>(src0, src1, dst, src1_q, stream);
            return;
        }
    }
    launch_mul_mat_q_few<type, 2, 1>(src0, src1, dst, src1_q, stream);
}

template <ggml_type type>
static void launch_mul_mat_q_few_switch_ncg(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, const int ncg, cudaStream_t stream, const int * pairs = nullptr, const int npairs_max = 0) {
    if (pairs) {
        launch_mul_mat_q_few<type, 1, 1, true>(src0, src1, dst, src1_q, stream, pairs, npairs_max);
        return;
    }
    switch (ncg) {
        case 1:
            launch_mul_mat_q_few<type, 1, 1>(src0, src1, dst, src1_q, stream);
            break;
        case 2:
            launch_mul_mat_q_few_tall<type>(src0, src1, dst, src1_q, stream);
            break;
        case 3:
            launch_mul_mat_q_few<type, 3, 1>(src0, src1, dst, src1_q, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(ne11 <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS);
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);

    cudaStream_t stream = ctx.stream();

    const int nunits = ne00 / QK_K;
    const int ncg    = mmq_few_ncg(ne11);

    static const bool log_each = [] {
        const char * e = getenv("GGML_CUDA_MMQ_FEW_LOG");
        return e != nullptr && atoi(e) != 0;
    }();
    if (log_each) {
        GGML_LOG_INFO("few-row MMQ: mmq-few %s %lld columns 1 matrices rows %lld K %lld\n", ggml_type_name(src0->type), (long long) ne11,
            (long long) ne01, (long long) ne00);
    }
    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("few-row MMQ: few-row MMQ (GGML_CUDA_MMQ_FEW=%d (-1 = AUTO), up to %d columns; =0 off) takes dense products (first: %s, %lld columns, K %lld)\n",
            mmq_few_env().mode, mmq_few_env().max_cols, ggml_type_name(src0->type), (long long) ne11, (long long) ne00);
    });

    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool(), int64_t(ne12)*ne13*nunits*ncg*MMQ_FEW_Q8_1_INT4);
    quantize_mmq_few_q8_1_cuda((const float *) src1->data, src1_q.get(), mmq_few_ds_layout(src0->type), ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
        ne11, ne12, ne13, stream);

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q8_0>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q3_K>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q4_K>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q5_K>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q6_K>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ3_S>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ4_NL>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ4_XS>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        default:
            GGML_ABORT("unsupported type for few-row MMQ");
    }
}

// Min. number of src1 columns for the few-row MMQ, 0 if it is not used for this type, GPU and row length.
static int ggml_cuda_mmq_few_min_ncols(const ggml_type type, const int cc, const int64_t ne00, const bool has_ids = false) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return 0;
#endif // GGML_CUDA_FORCE_CUBLAS

    // NVIDIA Ampere to Blackwell
    if (!ampere_mma_available(cc) || cc >= GGML_CUDA_CC_RUBIN) {
        return 0;
    }
    // on DGX Spark the kernel needs its L2 prefetch, which is only in the DGX Spark device code
    if (cc == GGML_CUDA_CC_DGX_SPARK && ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_DGX_SPARK) {
        return 0;
    }
    // with fewer than 8 units of 256 values per row the warps idle, MMVQ and MMQ are better there (MUL_MAT_ID splits the rows instead)
    if (mmq_few_unit_size(type) == 0 || (!has_ids && ne00 < 8*QK_K)) {
        return 0;
    }
    if (cc == GGML_CUDA_CC_DGX_SPARK) { // tuned on DGX Spark GB10
        return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL ? 8 : 2;
    }
    switch (type) { // all GPUs except DGX Spark, tuned on RTX 5090
        case GGML_TYPE_Q3_K:   return 3;
        case GGML_TYPE_Q5_K:   return 3;
        case GGML_TYPE_Q4_K:   return 4;
        case GGML_TYPE_IQ4_NL: return 4;
        case GGML_TYPE_Q6_K:   return 4;
        case GGML_TYPE_Q8_0:   return 5;
        case GGML_TYPE_IQ3_S:  return 5;
        case GGML_TYPE_IQ4_XS: return 6;
        default:               return 0;
    }
}

// AUTO table (min columns, 0 = off), from Qwen3.8-27B at 4 bits + MTP verify passes:
// - cc 8.0 (A100 / A30): K-quants from 2 columns (micro-bench 5 columns: Q4_K 53%, Q5_K 57% of the batched mmvq; Qwen3.8 dense MTP +41% chat,
//   +39% 8k on an A100 40 GB), Q6_K / IQ4_XS from 3, Q8_0 from 5 (from 2 it moved a 4-bit MoE's Q8_0 verify text: -5% chat by acceptance);
// - cc 8.9 (Ada, L4): the upstream table (L4 Qwen3.8 dense +8.5% chat, +5.4% 8k);
// - cc 12.0 (RTX PRO 6000 / RTX 50xx): off (GGML_CUDA_MMQ_FEW=1 = the upstream table). The table gained the Qwen3.8 dense +2.1% chat
//   (flat 8k) but cost Flash-Next, whose dense products are K 2560 Q6_K / Q8_0 / IQ4_XS: full card MTP chat 338.3 -> 325.1 tok/s at
//   the same text, MTP 8k 278.9 -> 251.2 (other text, +8.7% passes), DFlash2 48 GB chat 255.4 -> 248.8, and its verify rows broke the
//   row invariance of test-backend-ops VROW_INV (22 of 96);
// - other devices: off (not measured); GGML_CUDA_MMQ_FEW=1 / N turns it on.
static int ggml_cuda_mmq_few_auto_min_ncols(const ggml_type type, const int cc, const int64_t ne00) {
    const int up = ggml_cuda_mmq_few_min_ncols(type, cc, ne00);
    if (up == 0 || !GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return 0;
    }
    if (cc == GGML_CUDA_CC_AMPERE) {
        switch (type) {
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:   return 2;
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ4_XS: return 3;
            case GGML_TYPE_Q8_0:   return 5;
            default:               return up;
        }
    }
    if (cc == GGML_CUDA_CC_ADA_LOVELACE) {
        return up;
    }
    return 0;
}

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const int cc) {
    const mmq_few_env_t & env = mmq_few_env();
    if (env.mode == 0) {
        return false;
    }
    int min_ncols = env.mode < 0 ? ggml_cuda_mmq_few_auto_min_ncols(src0->type, cc, src0->ne[0]) :
                                   ggml_cuda_mmq_few_min_ncols(src0->type, cc, src0->ne[0]);
    if (min_ncols != 0 && env.mode >= 2) {
        min_ncols = env.mode;
    }
    if (min_ncols == 0 || src1->ne[1] < min_ncols || src1->ne[1] > env.max_cols) {
        return false;
    }
    // the row must hold a whole number of ring stages of the launched kernel
    const int ncg          = mmq_few_ncg(src1->ne[1]);
    const int nmt          = mmq_few_nmt(src0->type, ncg, src0->ne[1]);
    const int nunits_stage = mmq_few_nunits_stage(src0->type, ncg, nmt);
    if (src0->ne[0] % (nunits_stage*QK_K) != 0) {
        return false;
    }
    // unaligned data needs the alignment of the decoder loads
    const int x_alignment = mmq_few_unaligned(src0->type, nunits_stage) ? (ggml_type_size(src0->type) % 4 == 0 ? 4 : 2) : 16;
    return ggml_cuda_is_aligned(src0, x_alignment) && ggml_cuda_is_aligned(src1, 16);
}

// GGML_CUDA_MMQ_FEW_MOE: few-row MMQ for MUL_MAT_ID (one column group of up to 8 tokens per expert and CUDA block) from the first
// token count above the MoE mat-vec kernels (ggml_cuda_mmvq_mmid_max_batch + 1) up to GGML_CUDA_MMQ_FEW_MOE_MAX (default 24) tokens,
// instead of MMQ. Unset / 0 = off (MMQ as before, bit for bit): opt-in until its accuracy is gated where it engages (multi-slot
// decode). -1 = AUTO: NVIDIA cc 8.0 .. 8.9 and 10.x; off on cc 12.x, on other devices and on HIP / MUSA (no kernel).
// 1 = on every device the kernel supports (Ampere and newer).
// Not bit-identical to MMQ (another int8 tiling and float summation order of each output).
struct mmq_few_moe_env_t { int mode; int max_cols; };
static const mmq_few_moe_env_t & mmq_few_moe_env() {
    static const mmq_few_moe_env_t e = [] {
        mmq_few_moe_env_t r = { 0, MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS };
        if (const char * v = getenv("GGML_CUDA_MMQ_FEW_MOE")) {
            r.mode = atoi(v);
        }
        if (const char * v = getenv("GGML_CUDA_MMQ_FEW_MOE_MAX")) {
            r.max_cols = std::min(MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS, std::max(2, atoi(v)));
        }
        return r;
    }();
    return e;
}

bool ggml_cuda_should_use_mmq_few_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const int cc) {
    const mmq_few_moe_env_t & env = mmq_few_moe_env();
    if (env.mode == 0 || ids == nullptr || ids->type != GGML_TYPE_I32 || ids->nb[0] != ggml_type_size(ids->type)) {
        return false;
    }
    if (env.mode < 0) {
        const bool auto_cc = GGML_CUDA_CC_IS_NVIDIA(cc) && ((cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_HOPPER) ||
            (cc >= 1000 && cc < 1100));
        if (!auto_cc) {
            return false;
        }
    }
    if (src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float) || src0->nb[0] != ggml_type_size(src0->type) ||
            src0->ne[0] % QK_K != 0 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t ntokens = ids->ne[1];
    if (ggml_cuda_mmq_few_min_ncols(src0->type, cc, src0->ne[0], /*has_ids =*/ true) == 0 ||
            ntokens < ggml_cuda_mmvq_mmid_max_batch(src0->type, cc) + 1 || ntokens > env.max_cols) {
        return false;
    }
    // the pairs are built in shared memory
    if ((2*src0->ne[2] + 1)*sizeof(int) > ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        return false;
    }
    // the row must hold a whole number of ring stages of the launched kernel
    const int nunits_stage = mmq_few_nunits_stage(src0->type, 1, 1);
    if (src0->ne[0] % (nunits_stage*QK_K) != 0) {
        return false;
    }
    const int x_alignment = mmq_few_unaligned(src0->type, nunits_stage) ? (ggml_type_size(src0->type) % 4 == 0 ? 4 : 2) : 16;
    return ggml_cuda_is_aligned(src0, x_alignment) && ggml_cuda_is_aligned(src1, 16);
}

void ggml_cuda_mul_mat_q_few_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne1 == ids->ne[0] && ne13 == 1 && ne3 == 1);

    cudaStream_t stream = ctx.stream();

    const int     nunits        = ne00 / QK_K;
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t nentries      = n_tokens*n_expert_used;
    const int64_t nused         = std::min(ne02, nentries);
    // an expert with c > 0 entries (at most one per token) has ceil(c/MMQ_FEW_NCOLS) pairs
    const int npairs_max = (int) std::min(nused + (nentries - nused)/MMQ_FEW_NCOLS, ne02*((n_tokens + MMQ_FEW_NCOLS - 1) / MMQ_FEW_NCOLS));
    const int si1        = ids->nb[1] / sizeof(int32_t);

    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("few-row MMQ: few-row MMQ for MUL_MAT_ID engaged (GGML_CUDA_MMQ_FEW_MOE=%d (-1 = AUTO), up to %d tokens; =0 off; "
            "first: %s, %lld tokens, %lld experts, K %lld)\n", mmq_few_moe_env().mode, mmq_few_moe_env().max_cols, ggml_type_name(src0->type),
            (long long) n_tokens, (long long) ne02, (long long) ne00);
    });

    // the kernel writes the rows of the slots an expert owns, the skipped slots (id -1) are zeroed first
    if (ggml_cuda_mm_ids_may_skip(ids)) {
        ggml_cuda_launch_mm_ids_zero_dst((const int32_t *) ids->data, (float *) dst->data, (int) n_tokens, (int) n_expert_used, (int) ne0, si1,
            nb1/sizeof(float), nb2/sizeof(float), stream);
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_pool_alloc<int>  pairs(ctx.pool(), int64_t(npairs_max)*MMQ_FEW_PAIR_INTS);
    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool(), int64_t(npairs_max)*nunits*MMQ_FEW_Q8_1_INT4);
    quantize_mmq_few_id_q8_1_cuda((const float *) src1->data, (const int32_t *) ids->data, src1_q.get(), pairs.get(), mmq_few_ds_layout(src0->type),
        ne00, nb11/sizeof(float), nb12/sizeof(float), ne11, ne02, n_expert_used, n_tokens, si1, npairs_max, stream);

    switch (src0->type) {
#define MMQ_FEW_ID_CASE(T) case T: launch_mul_mat_q_few_switch_ncg<T>(src0, src1, dst, src1_q.get(), 1, stream, pairs.get(), npairs_max); break;
        MMQ_FEW_ID_CASE(GGML_TYPE_Q8_0)
        MMQ_FEW_ID_CASE(GGML_TYPE_Q3_K)
        MMQ_FEW_ID_CASE(GGML_TYPE_Q4_K)
        MMQ_FEW_ID_CASE(GGML_TYPE_Q5_K)
        MMQ_FEW_ID_CASE(GGML_TYPE_Q6_K)
        MMQ_FEW_ID_CASE(GGML_TYPE_IQ3_S)
        MMQ_FEW_ID_CASE(GGML_TYPE_IQ4_NL)
        MMQ_FEW_ID_CASE(GGML_TYPE_IQ4_XS)
#undef MMQ_FEW_ID_CASE
        default:
            GGML_ABORT("unsupported type for few-row MMQ");
    }
}

// the src1 quantization of the few-row MMQ
// Stores the 4 values xi of a thread, values [iu, iu + 4) of column g in a unit of 256 values, in the column group yg of the unit.
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void quantize_mmq_few_store(const float4 xi, int4 * yg, const int iu, const int g, const bool exact) {
    // the 32 values of a sub-block are held by 8 threads
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    amax = warp_reduce_max<8>(amax);

    const float d_inv = ggml_cuda_q8_1_d_inv(amax, exact, ds_layout != MMQ_Q8_1_DS_LAYOUT_D4);
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    // the sum of the quantized values (times the stored scale when exact); the int warp_reduce_sum reduces the whole warp on Ampere+
    [[maybe_unused]] float sum = 0.0f;
    if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = (exact ? __half2float(__float2half(d)) : d)*warp_reduce_sum<8>(float(q.x + q.y + q.z + q.w));
    }

    const int s = iu / QK8_1; // sub-block
    const int k = iu % QK8_1; // first value in sub-block
    ((int *) (yg + (s/2)*WARP_SIZE + 4*g + (k % 16)/4))[2*(s % 2) + k/16] = *(const int *) &q;
    if (k == 0) {
        int4 * ds = yg + 4*WARP_SIZE + 4*(g/2) + s/2;
        if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
            ((float *) ds)[2*(s % 2) + g % 2] = d;
        } else {
            ((half2 *) ds)[2*(s % 2) + g % 2] = make_half2(d, sum);
        }
    }
}

// Each thread quantizes 4 values of one column, see quantize.cuh for the layout.
template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_few_q8_1(
        const float * x_ptr, void * vy_ptr, const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int ne1, const int ne2, const bool exact) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    int4        * GGML_CUDA_RESTRICT vy = (int4 *) vy_ptr;
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne00) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const float4  xi = (int) blockIdx.x < ne1 ? *(const float4 *) (x + i3*s03 + i2*s02 + blockIdx.x*s01 + i0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const int g = blockIdx.x % MMQ_FEW_NCOLS; // column in group
    int4 * yg = vy + ((blockIdx.z*(ne00/QK_K) + i0/QK_K)*(gridDim.x/MMQ_FEW_NCOLS) + blockIdx.x/MMQ_FEW_NCOLS)*MMQ_FEW_Q8_1_INT4;
    quantize_mmq_few_store<ds_layout>(xi, yg, i0 % QK_K, g, exact);
}

void quantize_mmq_few_q8_1_cuda(
        const float * x, void * vy, const mmq_q8_1_ds_layout ds_layout, const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);

    const int64_t block_num_y = (ne00 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(GGML_PAD(ne1, MMQ_FEW_NCOLS), block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, block_size, 0, stream);
    switch (ds_layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            ggml_cuda_kernel_launch(quantize_mmq_few_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>, launch_params,
                x, vy, ne00, s01, s02, s03, (int) ne1, (int) ne2, ggml_cuda_q8_1_exact());
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            ggml_cuda_kernel_launch(quantize_mmq_few_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>, launch_params,
                x, vy, ne00, s01, s02, s03, (int) ne1, (int) ne2, ggml_cuda_q8_1_exact());
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// MUL_MAT_ID: block x is the (token, slot) entry, ids are as in mm_ids_helper. Each block finds the pair of its entry in shared memory,
// block (0, 0) also writes all pairs. Entry rank r = number of earlier tokens with the same expert, pair = first pair of the expert +
// r/MMQ_FEW_NCOLS, column = r % MMQ_FEW_NCOLS. An id outside [0, n_experts) (a skipped slot) belongs to no pair.
template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_few_id_q8_1(
        const float * x_ptr, const int32_t * ids, void * vy_ptr, int * pairs, const int64_t ne00, const int64_t s11, const int64_t s12,
        const int ne11, const int n_experts, const int n_expert_used, const int64_t si1, const int npairs_max, const bool exact) {
    // per expert: bit t set if token t uses it and the index of its first pair; then the total number of pairs
    extern __shared__ unsigned int data_mmq_few_id[];
    unsigned int * tokens = data_mmq_few_id;
    int          * pair0  = (int *) (tokens + n_experts);
    int          & npairs = pair0[n_experts];

    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    int4        * GGML_CUDA_RESTRICT vy = (int4 *) vy_ptr;
    ggml_cuda_pdl_sync();

    const int nentries = gridDim.x;
    auto expert_of = [&](const int i) {
        const int e = ids[(i / n_expert_used)*si1 + i % n_expert_used];
        return e >= 0 && e < n_experts ? e : -1;
    };

    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        tokens[e] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nentries; i += blockDim.x) {
        const int e = expert_of(i);
        if (e >= 0) {
            atomicOr(&tokens[e], 1u << (i / n_expert_used));
        }
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        const int per_lane = (n_experts + WARP_SIZE - 1) / WARP_SIZE;
        const int e0       = min(int(threadIdx.x)*per_lane, n_experts);
        const int e1       = min(e0 + per_lane, n_experts);
        int n = 0;
        for (int e = e0; e < e1; ++e) {
            n += (__popc(tokens[e]) + MMQ_FEW_NCOLS - 1) / MMQ_FEW_NCOLS;
        }
        int p = warp_prefix_inclusive_sum<int>(n) - n;
        for (int e = e0; e < e1; ++e) {
            pair0[e] = p;
            p += (__popc(tokens[e]) + MMQ_FEW_NCOLS - 1) / MMQ_FEW_NCOLS;
        }
        if (threadIdx.x == WARP_SIZE - 1) {
            npairs = p;
        }
    }
    __syncthreads();

    auto rank = [&](const int i, const int e) {
        return __popc(tokens[e] & ((1u << (i / n_expert_used)) - 1u));
    };

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i = threadIdx.x; i < nentries; i += blockDim.x) {
            const int e = expert_of(i);
            if (e < 0) {
                continue;
            }
            const int r = rank(i, e);
            int * pair = pairs + (pair0[e] + r/MMQ_FEW_NCOLS)*MMQ_FEW_PAIR_INTS;
            pair[2 + r % MMQ_FEW_NCOLS] = i;
            if (r % MMQ_FEW_NCOLS == 0) {
                pair[0] = e;
                pair[1] = min(__popc(tokens[e]) - r, MMQ_FEW_NCOLS);
            }
        }
        for (int p = npairs + threadIdx.x; p < npairs_max; p += blockDim.x) {
            pairs[p*MMQ_FEW_PAIR_INTS + 1] = 0;
        }
    }

    const int     e  = expert_of(blockIdx.x);
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if (e < 0 || i0 >= ne00) {
        return;
    }
    const int    it = blockIdx.x / n_expert_used;
    const int    is = blockIdx.x % n_expert_used;
    const float4 xi = *(const float4 *) (x + it*s12 + (is % ne11)*s11 + i0);

    const int r = rank(blockIdx.x, e);
    int4 * yg = vy + (int64_t(pair0[e] + r/MMQ_FEW_NCOLS)*(ne00/QK_K) + i0/QK_K)*MMQ_FEW_Q8_1_INT4;
    quantize_mmq_few_store<ds_layout>(xi, yg, i0 % QK_K, r % MMQ_FEW_NCOLS, exact);
}

void quantize_mmq_few_id_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, int * pairs, const mmq_q8_1_ds_layout ds_layout, const int64_t ne00, const int64_t s11,
        const int64_t s12, const int64_t ne11, const int64_t n_experts, const int64_t n_expert_used, const int64_t n_tokens, const int64_t si1,
        const int64_t npairs_max, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(n_tokens <= 32); // one bit per token

    const int64_t block_num_y   = (ne00 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const size_t  nbytes_shared = (2*n_experts + 1)*sizeof(int);
    GGML_ASSERT(nbytes_shared <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    const dim3 num_blocks(n_tokens*n_expert_used, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, block_size, nbytes_shared, stream);
    switch (ds_layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            ggml_cuda_kernel_launch(quantize_mmq_few_id_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>, launch_params,
                x, ids, vy, pairs, ne00, s11, s12, (int) ne11, (int) n_experts, (int) n_expert_used, si1, (int) npairs_max, ggml_cuda_q8_1_exact());
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            ggml_cuda_kernel_launch(quantize_mmq_few_id_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>, launch_params,
                x, ids, vy, pairs, ne00, s11, s12, (int) ne11, (int) n_experts, (int) n_expert_used, si1, (int) npairs_max, ggml_cuda_q8_1_exact());
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    GGML_ABORT("few-row MMQ is not supported on HIP or MUSA");
}

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const int cc) {
    GGML_UNUSED_VARS(src0, src1, cc);
    return false;
}

void ggml_cuda_mul_mat_q_few_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst);
    GGML_ABORT("few-row MMQ is not supported on HIP or MUSA");
}

bool ggml_cuda_should_use_mmq_few_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const int cc) {
    GGML_UNUSED_VARS(src0, src1, ids, cc);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
