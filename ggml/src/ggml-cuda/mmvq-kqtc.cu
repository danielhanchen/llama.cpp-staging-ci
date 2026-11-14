// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "mmvq-kqtc.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// k19-kqtc (see mmvq-kqtc.cuh). Layout of one launch:
// - a block has NW warps and owns 16*T output rows of one matrix (T row tiles of 16); its warps split the 256-value K units (warp w takes
//   units w, w + NW, ...) and the per-warp results are summed in shared memory in warp order;
// - per unit, a warp copies the q8_1 activations of its unit (<= 8 columns x 8 blocks of 32) into its own shared memory with coalesced
//   loads, ints permuted so that the B fragments are conflict-free reads, and the q8_1 scales as floats [block][column];
// - lane (g = lane/4, t = lane%4) reads the weights of rows g and g+8 of each tile from global memory (the 4 lanes of a row read
//   adjacent bytes), decodes them to int8 A fragments and runs m16n8k32 (Q6_K: m16n8k16, 16-value scales) int8 MMAs for all 8 columns;
//   C holds rows g / g+8 x columns 2t / 2t+1. Turing runs the same products as m8n8k16 pairs (exact integer sums either way).
// - The f32 part mirrors the per-block math of the mmvq vec_dot functions (Q4_K / Q5_K mins from the integer sums of the q8 values,
//   via an MMA with an all-ones A), in another order.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define KQTC_ENABLED 1
#endif

static constexpr int KQTC_SROW = 68; // ints per staged column: 8 blocks x 8 ints + 4 pad (column stride 68 = 4 banks apart)

struct kqtc_args {
    const char * x[GGML_CUDA_MMVQ_MULTI_MAX];
    float      * dst[GGML_CUDA_MMVQ_MULTI_MAX];
    int          nrows[GGML_CUDA_MMVQ_MULTI_MAX];
    int          block0[GGML_CUDA_MMVQ_MULTI_MAX];
    int          n_mat;
    const int  * y;              // q8_1 activations as ints (a block_q8_1 is 9 ints)
    int64_t      stride_col_y;   // in block_q8_1
    int64_t      stride_col_dst; // in floats
    int64_t      stride_row_x;   // in bytes
    int          nunits;         // 256-value units along K
    int          nb32;           // 32-value blocks along K (Q8_0 tail)
    int          ncols;
};

static __device__ __forceinline__ void kqtc_mma_k32(int (&c)[4], const int a0, const int a1, const int a2, const int a3, const int b0, const int b1) {
#if defined(KQTC_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(KQTC_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    // rows g (a0 / a2) and g+8 (a1 / a3), k 0..15 (b0) and 16..31 (b1)
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(b0));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(c[0]), "+r"(c[1]) : "r"(a2), "r"(b1));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(c[2]), "+r"(c[3]) : "r"(a1), "r"(b0));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(c[2]), "+r"(c[3]) : "r"(a3), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
#endif
}

static __device__ __forceinline__ void kqtc_mma_k16(int (&c)[4], const int a0, const int a1, const int b0) {
#if defined(KQTC_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
#elif defined(KQTC_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(b0));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(c[2]), "+r"(c[3]) : "r"(a1), "r"(b0));
#else
    GGML_UNUSED_VARS(c, a0, a1, b0);
#endif
}

static __device__ __forceinline__ int kqtc_byte(const int v, const int k) {
    return (v >> (8*k)) & 0xFF;
}

// Q4_K / Q5_K take the B ints of a q8_1 block as (2t, 2t+1) (lane t reads 8 adjacent weight bytes); the others as (t, t+4)
template <ggml_type type>
static constexpr __device__ bool kqtc_pair_ints() {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
}

// raw weights of one row and unit, as loaded (before the B staging, so the loads of both are in flight together)
template <ggml_type type> struct kqtc_raw;

template <> struct kqtc_raw<GGML_TYPE_Q4_K> {
    int4 head;  // dm, scales[12]
    int2 q[4];  // qs bytes 32c + 8t .. 32c + 8t + 7
    __device__ __forceinline__ void load(const char * row, const int u, const int t, const int /*nb32*/) {
        const char * b = row + (size_t) u*sizeof(block_q4_K);
        head = *(const int4 *) b;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            q[c] = *(const int2 *) (b + offsetof(block_q4_K, qs) + 32*c + 8*t);
        }
    }
};

template <> struct kqtc_raw<GGML_TYPE_Q5_K> {
    int4 head;  // dm, scales[12]
    int2 q[4];  // qs bytes 32c + 8t ..
    int2 qh;    // qh bytes 8t .. 8t + 7
    __device__ __forceinline__ void load(const char * row, const int u, const int t, const int /*nb32*/) {
        const char * b = row + (size_t) u*sizeof(block_q5_K);
        head = *(const int4 *) b;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            q[c] = *(const int2 *) (b + offsetof(block_q5_K, qs) + 32*c + 8*t);
        }
        qh = *(const int2 *) (b + offsetof(block_q5_K, qh) + 8*t);
    }
};

template <> struct kqtc_raw<GGML_TYPE_Q6_K> {
    int ql[2][2][2]; // [h][u][lo / hi 32 bytes]
    int qh[2][2];    // [h][u]
    int sc[4];       // scales[16]
    float d;
    __device__ __forceinline__ void load(const char * row, const int u0, const int t, const int /*nb32*/) {
        const block_q6_K * b = (const block_q6_K *) (row + (size_t) u0*sizeof(block_q6_K));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                ql[h][u][0] = get_int_b2(b->ql, 16*h + 4*u + t);
                ql[h][u][1] = get_int_b2(b->ql, 16*h + 8 + 4*u + t);
                qh[h][u]    = get_int_b2(b->qh, 8*h + 4*u + t);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc[i] = get_int_b2(b->scales, i);
        }
        d = __half2float(b->d);
    }
};

template <> struct kqtc_raw<GGML_TYPE_IQ4_XS> {
    int2 head;  // d | scales_h << 16, scales_l
    int  q[8];  // qs bytes 16s + 4t .. 16s + 4t + 3
    __device__ __forceinline__ void load(const char * row, const int u, const int t, const int /*nb32*/) {
        const char * b = row + (size_t) u*sizeof(block_iq4_xs);
        head = *(const int2 *) b;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            q[s] = *(const int *) (b + offsetof(block_iq4_xs, qs) + 16*s + 4*t);
        }
    }
};

template <> struct kqtc_raw<GGML_TYPE_Q8_0> {
    int   q[8][2];  // block s: ints t and t+4
    float d[8];
    __device__ __forceinline__ void load(const char * row, const int u, const int t, const int nb32) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int jb = 8*u + s;
            if (jb < nb32) {
                const block_q8_0 * b = (const block_q8_0 *) (row + (size_t) jb*sizeof(block_q8_0));
                q[s][0] = get_int_b2(b->qs, t);
                q[s][1] = get_int_b2(b->qs, t + 4);
                d[s]    = __half2float(b->d);
            } else {
                q[s][0] = 0;
                q[s][1] = 0;
                d[s]    = 0.0f;
            }
        }
    }
};

// per unit, the B fragments of column g and the q8_1 scales of columns 2t / 2t+1, shared by the tiles of a warp
struct kqtc_b {
    int    b[8][2];
    float2 d8[8];
    int    sum[8][2]; // Q4_K / Q5_K: integer sums of the q8 values of block s, columns 2t / 2t+1
};

// rows g (r0) and g+8 (r1) of one tile: acc[0..3] = (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)
template <ggml_type type>
static __device__ __forceinline__ void kqtc_tile(const kqtc_raw<type> & r0, const kqtc_raw<type> & r1, const kqtc_b & B, float (&acc)[4]) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const int u00 = r0.head.y, u01 = r0.head.z, u02 = r0.head.w;
        const int u10 = r1.head.y, u11 = r1.head.z, u12 = r1.head.w;
        const int sc0[2] = { u00 & 0x3F3F3F3F, (u02 & 0x0F0F0F0F) | ((u00 >> 2) & 0x30303030) };
        const int mn0[2] = { u01 & 0x3F3F3F3F, ((u02 >> 4) & 0x0F0F0F0F) | ((u01 >> 2) & 0x30303030) };
        const int sc1[2] = { u10 & 0x3F3F3F3F, (u12 & 0x0F0F0F0F) | ((u10 >> 2) & 0x30303030) };
        const int mn1[2] = { u11 & 0x3F3F3F3F, ((u12 >> 4) & 0x0F0F0F0F) | ((u11 >> 2) & 0x30303030) };
        float sd[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float sm[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int sh = 4*(s & 1);
            int a0 = (r0.q[s/2].x >> sh) & 0x0F0F0F0F;
            int a2 = (r0.q[s/2].y >> sh) & 0x0F0F0F0F;
            int a1 = (r1.q[s/2].x >> sh) & 0x0F0F0F0F;
            int a3 = (r1.q[s/2].y >> sh) & 0x0F0F0F0F;
            if constexpr (type == GGML_TYPE_Q5_K) {
                a0 |= ((r0.qh.x >> s) & 0x01010101) << 4;
                a2 |= ((r0.qh.y >> s) & 0x01010101) << 4;
                a1 |= ((r1.qh.x >> s) & 0x01010101) << 4;
                a3 |= ((r1.qh.y >> s) & 0x01010101) << 4;
            }
            int c[4] = { 0, 0, 0, 0 };
            kqtc_mma_k32(c, a0, a1, a2, a3, B.b[s][0], B.b[s][1]);
            const int s0 = kqtc_byte(sc0[s/4], s % 4), m0 = kqtc_byte(mn0[s/4], s % 4);
            const int s1 = kqtc_byte(sc1[s/4], s % 4), m1 = kqtc_byte(mn1[s/4], s % 4);
            sd[0] += B.d8[s].x * (c[0]*s0);
            sd[1] += B.d8[s].y * (c[1]*s0);
            sd[2] += B.d8[s].x * (c[2]*s1);
            sd[3] += B.d8[s].y * (c[3]*s1);
            sm[0] += B.d8[s].x * (B.sum[s][0]*m0);
            sm[1] += B.d8[s].y * (B.sum[s][1]*m0);
            sm[2] += B.d8[s].x * (B.sum[s][0]*m1);
            sm[3] += B.d8[s].y * (B.sum[s][1]*m1);
        }
        const float2 dm0 = __half22float2(*(const half2 *) &r0.head.x);
        const float2 dm1 = __half22float2(*(const half2 *) &r1.head.x);
        acc[0] += dm0.x*sd[0] - dm0.y*sm[0];
        acc[1] += dm0.x*sd[1] - dm0.y*sm[1];
        acc[2] += dm1.x*sd[2] - dm1.y*sm[2];
        acc[3] += dm1.x*sd[3] - dm1.y*sm[3];
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        float sf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int blk = 4*h + j; // q8_1 block of these 32 values
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int sh  = 4*(j >> 1);
                    const int a0  = __vsub4(((r0.ql[h][u][j & 1] >> sh) & 0x0F0F0F0F) | (((r0.qh[h][u] >> (2*j)) & 0x03030303) << 4), 0x20202020);
                    const int a1  = __vsub4(((r1.ql[h][u][j & 1] >> sh) & 0x0F0F0F0F) | (((r1.qh[h][u] >> (2*j)) & 0x03030303) << 4), 0x20202020);
                    int c[4] = { 0, 0, 0, 0 };
                    kqtc_mma_k16(c, a0, a1, B.b[blk][u]);
                    const int is = 8*h + 2*j + u; // scale of these 16 values
                    const int s0 = (int) (int8_t) kqtc_byte(r0.sc[is/4], is % 4);
                    const int s1 = (int) (int8_t) kqtc_byte(r1.sc[is/4], is % 4);
                    sf[0] += B.d8[blk].x * (c[0]*s0);
                    sf[1] += B.d8[blk].y * (c[1]*s0);
                    sf[2] += B.d8[blk].x * (c[2]*s1);
                    sf[3] += B.d8[blk].y * (c[3]*s1);
                }
            }
        }
        acc[0] += r0.d*sf[0];
        acc[1] += r0.d*sf[1];
        acc[2] += r1.d*sf[2];
        acc[3] += r1.d*sf[3];
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const int sh0 = r0.head.x >> 16, sh1 = r1.head.x >> 16; // scales_h
        float sf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int2 v0 = get_int_from_table_16(r0.q[s], kvalues_iq4nl);
            const int2 v1 = get_int_from_table_16(r1.q[s], kvalues_iq4nl);
            int c[4] = { 0, 0, 0, 0 };
            kqtc_mma_k32(c, v0.x, v1.x, v0.y, v1.y, B.b[s][0], B.b[s][1]);
            const int ls0 = (((r0.head.y >> (8*(s/2) + 4*(s & 1))) & 0x0F) | (((sh0 >> (2*s)) & 0x03) << 4)) - 32;
            const int ls1 = (((r1.head.y >> (8*(s/2) + 4*(s & 1))) & 0x0F) | (((sh1 >> (2*s)) & 0x03) << 4)) - 32;
            sf[0] += B.d8[s].x * (c[0]*ls0);
            sf[1] += B.d8[s].y * (c[1]*ls0);
            sf[2] += B.d8[s].x * (c[2]*ls1);
            sf[3] += B.d8[s].y * (c[3]*ls1);
        }
        const float d0 = __half2float(__ushort_as_half((unsigned short) (r0.head.x & 0xFFFF)));
        const float d1 = __half2float(__ushort_as_half((unsigned short) (r1.head.x & 0xFFFF)));
        acc[0] += d0*sf[0];
        acc[1] += d0*sf[1];
        acc[2] += d1*sf[2];
        acc[3] += d1*sf[3];
    } else if constexpr (type == GGML_TYPE_Q8_0) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            int c[4] = { 0, 0, 0, 0 };
            kqtc_mma_k32(c, r0.q[s][0], r1.q[s][0], r0.q[s][1], r1.q[s][1], B.b[s][0], B.b[s][1]);
            acc[0] += (r0.d[s]*B.d8[s].x) * (float) c[0];
            acc[1] += (r0.d[s]*B.d8[s].y) * (float) c[1];
            acc[2] += (r1.d[s]*B.d8[s].x) * (float) c[2];
            acc[3] += (r1.d[s]*B.d8[s].y) * (float) c[3];
        }
    }
}

template <ggml_type type, int nw, int ntiles>
static __global__ void __launch_bounds__(nw*WARP_SIZE, 16/nw) mul_mat_vec_q_kqtc(const kqtc_args args) {
#if defined(KQTC_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    __shared__ int   s_y[nw][8*KQTC_SROW];
    __shared__ float s_d[nw][8*8];               // [block][column]
    __shared__ float s_red[nw][ntiles*16*8];     // [row][column]

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    int m = 0;
#pragma unroll
    for (int i = 1; i < GGML_CUDA_MMVQ_MULTI_MAX; ++i) {
        if (i < args.n_mat && (int) blockIdx.x >= args.block0[i]) {
            m = i;
        }
    }
    const int    row0  = ((int) blockIdx.x - args.block0[m]) * (16*ntiles);
    const int    nrows = args.nrows[m];
    const char * x     = args.x[m];
    const int    ncols = args.ncols;

    // columns >= ncols stay zero (their results are not written)
    for (int i = lane; i < 8*KQTC_SROW; i += WARP_SIZE) {
        s_y[w][i] = 0;
    }
    for (int i = lane; i < 8*8; i += WARP_SIZE) {
        s_d[w][i] = 0.0f;
    }
    __syncwarp();

    const char * xr[ntiles][2];
#pragma unroll
    for (int tt = 0; tt < ntiles; ++tt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(row0 + 16*tt + 8*h + g, nrows - 1);
            xr[tt][h] = x + (int64_t) r*args.stride_row_x;
        }
    }

    float acc[ntiles][4];
#pragma unroll
    for (int tt = 0; tt < ntiles; ++tt) {
        acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.0f;
    }

    constexpr bool pair_ints = kqtc_pair_ints<type>();
    const int nints = ncols*72; // 8 blocks x 9 ints per column

    for (int u = w; u < args.nunits; u += nw) {
        kqtc_raw<type> r[ntiles][2];
#pragma unroll
        for (int tt = 0; tt < ntiles; ++tt) {
            r[tt][0].load(xr[tt][0], u, t, args.nb32);
            r[tt][1].load(xr[tt][1], u, t, args.nb32);
        }

        // stage the activations of this unit
        for (int i = lane; i < nints; i += WARP_SIZE) {
            const int c  = i / 72;
            const int j  = i - 72*c;
            const int bj = j / 9;
            const int wd = j - 9*bj;
            const int v  = args.y[(c*args.stride_col_y + 8*u)*9 + j];
            if (wd == 0) {
                s_d[w][8*bj + c] = __low2float(*(const half2 *) &v);
            } else {
                const int q = wd - 1;
                const int p = pair_ints ? 4*(q & 1) + (q >> 1) : q;
                s_y[w][c*KQTC_SROW + 8*bj + p] = v;
            }
        }
        __syncwarp();

        kqtc_b B;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            B.b[s][0] = s_y[w][g*KQTC_SROW + 8*s + t];
            B.b[s][1] = s_y[w][g*KQTC_SROW + 8*s + 4 + t];
            B.d8[s]   = *(const float2 *) &s_d[w][8*s + 2*t];
        }
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                int c[4] = { 0, 0, 0, 0 };
                kqtc_mma_k32(c, 0x01010101, 0x01010101, 0x01010101, 0x01010101, B.b[s][0], B.b[s][1]);
                B.sum[s][0] = c[0];
                B.sum[s][1] = c[1];
            }
        }
        __syncwarp();

#pragma unroll
        for (int tt = 0; tt < ntiles; ++tt) {
            kqtc_tile<type>(r[tt][0], r[tt][1], B, acc[tt]);
        }
    }

#pragma unroll
    for (int tt = 0; tt < ntiles; ++tt) {
        s_red[w][(16*tt + g    )*8 + 2*t    ] = acc[tt][0];
        s_red[w][(16*tt + g    )*8 + 2*t + 1] = acc[tt][1];
        s_red[w][(16*tt + g + 8)*8 + 2*t    ] = acc[tt][2];
        s_red[w][(16*tt + g + 8)*8 + 2*t + 1] = acc[tt][3];
    }
    __syncthreads();

    float * dst = args.dst[m];
    for (int i = w*WARP_SIZE + lane; i < ntiles*16*8; i += nw*WARP_SIZE) {
        const int row = i >> 3;
        const int col = i & 7;
        if (col >= ncols || row0 + row >= nrows) {
            continue;
        }
        float sum = s_red[0][i];
#pragma unroll
        for (int ww = 1; ww < nw; ++ww) {
            sum += s_red[ww][i];
        }
        dst[col*args.stride_col_dst + row0 + row] = sum;
    }
#else
    GGML_UNUSED(args);
    NO_DEVICE_CODE;
#endif
}

struct kqtc_cfg {
    int  mask     = 0;  // 1 Q4_K, 2 Q5_K, 4 Q6_K, 8 IQ4_XS, 16 Q8_0
    int  min_cols = 2;
    int  max_cols = 8;
    int  nw       = 0;  // 0 = automatic
    int  tiles    = 0;
    bool log      = false;
};

static int kqtc_type_bit(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:   return 1;
        case GGML_TYPE_Q5_K:   return 2;
        case GGML_TYPE_Q6_K:   return 4;
        case GGML_TYPE_IQ4_XS: return 8;
        case GGML_TYPE_Q8_0:   return 16;
        default:               return 0;
    }
}

static const kqtc_cfg & kqtc_config() {
    static const kqtc_cfg cfg = [] {
        kqtc_cfg c;
        if (const char * e = getenv("GGML_CUDA_MMVQ_KQTC")) {
            const std::string s = e;
            if (s == "1") {
                c.mask = 31;
            } else if (s != "0" && !s.empty()) {
                const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0 };
                for (const ggml_type tp : types) {
                    if (("," + s + ",").find("," + std::string(ggml_type_name(tp)) + ",") != std::string::npos) {
                        c.mask |= kqtc_type_bit(tp);
                    }
                }
            }
        }
        if (const char * e = getenv("GGML_CUDA_MMVQ_KQTC_MIN")) {
            c.min_cols = std::max(2, atoi(e));
        }
        if (const char * e = getenv("GGML_CUDA_MMVQ_KQTC_MAX")) {
            c.max_cols = std::min(8, atoi(e));
        }
        if (const char * e = getenv("GGML_CUDA_MMVQ_KQTC_CFG")) {
            int nw = 0, tiles = 0;
            if (sscanf(e, "%dx%d", &nw, &tiles) == 2 && (nw == 4 || nw == 8) && (tiles == 1 || tiles == 2)) {
                c.nw    = nw;
                c.tiles = tiles;
            }
        }
        if (const char * e = getenv("GGML_CUDA_MMVQ_KQTC_LOG")) {
            c.log = atoi(e) != 0;
        }
        return c;
    }();
    return cfg;
}

bool ggml_cuda_mmvq_kqtc_reassoc_ok() {
    static const bool reassoc = [] {
        const char * e = getenv("GGML_CUDA_VERIFY_REASSOC");
        return e == nullptr || atoi(e) != 0;
    }();
    return reassoc;
}

bool ggml_cuda_mmvq_kqtc_ok(const ggml_type type, const int64_t ncols_x, const int64_t ncols_dst, const int cc) {
#if defined(KQTC_ENABLED)
    const kqtc_cfg & cfg = kqtc_config();
    if ((cfg.mask & kqtc_type_bit(type)) == 0 || ncols_dst < cfg.min_cols || ncols_dst > cfg.max_cols || !turing_mma_available(cc)) {
        return false;
    }
    return ncols_x % ggml_blck_size(type) == 0 && ncols_x > 0;
#else
    GGML_UNUSED_VARS(type, ncols_x, ncols_dst, cc);
    return false;
#endif
}

template <ggml_type type, int nw, int ntiles>
static void kqtc_launch_t(const kqtc_args & args, const int nblocks, cudaStream_t stream) {
    mul_mat_vec_q_kqtc<type, nw, ntiles><<<nblocks, dim3(WARP_SIZE, nw, 1), 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

template <ggml_type type>
static void kqtc_launch_type(const kqtc_args & args, const int nblocks, const int nw, const int ntiles, cudaStream_t stream) {
    if (nw == 8) {
        if (ntiles == 2) {
            kqtc_launch_t<type, 8, 2>(args, nblocks, stream);
        } else {
            kqtc_launch_t<type, 8, 1>(args, nblocks, stream);
        }
    } else {
        if (ntiles == 2) {
            kqtc_launch_t<type, 4, 2>(args, nblocks, stream);
        } else {
            kqtc_launch_t<type, 4, 1>(args, nblocks, stream);
        }
    }
}

bool ggml_cuda_mmvq_kqtc_launch(const ggml_type type, const int n_mat, const void * const * vx, float * const * dst, const int64_t * nrows,
        const int64_t stride_row_x, const void * vy, const int64_t stride_col_y, const int64_t stride_col_dst, const int64_t ncols_x,
        const int64_t ncols_dst, cudaStream_t stream) {
#if defined(KQTC_ENABLED)
    GGML_ASSERT(n_mat >= 1 && n_mat <= GGML_CUDA_MMVQ_MULTI_MAX);
    GGML_ASSERT(ncols_dst >= 1 && ncols_dst <= 8);
    // alignment of the wide weight loads (Q4_K / Q5_K 16 bytes, IQ4_XS 8; Q6_K / Q8_0 read 16-bit halves) and of the q8_1 ints
    const int align = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 16 : type == GGML_TYPE_IQ4_XS ? 8 : 2;
    if (stride_row_x % align != 0 || ((uintptr_t) vy) % 4 != 0) {
        return false;
    }
    for (int i = 0; i < n_mat; ++i) {
        if (((uintptr_t) vx[i]) % align != 0 || nrows[i] < 1 || nrows[i] > INT_MAX/2) {
            return false;
        }
    }
    const kqtc_cfg & cfg = kqtc_config();
    const int nb32   = (int) (ncols_x / 32);
    const int nunits = (int) ((ncols_x + 255) / 256);
    // the warps per block (which fix the K split, so the f32 order) depend on K only: a column's bytes do not depend on the launch's
    // columns or matrices
    const int nw     = cfg.nw > 0 ? cfg.nw : nunits >= 32 ? 8 : 4;
    const int ntiles = cfg.tiles > 0 ? cfg.tiles : 1;

    kqtc_args args = {};
    int nblocks = 0;
    for (int i = 0; i < n_mat; ++i) {
        args.x[i]      = (const char *) vx[i];
        args.dst[i]    = dst[i];
        args.nrows[i]  = (int) nrows[i];
        args.block0[i] = nblocks;
        nblocks += (int) ((nrows[i] + 16*ntiles - 1) / (16*ntiles));
    }
    args.n_mat          = n_mat;
    args.y              = (const int *) vy;
    args.stride_col_y   = stride_col_y;
    args.stride_col_dst = stride_col_dst;
    args.stride_row_x   = stride_row_x;
    args.nunits         = nunits;
    args.nb32           = nb32;
    args.ncols          = (int) ncols_dst;

    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("k19-kqtc: dense %d..%d column products of the enabled types (mask %d) on int8 tensor cores (first: %s, %d columns, "
            "%d matrices, K %lld, %d warps x %d tiles; GGML_CUDA_MMVQ_KQTC=0 off)\n", cfg.min_cols, cfg.max_cols, cfg.mask,
            ggml_type_name(type), (int) ncols_dst, n_mat, (long long) ncols_x, nw, ntiles);
    });
    if (cfg.log) {
        GGML_LOG_INFO("k19-kqtc: %s %d columns %d matrices rows %lld K %lld %dx%d\n", ggml_type_name(type), (int) ncols_dst, n_mat,
            (long long) nrows[0], (long long) ncols_x, nw, ntiles);
    }

    switch (type) {
        case GGML_TYPE_Q4_K:   kqtc_launch_type<GGML_TYPE_Q4_K>  (args, nblocks, nw, ntiles, stream); break;
        case GGML_TYPE_Q5_K:   kqtc_launch_type<GGML_TYPE_Q5_K>  (args, nblocks, nw, ntiles, stream); break;
        case GGML_TYPE_Q6_K:   kqtc_launch_type<GGML_TYPE_Q6_K>  (args, nblocks, nw, ntiles, stream); break;
        case GGML_TYPE_IQ4_XS: kqtc_launch_type<GGML_TYPE_IQ4_XS>(args, nblocks, nw, ntiles, stream); break;
        case GGML_TYPE_Q8_0:   kqtc_launch_type<GGML_TYPE_Q8_0>  (args, nblocks, nw, ntiles, stream); break;
        default: return false;
    }
    return true;
#else
    GGML_UNUSED_VARS(type, n_mat, vx, dst, nrows, stride_row_x, vy, stride_col_y, stride_col_dst, ncols_x, ncols_dst, stream);
    return false;
#endif
}
