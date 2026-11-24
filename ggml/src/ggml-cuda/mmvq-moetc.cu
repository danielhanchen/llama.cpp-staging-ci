// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// The few-row int8-MMA structure and the Q4_K / Q5_K / Q6_K / IQ4_XS / Q8_0 / IQ4_NL decoders follow llama.cpp's mmq-few.cu
// (llama.cpp authors, MIT license); ported and extended with ids, gathered columns, K tails and a fused GLU.
#include "mmvq-moetc.cuh"
#include "cp-async.cuh"
#include "mmvf.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>

// (see mmvq-moetc.cuh). One launch:
// - grid (16*RW-row tiles, slots, tokens), block 4 warps = KW (split K) x RW (16-row groups). The block of entry (token, slot) leaves
//   unless it is the first entry (token-major) of its expert; then it takes the expert's entries as columns c = 0..ncols-1;
// - warp (kw, rw) runs the 256-value units kw, kw + KW, ... of its 16 rows (fused: + the 16 gate rows): per unit a cp.async ring stage
//   holds the raw weight rows and the q8_1 activations of the columns (36-byte blocks as in src1, column stride 304 bytes: conflict-free
//   B reads); lane (g, t) decodes rows g / g+8 into A fragments and runs m16n8k32 MMAs (Q6_K: m16n8k16) for all columns;
//   C: rows g / g+8 x columns 2t / 2t+1;
// - the KW partial sums are added in shared memory in warp order (fixed: the f32 order depends on K only), then the GLU, then dst.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define MOETC_ENABLED 1
#endif

static constexpr int MOETC_NW        = 4;   // warps per block
static constexpr int MOETC_ACT_CSTR  = 304; // bytes per staged activation column (8 x 36 + 16 pad)
static constexpr int MOETC_NSTAGES   = 2;

template <ggml_type type> struct moetc_traits;
template <> struct moetc_traits<GGML_TYPE_Q4_K>   { static constexpr int unit = sizeof(block_q4_K);    static constexpr int bs32 = 0; };
template <> struct moetc_traits<GGML_TYPE_Q5_K>   { static constexpr int unit = sizeof(block_q5_K);    static constexpr int bs32 = 0; };
template <> struct moetc_traits<GGML_TYPE_Q6_K>   { static constexpr int unit = sizeof(block_q6_K);    static constexpr int bs32 = 0; };
template <> struct moetc_traits<GGML_TYPE_IQ4_XS> { static constexpr int unit = sizeof(block_iq4_xs);  static constexpr int bs32 = 0; };
template <> struct moetc_traits<GGML_TYPE_Q8_0>   { static constexpr int unit = 8*sizeof(block_q8_0);  static constexpr int bs32 = sizeof(block_q8_0); };
template <> struct moetc_traits<GGML_TYPE_IQ4_NL> { static constexpr int unit = 8*sizeof(block_iq4_nl); static constexpr int bs32 = sizeof(block_iq4_nl); };
template <> struct moetc_traits<GGML_TYPE_Q5_0>   { static constexpr int unit = 8*sizeof(block_q5_0);  static constexpr int bs32 = sizeof(block_q5_0); };
template <> struct moetc_traits<GGML_TYPE_IQ3_XXS> { static constexpr int unit = sizeof(block_iq3_xxs); static constexpr int bs32 = 0; };
template <> struct moetc_traits<GGML_TYPE_IQ2_S>   { static constexpr int unit = sizeof(block_iq2_s);   static constexpr int bs32 = 0; };

// 16-byte chunks per staged weight row (rows not on a 16-byte boundary are copied from the preceding one) and the smem row stride
// (odd number of chunks against bank conflicts)
// rows of the 32-value block types have any even length (K 640 IQ4_NL: 360 bytes), so their units are copied as unaligned too
template <ggml_type type>
static constexpr __host__ __device__ bool moetc_unaligned() {
    return moetc_traits<type>::unit % 16 != 0 || moetc_traits<type>::bs32 != 0;
}
template <ggml_type type>
static constexpr __host__ __device__ int moetc_wchunks() {
    return moetc_unaligned<type>() ? (15 + moetc_traits<type>::unit + 15)/16 : moetc_traits<type>::unit/16;
}
template <ggml_type type>
static constexpr __host__ __device__ int moetc_wstride() {
    return 16*(moetc_wchunks<type>() % 2 == 0 ? moetc_wchunks<type>() + 1 : moetc_wchunks<type>());
}
template <ggml_type type, bool fused>
static constexpr __host__ __device__ int moetc_wbytes() {
    return (fused ? 32 : 16)*moetc_wstride<type>();
}
template <ggml_type type, bool fused>
static __host__ __device__ int moetc_stage_bytes(const int ntok) {
    return moetc_wbytes<type, fused>() + ntok*MOETC_ACT_CSTR;
}

struct moetc_args {
    const char    * x;
    const char    * xg;
    const char    * y;
    const int32_t * ids;
    float         * dst;
    int64_t stride_row_x;       // bytes
    int64_t stride_channel_x;   // bytes
    int64_t stride_col_y;       // bytes
    int64_t stride_channel_y;   // bytes
    int64_t stride_col_dst;     // floats
    int64_t stride_channel_dst; // floats
    int nrows;
    int nunits;
    int nb32;                   // 32-value blocks per row
    int nchannels_y;
    int nslots;
    int ntok;
    int ids_stride;
    int glu;                    // 0 SWIGLU, 1 GEGLU
    int iround;                 // IQ3_XXS / IQ2_S scales in mul_mat_vec_q's integer rounding (GGML_CUDA_MOE_TC_IROUND, default 1)
};

#if defined(MOETC_ENABLED)
static __device__ __forceinline__ void moetc_cp_commit() {
#if defined(CP_ASYNC_AVAILABLE)
    asm volatile("cp.async.commit_group;");
#endif
}
template <int n>
static __device__ __forceinline__ void moetc_cp_wait() {
#if defined(CP_ASYNC_AVAILABLE)
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif
}

// C (rows g / g+8 x columns 2t / 2t+1) = A (lo: values [4t, 4t+4), hi: [16+4t, 16+4t+4) of rows g (0) / g+8 (1)) x B (b.x / b.y)
static __device__ __forceinline__ void moetc_mma_k32(int (&c)[4], const int (&lo)[2], const int (&hi)[2], const int2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    c[0] = c[1] = c[2] = c[3] = 0;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(lo[0]), "r"(lo[1]), "r"(hi[0]), "r"(hi[1]), "r"(b.x), "r"(b.y));
#else
    GGML_UNUSED_VARS(c, lo, hi, b);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void moetc_mma_k16(int (&c)[4], const int (&a)[2], const int b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    c[0] = c[1] = c[2] = c[3] = 0;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
#else
    GGML_UNUSED_VARS(c, a, b);
    NO_DEVICE_CODE;
#endif
}

// the activation fragments of a lane for one unit: B of column g (q8_1 ints t / t+4 of block s), the q8_1 scales d of columns 2t / 2t+1
// and (Q4_K / Q5_K mins) d * the integer sum of the block's q8 values (as mul_mat_vec_q: an all-ones A MMA), columns >= ncols read
// column ncols-1 (inside the staged columns; their results are not stored).
// moetc_ysmem reads them from the staged stage when used (fewer registers); moetc_yreg loads them all first
struct moetc_ysmem {
    const char * bg;   // column g, + 4 (qs)
    const char * a0;   // column 2t
    const char * a1;   // column 2t+1
    int t;
    __device__ __forceinline__ void init(const char * act, const int g, const int t_, const int ncols) {
        t  = t_;
        bg = act + min(g, ncols - 1)*MOETC_ACT_CSTR + 4;
        a0 = act + min(2*t_, ncols - 1)*MOETC_ACT_CSTR;
        a1 = act + min(2*t_ + 1, ncols - 1)*MOETC_ACT_CSTR;
    }
    __device__ __forceinline__ int2 b(const int s) const {
        return make_int2(*(const int *) (bg + 36*s + 4*t), *(const int *) (bg + 36*s + 16 + 4*t));
    }
    __device__ __forceinline__ float2 d(const int s) const {
        return make_float2(__low2float(*(const half2 *) (a0 + 36*s)), __low2float(*(const half2 *) (a1 + 36*s)));
    }
};

// d * integer q8 sum of block s for columns 2t / 2t+1
template <typename Y>
static __device__ __forceinline__ float2 moetc_ysum(const Y & y, const int s) {
    const int ones[2] = {0x01010101, 0x01010101};
    int c[4];
    moetc_mma_k32(c, ones, ones, y.b(s));
    const float2 d = y.d(s);
    return make_float2(d.x*(float) c[0], d.y*(float) c[1]);
}

static __device__ __forceinline__ int moetc_unpack_scales_q45_K(const int * scales, const int ksc) {
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}

// sum[l] += d * C[l] * ls (row scale ls[0] for rows g, ls[1] for g+8) of sub-block s
static __device__ __forceinline__ void moetc_add(float (&sum)[4], const int (&c)[4], const int (&ls)[2], const float2 d) {
    sum[0] += d.x * float(c[0]*ls[0]);
    sum[1] += d.y * float(c[1]*ls[0]);
    sum[2] += d.x * float(c[2]*ls[1]);
    sum[3] += d.y * float(c[3]*ls[1]);
}

// one unit (256 values, or nvb <= 8 blocks of 32) of rows g (x[0]) and g+8 (x[1]) into sum
template <ggml_type type, typename Y>
static __device__ __forceinline__ void moetc_dot(const char * const (&x)[2], const Y & yf, float (&sum)[4], const int t, const int nvb,
        const bool iround) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
        int qh[2][2] = {{0, 0}, {0, 0}};
        int sc[2][2], mn[2][2];
        float d[2], dmin[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_t * bx = (const block_t *) x[r];
            const int4 hdr = *(const int4 *) bx; // dm and the 12 scale bytes
            const int scales[3] = {hdr.y, hdr.z, hdr.w};
            sc[r][0] = moetc_unpack_scales_q45_K(scales, 0);
            sc[r][1] = moetc_unpack_scales_q45_K(scales, 1);
            mn[r][0] = moetc_unpack_scales_q45_K(scales, 2);
            mn[r][1] = moetc_unpack_scales_q45_K(scales, 3);
            const float2 dm = __half22float2(*(const half2 *) &hdr.x);
            d[r]    = dm.x;
            dmin[r] = dm.y;
            if constexpr (type == GGML_TYPE_Q5_K) {
                qh[r][0] = get_int_b4(bx->qh,      t);
                qh[r][1] = get_int_b4(bx->qh + 16, t);
            }
        }
        float sum_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float sum_m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int q[2][2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const block_t * bx = (const block_t *) x[r];
                q[r][0] = get_int_b4(bx->qs + 32*j,      t);
                q[r][1] = get_int_b4(bx->qs + 32*j + 16, t);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int s = 2*j + h;
                int lo[2], hi[2];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    lo[r] = (q[r][0] >> (4*h)) & 0x0F0F0F0F;
                    hi[r] = (q[r][1] >> (4*h)) & 0x0F0F0F0F;
                    if constexpr (type == GGML_TYPE_Q5_K) {
                        lo[r] |= ((qh[r][0] >> s) & 0x01010101) << 4;
                        hi[r] |= ((qh[r][1] >> s) & 0x01010101) << 4;
                    }
                }
                int c[4];
                moetc_mma_k32(c, lo, hi, yf.b(s));
                const int ls[2] = {(sc[0][s/4] >> (8*(s % 4))) & 0xFF, (sc[1][s/4] >> (8*(s % 4))) & 0xFF};
                moetc_add(sum_d, c, ls, yf.d(s));
                const float m0 = (mn[0][s/4] >> (8*(s % 4))) & 0xFF;
                const float m1 = (mn[1][s/4] >> (8*(s % 4))) & 0xFF;
                const float2 ys = moetc_ysum(yf, s);
                sum_m[0] += m0 * ys.x;
                sum_m[1] += m0 * ys.y;
                sum_m[2] += m1 * ys.x;
                sum_m[3] += m1 * ys.y;
            }
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[l] += d[l/2]*sum_d[l] - dmin[l/2]*sum_m[l];
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        float d[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            d[r] = __half2float(((const block_q6_K *) x[r])->d);
        }
        float sum_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
                int lo[2], hi[2];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    lo[r] = __vsub4(((ql[r][j % 2][0] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][0] >> (2*j)) & 0x03030303) << 4), 0x20202020);
                    hi[r] = __vsub4(((ql[r][j % 2][1] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][1] >> (2*j)) & 0x03030303) << 4), 0x20202020);
                }
                const int sc_s[2] = {sc[0][j/2] >> (16*(j % 2)), sc[1][j/2] >> (16*(j % 2))}; // scales 2s and 2s+1
                const int sc_lo[2] = {(int8_t) sc_s[0], (int8_t) sc_s[1]};
                const int sc_hi[2] = {(int8_t) (sc_s[0] >> 8), (int8_t) (sc_s[1] >> 8)};
                int c_lo[4], c_hi[4];
                moetc_mma_k16(c_lo, lo, yf.b(s).x);
                moetc_mma_k16(c_hi, hi, yf.b(s).y);
                const float2 dy = yf.d(s);
                sum_d[0] += dy.x * float(c_lo[0]*sc_lo[0] + c_hi[0]*sc_hi[0]);
                sum_d[1] += dy.y * float(c_lo[1]*sc_lo[0] + c_hi[1]*sc_hi[0]);
                sum_d[2] += dy.x * float(c_lo[2]*sc_lo[1] + c_hi[2]*sc_hi[1]);
                sum_d[3] += dy.y * float(c_lo[3]*sc_lo[1] + c_hi[3]*sc_hi[1]);
            }
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[l] += d[l/2]*sum_d[l];
        }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        float d[2];
        int scales_h[2], scales_l[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_iq4_xs * bx = (const block_iq4_xs *) x[r];
            d[r]        = __half2float(bx->d);
            scales_h[r] = bx->scales_h;
            scales_l[r] = get_int_b4(bx->scales_l, 0);
        }
        float sum_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            int lo[2], hi[2], ls[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const block_iq4_xs * bx = (const block_iq4_xs *) x[r];
                const int2 v = get_int_from_table_16(get_int_b4(bx->qs + 16*s, t), kvalues_iq4nl);
                lo[r] = v.x;
                hi[r] = v.y;
                ls[r] = (((scales_l[r] >> (4*s)) & 0x0F) | (((scales_h[r] >> (2*s)) & 0x03) << 4)) - 32;
            }
            int c[4];
            moetc_mma_k32(c, lo, hi, yf.b(s));
            moetc_add(sum_d, c, ls, yf.d(s));
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[l] += d[l/2]*sum_d[l];
        }
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        // a sub-block of 32 = 8 grid entries of 4 values (lane t: entries t and 4 + t), signs per 8 values (7-bit index + parity),
        // scale (2*ls + 1)/4. iround: vec_dot_iq3_xxs_q8_1's integer form (ls*sumi + sumi/2)/2, so every integer term equals
        // mul_mat_vec_q's and only the f32 summation order differs; else the exact product
        float d[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            d[r] = __half2float(((const block_iq3_xxs *) x[r])->d);
        }
        float sum_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            int lo[2], hi[2], ls[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const block_iq3_xxs * bx = (const block_iq3_xxs *) x[r];
                const uint32_t aux = get_int_b2(bx->qs + QK_K/4, s);
                const int gl = iq3xxs_grid[bx->qs[8*s + t]];
                const int gh = iq3xxs_grid[bx->qs[8*s + 4 + t]];
                const int sl = (ksigns_iq2xs[(aux >> (7*(t/2)))     & 127] >> (4*(t%2))) & 0x0F;
                const int sh = (ksigns_iq2xs[(aux >> (7*(2 + t/2))) & 127] >> (4*(t%2))) & 0x0F;
                const int ml = __vcmpne4(((sl & 0x03) << 7) | ((sl & 0x0C) << 21), 0);
                const int mh = __vcmpne4(((sh & 0x03) << 7) | ((sh & 0x0C) << 21), 0);
                lo[r] = __vsub4(gl ^ ml, ml);
                hi[r] = __vsub4(gh ^ mh, mh);
                ls[r] = iround ? (int) (aux >> 28) : 2*(int) (aux >> 28) + 1;
            }
            int c[4];
            moetc_mma_k32(c, lo, hi, yf.b(s));
            if (iround) {
                const float2 dy = yf.d(s);
                sum_d[0] += dy.x * float((ls[0]*c[0] + c[0]/2)/2);
                sum_d[1] += dy.y * float((ls[0]*c[1] + c[1]/2)/2);
                sum_d[2] += dy.x * float((ls[1]*c[2] + c[2]/2)/2);
                sum_d[3] += dy.y * float((ls[1]*c[3] + c[3]/2)/2);
            } else {
                moetc_add(sum_d, c, ls, yf.d(s));
            }
        }
        const float f = iround ? 1.0f : 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[l] += (f*d[l/2])*sum_d[l];
        }
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        // a sub-block of 32 = 4 grid entries of 8 values (10-bit index: qs byte + 2 qh bits), signs bytes, one scale per 16 values
        // ((2*ls + 1)/8): one m16n8k16 per half (lane t: values [4t, 4t + 4) of entry t/2 and [16 + 4t, ..) of entry 2 + t/2)
        float d[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            d[r] = __half2float(((const block_iq2_s *) x[r])->d);
        }
        const int * grid32 = (const int *) iq2s_grid;
        float sum_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            int lo[2], hi[2], sl_[2], sh_[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const block_iq2_s * bx = (const block_iq2_s *) x[r];
                const int qh = bx->qh[s];
                const int il = t/2, ih = 2 + t/2;
                const int idxl = bx->qs[4*s + il] | ((qh << (8 - 2*il)) & 0x300);
                const int idxh = bx->qs[4*s + ih] | ((qh << (8 - 2*ih)) & 0x300);
                const int gl = grid32[2*idxl + (t%2)];
                const int gh = grid32[2*idxh + (t%2)];
                const int sl = (bx->qs[QK_K/8 + 4*s + il] >> (4*(t%2))) & 0x0F;
                const int sh = (bx->qs[QK_K/8 + 4*s + ih] >> (4*(t%2))) & 0x0F;
                const int ml = __vcmpne4(((sl & 0x03) << 7) | ((sl & 0x0C) << 21), 0);
                const int mh = __vcmpne4(((sh & 0x03) << 7) | ((sh & 0x0C) << 21), 0);
                lo[r] = __vsub4(gl ^ ml, ml);
                hi[r] = __vsub4(gh ^ mh, mh);
                const int sc = bx->scales[s];
                // iround: vec_dot_iq2_s_q8_1's integer form (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4
                sl_[r] = iround ? (sc & 0x0F) : 2*(sc & 0x0F) + 1;
                sh_[r] = iround ? (sc >> 4)   : 2*(sc >> 4) + 1;
            }
            int c_lo[4], c_hi[4];
            moetc_mma_k16(c_lo, lo, yf.b(s).x);
            moetc_mma_k16(c_hi, hi, yf.b(s).y);
            const float2 dy = yf.d(s);
            if (iround) {
                sum_d[0] += dy.x * float((c_lo[0]*sl_[0] + c_hi[0]*sh_[0] + (c_lo[0] + c_hi[0])/2)/4);
                sum_d[1] += dy.y * float((c_lo[1]*sl_[0] + c_hi[1]*sh_[0] + (c_lo[1] + c_hi[1])/2)/4);
                sum_d[2] += dy.x * float((c_lo[2]*sl_[1] + c_hi[2]*sh_[1] + (c_lo[2] + c_hi[2])/2)/4);
                sum_d[3] += dy.y * float((c_lo[3]*sl_[1] + c_hi[3]*sh_[1] + (c_lo[3] + c_hi[3])/2)/4);
            } else {
                sum_d[0] += dy.x * float(c_lo[0]*sl_[0] + c_hi[0]*sh_[0]);
                sum_d[1] += dy.y * float(c_lo[1]*sl_[0] + c_hi[1]*sh_[0]);
                sum_d[2] += dy.x * float(c_lo[2]*sl_[1] + c_hi[2]*sh_[1]);
                sum_d[3] += dy.y * float(c_lo[3]*sl_[1] + c_hi[3]*sh_[1]);
            }
        }
        const float f = iround ? 1.0f : 0.125f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[l] += (f*d[l/2])*sum_d[l];
        }
    } else {
        // Q8_0 / IQ4_NL / Q5_0: a scale per block of 32, nvb <= 8 blocks in this unit (K tail)
        using block_t = std::conditional_t<type == GGML_TYPE_Q8_0, block_q8_0,
                        std::conditional_t<type == GGML_TYPE_IQ4_NL, block_iq4_nl, block_q5_0>>;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= nvb) {
                break;
            }
            int   lo[2], hi[2];
            float d[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const block_t * bx = (const block_t *) x[r] + s;
                if constexpr (type == GGML_TYPE_Q8_0) {
                    lo[r] = get_int_b2(bx->qs,      t);
                    hi[r] = get_int_b2(bx->qs + 16, t);
                } else if constexpr (type == GGML_TYPE_IQ4_NL) {
                    const int2 v = get_int_from_table_16(get_int_b2(bx->qs, t), kvalues_iq4nl);
                    lo[r] = v.x;
                    hi[r] = v.y;
                } else {
                    // Q5_0: value j < 16 = (qs[j] & 0x0F) | bit j of qh << 4, value 16 + j = (qs[j] >> 4) | bit 16 + j << 4, minus 16
                    const int qs = get_int_b2(bx->qs, t);
                    const int qh = get_int_b2(bx->qh, 0) >> (4*t);
                    int vl = qs & 0x0F0F0F0F;
                    vl |= (qh <<  4) & 0x00000010;
                    vl |= (qh << 11) & 0x00001000;
                    vl |= (qh << 18) & 0x00100000;
                    vl |= (qh << 25) & 0x10000000;
                    int vh = (qs >> 4) & 0x0F0F0F0F;
                    vh |= (qh >> 12) & 0x00000010;
                    vh |= (qh >>  5) & 0x00001000;
                    vh |= (qh <<  2) & 0x00100000;
                    vh |= (qh <<  9) & 0x10000000;
                    lo[r] = __vsubss4(vl, 0x10101010);
                    hi[r] = __vsubss4(vh, 0x10101010);
                }
                d[r] = __half2float(bx->d);
            }
            int c[4];
            moetc_mma_k32(c, lo, hi, yf.b(s));
            const float2 dy = yf.d(s);
            sum[0] += (d[0]*dy.x) * float(c[0]);
            sum[1] += (d[0]*dy.y) * float(c[1]);
            sum[2] += (d[1]*dy.x) * float(c[2]);
            sum[3] += (d[1]*dy.y) * float(c[3]);
        }
    }
}
#endif // defined(MOETC_ENABLED)

template <ggml_type type, int KW, bool fused>
__launch_bounds__(MOETC_NW*WARP_SIZE, 3)
static __global__ void mul_mat_vec_q_moe_tc(const moetc_args args) {
#if defined(MOETC_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && defined(CP_ASYNC_AVAILABLE)
    constexpr int RW      = MOETC_NW / KW;
    constexpr int unit    = moetc_traits<type>::unit;
    constexpr int bs32    = moetc_traits<type>::bs32;
    constexpr int wchunks = moetc_wchunks<type>();
    constexpr int wstride = moetc_wstride<type>();
    constexpr int nmat    = fused ? 2 : 1;
    constexpr int wbytes  = moetc_wbytes<type, fused>();
    constexpr unsigned FULL = 0xFFFFFFFFu;

    extern __shared__ __align__(16) char moetc_smem[];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int kw   = w % KW;
    const int rw   = w / KW;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    const int tok    = blockIdx.z;
    const int slot   = blockIdx.y;
    const int nslots = args.nslots;
    const int row0   = blockIdx.x*(16*RW);

    ggml_cuda_pdl_sync();

    const int e = args.ids[slot + tok*args.ids_stride];
    if (e < 0) {
        // a skipped slot: zeros, as mul_mat_vec_q_moe writes
        const int i = w*WARP_SIZE + lane;
        if (i < 16*RW && row0 + i < args.nrows) {
            args.dst[slot*args.stride_channel_dst + tok*args.stride_col_dst + row0 + i] = 0.0f;
        }
        return;
    }

    // an expert's entries in token-major order go in groups of cap columns (a stage holds ntok activation columns): an expert can
    // repeat within a token (GroveMoE chunk experts, ids / n_group_experts), so it can have more than ntok (or 8) entries.
    // Leave unless this entry starts a group; every warp decides alike
    const int my   = tok*nslots + slot;
    const int nent = args.ntok*nslots;
    const int cap  = min(8, args.ntok);
    int nprev = 0;
    for (int i0 = 0; i0 < my; i0 += WARP_SIZE) {
        const int  i = i0 + lane;
        const bool m = i < my && args.ids[(i % nslots) + (i / nslots)*args.ids_stride] == e;
        nprev += __popc(__ballot_sync(FULL, m));
    }
    if (nprev % cap != 0) {
        return;
    }
    // the columns: this expert's entries in token-major order; lane c < ncols holds column c's offsets
    int     ncols = 0;
    int64_t yoff  = 0;
    int64_t doff  = 0;
    for (int i0 = my; i0 < nent && ncols < cap; i0 += WARP_SIZE) {
        const int  i = i0 + lane;
        const bool m = i < nent && args.ids[(i % nslots) + (i / nslots)*args.ids_stride] == e;
        unsigned b = __ballot_sync(FULL, m);
        while (b != 0 && ncols < cap) {
            const int ent = i0 + __ffs(b) - 1;
            b &= b - 1;
            if (lane == ncols) {
                const int tc = ent / nslots;
                const int sc = ent - tc*nslots;
                yoff = (int64_t) (sc % args.nchannels_y)*args.stride_channel_y + (int64_t) tc*args.stride_col_y;
                doff = (int64_t) sc*args.stride_channel_dst + (int64_t) tc*args.stride_col_dst;
            }
            ++ncols;
        }
    }

    const char * xe  = args.x + (int64_t) e*args.stride_channel_x;
    const char * xge = fused ? args.xg + (int64_t) e*args.stride_channel_x : nullptr;
    const int    rbase = row0 + 16*rw;
    const int    stage_bytes = wbytes + args.ntok*MOETC_ACT_CSTR;
    char * ring = moetc_smem + w*(MOETC_NSTAGES*stage_bytes);

    // global address of the bytes of staged row r (0..15 up, 16..31 gate) for unit u
    auto row_src = [&](const int r, const int u) -> const char * {
        const int row = min(rbase + (r & 15), args.nrows - 1);
        const char * base = (fused && r >= 16) ? xge : xe;
        return base + (int64_t) row*args.stride_row_x + (int64_t) u*unit;
    };
    auto nvb_of = [&](const int u) -> int {
        return bs32 ? min(8, args.nb32 - 8*u) : 8;
    };

    auto load_stage = [&](const int st, const int u) {
        char * sb = ring + st*stage_bytes;
        const unsigned sdst = ggml_cuda_cvta_generic_to_shared(sb);
        const int nvb    = nvb_of(u);
        const int wvalid = bs32 ? nvb*bs32 : unit;
#pragma unroll
        for (int c0 = 0; c0 < nmat*16*wchunks; c0 += WARP_SIZE) {
            const int c = c0 + lane;
            if ((nmat*16*wchunks) % WARP_SIZE == 0 || c < nmat*16*wchunks) {
                const int r  = c / wchunks;
                const int ic = c - r*wchunks;
                const char * src = row_src(r, u);
                const int off = moetc_unaligned<type>() ? (int) ((uintptr_t) src % 16) : 0;
                if (16*ic < off + wvalid) {
                    cp_async_cg_16<0>(sdst + r*wstride + 16*ic, src - off + 16*ic);
                }
            }
        }
        // activations: 18 chunks (8 blocks x 36 bytes) per column
        const int achunks = (nvb*36 + 15)/16;
        for (int c0 = 0; c0 < ncols*18; c0 += WARP_SIZE) {
            const int c   = c0 + lane;
            const int col = min(c / 18, 7);
            const int64_t yo = __shfl_sync(FULL, yoff, col);
            const int ic  = c - 18*(c / 18);
            if (c < ncols*18 && ic < achunks) {
                cp_async_cg_16<0>(sdst + wbytes + col*MOETC_ACT_CSTR + 16*ic, args.y + yo + (int64_t) u*288 + 16*ic);
            }
        }
    };

    float sum[nmat][4];
#pragma unroll
    for (int m = 0; m < nmat; ++m) {
        sum[m][0] = sum[m][1] = sum[m][2] = sum[m][3] = 0.0f;
    }

    // the ring: stage it % NSTAGES holds unit kw + it*KW
#pragma unroll
    for (int st = 0; st < MOETC_NSTAGES - 1; ++st) {
        const int u = kw + st*KW;
        if (u < args.nunits) {
            load_stage(st, u);
        }
        moetc_cp_commit();
    }
    for (int it = 0, u = kw; u < args.nunits; ++it, u += KW) {
        const int ul = u + (MOETC_NSTAGES - 1)*KW;
        if (ul < args.nunits) {
            load_stage((it + MOETC_NSTAGES - 1) % MOETC_NSTAGES, ul);
        }
        moetc_cp_commit();
        moetc_cp_wait<MOETC_NSTAGES - 1>();
        __syncwarp();

        const char * sb  = ring + (it % MOETC_NSTAGES)*stage_bytes;
        const int    nvb = nvb_of(u);
        moetc_ysmem yf;
        yf.init(sb + wbytes, g, t, ncols);
#pragma unroll
        for (int m = 0; m < nmat; ++m) {
            const int r0 = 16*m + g;
            const int off0 = moetc_unaligned<type>() ? (int) ((uintptr_t) row_src(r0,     u) % 16) : 0;
            const int off1 = moetc_unaligned<type>() ? (int) ((uintptr_t) row_src(r0 + 8, u) % 16) : 0;
            const char * const xr[2] = { sb + r0*wstride + off0, sb + (r0 + 8)*wstride + off1 };
            moetc_dot<type>(xr, yf, sum[m], t, nvb, args.iround != 0);
        }
        __syncwarp();
    }
    moetc_cp_wait<0>();

    ggml_cuda_pdl_lc();

    // KW partials in warp order; buffer [rw][m][kw][16 rows][8 columns]
    __syncthreads();
    float * red = (float *) moetc_smem;
#pragma unroll
    for (int m = 0; m < nmat; ++m) {
        float * rb = red + ((rw*nmat + m)*KW + kw)*128;
        rb[(g    )*8 + 2*t    ] = sum[m][0];
        rb[(g    )*8 + 2*t + 1] = sum[m][1];
        rb[(g + 8)*8 + 2*t    ] = sum[m][2];
        rb[(g + 8)*8 + 2*t + 1] = sum[m][3];
    }
    __syncthreads();
#pragma unroll
    for (int i0 = 0; i0 < RW*128; i0 += MOETC_NW*WARP_SIZE) {
        const int i   = i0 + w*WARP_SIZE + lane;
        const int rwi = i / 128;
        const int row = (i / 8) % 16;
        const int col = i % 8;
        const int64_t dof = __shfl_sync(FULL, doff, col);
        const int grow = row0 + 16*rwi + row;
        if (col < ncols && grow < args.nrows) {
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < KW; ++k) {
                v += red[((rwi*nmat + 0)*KW + k)*128 + row*8 + col];
            }
            if constexpr (fused) {
                float gv = 0.0f;
#pragma unroll
                for (int k = 0; k < KW; ++k) {
                    gv += red[((rwi*nmat + 1)*KW + k)*128 + row*8 + col];
                }
                v *= args.glu == 0 ? ggml_cuda_op_silu_single(gv) : ggml_cuda_op_gelu_single(gv);
            }
            args.dst[dof + grow] = v;
        }
    }
#else
    GGML_UNUSED(args);
    NO_DEVICE_CODE;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------------- host

struct moetc_cfg {
    int  mode  = 1;    // 0 off, 1 verify windows, 2 every 2..8-token MUL_MAT_ID
    // bit per moetc_type_bit. Default Q4_K / Q5_K / Q6_K: a long 48-prompt KLD run of a 4-bit MoE (A100) inside the -fa off band on every statistic. The
    // Flash-Next types (IQ4_XS / IQ4_NL / Q5_0 / Q8_0 / IQ3_XXS / IQ2_S) are opt-in (GGML_CUDA_MOE_TC_TYPES=all or a list): Flash-Next at 4 bits
    // full card long48 KLD mean / p99 / max inside the band, top-1 97.58% vs 97.70 / 98.14% for the two band samples (-fa off,
    // GGML_CUDA_VERIFY_WIDE=0), 0.12 pp below
    int  types = 0x7;
    int  kw    = 0;    // 0 = by K
    bool log   = false;
    int  min_tok = 0;   // GGML_CUDA_MOE_TC_MIN (0 = the per-device table)
    bool iround = true; // GGML_CUDA_MOE_TC_IROUND=0: IQ3_XXS / IQ2_S sub-block scales as the exact product (before: not mmvq's integers)
    bool any   = false; // GGML_CUDA_MOE_TC_ANYARCH=1: also sm_90+ (correctness runs on other hosts; not a tuned path there)
};

static int moetc_type_bit(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:   return 1;
        case GGML_TYPE_Q5_K:   return 2;
        case GGML_TYPE_Q6_K:   return 4;
        case GGML_TYPE_IQ4_XS: return 8;
        case GGML_TYPE_Q8_0:   return 16;
        case GGML_TYPE_IQ4_NL: return 32;
        case GGML_TYPE_Q5_0:   return 64;
        case GGML_TYPE_IQ3_XXS: return 128;
        case GGML_TYPE_IQ2_S:  return 256;
        default:               return 0;
    }
}

static const moetc_cfg & moetc_config() {
    static const moetc_cfg cfg = [] {
        moetc_cfg c;
        if (const char * e = getenv("GGML_CUDA_MOE_TC")) {
            c.mode = atoi(e);
        }
        if (const char * e = getenv("GGML_CUDA_MOE_TC_TYPES")) {
            const std::string s = e;
            c.types = s == "all" ? 0x1FF : 0;
            const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0,
                GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S };
            for (const ggml_type tp : types) {
                if (("," + s + ",").find("," + std::string(ggml_type_name(tp)) + ",") != std::string::npos) {
                    c.types |= moetc_type_bit(tp);
                }
            }
        }
        if (const char * e = getenv("GGML_CUDA_MOE_TC_KW")) {
            const int kw = atoi(e);
            c.kw = kw == 1 || kw == 2 || kw == 4 ? kw : 0;
        }
        if (const char * e = getenv("GGML_CUDA_MOE_TC_LOG")) {
            c.log = atoi(e) != 0;
        }
        if (const char * e = getenv("GGML_CUDA_MOE_TC_MIN")) {
            c.min_tok = atoi(e);
        }
        if (const char * e = getenv("GGML_CUDA_MOE_TC_IROUND")) {
            c.iround = atoi(e) != 0;
        }
        if (const char * e = getenv("GGML_CUDA_MOE_TC_ANYARCH")) {
            c.any = atoi(e) != 0;
        }
        return c;
    }();
    return cfg;
}

// the fewest tokens at which the kernel beats the mul_mat_vec_q MoE kernels it replaces, per type and device (test-moe-bw-exact --bench,
// verify-tagged); 9 = never. GGML_CUDA_MOE_TC_MIN=N: N for every type (bench)
static int moetc_min_tokens(const ggml_type type, const int cc, const int env_min) {
    if (env_min > 0) {
        return env_min;
    }
    // index: Q4_K Q5_K Q6_K IQ4_XS Q8_0 IQ4_NL Q5_0 IQ3_XXS IQ2_S
    // A100 40 (amp_ka2 / amp_ka3, two VMs): Q4_K gate / up 1.06-1.08x at 2..3 tokens, 1.2-1.4x at 4..8; Q5_K / Q6_K / Q8_0 / IQ4_NL
    // downs ~1.0-1.2x at 3, 1.5-2.2x at 5..8; IQ4_XS / Q5_0 from 4, IQ3_XXS / IQ2_S from 5 (4-bit MoE + MTP chat / 8k +3.2 / +2.9% e2e, KLD
    // inside the -fa off band). L4 (amp_kl2, at its DRAM roof): only the expert dedup pays; taken where the microbench gains >= 1.1x
    // (5..8-token windows). With Q4_K from 4 the per-pass time of the 4-bit MoE + MTP fell 1-5% but the changed verify text drafted fewer
    // tokens per pass at 8k (2.56 vs 2.84), -5% tok/s; its MTP windows (3..4 rows) stay on mmvq there
    static const int sm80[9] = { 2, 3, 3, 4, 3, 3, 4, 5, 5 };
    static const int sm89[9] = { 7, 7, 8, 7, 9, 5, 5, 5, 5 };
    int i = 0;
    for (int b = moetc_type_bit(type); b > 1; b >>= 1) {
        ++i;
    }
    if (moetc_type_bit(type) == 0) {
        return 9;
    }
    if (cc == GGML_CUDA_CC_AMPERE) {
        return sm80[i];
    }
    if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return sm89[i];
    }
    return std::max(sm80[i], sm89[i]); // sm_86 / sm_87: not measured, the later of the two
}

bool ggml_cuda_moe_tc_ok(const ggml_type type, const int64_t ncols_x, const int64_t ncols_dst, const int cc,
        const ggml_cuda_mm_fusion_args_device & fusion) {
#if defined(MOETC_ENABLED)
    const moetc_cfg & cfg = moetc_config();
    if (cfg.mode == 0 || (cfg.types & moetc_type_bit(type)) == 0 || ncols_dst < 2 || ncols_dst > 8) {
        return false;
    }
    // NVIDIA Ampere / Ada (sm_80 / sm_86 / sm_87 / sm_89); the kernel is compiled for the device's arch when it is >= sm_80
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE || (cc >= GGML_CUDA_CC_HOPPER && !cfg.any) ||
            ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    if (ncols_dst < moetc_min_tokens(type, cc, cfg.min_tok)) {
        return false;
    }
    if (cfg.mode == 1 && !ggml_cuda_verify_graph_active()) {
        return false;
    }
    if (fusion.x_bias != nullptr || fusion.gate_bias != nullptr || fusion.x_scale != nullptr || fusion.gate_scale != nullptr ||
            fusion.shared_up != nullptr || fusion.ew_src != nullptr) {
        return false;
    }
    if (fusion.gate != nullptr && fusion.glu_op != GGML_GLU_OP_SWIGLU && fusion.glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    const int bs = moetc_type_bit(type) & (16 | 32 | 64) ? 32 : 256;
    return ncols_x > 0 && ncols_x % bs == 0;
#else
    GGML_UNUSED_VARS(type, ncols_x, ncols_dst, cc, fusion);
    return false;
#endif
}

int ggml_cuda_moe_tc_max_batch(const ggml_type type, const int cc) {
#if defined(MOETC_ENABLED)
    ggml_cuda_mm_fusion_args_device none{};
    return ggml_cuda_moe_tc_ok(type, moetc_type_bit(type) & (16 | 32 | 64) ? 32 : 256, 8, cc, none) ? 8 : 0;
#else
    GGML_UNUSED_VARS(type, cc);
    return 0;
#endif
}

template <ggml_type type, int KW, bool fused>
static void moetc_launch_t(const moetc_args & args, const int nblocks_x, cudaStream_t stream) {
    const int smem = MOETC_NW*MOETC_NSTAGES*moetc_stage_bytes<type, fused>(args.ntok);
    const int smem_max = MOETC_NW*MOETC_NSTAGES*moetc_stage_bytes<type, fused>(8);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_vec_q_moe_tc<type, KW, fused>), smem_max);
    const dim3 grid(nblocks_x, args.nslots, args.ntok);
    const dim3 block(WARP_SIZE, MOETC_NW, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, (size_t) smem, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q_moe_tc<type, KW, fused>, params, args);
}

template <ggml_type type>
static void moetc_launch_type(const moetc_args & args, const int kw, const bool fused, cudaStream_t stream) {
    const int rpb = 16*(MOETC_NW/kw);
    const int nbx = (args.nrows + rpb - 1)/rpb;
#define MOETC_L(KW_) do { if (fused) { moetc_launch_t<type, KW_, true>(args, nbx, stream); } else { moetc_launch_t<type, KW_, false>(args, nbx, stream); } } while (0)
    switch (kw) {
        case 4:  MOETC_L(4); break;
        case 2:  MOETC_L(2); break;
        default: MOETC_L(1); break;
    }
#undef MOETC_L
}

bool ggml_cuda_moe_tc_launch(const ggml_type type, const void * vx, const void * vy, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device & fusion, float * dst, const int64_t ncols_x, const int64_t nrows_x,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t nchannels_y, const int64_t nslots, const int64_t ntok, const int64_t ids_stride, cudaStream_t stream) {
#if defined(MOETC_ENABLED)
    const moetc_cfg & cfg = moetc_config();
    const bool fused = fusion.gate != nullptr;
    const int64_t tsize = ggml_type_size(type);
    const int64_t bsize = ggml_blck_size(type);
    const int64_t nb32  = ncols_x / 32;
    const int64_t nunits = (ncols_x + 255) / 256;
    // the activation copies read whole units of 8 q8_1 blocks of every column (inside the padded src1 row) on 16-byte boundaries
    const int64_t ybytes_col = stride_col_y*(int64_t) sizeof(block_q8_1);
    const int64_t ybytes_ch  = stride_channel_y*(int64_t) sizeof(block_q8_1);
    if (((uintptr_t) vy) % 16 != 0 || ybytes_col % 16 != 0 || ybytes_ch % 16 != 0 || ntok < 2 || ntok > 8 ||
            nrows_x < 1 || nrows_x > INT_MAX/2 || nslots < 1 || nslots*ntok > 4096) {
        return false;
    }
    // every column's staged span (nunits*8 q8_1 blocks) inside its own padded src1 row
    const int64_t yspan = nunits*8;
    if (stride_col_y < yspan || (nchannels_y > 1 && stride_channel_y < yspan)) {
        return false;
    }
    // weights: 16-byte copies of the aligned types need 16-byte rows; the others are copied from the preceding boundary (2-byte rows)
    const int64_t rbytes = stride_row_x*tsize/1; // stride_row_x is in blocks of type
    if (((uintptr_t) vx) % 16 != 0 || (fused && ((uintptr_t) fusion.gate) % 16 != 0) || rbytes % 2 != 0) {
        return false;
    }
    const bool aligned_type = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
    if (aligned_type && ((rbytes % 16) != 0 || (stride_channel_x*tsize) % 16 != 0)) {
        return false;
    }
    GGML_UNUSED(bsize);

    moetc_args args = {};
    args.x                  = (const char *) vx;
    args.xg                 = (const char *) fusion.gate;
    args.y                  = (const char *) vy;
    args.ids                = ids;
    args.dst                = dst;
    args.stride_row_x       = rbytes;
    args.stride_channel_x   = stride_channel_x*tsize;
    args.stride_col_y       = ybytes_col;
    args.stride_channel_y   = ybytes_ch;
    args.stride_col_dst     = stride_col_dst;
    args.stride_channel_dst = stride_channel_dst;
    args.nrows              = (int) nrows_x;
    args.nunits             = (int) nunits;
    args.nb32               = (int) nb32;
    args.nchannels_y        = (int) std::max<int64_t>(1, nchannels_y);
    args.nslots             = (int) nslots;
    args.ntok               = (int) ntok;
    args.ids_stride         = (int) ids_stride;
    args.glu                = fused && fusion.glu_op == GGML_GLU_OP_GEGLU ? 1 : 0;
    args.iround             = moetc_config().iround ? 1 : 0;

    // warps splitting K: follows K only (so does the f32 order of a column)
    const int kw = cfg.kw > 0 ? cfg.kw : nunits >= 8 ? 4 : nunits >= 4 ? 2 : 1;

    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("int8 tensor-core MoE: multi-token MUL_MAT_IDs on int8 tensor cores (first: %s%s, %d tokens, K %lld, %d K warps; "
            "GGML_CUDA_MOE_TC=0 off)\n", ggml_type_name(type), fused ? " fused gate" : "", (int) ntok, (long long) ncols_x, kw);
    });
    if (cfg.log) {
        GGML_LOG_INFO("int8 tensor-core MoE: %s%s %d tokens %d slots rows %lld K %lld kw %d\n", ggml_type_name(type), fused ? " fused" : "",
            (int) ntok, (int) nslots, (long long) nrows_x, (long long) ncols_x, kw);
    }

    switch (type) {
        case GGML_TYPE_Q4_K:   moetc_launch_type<GGML_TYPE_Q4_K>  (args, kw, fused, stream); break;
        case GGML_TYPE_Q5_K:   moetc_launch_type<GGML_TYPE_Q5_K>  (args, kw, fused, stream); break;
        case GGML_TYPE_Q6_K:   moetc_launch_type<GGML_TYPE_Q6_K>  (args, kw, fused, stream); break;
        case GGML_TYPE_IQ4_XS: moetc_launch_type<GGML_TYPE_IQ4_XS>(args, kw, fused, stream); break;
        case GGML_TYPE_Q8_0:   moetc_launch_type<GGML_TYPE_Q8_0>  (args, kw, fused, stream); break;
        case GGML_TYPE_IQ4_NL: moetc_launch_type<GGML_TYPE_IQ4_NL>(args, kw, fused, stream); break;
        case GGML_TYPE_Q5_0:   moetc_launch_type<GGML_TYPE_Q5_0>  (args, kw, fused, stream); break;
        case GGML_TYPE_IQ3_XXS: moetc_launch_type<GGML_TYPE_IQ3_XXS>(args, kw, fused, stream); break;
        case GGML_TYPE_IQ2_S:  moetc_launch_type<GGML_TYPE_IQ2_S> (args, kw, fused, stream); break;
        default: return false;
    }
    return true;
#else
    GGML_UNUSED_VARS(type, vx, vy, ids, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x,
        stride_channel_y, stride_channel_dst, nchannels_y, nslots, ntok, ids_stride, stream);
    return false;
#endif
}
