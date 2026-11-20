// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "common.cuh"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <climits>
#include <cstdint>

#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K             256
#define MMQ_ITER_K_FP4         512
#define MMQ_NWARPS               8

typedef void (*ggml_cuda_mmq_load_tiles_t)(const char * __restrict__ x, int * x_tile, const int kbx0, const int i_max, const int stride);
typedef void (*ggml_cuda_mmq_vec_dot_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00);
typedef void (*ggml_cuda_mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ get_rows_to_sorted,
    float * __restrict__ dst, const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max);

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

static constexpr int QK8_1_MMQ  = 4*QK8_1;
static constexpr int QK_FP4_MMQ = 2*QK8_1_MMQ;

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[QK8_1_MMQ];
};

// this struct is used for fp4 data types (currently only used for Blackwell)
// mxfp4 has block size 32, each int32 of d4 contains 2 e8m0 scales in the lower 16 bits
// nvfp4 has block size 16, each int32 of d4 contains 4 ue4m3 scales
struct block_fp4_mmq {
    uint32_t d4[4];
    int8_t   qs[QK_FP4_MMQ / 2];
};

static_assert(sizeof(block_q8_1_mmq) == QK8_1_MMQ + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_fp4_mmq)  == sizeof(block_q8_1_mmq),    "Unexpected block_fp4_mmq size");

static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q5_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q5_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q8_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_MXFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_NVFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q2_K:
            return MMQ_Q8_1_DS_LAYOUT_D2S6;
        case GGML_TYPE_Q3_K:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_IQ1_S:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        default:
            GGML_ABORT("fatal error");
    }
}

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

// Decouple shared memory tile sizes from WARP_SIZE to allow for different warp sizes.
// The K dimension of the tiles has either,
// 1*MMQ_TILE_NE_K==32 (always for TILE_Y_K) or 2*MMQ_TILE_NE_K==64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// In other words, the size of the quantized data in the K dimension is a multiple of MMQ_TILE_NE_K.
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma.
#define MMQ_TILE_NE_K 32

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)
#define MMQ_TILE_Y_K     (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)
#define MMQ_TILE_Y_FP4_K MMQ_TILE_Y_K

enum ggml_cuda_mmq_sram_layout {
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_FP4,   // MXFP4 and NVFP4 on Blackwell.
    GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, // Generic NVFP4
};

static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_sram_layout sram_layout) {
    switch (sram_layout) {
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_1 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K         + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2       + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K   + MMQ_TILE_NE_K/8 + 7;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_FP4:
            return 2*MMQ_TILE_NE_K + 8                     + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2       + 4;
        default:
            return -1;
    }
}

static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_FP4)   % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4) % 8 == 4, "Wrong padding.");

static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_FP4) == ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1), "Wrong tile size for MXFP4");

// Config options for the MMQ kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct ggml_cuda_mmq_config {
    ggml_type                 type;        // src0->type
    int                       nthreads;    // Number of threads per CUDA block.
    int                       occupancy;   // Targeted occupancy for the MMA kernel.
    int                       I;           // SRAM tile width in src0->ne[1]/dst->ne[0] direction.
    int                       J;           // SRAM tile width in src1->ne[1]/dst->ne[1] direction.
    ggml_cuda_mmq_sram_layout sram_layout; // SRAM tile length in src0->ne[0]/src1->ne[0] direction (physical 32 bit elements).
    int                       K_vram;      // VRAM tile length in src0->ne[0]/src1->ne[0] direction (logical elements).
    bool                      stream_k;    // Whether or not to use stream-k decomposition.
    bool                      fallback;    // Whether a fallback for out-of-bounds check in src0->ne[1] direction is needed.

    constexpr __host__ __device__ ggml_cuda_mmq_config(
            ggml_type type, int nthreads, int occupancy, int I, int J, ggml_cuda_mmq_sram_layout sram_layout, int K_vram, bool stream_k, bool fallback) :
        type(type), nthreads(nthreads), occupancy(occupancy), I(I), J(J), sram_layout(sram_layout), K_vram(K_vram), stream_k(stream_k), fallback(fallback) {}

    constexpr __device__ int rows_per_warp() const {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        return 16;
#else
        return J >= 48 && J % 16 == 0 ? 32 : 16;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    // TODO transition all combinations of GPUs and quantizations to the MMA data layout.
    __host__ int use_mma_data_layout(const int cc) const {
        if (amd_mfma_available(cc) || amd_wmma_available(cc) || turing_mma_available(cc)) {
            return true;
        }
        return false;
    }

    constexpr __device__ bool use_mma_data_layout() const {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
        return true;
#else
        return false;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    }

};

#define CASE(type_, nthreads_, occupancy_, I_, J_, sram_layout_, K_vram_, stream_k_, fallback_)                                           \
    if (type == (type_) && J == (J_) && fallback == (fallback_)) {                                                                        \
        static_assert((nthreads_) %  32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                \
        static_assert(                          (occupancy_)      <=   8, "bad occupancy");                                               \
        static_assert((I_)        %  32 == 0,                             "bad I");                                                       \
        static_assert((J_)        %   8 == 0 && (J_)              <= 128, "bad J");                                                       \
        static_assert((K_vram_)   % 256 == 0,                             "bad K_vram");                                                  \
        return ggml_cuda_mmq_config((type_), (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_), (stream_k_), (fallback_)); \
    }                                                                                                                                     \

#include "mmq-config-pascal-older.cuh"
#include "mmq-config-pascal-dp4a.cuh"
#include "mmq-config-ampere.cuh"
#include "mmq-config-blackwell.cuh"

#include "mmq-config-gcn.cuh"
#include "mmq-config-cdna.cuh"
#include "mmq-config-rdna2.cuh"
#include "mmq-config-rdna3.cuh"
#include "mmq-config-rdna3-5.cuh"
#include "mmq-config-rdna4.cuh"

#undef CASE

static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return ggml_cuda_mmq_get_config_gcn(type, J, fallback);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return ggml_cuda_mmq_get_config_cdna(type, J, fallback);
        }
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return ggml_cuda_mmq_get_config_rdna4(type, J, fallback);
        }
        if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
            return ggml_cuda_mmq_get_config_rdna3_5(type, J, fallback);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {  // covers RDNA 3.0
            return ggml_cuda_mmq_get_config_rdna3(type, J, fallback);
        }
        return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
    }
    if (blackwell_mma_available(cc)) {
        // only src1 at Q4 uses the native FP4 config, higher precisions keep src1 at Q8_1
        if (prec_src1 != GGML_PREC_Q4 && (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4)) {
            return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
        }
        return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
    }
    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {
        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    }
    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A) {
        return ggml_cuda_mmq_get_config_pascal_dp4a(type, J, fallback);
    }
    return ggml_cuda_mmq_get_config_pascal_older(type, J, fallback);
}

static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
#ifdef GGML_USE_HIP
#ifdef GCN
    return ggml_cuda_mmq_get_config_gcn(type, J, fallback);
#elif defined(CDNA)
    return ggml_cuda_mmq_get_config_cdna(type, J, fallback);
#elif defined(RDNA4)
    return ggml_cuda_mmq_get_config_rdna4(type, J, fallback);
#elif defined(RDNA3_5)
    return ggml_cuda_mmq_get_config_rdna3_5(type, J, fallback);
#elif defined(RDNA3)
    return ggml_cuda_mmq_get_config_rdna3(type, J, fallback);
#else
    return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
#endif // CDNA
#else
#ifdef BLACKWELL_MMA_AVAILABLE
    // only src1 at Q4 uses the native FP4 config, higher precisions keep src1 at Q8_1
    if (prec_src1 != GGML_PREC_Q4 && (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4)) {
        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    }
    return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return ggml_cuda_mmq_get_config_pascal_dp4a(type, J, fallback);
#else
    return ggml_cuda_mmq_get_config_pascal_older(type, J, fallback);
#endif // BLACKWELL_MMA_AVAILABLE
#endif // GGML_USE_HIP
    GGML_UNUSED_VARS(type, J, fallback, prec_src1);
}

static __host__ int ggml_cuda_mmq_get_type(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1).type;
}

static constexpr __device__ int ggml_cuda_mmq_get_type(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).type;
}

static constexpr __device__ int ggml_cuda_mmq_get_nthreads(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).nthreads;
}

static constexpr __device__ int ggml_cuda_mmq_get_occupancy(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).occupancy;
}

static __host__ int ggml_cuda_mmq_get_I(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1).I;
}

static constexpr __device__ int ggml_cuda_mmq_get_I(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).I;
}

static __host__ int ggml_cuda_mmq_get_J(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1).J;
}

static constexpr __device__ int ggml_cuda_mmq_get_J(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).J;
}

static __host__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1).sram_layout;
}

static constexpr __device__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).sram_layout;
}

static __host__ int ggml_cuda_mmq_get_K_vram(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1).K_vram;
}

static constexpr __device__ int ggml_cuda_mmq_get_K_vram(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).K_vram;
}

static __host__ bool ggml_cuda_mmq_get_stream_k(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1).stream_k;
}

static constexpr __device__ bool ggml_cuda_mmq_get_stream_k(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).stream_k;
}

static __host__ int ggml_cuda_mmq_get_fallback(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1).fallback;
}

static constexpr __device__ int ggml_cuda_mmq_get_fallback(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).fallback;
}

// ---------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_mmq_get_sram_stride(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback, cc, prec_src1));
}

static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback, prec_src1));
}

static __host__ bool ggml_cuda_mmq_needs_fallback(const int64_t nrows_x) {
    return nrows_x % 128 != 0;
}

static constexpr __device__ int ggml_cuda_mmq_get_rows_per_warp(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {
    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).rows_per_warp();
}

#define MMQ_DP4A_TXS_Q4_0    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_0   + I/QI4_0,     0}
#define MMQ_DP4A_TXS_Q4_1    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_1   + I/QI4_1,     0}
#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*2/QI8_0 + I/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*4/QI8_0 + I/(QI8_0/4), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*2/QI8_1 + I/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q2_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K         + I,           0}
#define MMQ_DP4A_TXS_Q3_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I,                                     I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_K,                 I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K/QI5_K   + I/QI5_K,     I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q6_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K/QI6_K   + I/QI6_K,     I*MMQ_TILE_NE_K/8 + I/8}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int I) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q2_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q4_0:    return MMQ_DP4A_TXS_Q4_0;
        case GGML_TYPE_Q4_1:    return MMQ_DP4A_TXS_Q4_1;
        case GGML_TYPE_Q5_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q5_1:    return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_Q8_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_MXFP4:   return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_NVFP4:   return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_Q2_K:    return MMQ_DP4A_TXS_Q2_K;
        case GGML_TYPE_Q3_K:    return MMQ_DP4A_TXS_Q3_K;
        case GGML_TYPE_Q4_K:    return MMQ_DP4A_TXS_Q4_K;
        case GGML_TYPE_Q5_K:    return MMQ_DP4A_TXS_Q5_K;
        case GGML_TYPE_Q6_K:    return MMQ_DP4A_TXS_Q6_K;
        case GGML_TYPE_IQ2_XXS: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_XS:  return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ2_S:   return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ3_XXS: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ3_S:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ1_S:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_XS:  return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_NL:  return MMQ_DP4A_TXS_Q8_0;
        default:                return tile_x_sizes{0, 0, 0};
    }
}

// FIXME temporary until all combinations of data types and GPUs can use the MMA data layout
static __host__ int ggml_cuda_mmq_get_nbytes_shared_x(const ggml_cuda_mmq_config & config, const int cc) {
    if (config.use_mma_data_layout(cc)) {
        return config.I * ggml_cuda_mmq_get_sram_stride(config.sram_layout) * 4;
    }
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(config.type, config.I);
    return (txs.qs + txs.dm + txs.sc) * 4;
}

// ------------------------------------------------------------

#include "mmq-load-tiles.cuh"
#include "mmq-vec-dot.cuh"

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1> static __device__ __forceinline__ void ggml_cuda_mmq_write_back_dp4a(
        const float * __restrict__ sum, const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
        const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);

    const bool y_scale_used = y_scale != nullptr;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += warp_size) {
            const int i = i0 + threadIdx.x;

            if (fallback && i > i_max) {
                continue;
            }

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (y_scale_used) {
                    dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
                } else {
                    dst[ids_dst[j]*stride + i] = sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
                }
            } else {
                dst[ids_dst[j]*stride + i] = sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
                GGML_UNUSED(y_scale_used);
            }
        }
    }
}

template<ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
            const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
            const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max) {

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
#else
    typedef tile<16,  8, int> tile_C;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, prec_src1);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);

    const bool y_scale_used = y_scale != nullptr;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (fallback && i > i_max) {
                    continue;
                }

                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (y_scale_used) {
                        dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    } else {
                        dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    }
                } else {
                    dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    GGML_UNUSED(y_scale_used);
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------

// TODO remove this struct and use ggml_cuda_mmq_sram_layout instead.
struct ggml_cuda_mmq_util_funcs {
    int              vdr;
    ggml_cuda_mmq_load_tiles_t load_tiles;
    ggml_cuda_mmq_vec_dot_t    vec_dot;
    ggml_cuda_mmq_write_back_t write_back;

    constexpr __host__ __device__ ggml_cuda_mmq_util_funcs(
            int vdr, ggml_cuda_mmq_load_tiles_t load_tiles, ggml_cuda_mmq_vec_dot_t vec_dot, ggml_cuda_mmq_write_back_t write_back) :
        vdr(vdr), load_tiles(load_tiles), vec_dot(vec_dot), write_back(write_back) {}
};

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_funcs() {
    if (!ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).use_mma_data_layout()) {
        switch (type) {
            case GGML_TYPE_Q1_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q1_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q1_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q2_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q2_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q2_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q4_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q4_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q4_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q4_1:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q4_1_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q4_1<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q4_1_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q5_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q5_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q5_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q5_1:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q5_1_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q5_1<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q8_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q8_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
// ---------------------------------------------------------------------------------------------
            case GGML_TYPE_Q2_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q2_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q2_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q3_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q3_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q3_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q3_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q4_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q4_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q5_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q5_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q5_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q5_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q6_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q6_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
// ---------------------------------------------------------------------------------------------
            case GGML_TYPE_IQ1_S:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ1_S_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq1_s<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_IQ2_XXS:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ2_XXS_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq2_xxs<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_IQ2_XS:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ2_XS_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq2_xs<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_IQ2_S:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ2_S_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq2_s<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_IQ3_XXS:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ3_XXS_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq3_xxs<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_IQ3_S:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ3_S_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq3_s<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_IQ4_XS:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ4_XS_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq4_xs<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_IQ4_NL:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ4_NL_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq4_nl<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
// ---------------------------------------------------------------------------------------------
            case GGML_TYPE_MXFP4:
                return ggml_cuda_mmq_util_funcs(
                    VDR_MXFP4_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_mxfp4<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_NVFP4:
                return ggml_cuda_mmq_util_funcs(
                    VDR_NVFP4_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback, prec_src1>,
                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            default:
                return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
        }
    }

// ---------------------------------------------------------------------------------------------

#ifdef BLACKWELL_MMA_AVAILABLE
    switch (type) {
        case GGML_TYPE_MXFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                return ggml_cuda_mmq_util_funcs(
                    -1,
                    ggml_cuda_mmq_load_tiles_mxfp4_fp4<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
                    ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
            }
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                return ggml_cuda_mmq_util_funcs(
                    -1,
                    ggml_cuda_mmq_load_tiles_nvfp4_nvfp4<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
                    ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
            }
            break;
        default:
            break;
    }
#endif // BLACKWELL_MMA_AVAILABLE

// ---------------------------------------------------------------------------------------------

    switch (type) {
        case GGML_TYPE_Q1_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q1_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q2_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q2_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q4_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q4_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_DS4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q4_1:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q4_1<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q5_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q5_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q5_1:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q5_1<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q8_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
// ---------------------------------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q2_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q3_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q3_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback, prec_src1>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q4_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q5_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q5_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_Q6_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
// ---------------------------------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq1_s<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_IQ2_XXS:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq2_xxs<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_IQ2_XS:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq2_xs<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback, prec_src1>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_IQ2_S:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq2_s<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback, prec_src1>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_IQ3_XXS:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq3_xxs<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_IQ3_S:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq3_s<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_IQ4_XS:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq4_xs<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_IQ4_NL:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq4_nl<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
// ---------------------------------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_mxfp4<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        case GGML_TYPE_NVFP4:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback, prec_src1>,
                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback, prec_src1>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1>);
        default:
            return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
    }
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static constexpr __device__ int ggml_cuda_mmq_get_vdr() {
    return ggml_cuda_mmq_get_util_funcs<type, J, fallback, prec_src1>().vdr;
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static constexpr __device__ ggml_cuda_mmq_load_tiles_t ggml_cuda_mmq_get_load_tiles() {
    return ggml_cuda_mmq_get_util_funcs<type, J, fallback, prec_src1>().load_tiles;
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static constexpr __device__ ggml_cuda_mmq_vec_dot_t ggml_cuda_mmq_get_vec_dot() {
    return ggml_cuda_mmq_get_util_funcs<type, J, fallback, prec_src1>().vec_dot;
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static constexpr __device__ ggml_cuda_mmq_write_back_t ggml_cuda_mmq_get_write_back() {
    return ggml_cuda_mmq_get_util_funcs<type, J, fallback, prec_src1>().write_back;
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int              nwarps     = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              I          = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback, prec_src1>();
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback, prec_src1>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback, prec_src1>();

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + J;
    int * tile_x = tile_y + GGML_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);

#if defined(BLACKWELL_MMA_AVAILABLE)
    // FP4 tile stores 8 blocks. src1 above Q4 uses the generic
    // Q8_1 tile layout instead of the packed FP4 tile.
    constexpr int ne_block = ((type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) && prec_src1 == GGML_PREC_Q4) ?
        QK_FP4_MMQ : QK8_1_MMQ;
#else
    constexpr int ne_block = QK8_1_MMQ;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J*I / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J);
    } else {
        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

// k19-mmq-pipe: the same tile as mul_mat_q_process_tile with the loads issued earlier and more blocks per SM. Every vec_dot call gets
// the same tile_x / tile_y contents in the same order, so each output element sees the same K accumulation (bit-identical); only
// when a load is issued and where a lookup table is read from change.
//   MMQ_PIPE_YREG: the next half of the y tile is read into registers (16-byte loads) while vec_dot runs on the current one, and stored
//                  to shared memory after the barrier (the base copies global -> shared after the barrier, its latency exposed).
//   MMQ_PIPE_L2X:  the x rows and y columns of the next k iteration are prefetched into L2 at the start of the current one.
//   MMQ_PIPE_OCC2: the kernel is compiled for 2 blocks per SM (launch bounds, <= 128 registers), so one block's loads and conversion
//                  overlap the other's MMA. Used only where 2 blocks fit and the grid has one block per tile.
//   MMQ_PIPE_TAB:  IQ3_S only: the 2 KiB grid table is read from a copy in shared memory instead of through L1 (at 2 blocks per SM the
//                  L1 left beside the tiles does not hold it).
// The stream-k decomposition, the tile shapes and the grid are those of the base launch.
#define MMQ_PIPE_YREG 1
#define MMQ_PIPE_L2X  2
#define MMQ_PIPE_OCC2 4
#define MMQ_PIPE_TAB  8

static __device__ __forceinline__ void mmq_prefetch_l2(const void * p) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
    GGML_UNUSED(p);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1, int pipe>
static __device__ __forceinline__ void mul_mat_q_process_tile_pipe(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int              nthreads   = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1);
    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              bs         = ggml_cuda_type_traits<type>::bs;
    constexpr int              I          = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback, prec_src1>();
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback, prec_src1>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback, prec_src1>();

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + J;
    int * tile_x = tile_y + GGML_PAD(J*MMQ_TILE_Y_K, nthreads);

#if defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int ne_block = ((type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) && prec_src1 == GGML_PREC_Q4) ?
        QK_FP4_MMQ : QK8_1_MMQ;
#else
    constexpr int ne_block = QK8_1_MMQ;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
    constexpr int blocks_per_iter = ITER_K / qk;
    constexpr int sz              = sizeof(block_q8_1_mmq) / sizeof(int);

    // the base copy loop moves GGML_PAD(J*MMQ_TILE_Y_K, nthreads) ints per half (the padding reads the next columns, as there)
    constexpr int ny  = GGML_PAD(J*MMQ_TILE_Y_K, nthreads);
    constexpr int ny4 = ny/4;
    constexpr int nyr = (ny4 + nthreads - 1)/nthreads;
    static_assert(ny % 4 == 0, "bad y tile size");

    const int tid = threadIdx.y*warp_size + threadIdx.x;

    float sum[J*I / nthreads] = {0.0f};

    int4 yr[(pipe & MMQ_PIPE_YREG) ? nyr : 1];

    // the two halves of the y tile of k iteration kb0, as the base addresses them
    auto y_src = [&](const int kb0, const int half) -> const int * {
        return half == 0 ? y + ncols_y * (kb0 * qk / ne_block) * sz : y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
    };
    auto y_load = [&](const int * src) {
#pragma unroll
        for (int r = 0; r < nyr; ++r) {
            const int idx = r*nthreads + tid;
            if (ny4 % nthreads == 0 || idx < ny4) {
                yr[r] = ((const int4 *) src)[idx];
            }
        }
    };
    auto y_store = [&]() {
#pragma unroll
        for (int r = 0; r < nyr; ++r) {
            const int idx = r*nthreads + tid;
            if (ny4 % nthreads == 0 || idx < ny4) {
                ((int4 *) tile_y)[idx] = yr[r];
            }
        }
    };
    auto y_copy = [&](const int * by0) {
#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nthreads) {
            const int l = l0 + tid;
            tile_y[l] = by0[l];
        }
    };
    auto prefetch_next = [&](const int kb0n) {
        constexpr int row_bytes = blocks_per_iter*bs;
        constexpr int nlines_x  = (row_bytes + 127)/128 + 1; // + 1: the row's bytes need not start on a line
        for (int t = tid; t < I*nlines_x; t += nthreads) {
            const int i = fallback ? min(t / nlines_x, tile_x_max_i) : t / nlines_x;
            const char * row = x + (int64_t(offset_x) + kb0n + int64_t(i)*stride_row_x)*bs;
            const char * p   = (const char *) (uintptr_t(row) & ~uintptr_t(127)) + (t % nlines_x)*128;
            if (p < row + row_bytes) {
                mmq_prefetch_l2(p);
            }
        }
        constexpr int nlines_y = (J*MMQ_TILE_Y_K*4 + 127)/128 + 1;
        for (int t = tid; t < 2*nlines_y; t += nthreads) {
            const char * col = (const char *) y_src(kb0n, t / nlines_y);
            const char * p   = (const char *) (uintptr_t(col) & ~uintptr_t(127)) + (t % nlines_y)*128;
            if (p < col + J*MMQ_TILE_Y_K*4) {
                mmq_prefetch_l2(p);
            }
        }
    };

    constexpr bool tab = (pipe & MMQ_PIPE_TAB) && type == GGML_TYPE_IQ3_S;
    if constexpr (tab) {
        // the previous tile of this block (stream-k) ended with a barrier, so nothing reads the copy while it is rewritten
        for (int t = tid; t < 512; t += nthreads) {
            mmq_iq3s_grid_shared[t] = iq3s_grid[t];
        }
        __syncthreads();
    }

    if constexpr (pipe & MMQ_PIPE_YREG) {
        y_load(y_src(kb0_start, 0));
    }

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        const bool more = kb0 + blocks_per_iter < kb0_stop;
        if constexpr (pipe & MMQ_PIPE_L2X) {
            if (more) {
                prefetch_next(kb0 + blocks_per_iter);
            }
        }

        if constexpr (tab) {
            ggml_cuda_mmq_load_tiles_iq3_s<type, J, fallback, true>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        } else {
            load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        }
        if constexpr (pipe & MMQ_PIPE_YREG) {
            y_store();
        } else {
            y_copy(y_src(kb0, 0));
        }

        __syncthreads();

        if constexpr (pipe & MMQ_PIPE_YREG) {
            y_load(y_src(kb0, 1));
        }
        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        if constexpr (pipe & MMQ_PIPE_YREG) {
            y_store();
        } else {
            y_copy(y_src(kb0, 1));
        }

        __syncthreads();

        if constexpr (pipe & MMQ_PIPE_YREG) {
            if (more) {
                y_load(y_src(kb0 + blocks_per_iter, 0));
            }
        }
        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J);
    } else {
        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1, int pipe>
static __device__ __forceinline__ void mul_mat_q_process_tile_sel(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {
    if constexpr (pipe == 0) {
        mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1>
            (x, offset_x, y, ids_dst, dst, tmp_fixup, y_scale, stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
    } else {
        mul_mat_q_process_tile_pipe<type, J, fallback, fixup, prec_src1, pipe>
            (x, offset_x, y, ids_dst, dst, tmp_fixup, y_scale, stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
    }
}

// Compact MoE tiling: the column tiles of all experts are numbered consecutively, expert e starts at tile expert_bounds[e]/J + e.
// Tiles past the last tile of an expert are empty (jt*J >= col_diff). Finds the expert of tile v by a warp-wide search.
// k15-mm2: with tile_prefix (mmq_moe_tile_prefix) expert e starts at tile tile_prefix[e] = expert_bounds[e]/J + the number of non-empty
// experts before e, so an expert without columns gets no tile (a bank of MoE cache slots read in place has mostly empty ones). The starts
// are then non-decreasing; the search below finds the last expert starting at or before v, which is the non-empty one
template <int J>
static __device__ __forceinline__ int mmq_moe_tile_start(const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_prefix, const int e) {
    return tile_prefix ? tile_prefix[e] : expert_bounds[e]/J + e;
}

static __device__ __forceinline__ uint3 mmq_fastdiv_values_dev(const uint32_t d) {
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }
    const uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

template <int J>
static __device__ __forceinline__ void mmq_moe_compact_tile(
        const int32_t * __restrict__ expert_bounds, const int nexperts, const int v, int & zt, int & jt,
        const int32_t * __restrict__ tile_prefix = nullptr) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = threadIdx.x % warp_size;

    int lo = 0;        // tile_start(lo) <= v holds throughout since tile_start(0) == 0
    int n  = nexperts; // the expert is in [lo, lo + n)
    while (n > 1) {
        const int step = (n + warp_size - 1) / warp_size;
        const int e    = lo + lane*step;
        const bool le  = e < lo + n && mmq_moe_tile_start<J>(expert_bounds, tile_prefix, e) <= v;
        const int  nle = warp_reduce_sum<warp_size>(int(le)); // tile_start is increasing: the lanes with le are a prefix
        const int  lo_new = lo + (nle - 1)*step;
        n  = min(step, lo + n - lo_new);
        lo = lo_new;
    }
    zt = lo;
    jt = v - mmq_moe_tile_start<J>(expert_bounds, tile_prefix, lo);
}


// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

// k19-mmq-pipe: the kernel body, shared by mul_mat_q (pipe 0, the base) and mul_mat_q_pipe
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1, int pipe>
static __device__ __forceinline__ void mul_mat_q_body(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx_host, const int nexperts_compact, const int32_t * __restrict__ tile_prefix, const int exact_nsm) {
    // k15-mm2: tile_prefix: the column tile count is known on the device only (mmq_moe_tile_prefix), the stream-k split is over it.
    // k19-p5mmq: exact_nsm > 0 (stream-k kernel, skip_empty): the prefix kernel also wrote the exact prefix and chose the form (see
    // mmq_moe_tile_prefix): one block per non-empty tile over the exact prefix, or the stream-k split over exact_nsm blocks as before
    const bool exact_tiles = exact_nsm > 0 && tile_prefix[2*nexperts_compact + 2] != 0;
    const int32_t * prefix = exact_tiles ? tile_prefix + nexperts_compact + 1 : tile_prefix;
    const uint3 ntx = prefix ? mmq_fastdiv_values_dev(prefix[nexperts_compact]) : ntx_host;

    // Skip unused template specializations for faster compilation:
    if (ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);

    const uint32_t nty = (nrows_x + I - 1) / I; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback, prec_src1)) {
        const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
        const int wt = tmp2.x;
        int       zt = tmp2.y;
        int       jt = blockIdx.y;
        const int it = blockIdx.x;
        if (nexperts_compact > 0) {
            mmq_moe_compact_tile<J>(expert_bounds, nexperts_compact, jt, zt, jt, prefix);
        }

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
        int offset_y_scale;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
        } else {
            GGML_UNUSED(offset_y_scale);
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) {
                offset_y_scale = 0;
            }

            if (jt*J >= col_diff) {
                return;
            }

            // __syncthreads(); // There is no previous tile that could cause a race condition.
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > J && j >= J) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
            }
            __syncthreads();
        }

        offset_y   += (col_low + jt*J)*(sizeof(block_q8_1_mmq)/sizeof(int));
        offset_dst += it*I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt*J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

        constexpr bool fixup = false;
        mul_mat_q_process_tile_sel<type, J, fallback, fixup, prec_src1, pipe>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
        return;
    }

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
    constexpr int blocks_per_iter = ITER_K / qk;

    // k19-p5mmq: with exact_nsm the grid is the host's upper bound of the tile count: one tile per block (the blocks past the exact
    // count have nothing), or the base split over exact_nsm blocks (the others have nothing)
    const int nblocks = exact_nsm > 0 ? (exact_tiles ? int(nsamples_y.z*nchannels_y.z*ntx.z*nty) : exact_nsm) : int(gridDim.x);
    if (int(blockIdx.x) >= nblocks) {
        return;
    }

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / nblocks;
    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / nblocks;

    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        int tmp = fastdiv(kbc, blocks_per_ne00);
        uint2 tmp2 = fast_div_modulo(tmp, ntx);
        int jt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nchannels_y);
        int zt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nsamples_y);
        const int wt = tmp2.y;
        const int it = tmp2.x;
        if (nexperts_compact > 0) {
            mmq_moe_compact_tile<J>(expert_bounds, nexperts_compact, jt, zt, jt, prefix);
        }

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
        int offset_y_scale;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
        } else {
            GGML_UNUSED(offset_y_scale);
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) {
                offset_y_scale = 0;
            }

            if (jt*J >= col_diff) {
                kbc += blocks_per_ne00.z;
                kbc -= fastmodulo(kbc, blocks_per_ne00);

                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));

                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > J && j >= J) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it*I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt * J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile_sel<type, J, fallback, fixup, prec_src1, pipe>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = fastdiv(kbc, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;
    if (nexperts_compact > 0) {
        mmq_moe_compact_tile<J>(expert_bounds, nexperts_compact, jt, zt, jt, prefix);
    }

    // Defaults for regular matrix multiplication:
    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
    int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
    int offset_y_scale;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
    } else {
        GGML_UNUSED(offset_y_scale);
    }

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y   = 0;
        offset_dst = 0;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = 0;
        }

        if (jt*J >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
            const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

            if (j0 + nwarps*warp_size > J && j >= J) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it*I;
    const float * y_scale_tile = nullptr;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale += col_low + jt * J;
        y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
    }

    const int tile_x_max_i = nrows_x  - it*I - 1;
    const int tile_y_max_j = col_diff - jt*J - 1;

    const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile_sel<type, J, fallback, fixup, prec_src1, pipe>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1), ggml_cuda_mmq_get_occupancy(type, J, fallback, prec_src1))
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx_host, const int nexperts_compact, const int32_t * __restrict__ tile_prefix, const int exact_nsm) {
    mul_mat_q_body<type, J, fallback, prec_src1, 0>(
        x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale,
        blocks_per_ne00, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst,
        channel_ratio, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst,
        ntx_host, nexperts_compact, tile_prefix, exact_nsm);
}

// k19-mmq-pipe: compiled for sm 120 (consumer Blackwell) and, k19-a100tc, sm 80 (A100), k19-p5mmq, sm 75 (T4); the host launches it
// there only (mmq_pipe_kernel)
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1, int pipe>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1), (pipe & MMQ_PIPE_OCC2) ? 2 : 1)
static __global__ void mul_mat_q_pipe(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx_host, const int nexperts_compact, const int32_t * __restrict__ tile_prefix, const int exact_nsm) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && \
    ((__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN) || __CUDA_ARCH__ == GGML_CUDA_CC_AMPERE || \
     __CUDA_ARCH__ == GGML_CUDA_CC_TURING)
    mul_mat_q_body<type, J, fallback, prec_src1, pipe>(
        x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale,
        blocks_per_ne00, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst,
        channel_ratio, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst,
        ntx_host, nexperts_compact, tile_prefix, exact_nsm);
#else
    GGML_UNUSED_VARS(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, blocks_per_ne00, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, channel_ratio, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, ntx_host, nexperts_compact, tile_prefix, exact_nsm);
    NO_DEVICE_CODE;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (sm 12x || sm 80 || sm 75)
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1)/2, 1)
static __global__ void mul_mat_q_stream_k_fixup(
        const int32_t * ids_dst, const int32_t * expert_bounds, float * dst,
        float * tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
        const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
        const int stride_sample_dst, const uint3 ntx_host, const int nexperts_compact, const int32_t * __restrict__ tile_prefix,
        const int exact_nsm) {
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / 2) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
    constexpr int blocks_per_iter = ITER_K / qk;

    // k13-pp: launched with PDL (programmatic dependent launch) behind the stream-k kernel, so its launch overlaps that kernel's tail:
    // wait here for the stream-k kernel's partial sums (a no-op without PDL)
    ggml_cuda_pdl_sync();
    // k19-p5mmq: one block per tile chosen (mmq_moe_tile_prefix): nothing to fix up. Otherwise the grid is the exact_nsm blocks of the split
    if (exact_nsm > 0 && tile_prefix[2*nexperts_compact + 2] != 0) {
        return;
    }
    // k15-mm2: tile_prefix: the column tile count is known on the device only (mmq_moe_tile_prefix), the stream-k split is over it;
    // read after the PDL wait like every other input
    const uint3 ntx = tile_prefix ? mmq_fastdiv_values_dev(tile_prefix[nexperts_compact]) : ntx_host;

    float sum[J / nwarps] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int nty = (nrows_x + I - 1) / I;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;


#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(J*I) + j*I + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    int tmp = fastdiv(kbc0, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;
    if (nexperts_compact > 0) {
        mmq_moe_compact_tile<J>(expert_bounds, nexperts_compact, jt, zt, jt, tile_prefix);
    }

    if (!ids_dst) {
        const int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst + it*I;
        dst += offset_dst;

        const int i_max = nrows_x   - it*I - 1;
        const int j_max = ncols_dst - jt*J - 1;
        if (fallback && i > i_max) {
            return;
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j > j_max) {
                return;
            }

            dst[j*stride_col_dst + i] += sum[j0/nwarps];
        }
        return;
    }

    __shared__ int ids_dst_shared[J];
    const int col_low  = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;

    if (jt*J >= col_diff) {
        return; // empty tile
    }

    for (int j = threadIdx.y*warp_size + threadIdx.x; j < J; j += nwarps*warp_size) {
        ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
    }
    __syncthreads();

    const int offset_dst = it*I;
    dst += offset_dst;

    const int i_max = nrows_x  - it*I - 1;
    const int j_max = col_diff - jt*J - 1;
    if (fallback && i > i_max) {
        return;
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

        dst[ids_dst_shared[j]*stride_col_dst + i] += sum[j0/nwarps];
    }
}

// k19-p5mmq: whether a stream-k launch of ntiles output tiles runs one block per tile: the host's rule for products without
// skip_empty (launch_mul_mat_q), at least 90% of the last wave of nsm blocks filled
static __host__ __device__ __forceinline__ bool mmq_tiles_fill_sms(const int64_t ntiles, const int nsm) {
    if (ntiles <= 0) {
        return false;
    }
    const int64_t nwaves = (ntiles + nsm - 1) / nsm;
    return 100*ntiles / (nsm*nwaves) >= 90;
}

// k15-mm2: tile_prefix[e] = expert_bounds[e]/J + the number of non-empty experts before e, e in [0, nexperts]: the last entry is the
// column tile count. One block.
// k19-p5mmq: nty > 0 (exact tiles, see launch_mul_mat_q): also tile_prefix[nexperts + 1 + e] = the sum of ceil(count/J) over the
// experts before e (the exact prefix: no empty tile), and tile_prefix[2*nexperts + 2] = 1 if its nty*count tiles fill the nsm SMs
// (mmq_tiles_fill_sms: the kernels then run one block per tile over the exact prefix) or force_tiles and there is a tile, else 0 (the
// stream-k split as before)
static __global__ void mmq_moe_tile_prefix(const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_prefix,
        const int nexperts, const int J, const int nty, const int nsm, const bool force_tiles) {
    __shared__ int warp_sum[1024/WARP_SIZE];
    __shared__ int warp_sum_c[1024/WARP_SIZE];
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    int carry   = 0;
    int carry_c = 0;
    for (int e0 = 0; e0 <= nexperts; e0 += blockDim.x) {
        const int e = e0 + tid;
        const int n = e < nexperts ? expert_bounds[e + 1] - expert_bounds[e] : 0;
        const int f = n > 0 ? 1 : 0;
        const int c = (n + J - 1) / J;
        int incl   = f;
        int incl_c = c;
#pragma unroll
        for (int off = 1; off < WARP_SIZE; off *= 2) {
            const int v   = __shfl_up_sync(0xFFFFFFFF, incl,   off, WARP_SIZE);
            const int v_c = __shfl_up_sync(0xFFFFFFFF, incl_c, off, WARP_SIZE);
            incl   += lane >= off ? v   : 0;
            incl_c += lane >= off ? v_c : 0;
        }
        if (lane == WARP_SIZE - 1) {
            warp_sum[warp]   = incl;
            warp_sum_c[warp] = incl_c;
        }
        __syncthreads();
        int before   = 0;
        int total    = 0;
        int before_c = 0;
        int total_c  = 0;
        for (int w = 0; w < (int) blockDim.x/WARP_SIZE; ++w) {
            before   += w < warp ? warp_sum[w]   : 0;
            total    += warp_sum[w];
            before_c += w < warp ? warp_sum_c[w] : 0;
            total_c  += warp_sum_c[w];
        }
        if (e <= nexperts) {
            tile_prefix[e] = expert_bounds[e]/J + carry + before + incl - f;
            if (nty > 0) {
                const int start = carry_c + before_c + incl_c - c;
                tile_prefix[nexperts + 1 + e] = start;
                if (e == nexperts) {
                    tile_prefix[2*nexperts + 2] = (force_tiles && start > 0) || mmq_tiles_fill_sms(int64_t(start)*nty, nsm) ? 1 : 0;
                }
            }
        }
        carry   += total;
        carry_c += total_c;
        __syncthreads();
    }
}

// number of tiles of the compact MoE tiling, or 0 if it does not give fewer tiles or does not fit the grid
static int mmq_moe_compact_ntx(const int64_t ncols_y, const int64_t ncols_max, const int64_t nexperts, const int J) {
    const int64_t ntx_compact = ncols_y/J + nexperts;
    const int64_t ntx_expert  = (ncols_max + J - 1)/J;
    return ntx_compact < ntx_expert*nexperts && ntx_compact <= UINT16_MAX ? int(ntx_compact) : 0;
}

struct mmq_args {
    const char * x; ggml_type type_x; const int * y; const int32_t * ids_dst; const int32_t * expert_bounds; float * dst;
    const float * y_scale;
    int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst; int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
    int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
    int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
    int64_t ncols_max;
    int J_best; // Tile width in ne11(dense)/ne12(MoE) direction to use for optimal performance.
    bool skip_empty = false; // k15-mm2: the compact MoE tiling gives no tile to experts without columns (mmq_moe_tile_prefix)
};

static size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config & config, const int cc) {
    const size_t nbs_ids = config.J*sizeof(int);
    const size_t nbs_x = ggml_cuda_mmq_get_nbytes_shared_x(config, cc);
    const size_t nbs_y = config.J * (sizeof(block_q8_1_mmq));
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, config.nthreads*sizeof(int));
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// k19-mmq-pipe: GGML_CUDA_MMQ_PIPE: unset = the measured choice per type / tile width (mmq_pipe_policy), 0 = off (the base kernel),
// N > 0 = force the MMQ_PIPE_* bits N where a kernel for them is built (else the base). Only on sm 12x (consumer Blackwell) and sm 80, only for
// the MoE products (ids) of the types below at J <= 64 without the out-of-bounds fallback; everything else runs the base kernel.
static int ggml_cuda_mmq_pipe_env() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_MMQ_PIPE");
        return e == nullptr || *e == '\0' ? -1 : std::max(0, atoi(e));
    }();
    return v;
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static constexpr bool mmq_pipe_instantiated() {
    return !fallback && prec_src1 == GGML_PREC_Q8 && J <= 64 &&
        (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q4_K ||
         type == GGML_TYPE_Q5_K  || type == GGML_TYPE_Q6_K);
}

// The bits measured fastest on an RTX PRO 6000 (G4, tests/test-mmq-pipe) per type, tile width and grid form.
// one_tile_per_block: the grid has a block per output tile; otherwise it is the stream-k split over one block per SM, which has no
// second block to put on an SM
static int mmq_pipe_policy(const ggml_type type, const int J, const bool one_tile_per_block) {
    if (!one_tile_per_block) {
        return type == GGML_TYPE_IQ3_S ? (MMQ_PIPE_YREG | MMQ_PIPE_L2X | MMQ_PIPE_TAB) : 0;
    }
    switch (type) {
        case GGML_TYPE_IQ3_S:  return MMQ_PIPE_YREG | MMQ_PIPE_L2X | MMQ_PIPE_OCC2 | MMQ_PIPE_TAB;
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS: return MMQ_PIPE_YREG | MMQ_PIPE_L2X | MMQ_PIPE_OCC2;
        case GGML_TYPE_Q4_K:   return J <= 40 ? (MMQ_PIPE_YREG | MMQ_PIPE_L2X | MMQ_PIPE_OCC2) : (MMQ_PIPE_L2X | MMQ_PIPE_OCC2);
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:   return MMQ_PIPE_L2X | MMQ_PIPE_OCC2;
        default:               return 0;
    }
}

typedef void (*mmq_kernel_t)(
        const char *, const int *, const int32_t *, const int32_t *, float *, float *, const float *,
        const uint3, const int, const int, const int, const int, const int,
        const uint3, const uint3, const int, const int, const int,
        const uint3, const uint3, const int, const int, const int,
        const uint3, const int, const int32_t *, const int);

// the pipe kernel for code, with its shared memory limit raised; nullptr if its occupancy is not met (MMQ_PIPE_OCC2 needs 2 blocks per SM)
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1, int code>
static mmq_kernel_t mmq_pipe_kernel_code(const int nbytes_shared, const int nthreads) {
    static int ok[GGML_CUDA_MAX_DEVICES] = { 0 }; // 0 unknown, 1 usable, -1 not
    const int id = ggml_cuda_get_device();
    mmq_kernel_t k = mul_mat_q_pipe<type, J, fallback, prec_src1, code>;
    if (ok[id] == 0) {
        CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        int nblocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, k, nthreads, nbytes_shared));
        ok[id] = nblocks >= ((code & MMQ_PIPE_OCC2) ? 2 : 1) ? 1 : -1;
        GGML_LOG_INFO("%s: MMQ pipe %d %s J=%d: %d blocks/SM, %d B shared -> %s\n", __func__, code, ggml_type_name(type), J, nblocks,
            nbytes_shared, ok[id] == 1 ? "on" : "off (occupancy)");
    }
    return ok[id] == 1 ? k : nullptr;
}

// the codes with a kernel: the policy's (3 = TAB without OCC2 is IQ3_S's stream-k choice)
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static mmq_kernel_t mmq_pipe_kernel_switch(const int code, const int nbytes_shared, const int nthreads) {
    constexpr int YL = MMQ_PIPE_YREG | MMQ_PIPE_L2X;
    switch (code) {
        case MMQ_PIPE_L2X | MMQ_PIPE_OCC2:
            return mmq_pipe_kernel_code<type, J, fallback, prec_src1, MMQ_PIPE_L2X | MMQ_PIPE_OCC2>(nbytes_shared, nthreads);
        case YL | MMQ_PIPE_OCC2:
            return mmq_pipe_kernel_code<type, J, fallback, prec_src1, YL | MMQ_PIPE_OCC2>(nbytes_shared, nthreads);
        case YL | MMQ_PIPE_TAB:
            if constexpr (type == GGML_TYPE_IQ3_S) {
                return mmq_pipe_kernel_code<type, J, fallback, prec_src1, YL | MMQ_PIPE_TAB>(nbytes_shared, nthreads);
            }
            return nullptr;
        case YL | MMQ_PIPE_OCC2 | MMQ_PIPE_TAB:
            if constexpr (type == GGML_TYPE_IQ3_S) {
                return mmq_pipe_kernel_code<type, J, fallback, prec_src1, YL | MMQ_PIPE_OCC2 | MMQ_PIPE_TAB>(nbytes_shared, nthreads);
            }
            return nullptr;
        default:
            return nullptr;
    }
}

// k19-mmq-pipe engagement: MoE MMQ launches on the pipe kernel and on the base kernel, logged at 1, 10, 100, ... of each
static void mmq_pipe_count(const int code) {
    static std::atomic<int64_t> n_pipe{0};
    static std::atomic<int64_t> n_base{0};
    const int64_t n = (code != 0 ? n_pipe : n_base).fetch_add(1) + 1;
    int64_t p = 1;
    while (p < n) {
        p *= 10;
    }
    if (p == n) {
        GGML_LOG_INFO("ggml_cuda: MMQ pipe %s: %" PRId64 " MoE launches (%s)\n", code != 0 ? "engaged" : "not engaged", n,
            code != 0 ? "pipelined kernel" : "base kernel");
    }
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static mmq_kernel_t mmq_pipe_kernel(const mmq_args & args, const int cc, const int nbytes_shared, const int nthreads,
        const bool one_tile_per_block) {
    if (args.ids_dst == nullptr) {
        return nullptr;
    }
    const bool sm12x = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_BLACKWELL && cc < GGML_CUDA_CC_RUBIN &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL;
    // k19-a100tc: the same kernels on the A100 (cc 8.0 exactly, sm_80 code: the 164 KiB of shared memory per SM hold two 45-49 KiB
    // tiles; prefetch.global.L2 exists there). GGML_CUDA_MMQ_PIPE_SM80=0 is the kill switch of this addition (sm 12x only, as before)
    static const bool sm80_on = [] {
        const char * e = getenv("GGML_CUDA_MMQ_PIPE_SM80");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool sm80 = sm80_on && GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_AMPERE &&
        ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_AMPERE;
    // k19-p5mmq: the same kernels on Turing (cc 7.5, sm_75 code; 64 KiB of shared memory per SM: one J <= 64 tile, so OCC2 drops
    // itself where 2 blocks do not fit). GGML_CUDA_MMQ_PIPE_SM75=1 turns it on (default: off)
    static const bool sm75_on = [] {
        const char * e = getenv("GGML_CUDA_MMQ_PIPE_SM75");
        return e != nullptr && atoi(e) != 0;
    }();
    const bool sm75 = sm75_on && GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_TURING &&
        ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING;
    if (!sm12x && !sm80 && !sm75) {
        return nullptr;
    }
    mmq_kernel_t k = nullptr;
    int code = 0;
    if constexpr (mmq_pipe_instantiated<type, J, fallback, prec_src1>()) {
        const int env = ggml_cuda_mmq_pipe_env();
        if (env != 0 && uintptr_t(args.y) % 16 == 0) {
            code = env > 0 ? env : mmq_pipe_policy(type, J, one_tile_per_block);
            if (type != GGML_TYPE_IQ3_S) {
                code &= ~MMQ_PIPE_TAB;
            }
            if (!one_tile_per_block) {
                code &= ~MMQ_PIPE_OCC2;
            }
            k = code != 0 ? mmq_pipe_kernel_switch<type, J, fallback, prec_src1>(code, nbytes_shared, nthreads) : nullptr;
            if (k == nullptr && (code & MMQ_PIPE_OCC2)) {
                code &= ~MMQ_PIPE_OCC2;
                k = mmq_pipe_kernel_switch<type, J, fallback, prec_src1>(code, nbytes_shared, nthreads);
            }
        }
    } else {
        GGML_UNUSED_VARS(nbytes_shared, nthreads, one_tile_per_block);
    }
    mmq_pipe_count(k != nullptr ? code : 0);
    return k;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// k19-p5mmq engagement: MoE products over a cache bank (skip_empty) that may take exact tiles, logged at 1, 10, 100, ... (whether the
// host expected one block per tile: the device decides). GGML_CUDA_MMQ_EXACT_TILES_STATS=1 (diagnostic, synchronizes after each such
// product) also reads the device's choice and logs the counts of both forms at 1, 10, 100, ...
static void mmq_log_decades(std::atomic<int64_t> & c, const char * what) {
    const int64_t n = c.fetch_add(1) + 1;
    int64_t p = 1;
    while (p < n) {
        p *= 10;
    }
    if (p == n) {
        GGML_LOG_INFO("ggml_cuda: MMQ exact tiles %s: %" PRId64 " MoE launches (GGML_CUDA_MMQ_EXACT_TILES=0 turns it off)\n", what, n);
    }
}

static void mmq_exact_tiles_count(const bool likely) {
    static std::atomic<int64_t> n_likely{0};
    static std::atomic<int64_t> n_unlikely{0};
    mmq_log_decades(likely ? n_likely : n_unlikely, likely ? "engaged (one block per tile expected)" : "engaged (stream-k expected)");
}

static void mmq_exact_tiles_stats(const bool exact, const int32_t * tile_prefix, const int nexperts, cudaStream_t stream) {
    static const bool on = getenv("GGML_CUDA_MMQ_EXACT_TILES_STATS") != nullptr && atoi(getenv("GGML_CUDA_MMQ_EXACT_TILES_STATS")) != 0;
    if (!on || !exact) {
        return;
    }
    int32_t v[2] = { 0, 0 };
    CUDA_CHECK(cudaMemcpyAsync(v, tile_prefix + 2*nexperts + 1, 2*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    static std::atomic<int64_t> n_tiles{0};
    static std::atomic<int64_t> n_split{0};
    mmq_log_decades(v[1] ? n_tiles : n_split, v[1] ? "device form: one block per tile" : "device form: stream-k split");
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
    GGML_ASSERT(config.nthreads % warp_size == 0);
    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);

    const dim3 block_dims(warp_size, nwarps, 1);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false, prec_src1>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J,  true, prec_src1>), nbytes_shared);

    GGML_ASSERT(args.nchannels_y % args.nchannels_x == 0);
    GGML_ASSERT(args.nsamples_y  % args.nsamples_x  == 0);
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio  = args.nsamples_y  / args.nsamples_x;

    const int nty = (args.nrows_x   + config.I - 1) / config.I;
    int       ntx = (args.ncols_max + config.J - 1) / config.J;
    int nchannels_y_tiles = args.nchannels_y;

    // MoE: use the compact tiling (see mmq_moe_compact_tile) if it gives fewer tiles
    int nexperts_compact = 0;
    if (args.ids_dst != nullptr && channel_ratio == 1 && args.nsamples_y == 1) {
        const int ntx_compact = mmq_moe_compact_ntx(args.ncols_y, args.ncols_max, args.nchannels_y, config.J);
        if (ntx_compact > 0) {
            ntx               = ntx_compact;
            nchannels_y_tiles = 1;
            nexperts_compact  = args.nchannels_y;
        }
    }

    // k13-pp: GGML_CUDA_MMQ_TILING_MAX_N=N (opt-in, 0 = off): dense products (no ids) of at most N columns use the tiling (one block per
    // output tile, no stream-k fixup kernel). Other k split, so other rounding (KLD-gated)
    static const int tiling_max_n = getenv("GGML_CUDA_MMQ_TILING_MAX_N") ? atoi(getenv("GGML_CUDA_MMQ_TILING_MAX_N")) : 0;
    const bool use_tiling = !config.stream_k ||
        (tiling_max_n > 0 && args.ids_dst == nullptr && args.expert_bounds == nullptr && args.ncols_dst <= tiling_max_n);

    // k15-mm2: skip_empty: the tile count is computed on the device, so the grid of the tiling path stays the upper bound (its extra
    // blocks find no columns) and the stream-k path always splits over nsm blocks and runs the fixup.
    // k19-p5mmq: exact tiles (NVIDIA stream-k kernel, GGML_CUDA_MMQ_EXACT_TILES=0 turns it off): the prefix kernel also builds the
    // exact prefix of ceil(count/J) tiles and decides on the device: if those tiles fill the SMs (the >= 90% wave rule of the products
    // without skip_empty), one block per non-empty tile (each sums all of K, no fixup work: other rounding than the split, KLD-gated),
    // else the stream-k split over nsm blocks exactly as before (byte-identical). The grid is the host's upper bound of the tile count.
    // GGML_CUDA_MMQ_EXACT_TILES=2: one block per tile whenever there is a tile (tests)
    static const int exact_env = [] {
        const char * e = getenv("GGML_CUDA_MMQ_EXACT_TILES");
        return e == nullptr ? 1 : atoi(e);
    }();
    const bool exact = exact_env != 0 && nexperts_compact > 0 && args.skip_empty && !use_tiling && GGML_CUDA_CC_IS_NVIDIA(cc);
    ggml_cuda_pool_alloc<int32_t> tile_prefix(ctx.pool(id));
    if (nexperts_compact > 0 && args.skip_empty) {
        tile_prefix.alloc(exact ? 2*nexperts_compact + 3 : nexperts_compact + 1);
        mmq_moe_tile_prefix<<<1, 1024, 0, stream>>>(args.expert_bounds, tile_prefix.ptr, nexperts_compact, config.J, exact ? nty : 0, nsm,
            exact_env == 2);
        CUDA_CHECK(cudaGetLastError());
    }
    const int32_t * tile_prefix_p = tile_prefix.ptr;
    const int exact_nsm = exact ? nsm : 0;

    const int ntzw = nchannels_y_tiles * args.nsamples_y;
    const dim3 block_nums_xy_tiling(nty, ntx, ntzw);

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const uint3 ntx_fd             = init_fastdiv_values(ntx);
    const uint3 nchannels_y_fd     = init_fastdiv_values(nchannels_y_tiles);
    const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);

    // For the stream-k kernel it is possible to run it with tiling by setting the number of CUDA blocks equal to the number of tiles.
    // This is worthwhile if the efficiency of tiling is high and skipping the fixup kernel is more important.
    const int ntiles_dst = ntx * nty * ntzw;
    const bool tiles_fill = GGML_CUDA_CC_IS_NVIDIA(cc) && mmq_tiles_fill_sms(ntiles_dst, nsm);
    const dim3 block_nums_stream_k(exact ? std::max(nsm, ntiles_dst) : (tiles_fill && !tile_prefix_p ? ntiles_dst : nsm), 1, 1);
    // k19-p5mmq: the blocks of the split and of the fixup (with exact tiles the grid is larger, only the first nsm split)
    const int nblocks_split = exact ? nsm : int(block_nums_stream_k.x);

    // k19-mmq-pipe: same grid, same arguments, the pipelined tile loop (bit-identical, see mul_mat_q_process_tile_pipe).
    // k19-p5mmq: with exact tiles the form is chosen on the device; the kernel is picked for the likely one: one block per tile if the
    // columns without skipped slots would fill the SMs (a pipe kernel in the other form is still exact, only maybe slower)
    const bool exact_likely = exact && mmq_tiles_fill_sms(int64_t(nty)*((args.ncols_y + config.J - 1)/config.J), nsm);
    const bool one_tile_per_block = use_tiling || (exact ? exact_likely : int(block_nums_stream_k.x) == ntiles_dst);
    if (exact) {
        mmq_exact_tiles_count(exact_likely);
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    mmq_kernel_t kernel = mmq_pipe_kernel<type, J, fallback, prec_src1>(args, cc, nbytes_shared, config.nthreads, one_tile_per_block);
    if (kernel == nullptr) {
        kernel = mul_mat_q<type, J, fallback, prec_src1>;
    }
#define MMQ_LAUNCH_KERNEL kernel
#else
    GGML_UNUSED(one_tile_per_block);
#define MMQ_LAUNCH_KERNEL mul_mat_q<type, J, fallback, prec_src1>
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if (use_tiling) {
        MMQ_LAUNCH_KERNEL<<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr, args.y_scale,
             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
             ntx_fd, nexperts_compact, tile_prefix_p, 0);
        return;
    }

    GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30)); // Assert that variable kbc will not overflow.

    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0 || tile_prefix_p;

    ggml_cuda_pool & pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool);
    if (fixup_needed) {
        tmp_fixup.alloc(nblocks_split * config.J*config.I);
    }

    const dim3 block_nums_fixup(nblocks_split, config.I/warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y/2, block_dims.z);

    MMQ_LAUNCH_KERNEL<<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.y_scale,
         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
         ntx_fd, nexperts_compact, tile_prefix_p, exact_nsm);
#undef MMQ_LAUNCH_KERNEL

    if (!fixup_needed) {
        return;
    }

    CUDA_CHECK(cudaGetLastError());
    // k13-pp: the fixup is launched with PDL where available (GGML_CUDA_MMQ_FIXUP_PDL=0: a plain launch, as before; same results either way)
    static const bool fixup_pdl = getenv("GGML_CUDA_MMQ_FIXUP_PDL") == nullptr || atoi(getenv("GGML_CUDA_MMQ_FIXUP_PDL")) != 0;
    if (fixup_pdl) {
        ggml_cuda_kernel_launch(mul_mat_q_stream_k_fixup<type, J, fallback, prec_src1>,
            ggml_cuda_kernel_launch_params(block_nums_fixup, block_dims_fixup, 0, stream),
            (const int32_t *) args.ids_dst, (const int32_t *) args.expert_bounds, (float *) args.dst, (float *) tmp_fixup.ptr, blocks_per_ne00_fd,
            (int) args.nrows_x, (int) args.ncols_dst, (int) args.nrows_dst, nchannels_y_fd, (int) args.stride_channel_dst, nsamples_y_fd,
            (int) args.stride_sample_dst, ntx_fd, (int) nexperts_compact, tile_prefix_p, (int) exact_nsm);
        mmq_exact_tiles_stats(exact, tile_prefix_p, nexperts_compact, stream);
        return;
    }
    mul_mat_q_stream_k_fixup<type, J, fallback, prec_src1><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
        (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
         args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
         ntx_fd, nexperts_compact, tile_prefix_p, exact_nsm);
    mmq_exact_tiles_stats(exact, tile_prefix_p, nexperts_compact, stream);
}

template <ggml_type type, bool fallback, ggml_prec prec_src1>
void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.J_best) {
        case   8:
            launch_mul_mat_q<type,   8, fallback, prec_src1>(ctx, args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16, fallback, prec_src1>(ctx, args, stream);
            break;
        case  24:
            launch_mul_mat_q<type,  24, fallback, prec_src1>(ctx, args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32, fallback, prec_src1>(ctx, args, stream);
            break;
        case  40:
            launch_mul_mat_q<type,  40, fallback, prec_src1>(ctx, args, stream);
            break;
        case  48:
            launch_mul_mat_q<type,  48, fallback, prec_src1>(ctx, args, stream);
            break;
        case  56:
            launch_mul_mat_q<type,  56, fallback, prec_src1>(ctx, args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64, fallback, prec_src1>(ctx, args, stream);
            break;
        case  72:
            launch_mul_mat_q<type,  72, fallback, prec_src1>(ctx, args, stream);
            break;
        case  80:
            launch_mul_mat_q<type,  80, fallback, prec_src1>(ctx, args, stream);
            break;
        case  88:
            launch_mul_mat_q<type,  88, fallback, prec_src1>(ctx, args, stream);
            break;
        case  96:
            launch_mul_mat_q<type,  96, fallback, prec_src1>(ctx, args, stream);
            break;
        case 104:
            launch_mul_mat_q<type, 104, fallback, prec_src1>(ctx, args, stream);
            break;
        case 112:
            launch_mul_mat_q<type, 112, fallback, prec_src1>(ctx, args, stream);
            break;
        case 120:
            launch_mul_mat_q<type, 120, fallback, prec_src1>(ctx, args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128, fallback, prec_src1>(ctx, args, stream);
            break;
        default:
            fprintf(stderr, "J_best=%d\n", args.J_best);
            GGML_ABORT("fatal error");
    }
}

template <ggml_type type, ggml_prec prec_src1>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    if (ggml_cuda_mmq_needs_fallback(args.nrows_x)) {
        constexpr bool fallback = true;
        mul_mat_q_switch_J<type, fallback, prec_src1>(ctx, args, stream);
    } else {
        constexpr bool fallback = false;
        mul_mat_q_switch_J<type, fallback, prec_src1>(ctx, args, stream);
    }
}

#define DECL_MMQ_CASE(type)                                                        \
    template void mul_mat_q_case<type, GGML_PREC_Q8>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \

// FP4 variant: uses native FP4 MMA instead of keeping src1 at Q8_1.
#define DECL_MMQ_CASE_W4A4(type)                                                   \
    template void mul_mat_q_case<type, GGML_PREC_Q4>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \

extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q2_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q8_0);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q6_K);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_NL);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_MXFP4);
extern DECL_MMQ_CASE(GGML_TYPE_NVFP4);
extern DECL_MMQ_CASE_W4A4(GGML_TYPE_MXFP4);
extern DECL_MMQ_CASE_W4A4(GGML_TYPE_NVFP4);

// -------------------------------------------------------------------------------------------------------------------------

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts);

// k19-fu2 F3: see mmq.cu
bool ggml_cuda_mmq_id_swiglu_ok(const ggml_tensor * src0, const ggml_tensor * glu, const ggml_tensor * ids, const ggml_tensor * dst, int cc);
