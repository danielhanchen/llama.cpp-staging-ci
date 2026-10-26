#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// c_nwarps > 0 overrides the number of warps (batch invariant verify: the K split and the reduction of one column)
// n_multi > 1: n_multi dense matrices of one type and one K over the same activation (fusion.multi_*), each block computes the
// rows of one of them exactly as that matrix's own launch would (same warps, rows per block and reduction)
// c_unroll > 1: the K loop unrolled c_unroll times, so that a thread has the weight loads of several iterations in flight (few rows,
// long K: the grid covers few SMs and waits on memory latency); every thread adds its blocks in the same order, so the bytes are the same
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, int c_nwarps = 0, int n_multi = 0, int c_rpb = 0, int c_unroll = 1, bool c_ewq = false>
__launch_bounds__((c_nwarps > 0 ? c_nwarps : calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters))*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = c_nwarps > 0 ? c_nwarps : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    // c_rpb > 0 overrides the rows per block: a row's sum does not depend on it (the K split and the reduction are per row)
    constexpr int rows_per_cuda_block = c_rpb > 0 ? c_rpb : calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
              int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    if constexpr (n_multi > 1) {
        static_assert(!has_fusion && n_multi <= GGML_CUDA_MMVQ_MULTI_MAX, "multi-matrix mul_mat_vec_q");
        int m = 0;
#pragma unroll
        for (int k = 1; k < n_multi; ++k) {
            m = blockIdx.x >= fusion.multi_block0[k] ? k : m;
        }
        vx             = fusion.multi_x[m];
        dst            = fusion.multi_dst[m];
        stride_col_dst = fusion.multi_nrows[m];
        row0           = rows_per_cuda_block*(blockIdx.x - fusion.multi_block0[m]);
    }

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    // let the next kernel launch while this one runs, its ggml_cuda_pdl_sync() still waits for the completion
    ggml_cuda_pdl_lc();
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (c_ewq) {
        // the weights do not depend on the previous kernel: start pulling this block's rows (contiguous) into L2 while it finishes,
        // the chain prologue below then overlaps their latency
        const char * w0 = (const char *) vx + (size_t) row0*stride_row_x*ggml_cuda_type_traits<type>::bs;
        const int nbytes = rows_per_cuda_block*stride_row_x*ggml_cuda_type_traits<type>::bs;
        for (int off = 128*tid; off < nbytes; off += 128*nwarps*warp_size) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(w0 + off));
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_cuda_pdl_sync();
    sample_dst = blockIdx.z;
    // k11-decode: a single column MUL_MAT_ID launch over several tokens (batch invariant verify) takes the tokens as samples:
    // token z reads the experts of its ids row (one sample, z = 0: as before)
    if (ncols_dst == 1 && ids) {
        ids += sample_dst*ids_stride;
    }
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;

    // the selected expert is uniform in the block, so the whole block leaves on a skipped slot
    if (!shared_expert && ncols_dst == 1 && ids && ids[channel_dst] == -1) {
        dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
        if (threadIdx.y == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                        dst[j*stride_col_dst + i] = 0.0f;
                    }
                }
            }
        }
        return;
    }

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // k15-mm2 (c_ewq, see ggml_cuda_mul_mat_vec_q_ew): src1 is the F32 input of a SCALE / SILU chain; the block applies the chain
    // and quantizes the result into shared memory, with the expressions of the chain kernels (ew_chain_apply, scale_f32,
    // unary_op_kernel) and of quantize_q8_1 (one value per lane, the same 32-lane reductions), so y holds the bytes the
    // unfused chain + quantize_q8_1 would have written
    if constexpr (c_ewq) {
        static_assert(!has_fusion && n_multi == 0, "c_ewq: dense product without other fusion");
        __shared__ block_q8_1 y_ew[ncols_dst*(GGML_CUDA_MMVQ_EW_MAX_K/QK8_1)];
        const int nv = ncols_dst*blocks_per_row_x*QK8_1; // a multiple of the warp size: a warp holds whole q8_1 blocks
        for (int e = tid; e < nv; e += nwarps*warp_size) {
            const int j = e / (blocks_per_row_x*QK8_1);
            const int k = e - j*(blocks_per_row_x*QK8_1);
            float v = fusion.ew_src[j*fusion.ew_stride_col + k];
#pragma unroll
            for (int o = 0; o < GGML_CUDA_MMVQ_EW_MAX; ++o) {
                if (o < fusion.ew_n) {
                    v = fusion.ew_op[o] == 0 ? fusion.ew_p0[o] * v + fusion.ew_p1[o] : ggml_cuda_op_silu_single(v);
                }
            }
            float amax = fabsf(v);
            float sum  = v;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
            y_ew[e/QK8_1].qs[e % QK8_1] = q;
            if (e % QK8_1 == 0) {
                y_ew[e/QK8_1].ds = make_half2(d, sum);
            }
        }
        __syncthreads();
        vy = y_ew;
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    // c_unroll == 1 keeps the loop as it was (no pragma: the compiler's own unrolling), c_unroll > 1 unrolls it that many times
    if constexpr (c_unroll > 1) {
#pragma unroll c_unroll
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
        }
    } else {
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const int32_t expert = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];

    // each warp owns one token, so the whole warp leaves on a skipped slot
    if (expert == -1) {
        if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = 0.0f;
        }
        return;
    }

    const uint32_t channel_x = expert;
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

// the types with a multi-matrix kernel: the dense / attention / shared-expert projections of the models we ship
// (Q8_0 everywhere, Q4_K/Q5_K/Q6_K and IQ4_XS/IQ3_XXS in low-bit mixes)
static constexpr bool mul_mat_vec_q_multi_type(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_XXS:
            return true;
        default:
            return false;
    }
}


// k15-mm: whether a dense launch of nrows rows (over all channels / samples) of K = ncols_x unrolls its K loop (c_unroll = 8): few
// rows and at least two K iterations per thread, where the latency of the weight loads, not the bandwidth, sets the time
// (GGML_CUDA_MMVQ_UNROLL=0 turns it off)
template <ggml_type type>
static bool mul_mat_vec_q_unroll(const int ncols_x, const int64_t nrows, const int ncols_dst, const mmvq_parameter_table_id table_id, const int warp_size) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_MMVQ_UNROLL"); return e == nullptr || atoi(e) != 0; }();
    if (!on || !mul_mat_vec_q_multi_type(type) || nrows > 512 ||
            (table_id != MMVQ_PARAMETERS_GENERIC && table_id != MMVQ_PARAMETERS_TURING && table_id != MMVQ_PARAMETERS_GB10)) {
        return false;
    }
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    const int blocks_per_iter = vdr * calc_nwarps(type, ncols_dst, table_id) * warp_size / qi;
    // at least 4 K iterations per thread (Q8_0: K >= 4096): measured faster on B200 for 320 x 10240 (1 and 3 columns: 0.78x / 0.42x),
    // slower for 640 x 2560 (2.5 iterations)
    return ncols_x / qk >= 4 * blocks_per_iter;
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, int c_nwarps = 0>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream, const bool unroll = false) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    if constexpr (mul_mat_vec_q_multi_type(type) && c_nwarps == 0 && !small_k) {
        if (unroll) {
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, 0, 0, 0, 8>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }
    GGML_UNUSED(unroll);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, c_nwarps>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

// the warps of a single column launch: should_use_small_k / should_halve_iters of mul_mat_vec_q_switch_ncols_dst at ncols_dst 1
// (keep in sync with the lambdas there)
template <ggml_type type>
static int mul_mat_vec_q_nwarps_1(const int ncols_x, const bool has_ids, const int cc, const int warp_size, const mmvq_parameter_table_id table_id,
        bool * small_k_out = nullptr, bool * halve_iters_out = nullptr) {
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    bool small_k;
    {
        const int nwarps = calc_nwarps(type, 1, table_id);
        bool      use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = { GGML_TYPE_IQ3_S, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if (std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end() ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }
        small_k = use;
    }

    bool halve_iters = false;
    if (!small_k && table_id == MMVQ_PARAMETERS_GB10 && !has_ids) {
        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);
        const int idle            = iters_wide * 2 - iters;
        halve_iters = idle * 8 <= iters_wide * 2;
    }

    if (small_k_out) {
        *small_k_out = small_k;
    }
    if (halve_iters_out) {
        *halve_iters_out = halve_iters;
    }
    return calc_nwarps(type, 1, table_id, small_k, halve_iters);
}

template <ggml_type type, int c_ncols_dst, int c_nwarps>
static void mul_mat_vec_q_launch_nwarps(
        const void * vx, const void * vy, float * dst,
        const uint32_t ncols_x, const int nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const int nchannels_dst, const int nsamples_dst, const int warp_size, cudaStream_t stream) {
    static_assert(c_nwarps > 0, "c_nwarps");
    constexpr int rpb = 2; // calc_rows_per_block for 2..8 columns (generic, Turing and GB10 tables)
    const dim3 block_nums((nrows_x + rpb - 1) / rpb, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, c_nwarps, 1);
    const ggml_cuda_mm_fusion_args_device no_fusion{};
    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, c_nwarps>(
        vx, vy, nullptr, no_fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio,
        stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, 0, 0, stream);
}

// batch invariant dense MUL_MAT with 2..8 columns: launch the batched kernel with the warps of the single column launch when the
// table gives other warps to this width. Returns 0 when the normal batched kernel already splits K like one column,
// 1 when launched here, -1 when there is no such kernel (the caller loops over the columns)
template <ggml_type type>
static int mul_mat_vec_q_exact_batched(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {
    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.shared_up != nullptr;
    if (has_fusion) {
        return -1;
    }
    if (table_id != MMVQ_PARAMETERS_GENERIC && table_id != MMVQ_PARAMETERS_TURING && table_id != MMVQ_PARAMETERS_GB10) {
        return -1;
    }

    const int nw1 = mul_mat_vec_q_nwarps_1<type>(ncols_x, false, cc, warp_size, table_id);
    const int nwn = calc_nwarps(type, ncols_dst, table_id);

    const uint3 nchannels_y_fd   = make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);
    GGML_UNUSED(nchannels_y);

#define MMVQ_EXACT_LAUNCH(NC, NW) \
    mul_mat_vec_q_launch_nwarps<type, NC, NW>(vx, vy, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, \
        stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, \
        stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, warp_size, stream)

    if (nw1 == nwn) {
        return 0; // the batched kernel already splits K like a single column launch
    }
    // Turing K-quants: 2 warps at one column, 4 at 2..4; generic: 4 warps at one column, 2 at 5..8
    if (nw1 == 2 && ncols_dst <= 4) {
        switch (ncols_dst) {
            case 2: MMVQ_EXACT_LAUNCH(2, 2); return 1;
            case 3: MMVQ_EXACT_LAUNCH(3, 2); return 1;
            case 4: MMVQ_EXACT_LAUNCH(4, 2); return 1;
            default: return -1;
        }
    }
    if (nw1 == 4 && ncols_dst >= 5) {
        switch (ncols_dst) {
            case 5: MMVQ_EXACT_LAUNCH(5, 4); return 1;
            case 6: MMVQ_EXACT_LAUNCH(6, 4); return 1;
            case 7: MMVQ_EXACT_LAUNCH(7, 4); return 1;
            case 8: MMVQ_EXACT_LAUNCH(8, 4); return 1;
            default: return -1;
        }
    }
#undef MMVQ_EXACT_LAUNCH
    return -1;
}



// k15-mm: rows per block of a 2..8 column launch, by the shape (a row's sum does not depend on it, so every output keeps its bytes):
//  - few K blocks per row (the hyper-connection up projection, K = 320): most thread groups have no block, so a block takes as many
//    rows as it has warps, as the single column small-K launch does (GGML_CUDA_MMVQ_BATCH_SMALL_K=0 turns this off);
//  - few rows (the hyper-connection down projection, 320 rows of K = 10240): one row per block, so that the grid covers every SM
//    at least twice (GGML_CUDA_MMVQ_BATCH_ROWS1=0 turns this off).
// Returns 0 for the default (calc_rows_per_block).
template <ggml_type type>
static int mul_mat_vec_q_batch_rpb(const int ncols_x, const int nrows_x, const int ncols_dst, const int cc, const int warp_size,
        const mmvq_parameter_table_id table_id) {
    static const bool small_k_on = [] { const char * e = getenv("GGML_CUDA_MMVQ_BATCH_SMALL_K"); return e == nullptr || atoi(e) != 0; }();
    static const bool rows1_on   = [] { const char * e = getenv("GGML_CUDA_MMVQ_BATCH_ROWS1");  return e == nullptr || atoi(e) != 0; }();
    if (ncols_dst < 2 || ncols_dst > 8 || !mul_mat_vec_q_multi_type(type) ||
            (table_id != MMVQ_PARAMETERS_GENERIC && table_id != MMVQ_PARAMETERS_TURING && table_id != MMVQ_PARAMETERS_GB10)) {
        return 0;
    }
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int blocks_per_row_x = ncols_x / qk;
    // 4 rows per block only where the rows divide evenly: a block reads the weights of all its rows (the last block would read past
    // the matrix) and guards its writes by stride_col_dst
    if (small_k_on && nwarps == 4 && blocks_per_row_x < nwarps * (vdr * warp_size / qi) && nrows_x % 4 == 0) {
        return 4;
    }
    // one row per block only with the unrolled long-K launch (mul_mat_vec_q_unroll): 640 rows of K = 2560 were slower with it
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    if (rows1_on && (nrows_x + 1) / 2 < 2 * nsm && mul_mat_vec_q_unroll<type>(ncols_x, nrows_x, ncols_dst, table_id, warp_size)) {
        return 1;
    }
    return 0;
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const bool exact = false) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    // k15-fv, GGML_CUDA_MMVQ_MOE_ONE=1 (opt-in, changes the single token MUL_MAT_ID sums): one token MUL_MAT_ID runs the multi-token MoE
    // kernel too. That kernel gives each token one warp of its own over the whole K loop, so a token's sums do not depend on the other
    // tokens of the launch: a batch invariant verify batch is then that one launch (in chunks of the kernel's batch limit), no longer
    // one single token launch per token
    static const bool moe_one = [] {
        const char * e = getenv("GGML_CUDA_MMVQ_MOE_ONE");
        return e != nullptr && atoi(e) != 0;
    }();
    const bool moe_path = moe_one && ids != nullptr && nsamples_x == 1 && nsamples_dst == 1;
    // k15-fv, GGML_CUDA_VERIFY_REASSOC (default 1; 0 = the bit-exact per-token verify below): a batch invariant verify batch runs the
    // ordinary multi-column kernels (MUL_MAT_ID: mul_mat_vec_q_moe; dense without fusion: the batched mmvq). Every row then reads the same
    // q8_1 activations into the same int8 dot products with f32 accumulation as its single token launch, only the order of the f32
    // sums differs (plans/k11.md equivalence rule); the dispatcher still never routes these nodes to MMQ / cuBLAS
    static const bool reassoc = [] {
        const char * e = getenv("GGML_CUDA_VERIFY_REASSOC");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool has_fusion_any = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.shared_up != nullptr;
    const bool reassoc_path = exact && reassoc && ncols_dst > 1 && (ids != nullptr ? nsamples_x == 1 && nsamples_dst == 1 : !has_fusion_any);
    if ((moe_path || (reassoc_path && ids != nullptr)) && ncols_dst > 1) {
        const int cc_m  = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const int max_b = get_mmvq_mmid_max_batch(type, cc_m);
        if (ncols_dst > max_b) {
            for (int j0 = 0; j0 < ncols_dst; j0 += max_b) {
                ggml_cuda_mm_fusion_args_device fusion_j = fusion;
                if (fusion_j.shared_dst != nullptr) {
                    fusion_j.shared_dst = fusion.shared_dst + (size_t) j0*fusion.shared_stride_col_dst;
                }
                mul_mat_vec_q_switch_ncols_dst<type>(
                    vx, (const char *) vy + (size_t) j0*stride_col_y*sizeof(block_q8_1), ids + (size_t) j0*ids_stride, fusion_j,
                    dst + (size_t) j0*stride_col_dst, ncols_x, nrows_x, std::min(max_b, ncols_dst - j0), stride_row_x, stride_col_y,
                    stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, false);
            }
            return;
        }
    }

    // batch invariant (verify batches, GGML_TENSOR_FLAG_BATCH_INVARIANT): every column gets the bytes of a single column launch
    if (exact && ncols_dst > 1 && !moe_path && !reassoc_path) {
        const auto one_column = [&](const int j) {
            mul_mat_vec_q_switch_ncols_dst<type>(
                vx, (const char *) vy + (size_t) j*stride_col_y*sizeof(block_q8_1), ids ? ids + (size_t) j*ids_stride : nullptr,
                fusion, dst + (size_t) j*stride_col_dst, ncols_x, nrows_x, 1, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, false);
        };
        // k11-decode: MUL_MAT_ID without a shared expert or biases: one launch of the single token kernel, the tokens as samples (grid z),
        // so every block computes what it computes in the token's own launch (GGML_CUDA_MMVQ_EXACT_IDS_MERGE=0: one launch per token)
        static const bool ids_merge = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_EXACT_IDS_MERGE");
            return e == nullptr || atoi(e) != 0;
        }();
        if (ids != nullptr && ids_merge && fusion.shared_up == nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                nsamples_x == 1 && nsamples_dst == 1) {
            mul_mat_vec_q_switch_ncols_dst<type>(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, 1, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                1, ncols_dst, stride_sample_x, stride_col_y, stride_col_dst, ids_stride, stream, false);
            return;
        }
        if (ids != nullptr) {
            // MUL_MAT_ID: the single token kernel per token (the experts differ per token, there is little to share), fusion
            // included: the shared expert of the fused path is written per token too
            GGML_ASSERT(fusion.shared_up == nullptr || fusion.shared_stride_col_dst > 0);
            for (int j = 0; j < ncols_dst; ++j) {
                ggml_cuda_mm_fusion_args_device fusion_j = fusion;
                if (fusion_j.shared_dst != nullptr) {
                    fusion_j.shared_dst = fusion.shared_dst + (size_t) j*fusion.shared_stride_col_dst;
                }
                mul_mat_vec_q_switch_ncols_dst<type>(
                    vx, (const char *) vy + (size_t) j*stride_col_y*sizeof(block_q8_1), ids + (size_t) j*ids_stride,
                    fusion_j, dst + (size_t) j*stride_col_dst, ncols_x, nrows_x, 1, stride_row_x, stride_col_y, stride_col_dst,
                    nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, false);
            }
            return;
        }
        // dense: a column only depends on how the K loop is split, i.e. on the number of warps per block (the rows per
        // block do not change a row's sum); the batched kernel is kept when its warps match the single column launch
        const int rc = mul_mat_vec_q_exact_batched<type>(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        if (rc > 0) {
            return;
        }
        if (rc < 0) {
            for (int j = 0; j < ncols_dst; ++j) {
                one_column(j);
            }
            return;
        }
        // rc == 0: the batched launch below
    }

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && (ncols_dst > 1 || moe_path)) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel (GGML_CUDA_MMVQ_MOE_ONE=1: one token too)
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    // k15-mm: few rows of a long K: the K loop unrolled (mul_mat_vec_q_unroll)
    const bool unroll = !has_ids && mul_mat_vec_q_unroll<type>(ncols_x, (int64_t) nrows_x*nchannels_dst*nsamples_dst, ncols_dst, table_id, warp_size);

    // k15-mm: a 2..8 column dense launch with other rows per block (mul_mat_vec_q_batch_rpb); false = the default launch
    const auto batch_rpb_launch = [&](auto nc_tag) -> bool {
        constexpr int c_ncols_dst = decltype(nc_tag)::value;
        if constexpr (!mul_mat_vec_q_multi_type(type)) {
            return false;
        } else {
            const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                    fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.shared_up != nullptr;
            if (ids != nullptr || has_fusion) {
                return false;
            }
            const int rpb = mul_mat_vec_q_batch_rpb<type>(ncols_x, nrows_x, c_ncols_dst, cc, warp_size, table_id);
            if (rpb == 0) {
                return false;
            }
            const int nwarps = calc_nwarps(type, c_ncols_dst, table_id);
            const dim3 block_nums((nrows_x + rpb - 1) / rpb, nchannels_dst, nsamples_dst);
            const dim3 block_dims(warp_size, nwarps, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
#define MMVQ_RPB_LAUNCH(RPB, UNR) ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, false, false, 0, 0, RPB, UNR>, launch_params, \
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst, \
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, \
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride)
            if (rpb == 1 && unroll) {
                MMVQ_RPB_LAUNCH(1, 8);
            } else if (rpb == 1) {
                MMVQ_RPB_LAUNCH(1, 1);
            } else {
                MMVQ_RPB_LAUNCH(4, 1);
            }
#undef MMVQ_RPB_LAUNCH
            return true;
        }
    };

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream, unroll);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            if (batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, unroll);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            if (batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, unroll);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            if (batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, unroll);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            if (batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, unroll);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            if (batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, unroll);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            if (batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, unroll);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            if (batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, unroll);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

// k15-mm: n_multi dense matrices of one type and one K that read the same activation, in one launch (see ggml_cuda_mul_mat_vec_q_multi).
// The launch configuration is the one each matrix's own (non batch invariant, or batch invariant with the K split of one column) launch
// takes; it depends only on the type, K and the number of columns, so every output gets the bytes of that launch.
template <ggml_type type, int c_ncols_dst, int n_multi, bool c_small_k = false, bool c_halve_iters = false, int c_rpb = 0, int c_unroll = 1>
static void mul_mat_vec_q_multi_launch(
        ggml_cuda_mm_fusion_args_device fusion, const int * nrows, const void * vy, const int ncols_x,
        const int stride_row_x, const int stride_col_y, const int warp_size, const mmvq_parameter_table_id table_id, cudaStream_t stream) {
    const int nwarps = calc_nwarps(type, c_ncols_dst, table_id, c_small_k, c_halve_iters);
    const int rpb    = c_rpb > 0 ? c_rpb : calc_rows_per_block(type, c_ncols_dst, table_id, c_small_k, nwarps);
    uint32_t nblocks = 0;
    for (int m = 0; m < n_multi; ++m) {
        fusion.multi_block0[m] = nblocks;
        fusion.multi_nrows[m]  = nrows[m];
        nblocks += (nrows[m] + rpb - 1) / rpb;
    }
    const uint3 zero = make_uint3(0, 0, 0);
    const uint3 one  = init_fastdiv_values(1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(dim3(nblocks, 1, 1), dim3(warp_size, nwarps, 1), 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, c_small_k, c_halve_iters, 0, n_multi, c_rpb, c_unroll>, launch_params,
        fusion.multi_x[0], vy, (const int32_t *) nullptr, fusion, fusion.multi_dst[0], (uint32_t) ncols_x, zero, (uint32_t) stride_row_x,
        (uint32_t) stride_col_y, (uint32_t) nrows[0], one, 0u, 0u, 0u, one, 0u, 0u, 0u, 0u);
}

template <ggml_type type, int n_multi>
static void mul_mat_vec_q_multi_switch_ncols(
        const ggml_cuda_mm_fusion_args_device & fusion, const int * nrows, const void * vy, const int ncols_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, cudaStream_t stream) {
    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);

    // 2..8 columns: the rows per block of the single launch of the shortest K / the fewest rows (mul_mat_vec_q_batch_rpb); a row's
    // sum does not depend on it, so this only keeps the launch shape the single products would get
    int rows_min = nrows[0];
    for (int m = 1; m < n_multi; ++m) {
        rows_min = std::min(rows_min, nrows[m]);
    }
    int total_rows = 0;
    for (int m = 0; m < n_multi; ++m) {
        total_rows += nrows[m];
    }
    int brpb = ncols_dst > 1 ? mul_mat_vec_q_batch_rpb<type>(ncols_x, total_rows, ncols_dst, cc, warp_size, table_id) : 0;
    for (int m = 0; m < n_multi; ++m) {
        if (brpb == 4 && nrows[m] % 4 != 0) {
            brpb = 0;
        }
    }
    GGML_UNUSED(rows_min);
    const bool unroll = mul_mat_vec_q_unroll<type>(ncols_x, total_rows, ncols_dst, table_id, warp_size);
#define MMVQ_MULTI_L(NC, RPB, UNR) mul_mat_vec_q_multi_launch<type, NC, n_multi, false, false, RPB, UNR>(fusion, nrows, vy, ncols_x, stride_row_x, stride_col_y, warp_size, table_id, stream)
#define MMVQ_MULTI(NC) (brpb == 1 ? (unroll ? MMVQ_MULTI_L(NC, 1, 8) : MMVQ_MULTI_L(NC, 1, 1)) : brpb == 4 ? MMVQ_MULTI_L(NC, 4, 1) : \
                        unroll ? MMVQ_MULTI_L(NC, 0, 8) : MMVQ_MULTI_L(NC, 0, 1))
    switch (ncols_dst) {
        case 1: {
            bool small_k = false;
            bool halve   = false;
            mul_mat_vec_q_nwarps_1<type>(ncols_x, false, cc, warp_size, table_id, &small_k, &halve);
            constexpr bool c_promoted = calc_nwarps(type, 1, MMVQ_PARAMETERS_GB10, false, true) != calc_nwarps(type, 1, MMVQ_PARAMETERS_GB10, false, false);
            if (small_k) {
                mul_mat_vec_q_multi_launch<type, 1, n_multi, true, false>(fusion, nrows, vy, ncols_x, stride_row_x, stride_col_y, warp_size, table_id, stream);
            } else if (halve && c_promoted) {
                mul_mat_vec_q_multi_launch<type, 1, n_multi, false, c_promoted>(fusion, nrows, vy, ncols_x, stride_row_x, stride_col_y, warp_size, table_id, stream);
            } else {
                MMVQ_MULTI(1);
            }
        } break;
        case 2: MMVQ_MULTI(2); break;
        case 3: MMVQ_MULTI(3); break;
        case 4: MMVQ_MULTI(4); break;
        case 5: MMVQ_MULTI(5); break;
        case 6: MMVQ_MULTI(6); break;
        case 7: MMVQ_MULTI(7); break;
        case 8: MMVQ_MULTI(8); break;
        default: GGML_ABORT("fatal error");
    }
#undef MMVQ_MULTI
#undef MMVQ_MULTI_L
}

template <ggml_type type>
static bool mul_mat_vec_q_multi_switch_n(
        const ggml_cuda_mm_fusion_args_device & fusion, const int n, const int * nrows, const void * vy, const int ncols_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, cudaStream_t stream) {
    switch (n) {
        case 2: mul_mat_vec_q_multi_switch_ncols<type, 2>(fusion, nrows, vy, ncols_x, ncols_dst, stride_row_x, stride_col_y, stream); return true;
        case 3: mul_mat_vec_q_multi_switch_ncols<type, 3>(fusion, nrows, vy, ncols_x, ncols_dst, stride_row_x, stride_col_y, stream); return true;
        default: return false;
    }
}

static bool mul_mat_vec_q_multi_switch_type(
        const ggml_type type, const ggml_cuda_mm_fusion_args_device & fusion, const int n, const int * nrows, const void * vy,
        const int ncols_x, const int ncols_dst, const int stride_row_x, const int stride_col_y, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q8_0:    return mul_mat_vec_q_multi_switch_n<GGML_TYPE_Q8_0>   (fusion, n, nrows, vy, ncols_x, ncols_dst, stride_row_x, stride_col_y, stream);
        case GGML_TYPE_Q4_K:    return mul_mat_vec_q_multi_switch_n<GGML_TYPE_Q4_K>   (fusion, n, nrows, vy, ncols_x, ncols_dst, stride_row_x, stride_col_y, stream);
        case GGML_TYPE_Q5_K:    return mul_mat_vec_q_multi_switch_n<GGML_TYPE_Q5_K>   (fusion, n, nrows, vy, ncols_x, ncols_dst, stride_row_x, stride_col_y, stream);
        case GGML_TYPE_Q6_K:    return mul_mat_vec_q_multi_switch_n<GGML_TYPE_Q6_K>   (fusion, n, nrows, vy, ncols_x, ncols_dst, stride_row_x, stride_col_y, stream);
        case GGML_TYPE_IQ4_XS:  return mul_mat_vec_q_multi_switch_n<GGML_TYPE_IQ4_XS> (fusion, n, nrows, vy, ncols_x, ncols_dst, stride_row_x, stride_col_y, stream);
        case GGML_TYPE_IQ3_XXS: return mul_mat_vec_q_multi_switch_n<GGML_TYPE_IQ3_XXS>(fusion, n, nrows, vy, ncols_x, ncols_dst, stride_row_x, stride_col_y, stream);
        default: return false;
    }
}

// batch invariant columns: the multi launch is the batched launch, so it is exact only where that launch already splits K like one
// column (mul_mat_vec_q_exact_batched returns 0)
template <ggml_type type>
static bool mul_mat_vec_q_multi_exact_ok_t(const int ncols_x, const int ncols_dst, const int cc, const int warp_size) {
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);
    if (table_id != MMVQ_PARAMETERS_GENERIC && table_id != MMVQ_PARAMETERS_TURING && table_id != MMVQ_PARAMETERS_GB10) {
        return false;
    }
    return mul_mat_vec_q_nwarps_1<type>(ncols_x, false, cc, warp_size, table_id) == calc_nwarps(type, ncols_dst, table_id);
}

static bool mul_mat_vec_q_multi_exact_ok(const ggml_type type, const int ncols_x, const int ncols_dst, const int cc, const int warp_size) {
    switch (type) {
        case GGML_TYPE_Q8_0:    return mul_mat_vec_q_multi_exact_ok_t<GGML_TYPE_Q8_0>   (ncols_x, ncols_dst, cc, warp_size);
        case GGML_TYPE_Q4_K:    return mul_mat_vec_q_multi_exact_ok_t<GGML_TYPE_Q4_K>   (ncols_x, ncols_dst, cc, warp_size);
        case GGML_TYPE_Q5_K:    return mul_mat_vec_q_multi_exact_ok_t<GGML_TYPE_Q5_K>   (ncols_x, ncols_dst, cc, warp_size);
        case GGML_TYPE_Q6_K:    return mul_mat_vec_q_multi_exact_ok_t<GGML_TYPE_Q6_K>   (ncols_x, ncols_dst, cc, warp_size);
        case GGML_TYPE_IQ4_XS:  return mul_mat_vec_q_multi_exact_ok_t<GGML_TYPE_IQ4_XS> (ncols_x, ncols_dst, cc, warp_size);
        case GGML_TYPE_IQ3_XXS: return mul_mat_vec_q_multi_exact_ok_t<GGML_TYPE_IQ3_XXS>(ncols_x, ncols_dst, cc, warp_size);
        default: return false;
    }
}

static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const bool exact = false) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, exact);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

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
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_1_size   = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_alloc(ctx.pool());
    char * src1_q8_1 = nullptr;
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        // reuse the q8_1 copy of src1 if an earlier mul_mat_vec_q of this graph already quantized it
        bool quantized = false;
        if (ctx.curr_stream_no == 0) {
            // the q8_1 rows are written in logical order: a contiguous src1 only depends on the number of rows,
            // e.g. [ne10, 1, n_tokens] for MUL_MAT_ID and [ne10, n_tokens] for MUL_MAT share one copy;
            // its strides are keyed as -1, which no real stride (0 for a broadcast view) can match
            const bool contiguous = ggml_is_contiguous(src1);
            int64_t    key[ggml_cuda_q8_1_cache::n_key] = {
                ne10,
                ne11,
                ne12,
                ne13,
                ne11 == 1 ? 0 : s11,
                ne12 == 1 ? 0 : s12,
                ne13 == 1 ? 0 : s13,
            };
            if (contiguous) {
                ggml_cuda_q8_1_cache::key_contiguous(ne10, ne11*ne12*ne13, key);
            }
            src1_q8_1 = ctx.q8_1_cache.get(src1->data, ggml_nbytes(src1), key, q8_1_size, quantized);
        }
        if (src1_q8_1 == nullptr) {
            src1_q8_1 = src1_q8_1_alloc.alloc(q8_1_size);
        }
        if (!quantized) {
            quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        }
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    const bool exact = (dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) != 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream, exact);
}

bool ggml_cuda_mul_mat_vec_q_multi_ok(const ggml_type type, const int64_t ncols_x, const int64_t ncols_dst, const bool exact, const int cc) {
    if (!mul_mat_vec_q_multi_type(type) || ncols_dst < 1 || ncols_dst > MMVQ_MAX_BATCH_SIZE || ncols_x % ggml_blck_size(type) != 0) {
        return false;
    }
    if (exact && ncols_dst > 1) {
        return mul_mat_vec_q_multi_exact_ok(type, ncols_x, ncols_dst, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size);
    }
    return true;
}

void ggml_cuda_mul_mat_vec_q_multi(ggml_backend_cuda_context & ctx, const int n, const ggml_tensor * const * src0, const ggml_tensor * src1,
        ggml_tensor * const * dst) {
    GGML_ASSERT(n >= 2 && n <= GGML_CUDA_MMVQ_MULTI_MAX);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[2] == 1 && src1->ne[3] == 1);
    const ggml_type type = src0[0]->type;
    const int64_t   ne10 = src1->ne[0];
    const int64_t   ne11 = src1->ne[1];
    GGML_ASSERT(ne11 <= MMVQ_MAX_BATCH_SIZE);

    ggml_cuda_mm_fusion_args_device fusion{};
    int nrows[GGML_CUDA_MMVQ_MULTI_MAX] = {};
    for (int m = 0; m < n; ++m) {
        GGML_ASSERT(src0[m]->type == type && src0[m]->ne[0] == ne10 && ggml_is_contiguous(src0[m]) && ggml_is_matrix(src0[m]));
        GGML_ASSERT(src0[m]->nb[1] == src0[0]->nb[1]);
        GGML_ASSERT(dst[m]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst[m]) && dst[m]->ne[0] == src0[m]->ne[1] && dst[m]->ne[1] == ne11);
        fusion.multi_x[m]   = src0[m]->data;
        fusion.multi_dst[m] = (float *) dst[m]->data;
        nrows[m]            = (int) src0[m]->ne[1];
    }

    cudaStream_t stream = ctx.stream();

    // the q8_1 copy of src1: as ggml_cuda_mul_mat_vec_q does it for a contiguous src1 (the q8_1 cache entry it would use)
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_1_size   = ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_alloc(ctx.pool());
    char * src1_q8_1 = nullptr;
    bool quantized = false;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    if (ctx.curr_stream_no == 0) {
        int64_t key[ggml_cuda_q8_1_cache::n_key];
        ggml_cuda_q8_1_cache::key_contiguous(ne10, ne11, key);
        src1_q8_1 = ctx.q8_1_cache.get(src1->data, ggml_nbytes(src1), key, q8_1_size, quantized);
    }
    if (src1_q8_1 == nullptr) {
        src1_q8_1 = src1_q8_1_alloc.alloc(q8_1_size);
    }
    if (!quantized) {
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1, type, ne10, s11, s11*ne11, s11*ne11, ne10_padded, ne11, 1, 1, stream);
    }

    const int stride_row_x = src0[0]->nb[1] / ggml_type_size(type);
    const int stride_col_y = ne10_padded / QK8_1;
    const bool ok = mul_mat_vec_q_multi_switch_type(type, fusion, n, nrows, src1_q8_1, ne10, ne11, stride_row_x, stride_col_y, stream);
    GGML_ASSERT(ok);
}

// k15-mm2: a short-K dense Q8_0 product of 1..4 columns whose F32 src1 is the output of a SCALE / SILU chain (the hyper-connection
// up projection, K = 320, behind silu(scale(down))): one launch in place of the chain kernel, quantize_q8_1 and mul_mat_vec_q.
// Only where the unfused mul_mat_vec_q runs 4 warps (generic and Turing tables, 1..4 columns, batch invariant or not): a row's sum
// depends on the warps only (rows per block and unrolling do not change it), so every output keeps its bytes.
bool ggml_cuda_mul_mat_vec_q_ew_ok(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    if (src0->type != GGML_TYPE_Q8_0 || !ggml_is_contiguous(src0) || !ggml_is_matrix(src0) || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(dst) || dst->ne[2] != 1 || dst->ne[3] != 1 || dst->ne[1] < 1 || dst->ne[1] > 4) {
        return false;
    }
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32 ||
            (table_id != MMVQ_PARAMETERS_GENERIC && table_id != MMVQ_PARAMETERS_TURING)) {
        return false;
    }
    const int ncols_dst = (int) dst->ne[1];
    if (calc_nwarps(GGML_TYPE_Q8_0, ncols_dst, table_id) != 4 || calc_nwarps(GGML_TYPE_Q8_0, 1, table_id, true) != 4 ||
            calc_nwarps(GGML_TYPE_Q8_0, 1, table_id, false) != 4) {
        return false;
    }
    const int64_t K = src0->ne[0];
    return K % QK8_1 == 0 && K <= GGML_CUDA_MMVQ_EW_MAX_K && src0->ne[1] % 4 == 0 && dst->ne[0] == src0->ne[1];
}

void ggml_cuda_mul_mat_vec_q_ew(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * ew_src, const int n_ops,
        const int * ops, const float * p0, const float * p1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mul_mat_vec_q_ew_ok(src0, dst, ggml_cuda_info().devices[ctx.device].cc));
    GGML_ASSERT(ew_src->type == GGML_TYPE_F32 && ew_src->ne[0] == src0->ne[0] && ew_src->ne[1] == dst->ne[1] && ew_src->nb[0] == sizeof(float));
    GGML_ASSERT(n_ops >= 1 && n_ops <= GGML_CUDA_MMVQ_EW_MAX);

    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.ew_src        = (const float *) ew_src->data;
    fusion.ew_stride_col = ew_src->nb[1] / sizeof(float);
    fusion.ew_n          = n_ops;
    for (int o = 0; o < n_ops; ++o) {
        fusion.ew_op[o] = ops[o];
        fusion.ew_p0[o] = p0[o];
        fusion.ew_p1[o] = p1[o];
    }

    const int      ncols_x        = (int) src0->ne[0];
    const int      nrows_x        = (int) src0->ne[1];
    const int      ncols_dst      = (int) dst->ne[1];
    const uint32_t stride_row_x   = src0->nb[1] / ggml_type_size(src0->type);
    const uint32_t stride_col_y   = ncols_x / QK8_1;
    const uint32_t stride_col_dst = dst->nb[1] / sizeof(float);
    const uint3    one            = init_fastdiv_values(1);
    const uint3    zero           = make_uint3(0, 0, 0);

    // 16 rows per block (the rows do not change a row's sum): every block applies the chain to the whole src1, so fewer, longer
    // blocks (10240 rows: 640 blocks, about two per SM on B200); GGML_CUDA_MMVQ_EW_RPB=4 for 4
    static const int rpb_env = getenv("GGML_CUDA_MMVQ_EW_RPB") ? atoi(getenv("GGML_CUDA_MMVQ_EW_RPB")) : 16;
    const int rpb = rpb_env == 4 || nrows_x % 16 != 0 ? 4 : 16;
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(dim3((nrows_x + rpb - 1) / rpb, 1, 1), dim3(32, 4, 1), 0, ctx.stream());
#define MMVQ_EW_LAUNCH_R(NC, RPB) ggml_cuda_kernel_launch(mul_mat_vec_q<GGML_TYPE_Q8_0, NC, false, false, false, 4, 0, RPB, 1, true>, \
        launch_params, src0->data, nullptr, nullptr, fusion, (float *) dst->data, ncols_x, zero, stride_row_x, stride_col_y, \
        stride_col_dst, one, 0, 0, 0, one, 0, 0, 0, 0)
#define MMVQ_EW_LAUNCH(NC) if (rpb == 16) { MMVQ_EW_LAUNCH_R(NC, 16); } else { MMVQ_EW_LAUNCH_R(NC, 4); }
    switch (ncols_dst) {
        case 1: MMVQ_EW_LAUNCH(1); break;
        case 2: MMVQ_EW_LAUNCH(2); break;
        case 3: MMVQ_EW_LAUNCH(3); break;
        case 4: MMVQ_EW_LAUNCH(4); break;
        default: GGML_ABORT("fatal error");
    }
#undef MMVQ_EW_LAUNCH
#undef MMVQ_EW_LAUNCH_R
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
