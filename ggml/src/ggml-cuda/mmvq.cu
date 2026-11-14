// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"
#include "cp-async.cuh"
#include "mmvf-splitk.cuh"
#include "mmvf.cuh"
#include "mmvq-kqtc.cuh"

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

// k19-ampere (GGML_CUDA_VERIFY_WIDE bit 8): the Turing / Ampere table sends some MUL_MAT_IDs of 7..8 tokens to MMQ (IQ3_S from 7: the
// Flash-Next gate / up of a DFlash2 n_max 6 window). Inside a verify-tagged graph those columns keep the multi-token mmvq kernels a
// 2..6-row window and every Ada / Blackwell window runs (mul_mat_vec_q_moe_vl for the fused IQ3_S gate / up). Not bit-identical to the
// MMQ launch it replaces (another f32 summation order: reassociation, KLD-gated, plans/k19_ampere.md)
int ggml_cuda_mmvq_mmid_max_batch(ggml_type type, int cc) {
    const int table = get_mmvq_mmid_max_batch(type, cc);
    if (table >= MMVQ_MAX_BATCH_SIZE) {
        return table;
    }
    return std::max(table, ggml_cuda_verify_wide_cols(cc, 8));
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts);

// k19-a80verify (GGML_CUDA_VERIFY_DENSE_MMQ_SM80="min[:mask]", opt-in: unset / 0 = off, "5" = Q6_K from 5 columns): on cc 8.0 (A100 / A30) a dense quantized product
// of min..8 columns inside a verify-tagged graph runs MMQ (int8 tensor cores) instead of the batched mmvq, for the types of the mask
// (1 Q6_K, 2 Q8_0, 4 the other K / legacy quants MMQ takes). The A100 mmvq pays every further column of these products in full (Colab
// A100 80 GB, Qwen3.8-Flash-Next fn4n full card, node trace of a 7-row DFlash2 verify: the 8192-row Q6_K in-projections 87.5 us at 7
// columns vs 47.4 at 3, the dense mat-vecs 1.36 ms of the 2.9 ms each verify row costs, plans/k19_a80verify.md). MUL_MAT_IDs keep their
// rule (ggml_cuda_mmvq_mmid_max_batch); 1..min-1 columns, prompt chunks (untagged graphs) and the exact (batch invariant) verify keep
// mmvq. Not bit-identical to the mmvq launch it replaces (another f32 order of the same int8 dot products: reassociation, KLD-gated).
// Opt-in: full card 7-row DFlash2 chat +4.0% (165.8 -> 172.5 tok/s, md5 equal), but MMQ's pool scratch beside a MoE cache that fills
// the card ran the 48 GB tier out of memory (cuMemCreate in ggml_cuda_mul_mat_q at the first verify); Q8_0 (mask 2) is slower on MMQ
// at these sizes
static bool ggml_cuda_verify_dense_mmq_sm80(const ggml_type type, const int cc, const int64_t ne11) {
    struct cfg_t { int min_cols; int mask; };
    static const cfg_t cfg = [] {
        cfg_t c = { 0, 1 };
        if (const char * e = getenv("GGML_CUDA_VERIFY_DENSE_MMQ_SM80")) {
            int mc = 0, mask = 1;
            const int n = sscanf(e, "%d:%d", &mc, &mask);
            c.min_cols = n >= 1 ? mc : 0;
            c.mask     = n >= 2 ? mask : 1;
        }
        return c;
    }();
    if (cfg.min_cols <= 0 || ne11 < std::max(2, cfg.min_cols) || ne11 > MMVQ_MAX_BATCH_SIZE || !GGML_CUDA_CC_IS_NVIDIA(cc) ||
            cc != GGML_CUDA_CC_AMPERE || !ggml_cuda_verify_graph_active()) {
        return false;
    }
    const int bit = type == GGML_TYPE_Q6_K ? 1 : type == GGML_TYPE_Q8_0 ? 2 : 4;
    if ((cfg.mask & bit) == 0 || !ggml_cuda_should_use_mmq(type, cc, ne11, /*n_experts =*/ 0)) {
        return false;
    }
    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("k19-a80verify: dense verify products of %d..%d columns on MMQ on cc 8.0 (types mask %d: 1 Q6_K, 2 Q8_0, 4 others; "
            "first: %s, %lld columns; GGML_CUDA_VERIFY_DENSE_MMQ_SM80=0 off)\n", std::max(2, cfg.min_cols), MMVQ_MAX_BATCH_SIZE, cfg.mask,
            ggml_type_name(type), (long long) ne11);
    });
    return true;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    if (ggml_cuda_verify_dense_mmq_sm80(type, cc, ne11)) {
        return false;
    }
    // k19-vrow3: a verify window of 5..8 rows keeps mmvq where the tables below send K-quants to MMQ (Blackwell: Q6_K at 8, the lm head
    // of a DFlash2 n_max 7 window), so its rows keep the sums of a 1..4-row window (GGML_CUDA_VERIFY_WIDE bit 2)
    if (ne11 >= 5 && ne11 <= ggml_cuda_verify_wide_cols(cc, 2)) {
        return true;
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

// k19-q38t: the IQ3_S / IQ3_XXS grid tables (2 / 1 KiB) in shared memory for the dense mul_mat_vec_q. Each lane looks up 8 grid
// entries per 32 weights at random indices; from the global table a warp's lookup touches ~14 of the 16 cache lines of iq3s_grid
// (one L1 wavefront each), from shared memory it costs the bank conflicts of 32 random words (~3-4). On a T4 a 1-column IQ3_S
// product ran at ~104 GB/s against ~258 GB/s for IQ4_XS (k19-mmvqcols micro-bench), i.e. bound by these lookups. The table
// values, the int arithmetic and the float expressions are those of vec_dot_iq3_s_q8_1 / vec_dot_iq3_xxs_q8_1 (vecdotq.cuh),
// only the table the entries come from differs, so every output keeps its bytes. GGML_CUDA_MMVQ_GRID_SMEM=0: the global tables
// (set per device in ggml_backend_cuda_init, ggml_cuda_mmvq_grid_init)
static __device__ int mmvq_grid_smem_off = 0;

void ggml_cuda_mmvq_grid_init(int device);

void ggml_cuda_mmvq_grid_init(int device) {
    const char * e = getenv("GGML_CUDA_MMVQ_GRID_SMEM");
    if (e != nullptr && atoi(e) == 0) {
        const int off = 1;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMemcpyToSymbol(mmvq_grid_smem_off, &off, sizeof(off)));
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: GGML_CUDA_MMVQ_GRID_SMEM=0: IQ3_S / IQ3_XXS mat-vec grid lookups from the global tables\n", __func__);
        }
    }
}

static constexpr __host__ __device__ int mmvq_grid_smem_n(ggml_type type) {
    return type == GGML_TYPE_IQ3_S ? 512 : type == GGML_TYPE_IQ3_XXS ? 256 : 0;
}

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_grid(const uint32_t * grid,
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int sb0 = ((signs_packed_8[l0/2] & 0x0F) * 0x00204081) & 0x01010101;
        const int sb1 = ((signs_packed_8[l0/2] >>   4) * 0x00204081) & 0x01010101;

        const int grid_l = (grid_pos.x ^ (sb0 * 0xFF)) + sb0;
        const int grid_h = (grid_pos.y ^ (sb1 * 0xFF)) + sb1;

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_grid(const uint32_t * grid,
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// k19-dsv4p (GGML_CUDA_MMVQ_IQ_TAB): IQ1_S / IQ2_XXS single column mat-vec with the codebook in shared memory. Each vec_dot of these
// types gathers 4 codebook entries at random indices; from global memory (L1) a warp-wide gather of 32 random entries of the 8 KB
// IQ1_S grid touches ~25 cache lines and is replayed per line, from shared memory it costs a few bank conflicts. The entries and the
// integer / float arithmetic are those of vec_dot_iq1_s_q8_1 / vec_dot_iq2_xxs_q8_1, so every sum keeps its bytes.
// k19-iqsmem: also IQ2_XS (iq2xs_grid, 4 KiB) and IQ2_S (iq2s_grid, 8 KiB), under their own switch GGML_CUDA_MMVQ_IQ2_TAB
// (mul_mat_vec_q_tab_launch); vec_dot_iq2_xs_q8_1_tab / vec_dot_iq2_s_q8_1_tab are vec_dot_iq2_xs_q8_1 / vec_dot_iq2_s_q8_1 with the
// entries read from the shared copy (same 64-bit entries, same int / float expressions in the same order)
static constexpr __host__ __device__ bool mmvq_tab_type(ggml_type type) {
    return type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S;
}

static constexpr __host__ __device__ bool mmvq_tab_iq2_type(ggml_type type) {
    return type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S;
}

static constexpr __host__ __device__ int mmvq_tab_words(ggml_type type) {
    return type == GGML_TYPE_IQ1_S ? NGRID_IQ1S : type == GGML_TYPE_IQ2_XXS ? 2*256 : type == GGML_TYPE_IQ2_XS ? 2*512 :
        type == GGML_TYPE_IQ2_S ? 2*1024 : 1;
}

static __device__ __forceinline__ const uint32_t * mmvq_tab_src(ggml_type type) {
    return type == GGML_TYPE_IQ1_S ? (const uint32_t *) iq1s_grid_gpu : type == GGML_TYPE_IQ2_XXS ? (const uint32_t *) iq2xxs_grid :
        type == GGML_TYPE_IQ2_XS ? (const uint32_t *) iq2xs_grid : (const uint32_t *) iq2s_grid;
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_tab(
    const uint2 * __restrict__ tab, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = tab[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1_tab(
    const uint2 * __restrict__ tab, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = tab[qs[l0/2] | ((qh << (8-l0)) & 0x300)];

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4((int) grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4((int) grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1_tab(
    const uint32_t * __restrict__ tab, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq1->qh[iqs];

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = tab[qs[l0/2] | (((qh >> 3*(l0/2)) & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const float  d1q   = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
    const float2 ds    = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x*sumi + ds.y*delta);
}

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1_tab(
    const uint2 * __restrict__ tab, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = tab[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// the vec_dot of the kernel: with c_tab the shared memory codebook variant of an mmvq_tab_type (k19-dsv4p), the IQ3_S / IQ3_XXS
// grid through grid (k19-q38t: shared memory, or the global table under GGML_CUDA_MMVQ_GRID_SMEM=0), else vec_dot_q_cuda
template <ggml_type type, bool c_tab>
static __device__ __forceinline__ float mmvq_vec_dot(const void * tab, const uint32_t * grid,
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int kbx, const int kqs) {
    if constexpr (c_tab && type == GGML_TYPE_IQ1_S) {
        GGML_UNUSED(grid);
        return vec_dot_iq1_s_q8_1_tab((const uint32_t *) tab, vx, y, kbx, kqs);
    } else if constexpr (c_tab && type == GGML_TYPE_IQ2_XXS) {
        GGML_UNUSED(grid);
        return vec_dot_iq2_xxs_q8_1_tab((const uint2 *) tab, vx, y, kbx, kqs);
    } else if constexpr (c_tab && type == GGML_TYPE_IQ2_XS) {
        GGML_UNUSED(grid);
        return vec_dot_iq2_xs_q8_1_tab((const uint2 *) tab, vx, y, kbx, kqs);
    } else if constexpr (c_tab && type == GGML_TYPE_IQ2_S) {
        GGML_UNUSED(grid);
        return vec_dot_iq2_s_q8_1_tab((const uint2 *) tab, vx, y, kbx, kqs);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        GGML_UNUSED(tab);
        return vec_dot_iq3_s_q8_1_grid(grid, vx, y, kbx, kqs);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        GGML_UNUSED(tab);
        return vec_dot_iq3_xxs_q8_1_grid(grid, vx, y, kbx, kqs);
    } else {
        GGML_UNUSED(tab);
        GGML_UNUSED(grid);
        constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
        return vec_dot_q_cuda(vx, y, kbx, kqs);
    }
}

// c_nwarps > 0 overrides the number of warps (batch invariant verify: the K split and the reduction of one column)
// n_multi > 1: n_multi dense matrices of one type and one K over the same activation (fusion.multi_*), each block computes the
// rows of one of them exactly as that matrix's own launch would (same warps, rows per block and reduction)
// c_unroll > 1: the K loop unrolled c_unroll times, so that a thread has the weight loads of several iterations in flight (few rows,
// long K: the grid covers few SMs and waits on memory latency); every thread adds its blocks in the same order, so the bytes are the same
// c_tab: the codebook of an mmvq_tab_type in shared memory (k19-dsv4p, mmvq_vec_dot)
// c_wpr (k19-q4xt): every warp of the block is a block of its own (one warp, c_rpb rows): warp w takes rows
// c_rpb*(c_wpr*blockIdx.x + w), no shared memory reduction (c_nwarps = c_wpr warps per block, single column only)
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, int c_nwarps = 0, int n_multi = 0, int c_rpb = 0, int c_unroll = 1, bool c_ewq = false, bool c_tab = false, int c_wpr = 0>
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
    constexpr bool wpr = c_wpr > 0;
    static_assert(!wpr || (ncols_dst == 1 && c_nwarps == c_wpr && n_multi == 0 && !c_ewq && !c_tab && mmvq_grid_smem_n(type) == 0),
        "c_wpr: single column, plain types");
    // the warps whose partial sums are reduced together (c_wpr: each warp alone)
    constexpr int nwarps_red = wpr ? 1 : nwarps;

    const     int tid = wpr ? (int) threadIdx.x : warp_size*threadIdx.y + threadIdx.x;
              int row0 = wpr ? rows_per_cuda_block*(c_wpr*blockIdx.x + threadIdx.y) : rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps_red*warp_size / qi;

    // k19-dsv4p: the codebook into shared memory (constant data: before the PDL wait, every thread of the block passes here)
    constexpr bool use_tab = c_tab && mmvq_tab_type(type);
    [[maybe_unused]] const void * tab = nullptr;
    if constexpr (use_tab) {
        __shared__ alignas(16) uint32_t tab_shared[mmvq_tab_words(type)];
        const uint32_t * tab_src = mmvq_tab_src(type);
        for (int i = tid; i < mmvq_tab_words(type); i += nwarps*warp_size) {
            tab_shared[i] = tab_src[i];
        }
        __syncthreads();
        tab = tab_shared;
    }

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
    // k19-exactdet: the rows of dst, the bound of a partial last row block (rows_per_cuda_block > 1: small K, rows not a multiple of the
    // rows per block). Dense: the column stride. MUL_MAT_ID at one column: the slot stride (stride_col_dst is the token stride there, so
    // the last block of a slot wrote its rows past the expert into the next slot's first rows, which that slot's block also writes: the
    // bytes depended on which block wrote last, and the last slot of the last token wrote past dst)
    const uint32_t nrows_dst = ncols_dst == 1 && ids && !shared_expert ? stride_channel_dst : stride_col_dst;
    // k19-ikfix: the last row a partial last row block reads. Its rows past nrows_dst read the last row again instead of the bytes after
    // the matrix (past the end of the weight buffer for the last matrix: an out-of-bounds read); their sums are not stored (the row guard
    // below), the rows that are stored read what they read before, so every output keeps its bytes
    const int i_last = rows_per_cuda_block == 1 ? 0 : max(0, min(rows_per_cuda_block, (int) nrows_dst - row0) - 1);

    // the selected expert is uniform in the block, so the whole block leaves on a skipped slot
    if (!shared_expert && ncols_dst == 1 && ids && ids[channel_dst] == -1) {
        dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
        if (wpr || threadIdx.y == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_dst)) {
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
        if (threadIdx.x < rows_per_cuda_block && (wpr || threadIdx.y == 0) &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_dst)) {
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

    // k19-q38t: grid table in shared memory (see mmvq_grid_smem_off)
    constexpr int grid_n = mmvq_grid_smem_n(type);
    __shared__ uint32_t grid_sm[grid_n > 0 ? grid_n : 1];
    const uint32_t * grid = grid_sm;
    if constexpr (grid_n > 0) {
        const uint32_t * grid_g = type == GGML_TYPE_IQ3_S ? iq3s_grid : iq3xxs_grid;
        if (mmvq_grid_smem_off) {
            grid = grid_g;
        } else {
            for (int e = tid; e < grid_n; e += nwarps*warp_size) {
                grid_sm[e] = grid_g[e];
            }
            __syncthreads();
        }
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
                    const size_t off = (size_t)(kbx_offset + min(i, i_last)*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
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
                tmp[j][i] += mmvq_vec_dot<type, use_tab>(tab, grid,
                    vx, &y[j*stride_col_y + kby], kbx_offset + min(i, i_last)*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += mmvq_vec_dot<type, use_tab>(tab, grid,
                            vgate, &y[j*stride_col_y + kby], kbx_offset + min(i, i_last)*stride_row_x + kbx, kqs);
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
                    const size_t off = (size_t)(kbx_offset + min(i, i_last)*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
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
                tmp[j][i] += mmvq_vec_dot<type, use_tab>(tab, grid,
                    vx, &y[j*stride_col_y + kby], kbx_offset + min(i, i_last)*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += mmvq_vec_dot<type, use_tab>(tab, grid,
                            vgate, &y[j*stride_col_y + kby], kbx_offset + min(i, i_last)*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
        }
    }

    __shared__ float tmp_shared[nwarps_red-1 > 0 ? nwarps_red-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps_red-1 > 0)) ? nwarps_red-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if constexpr (!wpr) {
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
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps_red-1; ++l) {
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

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_dst)) {
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
// k19-moebw: c_unroll > 1 runs c_unroll K iterations of a lane per pass of its loop (the later ones guarded), so their weight and
// activation loads are in flight together; a lane still adds its blocks in the same order, so every sum keeps its bytes
template <ggml_type type, int c_rows_per_block, bool has_fusion = false, int c_unroll = 1>
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
    // k19-ikfix: rows past nrows_x of a partial last row block read the last row (not the bytes after the expert); their sums are not stored
    const int i_last = c_rows_per_block == 1 ? 0 : max(0, min(c_rows_per_block, (int) nrows_x - row0) - 1);

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    if constexpr (c_unroll > 1) {
        const int kqs = vdr * (threadIdx.x % (qi/vdr));
        for (int kb0 = threadIdx.x / (qi/vdr); kb0 < blocks_per_row_x; kb0 += c_unroll*blocks_per_iter) {
#pragma unroll
            for (int u = 0; u < c_unroll; ++u) {
                const int kbx = kb0 + u*blocks_per_iter;
                if (u == 0 || kbx < blocks_per_row_x) {
                    const int kby = kbx * (qk/QK8_1);
#pragma unroll
                    for (int i = 0; i < c_rows_per_block; ++i) {
                        tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + min(i, i_last)*stride_row_x + kbx, kqs);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + min(i, i_last)*stride_row_x + kbx, kqs);
                            }
                        }
                    }
                }
            }
        }
    } else {
    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + min(i, i_last)*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + min(i, i_last)*stride_row_x + kbx, kqs);
                }
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

// k19-vrow: the weight operand of one vec_dot_q_cuda call of a MoE row, decoded once and applied to every token of the expert's group
// (mul_mat_vec_q_moe_group). apply(y) is the type's vec_dot_q_cuda(vx, y, kbx, kqs) with the weight side precomputed: the same int8
// dot products into the same int sum and the same float products in the same order, so a token's value is bitwise its own call's.
// Types without a specialisation keep the call itself (c_dec false, or no decode below)
template <ggml_type type, bool c_dec>
struct mmvq_moe_wdec {
    const void * vx;
    int          kbx;
    int          kqs;
    __device__ __forceinline__ void load(const void * vx_, const int kbx_, const int kqs_) {
        vx = vx_; kbx = kbx_; kqs = kqs_;
    }
    __device__ __forceinline__ float apply(const block_q8_1 * y) const {
        constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
        return vec_dot_q_cuda(vx, y, kbx, kqs);
    }
};

// vec_dot_iq3_s_q8_1
template <>
struct mmvq_moe_wdec<GGML_TYPE_IQ3_S, true> {
    int   grid[8];
    int   scale;
    float d;
    int   iqs;
    __device__ __forceinline__ void load(const void * vbq, const int kbx, const int iqs_) {
        const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;
        iqs = iqs_;

        const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bq3->qh[iqs/2];

        const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

            // k19-moebw: the signs as in vec_dot_iq3_s_q8_1 (0x01 per negative byte, -g = (g ^ 0xFF) + 1 without a carry)
            const int sb0 = ((signs_packed_8[l0/2] & 0x0F) * 0x00204081) & 0x01010101;
            const int sb1 = ((signs_packed_8[l0/2] >>   4) * 0x00204081) & 0x01010101;

            grid[l0 + 0] = (grid_pos.x ^ (sb0 * 0xFF)) + sb0;
            grid[l0 + 1] = (grid_pos.y ^ (sb1 * 0xFF)) + sb1;
        }

        scale = 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
        d     = __half2float(bq3->d);
    }
    __device__ __forceinline__ float apply(const block_q8_1 * bq8_1) const {
        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

            sumi = ggml_cuda_dp4a(grid[l0 + 0], u0, sumi);
            sumi = ggml_cuda_dp4a(grid[l0 + 1], u1, sumi);
        }
        sumi *= scale;

        const float dd = d * __low2float(bq8_1[iqs/2].ds);
        return dd * sumi;
    }
};

// vec_dot_iq4_nl_q8_1
template <>
struct mmvq_moe_wdec<GGML_TYPE_IQ4_NL, true> {
    int2  v[VDR_Q4_0_Q8_1_MMVQ];
    float d;
    int   iqs;
    __device__ __forceinline__ void load(const void * vbq, const int kbx, const int iqs_) {
        const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;
        iqs = iqs_;
#pragma unroll
        for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
            v[l] = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
        }
        d = __half2float(bq4->d);
    }
    __device__ __forceinline__ float apply(const block_q8_1 * bq8_1) const {
        const int * q8 = (const int *) bq8_1->qs + iqs;

        int sumi = 0;
#pragma unroll
        for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
            sumi = ggml_cuda_dp4a(v[l].x, q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(v[l].y, q8[l + 4], sumi);
        }

        const float dd = d * __low2float(bq8_1->ds);
        return dd * sumi;
    }
};

// k19-vrow: mul_mat_vec_q_moe with the (token, slot) entries grouped by expert. A verify window of T tokens x k slots names fewer
// distinct experts than entries (Flash-Next: ~1.9 x k for T = 3); mul_mat_vec_q_moe gives each entry a warp of its own, which reads and
// decodes the expert's rows again for every token that chose it. Here the warp of the first entry of an expert (token-major order)
// takes every entry of that expert, decodes each weight block once and applies it to all of them; the other entries' warps leave.
// Every (token, slot, row) value is bitwise mul_mat_vec_q_moe's: the same lane runs the same vec_dot_q_cuda sums over the same
// K blocks in the same order (mmvq_moe_wdec), the same warp_reduce_sum and the same epilogue; only which warp computes a row changes.
// Grid: (ceil(nrows_x / (c_rpw*c_nwarps)), nchannels_dst (slots), ncols_dst (tokens)); block (warp_size, c_nwarps), warp y owns c_rpw
// rows and takes c_group entries of its expert per pass over K (more: further passes over K, the weights then come from L1 / L2).
// NVIDIA only (ballots on a 32 lane warp); no shared expert
// k19-moebw: c_unroll > 1 as in mul_mat_vec_q_moe (c_unroll K iterations per pass, the same order of every lane's sums)
template <ggml_type type, int c_rpw, int c_nwarps, int c_group, bool c_dec, bool has_fusion, int c_unroll = 1>
__launch_bounds__(c_nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe_group(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(vx_ptr, vy_ptr, ids_ptr, fusion, dst_ptr, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride);
    NO_DEVICE_CODE;
#else
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int warp_size = 32;

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
        vgate      = fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const int      lane   = threadIdx.x;
    const int      row0   = (blockIdx.x*c_nwarps + threadIdx.y)*c_rpw;
    const uint32_t slot   = blockIdx.y;
    const uint32_t token  = blockIdx.z;
    const uint32_t nslots = gridDim.y;
    const int      n_ent  = nslots*ncols_dst;   // <= 128 (host)
    const int      my_e   = token*nslots + slot;

    if (uint32_t(row0) >= nrows_x) {
        return;
    }

    ggml_cuda_pdl_sync();
    const int32_t expert = ids[slot + token*ids_stride];

    if (expert == -1) {
        if (lane < c_rpw) {
            dst[slot*stride_channel_dst + token*stride_col_dst + row0 + lane] = 0.0f;
        }
        return;
    }

    // the entries of this expert, token-major, as a 128-bit set (two words); an earlier one owns them
    uint64_t members[2] = { 0, 0 };
    for (int e0 = 0; e0 < n_ent; e0 += warp_size) {
        const int e = e0 + lane;
        bool hit = false;
        if (e < n_ent) {
            const uint32_t t = e / nslots;
            hit = ids[(e - t*nslots) + t*ids_stride] == expert;
        }
        members[e0 >> 6] |= uint64_t(__ballot_sync(0xFFFFFFFF, hit)) << (e0 & 63);
    }
    const bool owned_earlier = my_e < 64 ? (members[0] & ((uint64_t(1) << my_e) - 1)) != 0
                                         : members[0] != 0 || (members[1] & ((uint64_t(1) << (my_e - 64)) - 1)) != 0;
    if (owned_earlier) {
        return;
    }

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    const int kbx_offset = expert*stride_channel_x + row0*stride_row_x;

    while ((members[0] | members[1]) != 0) {
        int n = 0;
        const block_q8_1 * y[c_group];
        uint32_t           dst_off[c_group];
#pragma unroll
        for (int j = 0; j < c_group; ++j) {
            y[j]       = (const block_q8_1 *) vy;
            dst_off[j] = 0;
            if ((members[0] | members[1]) != 0) {
                const int      w = members[0] != 0 ? 0 : 1;
                const int      e = 64*w + __ffsll((long long) members[w]) - 1;
                members[w] &= members[w] - 1;
                const uint32_t t = e / nslots;
                const uint32_t s = e - t*nslots;
                y[j]       = ((const block_q8_1 *) vy) + fastmodulo(s, nchannels_y)*stride_channel_y + t*stride_col_y;
                dst_off[j] = s*stride_channel_dst + t*stride_col_dst;
                n = j + 1;
            }
        }

        float tmp[c_group][c_rpw]      = {{0.0f}};
        float tmp_gate[c_group][c_rpw] = {{0.0f}};

        const auto k_iter = [&](const int kbx) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (lane % (qi/vdr));

#pragma unroll
            for (int i = 0; i < c_rpw; ++i) {
                mmvq_moe_wdec<type, c_dec> w;
                w.load(vx, kbx_offset + i*stride_row_x + kbx, kqs);
                mmvq_moe_wdec<type, c_dec> g;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        g.load(vgate, kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
#pragma unroll
                for (int j = 0; j < c_group; ++j) {
                    if (j < n) {
                        tmp[j][i] += w.apply(&y[j][kby]);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                tmp_gate[j][i] += g.apply(&y[j][kby]);
                            }
                        }
                    }
                }
            }
        };
        if constexpr (c_unroll > 1) {
            for (int kb0 = lane / (qi/vdr); kb0 < blocks_per_row_x; kb0 += c_unroll*blocks_per_iter) {
#pragma unroll
                for (int u = 0; u < c_unroll; ++u) {
                    if (u == 0 || kb0 + u*blocks_per_iter < blocks_per_row_x) {
                        k_iter(kb0 + u*blocks_per_iter);
                    }
                }
            }
        } else {
        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (lane % (qi/vdr));

#pragma unroll
            for (int i = 0; i < c_rpw; ++i) {
                mmvq_moe_wdec<type, c_dec> w;
                w.load(vx, kbx_offset + i*stride_row_x + kbx, kqs);
                mmvq_moe_wdec<type, c_dec> g;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        g.load(vgate, kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
#pragma unroll
                for (int j = 0; j < c_group; ++j) {
                    if (j < n) {
                        tmp[j][i] += w.apply(&y[j][kby]);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                tmp_gate[j][i] += g.apply(&y[j][kby]);
                            }
                        }
                    }
                }
            }
        }
        }
        GGML_UNUSED(k_iter);

        ggml_cuda_pdl_lc();

#pragma unroll
        for (int j = 0; j < c_group; ++j) {
            if (j >= n) {
                break;
            }
#pragma unroll
            for (int i = 0; i < c_rpw; ++i) {
                tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                    }
                }
            }

            // mul_mat_vec_q_moe's epilogue
            if (lane < c_rpw) {
                float result = tmp[j][0];
                float gate_value = tmp_gate[j][0];
#pragma unroll
                for (int i = 1; i < c_rpw; ++i) {
                    if (lane == i) {
                        result     = tmp[j][i];
                        gate_value = tmp_gate[j][i];
                    }
                }
                if constexpr (has_fusion) {
                    const uint32_t bias_idx = expert*stride_channel_dst + row0 + lane;

                    if constexpr (type == GGML_TYPE_NVFP4) {
                        if (x_scale) {
                            result *= x_scale[expert];
                        }
                    }
                    if (x_bias) {
                        result += x_bias[bias_idx];
                    }
                    if (use_gate) {
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            if (gate_scale) {
                                gate_value *= gate_scale[expert];
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
                GGML_UNUSED(gate_value);
                dst[dst_off[j] + row0 + lane] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// k19-moebw: the multi-token routed-expert mat-vec with the weight rows of a warp fetched as whole 16 byte chunks into shared memory
// before any of them is used, then the K loop of mul_mat_vec_q_moe on that copy: lane l takes the K blocks l / (qi / vdr) +
// n * vdr * 32 / qi with kqs = vdr * (l % (qi / vdr)), sums them in that order, then warp_reduce_sum and the moe epilogue: the same
// vec_dot_q_cuda calls on the same bytes, so every output is bitwise mul_mat_vec_q_moe's (and mul_mat_vec_q_moe_group's). As in
// mul_mat_vec_q_moe_group, a warp takes every entry (token, slot) that chose its expert, so an expert's rows are read from DRAM once.
// Turing has no cp.async: 16 byte loads through registers. On Turing (T4) the multi-token kernels run at ~20% of the DRAM peak and
// this one is 20-35% faster at 2..8 tokens; on Blackwell (G4) it is slower than the K-unrolled mul_mat_vec_q_moe (plans/k19_moebw.md)
// Grid: (ceil(nrows_x / (rpw * nwarps)), slots, tokens); block (32, nwarps); warp y owns rows [row0, row0 + rpw).
template <ggml_type type, bool has_fusion>
__launch_bounds__(8*WARP_SIZE)
static __global__ void mul_mat_vec_q_moe_bw(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ntok, const uint32_t ids_stride, const int rpw, const uint32_t smem_mat) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(vx_ptr, vy_ptr, ids_ptr, fusion, dst_ptr, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst, ntok, ids_stride, rpw, smem_mat);
    NO_DEVICE_CODE;
#else
    extern __shared__ int4 mmvq_moe_bw_smem[];

    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int bs        = ggml_cuda_type_traits<type>::bs;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int warp_size = 32;
    constexpr int lpb       = qi/vdr;                // lanes per weight block
    constexpr int bpi       = vdr*warp_size/qi;      // weight blocks per K iteration
    constexpr int c_rr      = 2;                     // rows computed together
    static_assert(warp_size % lpb == 0, "lanes per block");

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x*blockDim.y + threadIdx.y)*rpw;

    if (uint32_t(row0) >= nrows_x) {
        return;
    }

    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
    }

    const uint32_t slot   = blockIdx.y;
    const uint32_t token  = blockIdx.z;
    const uint32_t nslots = gridDim.y;
    const int      my_e   = token*nslots + slot;

    ggml_cuda_pdl_sync();
    const int32_t expert = ids[slot + token*ids_stride];

    if (expert == -1) {
        for (int i = lane; i < rpw; i += warp_size) {
            dst[slot*stride_channel_dst + token*stride_col_dst + row0 + i] = 0.0f;
        }
        return;
    }

    // the entries of this expert, token-major, as a 128-bit set; the first one owns them (mul_mat_vec_q_moe_group)
    uint64_t mem_lo = 0;   // entries 0..63
    uint64_t mem_hi = 0;   // entries 64..127
    {
        const int n_ent = nslots*ntok; // <= 128 (host)
        for (int e0 = 0; e0 < n_ent; e0 += warp_size) {
            const int e = e0 + lane;
            bool hit = false;
            if (e < n_ent) {
                const uint32_t t = e / nslots;
                hit = ids[(e - t*nslots) + t*ids_stride] == expert;
            }
            const uint64_t b = uint64_t(__ballot_sync(0xFFFFFFFF, hit)) << (e0 & 63);
            if (e0 < 64) {
                mem_lo |= b;
            } else {
                mem_hi |= b;
            }
        }
        const bool owned_earlier = my_e < 64 ? (mem_lo & ((uint64_t(1) << my_e) - 1)) != 0
                                             : mem_lo != 0 || (mem_hi & ((uint64_t(1) << (my_e - 64)) - 1)) != 0;
        if (owned_earlier) {
            return;
        }
    }

    // this warp's rows of the expert (contiguous bytes) -> shared memory, at the same address modulo 16
    const size_t tile_off = ((size_t) expert*stride_channel_x + (size_t) row0*stride_row_x)*bs;
    const int    nbytes   = rpw*stride_row_x*bs;
    char * sw = (char *) mmvq_moe_bw_smem + threadIdx.y*(use_gate ? 2 : 1)*smem_mat;
    const auto copy = [&](char * sm, const char * g) -> const void * {
        const char * g0  = (const char *) ((uintptr_t) g & ~uintptr_t(15));
        const int    nch = (int) ((g + nbytes - g0 + 15) / 16);
#if defined(CP_ASYNC_AVAILABLE)
        const unsigned int s32 = ggml_cuda_cvta_generic_to_shared(sm);
        for (int c = lane; c < nch; c += warp_size) {
            cp_async_cg_16<0>(s32 + 16*c, g0 + 16*c);
        }
#else
        constexpr int U = 4;
        for (int c0 = 0; c0 < nch; c0 += U*warp_size) {
            int4 v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = c0 + u*warp_size + lane;
                if (c < nch) {
                    v[u] = ((const int4 *) g0)[c];
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = c0 + u*warp_size + lane;
                if (c < nch) {
                    ((int4 *) sm)[c] = v[u];
                }
            }
        }
#endif // defined(CP_ASYNC_AVAILABLE)
        return sm + (g - g0);
    };
    const void * xs = copy(sw, (const char *) vx + tile_off);
    const void * gs = use_gate ? copy(sw + smem_mat, (const char *) vgate + tile_off) : nullptr;
#if defined(CP_ASYNC_AVAILABLE)
    cp_async_wait_all();
#endif // defined(CP_ASYNC_AVAILABLE)
    __syncwarp();

    ggml_cuda_pdl_lc();

    const int blocks_per_row_x = ncols_x / qk;
    const int kqs              = vdr * (lane % lpb);

    while ((mem_lo | mem_hi) != 0) {
        int e;
        if (mem_lo != 0) {
            e = __ffsll((long long) mem_lo) - 1;
            mem_lo &= mem_lo - 1;
        } else {
            e = 64 + __ffsll((long long) mem_hi) - 1;
            mem_hi &= mem_hi - 1;
        }
        const uint32_t t = e / nslots;
        const uint32_t s = e - t*nslots;
        const block_q8_1 * y  = ((const block_q8_1 *) vy) + fastmodulo(s, nchannels_y)*stride_channel_y + t*stride_col_y;
        float            * de = dst + s*stride_channel_dst + t*stride_col_dst + row0;

        for (int i0 = 0; i0 < rpw; i0 += c_rr) {
            float tmp[c_rr]      = {0.0f};
            float tmp_gate[c_rr] = {0.0f};

            for (int kbx = lane / lpb; kbx < blocks_per_row_x; kbx += bpi) {
                const int kby = kbx * (qk/QK8_1);
#pragma unroll
                for (int r = 0; r < c_rr; ++r) {
                    tmp[r] += vec_dot_q_cuda(xs, &y[kby], (i0 + r)*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[r] += vec_dot_q_cuda(gs, &y[kby], (i0 + r)*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }

#pragma unroll
            for (int r = 0; r < c_rr; ++r) {
                tmp[r] = warp_reduce_sum<warp_size>(tmp[r]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[r] = warp_reduce_sum<warp_size>(tmp_gate[r]);
                    }
                }
                // mul_mat_vec_q_moe's epilogue
                if (lane == r) {
                    const int row = row0 + i0 + r;
                    float result = tmp[r];
                    if constexpr (has_fusion) {
                        const uint32_t bias_idx = expert*stride_channel_dst + row;
                        if (x_bias) {
                            result += x_bias[bias_idx];
                        }
                        if (use_gate) {
                            float gate_value = tmp_gate[r];
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
                    de[i0 + r] = result;
                }
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(vgate, x_bias, gate_bias, active_glu, glu_limit);
    }
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// k19-moegrp: the multi-token routed-expert mat-vec with c_tpr threads per row and the entries of an expert c_g at a time, bitwise
// mul_mat_vec_q_moe. That kernel gives a row a warp: lane l (a "virtual lane" here) sums vec_dot_q_cuda over the K blocks
// l / (qi / vdr) + n * vdr * 32 / qi (kqs = vdr * (l % (qi / vdr))) in that order, then warp_reduce_sum adds lane l ^ o to lane l for
// o = 16, 8, 4, 2, 1. Here thread t of a row computes the sums of the virtual lanes t + c_tpr * m (m < 32 / c_tpr) in the same order,
// adds them as the levels o >= c_tpr of that butterfly would (x_v + x_(v ^ o), in registers), then runs the levels o < c_tpr with
// shuffles among the row's c_tpr threads: every addition has the same operands, so every (entry, row) value is the base's bit for bit.
// Each weight operand is decoded once for the c_g entries (mmvq_moe_wdec), a warp holds 32 / c_tpr rows, and as in
// mul_mat_vec_q_moe_group the block of an expert's first entry (token-major) takes every entry of that expert, the others leave.
// Grid: (nrows_x / (32 / c_tpr * c_nwarps), slots, tokens); block (32, c_nwarps). NVIDIA only; no shared expert.
// On G4 it is faster than mul_mat_vec_q_moe_group / mul_mat_vec_q_moe_bw for the IQ3_S gate / up only (mul_mat_vec_q_moe_vl_launch)
template <ggml_type type, int c_tpr, int c_g, int c_nwarps, bool has_fusion>
__launch_bounds__(c_nwarps*32, 1)
static __global__ void mul_mat_vec_q_moe_vl(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(vx_ptr, vy_ptr, ids_ptr, fusion, dst_ptr, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride);
    NO_DEVICE_CODE;
#else
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int warp_size = 32;
    constexpr int lpb       = qi/vdr;               // virtual lanes per weight block
    constexpr int bpi       = vdr*warp_size/qi;     // weight blocks per K iteration of a virtual lane
    constexpr int c_vl      = warp_size/c_tpr;      // virtual lanes per thread = rows per warp
    constexpr bool c_dec    = type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL;
    static_assert(warp_size % c_tpr == 0 && c_tpr >= 1, "threads per row");

    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
    }

    const int      lane   = threadIdx.x;
    const int      t      = lane % c_tpr;            // thread of the row
    const int      warp0  = (blockIdx.x*c_nwarps + threadIdx.y)*c_vl;
    const int      row    = warp0 + lane / c_tpr;
    const uint32_t slot   = blockIdx.y;
    const uint32_t token  = blockIdx.z;
    const uint32_t nslots = gridDim.y;
    const int      n_ent  = nslots*ncols_dst;   // <= 128 (host)
    const int      my_e   = token*nslots + slot;

    if (uint32_t(warp0) >= nrows_x) {   // nrows_x is a multiple of c_vl (host): a warp has all its rows or none
        return;
    }

    ggml_cuda_pdl_sync();
    const int32_t expert = ids[slot + token*ids_stride];

    if (expert == -1) {
        if (t == 0) {
            dst[slot*stride_channel_dst + token*stride_col_dst + row] = 0.0f;
        }
        return;
    }

    // the entries of this expert, token-major, as a 128-bit set; an earlier one owns them (mul_mat_vec_q_moe_group)
    uint64_t members[2] = { 0, 0 };
    for (int e0 = 0; e0 < n_ent; e0 += warp_size) {
        const int e = e0 + lane;
        bool hit = false;
        if (e < n_ent) {
            const uint32_t te = e / nslots;
            hit = ids[(e - te*nslots) + te*ids_stride] == expert;
        }
        members[e0 >> 6] |= uint64_t(__ballot_sync(0xFFFFFFFF, hit)) << (e0 & 63);
    }
    const bool owned_earlier = my_e < 64 ? (members[0] & ((uint64_t(1) << my_e) - 1)) != 0
                                         : members[0] != 0 || (members[1] & ((uint64_t(1) << (my_e - 64)) - 1)) != 0;
    if (owned_earlier) {
        return;
    }

    const int blocks_per_row_x = ncols_x / qk;
    const int kbx_offset       = expert*stride_channel_x + row*stride_row_x;

    while ((members[0] | members[1]) != 0) {
        int n = 0;
        const block_q8_1 * y[c_g];
        uint32_t           dst_off[c_g];
#pragma unroll
        for (int j = 0; j < c_g; ++j) {
            y[j]       = (const block_q8_1 *) vy;
            dst_off[j] = 0;
            if ((members[0] | members[1]) != 0) {
                const int      w = members[0] != 0 ? 0 : 1;
                const int      e = 64*w + __ffsll((long long) members[w]) - 1;
                members[w] &= members[w] - 1;
                const uint32_t te = e / nslots;
                const uint32_t s  = e - te*nslots;
                y[j]       = ((const block_q8_1 *) vy) + fastmodulo(s, nchannels_y)*stride_channel_y + te*stride_col_y;
                dst_off[j] = s*stride_channel_dst + te*stride_col_dst;
                n = j + 1;
            }
        }

        float tmp[c_g][c_vl]      = {{0.0f}};
        float tmp_gate[c_g][c_vl] = {{0.0f}};

#pragma unroll
        for (int m = 0; m < c_vl; ++m) {
            const int v   = t + c_tpr*m;               // the virtual lane
            const int kqs = vdr * (v % lpb);
            for (int kbx = v / lpb; kbx < blocks_per_row_x; kbx += bpi) {
                const int kby = kbx * (qk/QK8_1);
                mmvq_moe_wdec<type, c_dec> w;
                w.load(vx, kbx_offset + kbx, kqs);
                mmvq_moe_wdec<type, c_dec> g;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        g.load(vgate, kbx_offset + kbx, kqs);
                    }
                }
#pragma unroll
                for (int j = 0; j < c_g; ++j) {
                    if (j < n) {
                        tmp[j][m] += w.apply(&y[j][kby]);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                tmp_gate[j][m] += g.apply(&y[j][kby]);
                            }
                        }
                    }
                }
            }
        }

        ggml_cuda_pdl_lc();

        // the butterfly levels o >= c_tpr in registers: virtual lane t + c_tpr*m takes x_v + x_(v ^ o), v ^ o = t + c_tpr*(m ^ o/c_tpr)
        const auto reduce = [&](float (&x)[c_vl]) -> float {
#pragma unroll
            for (int o = warp_size/2; o >= c_tpr; o >>= 1) {
                float nx[c_vl];
#pragma unroll
                for (int m = 0; m < c_vl; ++m) {
                    nx[m] = x[m] + x[m ^ (o/c_tpr)];
                }
#pragma unroll
                for (int m = 0; m < c_vl; ++m) {
                    x[m] = nx[m];
                }
            }
            return warp_reduce_sum<c_tpr>(x[0]);   // the levels o < c_tpr among the row's threads (virtual lane t)
        };

#pragma unroll
        for (int j = 0; j < c_g; ++j) {
            if (j >= n) {
                break;
            }
            float result     = reduce(tmp[j]);
            float gate_value = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    gate_value = reduce(tmp_gate[j]);
                }
            }
            // mul_mat_vec_q_moe's epilogue
            if (t == 0) {
                if constexpr (has_fusion) {
                    const uint32_t bias_idx = expert*stride_channel_dst + row;
                    if (x_bias) {
                        result += x_bias[bias_idx];
                    }
                    if (use_gate) {
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
                GGML_UNUSED(gate_value);
                dst[dst_off[j] + row] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, vgate, x_bias, gate_bias, active_glu, glu_limit);
    }
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
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

// k19-moebw: the types whose mul_mat_vec_q_moe has K-unrolled instances (the routed experts of the models we ship)
static constexpr bool mmvq_moe_unroll_type(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

// k19-ampere, GGML_CUDA_MOE_AMPERE (default 1; 0 = the Blackwell-tuned shapes there too): on sm_80 (A100; measured there only, sm_86 /
// sm_87 keep the old rules) the multi-token MoE mat-vecs pick other shapes of the same kernels, every one bitwise the kernel it replaces
// (test-moe-bw-exact on A100: rpb8_g0 / bw1 / vl16_4 0 floats differ). A100 microbench (test-moe-bw-exact --bench, Flash-Next shapes,
// 128 experts, P 0.55; us per layer vs the old rule): IQ4_NL down at 5..8 tokens 8 rows per warp block 0.86 / 0.84 / 0.88 / 0.83;
// IQ3_S gate or up without a fused gate (MoE cache tiers) at 5..8 tokens the staged kernel 0.90 / 0.92 / 0.87 / 0.87, the fused gate / up
// at 2 tokens 0.81; mul_mat_vec_q_moe_vl with 4 entries per decode at 3..5 tokens 0.91 / 0.88 / 0.86 (plans/k19_ampere.md)
static bool mmvq_moe_ampere(const int cc) {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MOE_AMPERE");
        return e == nullptr || atoi(e) != 0;
    }();
    return on && GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_AMPERE;
}

static void mmvq_moe_ampere_log(const char * what, const ggml_type type, const int ntok) {
    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("k19-ampere: sm_80 MoE mat-vec shapes engaged (first: %s, %s, %d tokens; GGML_CUDA_MOE_AMPERE=0 off)\n", what,
            ggml_type_name(type), ntok);
    });
}

// k19-moebw, GGML_CUDA_MOE_UNROLL (default 4; 0 or 1 = off): a multi-token MUL_MAT_ID (verify window) lane runs up to this many of its
// K iterations per pass (2 or 4), so their loads are in flight together instead of one iteration's at a time; the warps of the MoE
// kernels are few and short (one row pair per token, 1..4 iterations of K), so they wait on L1TEX / L2 / DRAM latency (Nsight Compute
// on G4: 40-78% of the stalls). A lane adds its K blocks in the same order, so every output keeps its bytes. 1 for a K of one iteration
template <ggml_type type>
static int mmvq_moe_unroll(const int ncols_x, const int warp_size) {
    static const int env = [] {
        const char * e = getenv("GGML_CUDA_MOE_UNROLL");
        return e == nullptr ? 1 : atoi(e);
    }();
    if constexpr (!mmvq_moe_unroll_type(type)) {
        GGML_UNUSED_VARS(ncols_x, warp_size);
        return 1;
    } else {
        constexpr int qk  = ggml_cuda_type_traits<type>::qk;
        constexpr int qi  = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr = get_vdr_mmvq(type);
        const int bpi   = vdr * warp_size / qi;
        const int iters = (ncols_x / qk + bpi - 1) / bpi;
        if (env <= 1 || iters <= 1) {
            return 1;
        }
        return env >= 4 && iters > 2 ? 4 : 2;
    }
}

// one INFO line per process at the first K-unrolled MoE launch (GGML_CUDA_MOE_UNROLL_LOG=1: every launch, for test-moe-bw-exact; not
// repeated by CUDA graph replays): the engagement check of runs and kill-switch arms
static void mmvq_moe_unroll_log(const char * kernel, const ggml_type type, const bool fused, const int unroll, const int ncols_dst) {
    static const bool every = [] { const char * e = getenv("GGML_CUDA_MOE_UNROLL_LOG"); return e != nullptr && atoi(e) != 0; }();
    if (every) {
        GGML_LOG_INFO("k19-moebw: %s %s%s unroll %d, %d tokens\n", kernel, ggml_type_name(type), fused ? " fused" : "", unroll, ncols_dst);
        return;
    }
    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("k19-moebw: multi-token MoE mat-vec with K unroll engaged (first: %s %s%s, unroll %d, %d tokens)\n", kernel,
            ggml_type_name(type), fused ? " with fused gate" : "", unroll, ncols_dst);
    });
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

    // k19-moebw, GGML_CUDA_MOE_RPB (unset = auto, 2 = off, 4 / 8 = always): the rows a warp takes; 4 = fewer and longer blocks (a row's
    // sum is its own: the same lanes, K order and reduction). Auto: 4 for a product without a fused gate whose K is 1..2 iterations of a
    // lane at 2..4 tokens (the blocks of 2..4 warps of the down projections are too short there: G4 Flash-Next IQ4_NL down 0.71 / 0.86 /
    // 0.94x the time at 2 / 3 / 4 tokens, Qwen3.6 Q5_K down 0.74 / 0.87 / 0.96x, Q6_K 0.85 / 0.94 / 1.00x; Q8_0 / Q5_K with 3 iterations
    // and every fused gate / up: no gain or slower), else 2. Rows a multiple of 4, no shared expert
    static const int rpb_env = [] {
        const char * e = getenv("GGML_CUDA_MOE_RPB");
        return e == nullptr ? 0 : atoi(e);
    }();
    int rpb_moe = rpb_env;
    if constexpr (mmvq_moe_unroll_type(type)) {
        if (rpb_moe == 0) {
            constexpr int qk  = ggml_cuda_type_traits<type>::qk;
            constexpr int qi  = ggml_cuda_type_traits<type>::qi;
            constexpr int vdr = get_vdr_mmvq(type);
            const int bpi   = vdr * warp_size / qi;
            const int iters = (ncols_x / qk + bpi - 1) / bpi;
            rpb_moe = !has_fusion && iters <= 2 && ncols_dst >= 2 && ncols_dst <= 4 ? 4 : 2;
            // k19-ampere: sm_80, 5..8 tokens: 8 rows per warp block
            const int cc_rpb = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (!has_fusion && iters <= 2 && ncols_dst >= 5 && ncols_dst <= 8 && mmvq_moe_ampere(cc_rpb) && nrows_x % 8 == 0 &&
                    fusion.shared_up == nullptr) {
                rpb_moe = 8;
                mmvq_moe_ampere_log("8 rows per warp block", type, (int) ncols_dst);
            }
        }
    }
    if constexpr (mmvq_moe_unroll_type(type)) {
        if ((rpb_moe == 4 || rpb_moe == 8) && nrows_x % rpb_moe == 0 && fusion.shared_up == nullptr) {
            const dim3 block_nums_r(nrows_x / rpb_moe, nchannels_dst);
            const ggml_cuda_kernel_launch_params launch_params_r = ggml_cuda_kernel_launch_params(block_nums_r, block_dims, 0, stream);
#define MMVQ_MOE_LAUNCH_R(RPB, FUS) \
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, RPB, FUS, 1>, launch_params_r, \
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
                stride_row_x, stride_col_y, stride_col_dst, \
                stride_channel_x, stride_channel_y, stride_channel_dst, \
                ncols_dst, ids_stride)
            if (rpb_moe == 4) {
                if (has_fusion) { MMVQ_MOE_LAUNCH_R(4, true); } else { MMVQ_MOE_LAUNCH_R(4, false); }
            } else {
                if (has_fusion) { MMVQ_MOE_LAUNCH_R(8, true); } else { MMVQ_MOE_LAUNCH_R(8, false); }
            }
#undef MMVQ_MOE_LAUNCH_R
            mmvq_moe_unroll_log("mul_mat_vec_q_moe rpb", type, has_fusion, rpb_moe, ncols_dst);
            return;
        }
    }

    // k19-moebw: the K iterations a lane runs per pass of its loop (mul_mat_vec_q_moe c_unroll, mmvq_moe_unroll)
    const int unroll = mmvq_moe_unroll<type>(ncols_x, warp_size);
#define MMVQ_MOE_LAUNCH(FUS, UNR) \
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, FUS, UNR>, launch_params, \
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
            stride_row_x, stride_col_y, stride_col_dst, \
            stride_channel_x, stride_channel_y, stride_channel_dst, \
            ncols_dst, ids_stride)
    if constexpr (mmvq_moe_unroll_type(type)) {
        if (unroll == 4) {
            if (has_fusion) { MMVQ_MOE_LAUNCH(true, 4); } else { MMVQ_MOE_LAUNCH(false, 4); }
            mmvq_moe_unroll_log("mul_mat_vec_q_moe", type, has_fusion, 4, ncols_dst);
            return;
        }
        if (unroll == 2) {
            if (has_fusion) { MMVQ_MOE_LAUNCH(true, 2); } else { MMVQ_MOE_LAUNCH(false, 2); }
            mmvq_moe_unroll_log("mul_mat_vec_q_moe", type, has_fusion, 2, ncols_dst);
            return;
        }
    }
    if (has_fusion) {
        MMVQ_MOE_LAUNCH(true, 1);
    } else {
        MMVQ_MOE_LAUNCH(false, 1);
    }
#undef MMVQ_MOE_LAUNCH
}

// k19-vrow: GGML_CUDA_MMVQ_MOE_GROUP (default 1; 0 = mul_mat_vec_q_moe everywhere): a bit mask of the multi-token IQ3_S / IQ4_NL
// MUL_MAT_IDs that run mul_mat_vec_q_moe_group instead, bitwise the same values: 1 = with a fused gate (gate / up + GLU), 2 = the
// others (down). Flash-Next shapes on G4, random weights, real-like routing (1.85x top-k distinct experts at 3 tokens), us per layer
// of gate / up + down (scripts/k19vr, plans/k19_vrow.md): 3 tokens 49.6 -> 47.0, 5 tokens 69.9 -> 66.2, 6 tokens 82.9 -> 76.5 with
// bit 1 and the shape below; the grouped down is slower at every shape (its base kernel is 40 registers, all lanes busy on L2 hits),
// and without shared experts across tokens the grouped gate / up is 2-4% slower.
// GGML_CUDA_MMVQ_MOE_GROUP_CFG=rpw,nwarps,dec,group (microbench builds, GGML_CUDA_MMVQ_MOE_GROUP_BENCH) picks another shape
struct mmvq_moe_group_cfg {
    int  rpw    = 1;   // rows per warp
    int  nwarps = 4;   // warps per block
    bool dec    = true;
    int  group  = 2;   // entries per pass over K (registers: 2 keeps the gate / up kernel near mul_mat_vec_q_moe's); 0 = the token count
};

static const mmvq_moe_group_cfg & mmvq_moe_group_config() {
    static const mmvq_moe_group_cfg cfg = [] {
        mmvq_moe_group_cfg c;
        const char * e = getenv("GGML_CUDA_MMVQ_MOE_GROUP_CFG");
        int rpw = 0, nw = 0, dec = 1, group = 0;
        if (e != nullptr && sscanf(e, "%d,%d,%d,%d", &rpw, &nw, &dec, &group) >= 2) {
            c.rpw = rpw; c.nwarps = nw; c.dec = dec != 0; c.group = group;
        }
        return c;
    }();
    return cfg;
}

// one INFO line per process at the first grouped launch: the engagement check for runs and kill-switch arms (absent with
// GGML_CUDA_MMVQ_MOE_GROUP=0, on HIP / MUSA, or when no product qualifies)
static void mmvq_moe_group_log(const ggml_type type, const bool fused, const int ncols_dst, const int nchannels_dst) {
    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("mmvq_moe_group: grouped MoE mat-vec engaged (first: %s%s, %d tokens x %d slots)\n", ggml_type_name(type),
            fused ? " with fused gate" : "", ncols_dst, nchannels_dst);
    });
}

template <ggml_type type, int c_rpw, int c_nwarps, int c_group, bool c_dec>
static void mul_mat_vec_q_moe_group_launch_cfg(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const int nchannels_dst, cudaStream_t stream) {
    const int64_t nblocks_rows = (nrows_x + c_rpw*c_nwarps - 1) / (c_rpw*c_nwarps);
    const dim3 block_nums(nblocks_rows, nchannels_dst, ncols_dst);
    const dim3 block_dims(32, c_nwarps);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    // k19-moebw, GGML_CUDA_MMVQ_MOE_GROUP_UNROLL (default 3; 0 or 1 = off): the fused gate / up runs up to 3 K iterations per pass
    // (IQ3_S at K 2560: all 3 of a lane's), the same order of its sums
    static const int unroll_env = [] {
        const char * e = getenv("GGML_CUDA_MMVQ_MOE_GROUP_UNROLL");
        return e == nullptr ? 1 : atoi(e);
    }();
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    const int iters = (ncols_x / qk + vdr * 32 / qi - 1) / (vdr * 32 / qi);
    if (has_fusion && unroll_env >= 2 && iters >= 2) {
        if (unroll_env == 2 || iters == 2) {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_group<type, c_rpw, c_nwarps, c_group, c_dec, true, 2>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride);
            mmvq_moe_unroll_log("mul_mat_vec_q_moe_group", type, true, 2, ncols_dst);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_group<type, c_rpw, c_nwarps, c_group, c_dec, true, 3>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride);
            mmvq_moe_unroll_log("mul_mat_vec_q_moe_group", type, true, 3, ncols_dst);
        }
        return;
    }
    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe_group<type, c_rpw, c_nwarps, c_group, c_dec, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe_group<type, c_rpw, c_nwarps, c_group, c_dec, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride);
    }
}

// false: not taken (switch off, a shared expert, HIP / MUSA, more than 128 entries, rows not a multiple of the shape's rows per warp)
template <ggml_type type>
static bool mul_mat_vec_q_moe_group_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const int warp_size, const int nchannels_dst, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
    return false;
#else
    // bit 1: products with a fused gate (gate / up + GLU), bit 2: the others (down)
    static const int on = [] {
        const char * e = getenv("GGML_CUDA_MMVQ_MOE_GROUP");
        return e == nullptr ? 1 : atoi(e);
    }();
    if constexpr (!(type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL)) {
        GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return false;
    } else {
        const mmvq_moe_group_cfg & cfg = mmvq_moe_group_config();
        const bool fused = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                           fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
        if (!(on & (fused ? 1 : 2)) || warp_size != 32 || fusion.shared_up != nullptr || (int64_t) nchannels_dst*ncols_dst > 128 || nrows_x % cfg.rpw != 0) {
            return false;
        }
        const int group = cfg.group > 0 ? cfg.group : (int) ncols_dst;
#define MMVQ_MOE_GROUP_CFG(R, W, G, D) \
        if (cfg.rpw == (R) && cfg.nwarps == (W) && group == (G) && cfg.dec == (D)) { \
            mul_mat_vec_q_moe_group_launch_cfg<type, R, W, G, D>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, \
                stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride, \
                nchannels_dst, stream); \
            mmvq_moe_group_log(type, fused, ncols_dst, nchannels_dst); \
            return true; \
        }
        MMVQ_MOE_GROUP_CFG(1, 4, 2, true)
#ifdef GGML_CUDA_MMVQ_MOE_GROUP_BENCH
        MMVQ_MOE_GROUP_CFG(2, 4, 2, true)
        MMVQ_MOE_GROUP_CFG(2, 4, 3, true)
        MMVQ_MOE_GROUP_CFG(2, 4, 4, true)
        MMVQ_MOE_GROUP_CFG(2, 4, 5, true)
        MMVQ_MOE_GROUP_CFG(2, 4, 6, true)
        MMVQ_MOE_GROUP_CFG(1, 4, 3, true)
        MMVQ_MOE_GROUP_CFG(1, 4, 4, true)
        MMVQ_MOE_GROUP_CFG(1, 4, 5, true)
        MMVQ_MOE_GROUP_CFG(1, 4, 6, true)
        MMVQ_MOE_GROUP_CFG(1, 2, 2, true)
        MMVQ_MOE_GROUP_CFG(1, 2, 3, true)
        MMVQ_MOE_GROUP_CFG(1, 2, 4, true)
        MMVQ_MOE_GROUP_CFG(1, 2, 5, true)
        MMVQ_MOE_GROUP_CFG(1, 2, 6, true)
        MMVQ_MOE_GROUP_CFG(2, 4, 3, false)
#endif // GGML_CUDA_MMVQ_MOE_GROUP_BENCH
#undef MMVQ_MOE_GROUP_CFG
        return false;   // a token count / shape without an instance: mul_mat_vec_q_moe
    }
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// k19-moebw, GGML_CUDA_MOE_BW (unset = auto, 0 = off, 1 = on wherever it can run): multi-token MUL_MAT_IDs (2..8 tokens) of IQ3_S /
// IQ4_NL / IQ4_XS / Q4_K / Q5_K / Q6_K / Q8_0 experts run mul_mat_vec_q_moe_bw, bitwise the same values; NVIDIA Turing and newer (not
// GB10), no shared expert in the launch, at most 128 (token, slot) entries. Auto: Ampere and newer, the IQ3_S gate / up with a fused
// gate at 6..8 tokens, where the grouped kernel's per-pass decode costs more than the copy (G4 Flash-Next: 0.976 / 0.950 / 0.933x the
// time of mul_mat_vec_q_moe_group at 6 / 7 / 8 tokens, slower below 5 and for every other type there; mul_mat_vec_q_moe_vl takes
// that product first unless GGML_CUDA_MOE_VL=0). Not on Turing: on a T4 the
// kernel is 0.6-0.85x the time of the ones it replaces at 3..8 tokens, but Qwen3.6 UD-Q4_K_XL at 12 GB decodes 1.8% slower with it.
// GGML_CUDA_MOE_BW_CFG=rpw,nwarps,bytes picks the shape (0 = auto: rows per warp up to `bytes` (4096) of weights, 4 warps)
static bool mmvq_moe_bw_on(const int cc, const ggml_type type, const bool fused, const int ntok, bool * ampere_rule = nullptr) {
    static const int env = [] {
        const char * e = getenv("GGML_CUDA_MOE_BW");
        return e == nullptr ? -1 : atoi(e);
    }();
    if (env >= 0) {
        return env != 0;
    }
    // k19-ampere: sm_80 also the IQ3_S gate or up without a fused gate at 5..8 tokens and the fused gate / up at 2
    if (mmvq_moe_ampere(cc) && type == GGML_TYPE_IQ3_S && ((!fused && ntok >= 5) || (fused && ntok == 2))) {
        if (ampere_rule) {
            *ampere_rule = true;
        }
        return true;
    }
    return cc >= GGML_CUDA_CC_AMPERE && type == GGML_TYPE_IQ3_S && fused && ntok >= 6;
}

static constexpr bool mmvq_moe_bw_type(const ggml_type type) {
    return mmvq_moe_unroll_type(type);
}

static void mmvq_moe_bw_log(const ggml_type type, const bool fused, const int ntok, const int nslots, const int rpw, const int nwarps) {
    static const bool every = [] { const char * e = getenv("GGML_CUDA_MOE_UNROLL_LOG"); return e != nullptr && atoi(e) != 0; }();
    if (every) {
        GGML_LOG_INFO("k19-moebw: mul_mat_vec_q_moe_bw %s%s %d x %d rpw %d nwarps %d\n", ggml_type_name(type), fused ? " fused" : "",
            ntok, nslots, rpw, nwarps);
        return;
    }
    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("k19-moebw: multi-token MoE mat-vec with staged weight rows engaged (first: %s%s, %d tokens x %d slots, "
            "%d rows per warp, %d warps)\n", ggml_type_name(type), fused ? " with fused gate" : "", ntok, nslots, rpw, nwarps);
    });
}

// false: not taken
template <ggml_type type>
static bool mul_mat_vec_q_moe_bw_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ntok, const uint32_t ids_stride, const int nchannels_dst, const int cc, const int warp_size, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst, ntok, ids_stride, nchannels_dst, cc, warp_size, stream);
    return false;
#else
    if constexpr (!mmvq_moe_bw_type(type)) {
        GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst, ntok, ids_stride, nchannels_dst, cc, warp_size, stream);
        return false;
    } else {
        constexpr int qk = ggml_cuda_type_traits<type>::qk;
        constexpr int bs = ggml_cuda_type_traits<type>::bs;
        bool ampere_rule = false;
        if (ntok < 2 || !mmvq_moe_bw_on(cc, type, fusion.gate != nullptr, ntok, &ampere_rule) || warp_size != 32 || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_TURING ||
                get_device_table_id(cc) == MMVQ_PARAMETERS_GB10 ||
                fusion.shared_up != nullptr || fusion.x_scale != nullptr || fusion.gate_scale != nullptr ||
                (int64_t) stride_row_x*qk != (int64_t) ncols_x || (int64_t) nchannels_dst*ntok > 128) {
            return false;
        }
        static const int * c = [] {
            static int v[3] = { 0, 0, 4096 };
            if (const char * e = getenv("GGML_CUDA_MOE_BW_CFG")) {
                int r = 0, w = 0, b = 4096;
                if (sscanf(e, "%d,%d,%d", &r, &w, &b) >= 1) {
                    v[0] = r; v[1] = w; v[2] = b > 0 ? b : 4096;
                }
            }
            return v;
        }();
        const bool fused = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
        const int  nmat  = fusion.gate != nullptr ? 2 : 1;
        const int64_t row_bytes = (int64_t) stride_row_x*bs;
        int rpw = c[0];
        if (rpw <= 0) {
            rpw = 2;
            while (rpw < 8 && 2*rpw*row_bytes*nmat <= c[2] && nrows_x % (2*rpw) == 0) {
                rpw *= 2;
            }
        }
        if (rpw % 2 != 0 || nrows_x % rpw != 0) {
            return false;
        }
        int nwarps = c[1] > 0 ? std::min(c[1], 8) : 4;
        const uint32_t smem_mat = (uint32_t) GGML_PAD(rpw*row_bytes + 32, 16);
        while (nwarps > 1 && (size_t) nwarps*nmat*smem_mat > 48*1024) {
            nwarps /= 2;
        }
        if ((size_t) nwarps*nmat*smem_mat > 48*1024) {
            return false;
        }
        const int64_t nblocks = (nrows_x + rpw*nwarps - 1) / (rpw*nwarps);
        const dim3 block_nums(nblocks, nchannels_dst, ntok);
        const dim3 block_dims(32, nwarps);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, (size_t) nwarps*nmat*smem_mat, stream);
        if (fused) {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_bw<type, true>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, ntok, ids_stride, rpw, smem_mat);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_bw<type, false>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, ntok, ids_stride, rpw, smem_mat);
        }
        mmvq_moe_bw_log(type, fusion.gate != nullptr, ntok, nchannels_dst, rpw, nwarps);
        if (ampere_rule) {
            mmvq_moe_ampere_log(fusion.gate != nullptr ? "staged weights, fused gate" : "staged weights", type, (int) ntok);
        }
        return true;
    }
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// k19-moegrp, GGML_CUDA_MOE_VL (unset = auto, 0 = off, "tpr,g" = c_tpr threads per row and c_g entries per decode wherever it can
// run): multi-token MUL_MAT_IDs of IQ3_S experts with a fused gate (the Flash-Next gate / up) run mul_mat_vec_q_moe_vl, bitwise the
// same values; NVIDIA Turing and newer (not GB10), no shared expert, at most 128 entries. Auto: Ampere and newer at 3..8 tokens,
// 16 threads per row, 2 entries per decode up to 5 tokens and 4 above (G4 Flash-Next, the time of the grouped / staged kernels it
// replaces: 0.92 / 0.90 / 0.88 / 0.90 / 0.90 / 0.91 at 3..8 tokens). The IQ4_NL / Q8_0 downs and the IQ4_XS gate / up measured slower
// with it at every width (1.04-1.25x), so it has no instances for them
struct mmvq_moe_vl_cfg {
    int tpr = 0;
    int g   = 0;
};

static mmvq_moe_vl_cfg mmvq_moe_vl_config(const ggml_type type, const bool fused, const int ntok, const int cc) {
    static const mmvq_moe_vl_cfg env = [] {
        mmvq_moe_vl_cfg c;
        c.tpr = -1;
        if (const char * e = getenv("GGML_CUDA_MOE_VL")) {
            int tpr = 0, g = 4;
            const int n = sscanf(e, "%d,%d", &tpr, &g);
            c.tpr = n >= 1 ? tpr : 0;
            c.g   = g;
        }
        return c;
    }();
    if (env.tpr >= 0) {
        return env;
    }
    mmvq_moe_vl_cfg c;
    if (cc >= GGML_CUDA_CC_AMPERE && type == GGML_TYPE_IQ3_S && fused && ntok >= 3) {
        c.tpr = 16;
        c.g   = ntok <= 5 ? 2 : 4;
        // k19-ampere: sm_80, 4 entries per decode from 3 tokens
        if (ntok <= 5 && mmvq_moe_ampere(cc)) {
            c.g = 4;
            mmvq_moe_ampere_log("4 entries per decode", type, ntok);
        }
    }
    return c;
}

static void mmvq_moe_vl_log(const ggml_type type, const bool fused, const int ntok, const int nslots, const int tpr, const int g) {
    static const bool every = [] { const char * e = getenv("GGML_CUDA_MOE_UNROLL_LOG"); return e != nullptr && atoi(e) != 0; }();
    if (every) {
        GGML_LOG_INFO("k19-moebw: mul_mat_vec_q_moe_vl %s%s %d x %d vl%d,%d\n", ggml_type_name(type), fused ? " fused" : "", ntok, nslots,
            tpr, g);
        return;
    }
    static std::once_flag once;
    std::call_once(once, [&] {
        GGML_LOG_INFO("k19-moegrp: multi-token MoE mat-vec with %d threads per row, %d entries per decode engaged (first: %s%s, "
            "%d tokens x %d slots)\n", tpr, g, ggml_type_name(type), fused ? " with fused gate" : "", ntok, nslots);
    });
}

// false: not taken
template <ggml_type type>
static bool mul_mat_vec_q_moe_vl_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ntok, const uint32_t ids_stride, const int nchannels_dst, const int cc, const int warp_size, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst, ntok, ids_stride, nchannels_dst, cc, warp_size, stream);
    return false;
#else
    constexpr bool c_fused = true;   // the instances: the IQ3_S gate / up with a fused gate
    if constexpr (type != GGML_TYPE_IQ3_S) {
        GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst, ntok, ids_stride, nchannels_dst, cc, warp_size, stream);
        return false;
    } else {
        const bool fused = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
        if (ntok < 2 || fused != c_fused || fusion.gate == nullptr || warp_size != 32 || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_TURING ||
                get_device_table_id(cc) == MMVQ_PARAMETERS_GB10 ||
                fusion.shared_up != nullptr || fusion.x_scale != nullptr || fusion.gate_scale != nullptr ||
                (int64_t) nchannels_dst*ntok > 128) {
            return false;
        }
        const mmvq_moe_vl_cfg cfg = mmvq_moe_vl_config(type, fused, ntok, cc);
        if (cfg.tpr <= 0) {
            return false;
        }
        constexpr int nwarps = 4;
        const int rows_per_block = 32/cfg.tpr*nwarps;
        if (cfg.tpr > 32 || 32 % cfg.tpr != 0 || nrows_x % (32/cfg.tpr) != 0) {
            return false;
        }
        const dim3 block_nums((nrows_x + rows_per_block - 1) / rows_per_block, nchannels_dst, ntok);
        const dim3 block_dims(32, nwarps);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        bool done = false;
#define MMVQ_MOE_VL(TPR, G) \
        if (!done && cfg.tpr == (TPR) && cfg.g == (G)) { \
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_vl<type, TPR, G, nwarps, c_fused>, launch_params, \
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst, \
                stride_channel_x, stride_channel_y, stride_channel_dst, ntok, ids_stride); \
            done = true; \
        }
        MMVQ_MOE_VL(16, 2)
        MMVQ_MOE_VL(16, 4)
        MMVQ_MOE_VL(8, 4)
#undef MMVQ_MOE_VL
        if (done) {
            mmvq_moe_vl_log(type, fusion.gate != nullptr, ntok, nchannels_dst, cfg.tpr, cfg.g);
        }
        return done;
    }
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
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

// k19-mmvqcols: more rows per block for a dense 2..8 column launch (speculative verify windows) of the types whose lanes each read one
// q8_1 block per 32 weights (IQ3_S, IQ2_XXS, Q3_K): a warp's 32-bit activation load then spans 9 cache lines, the per-column cost of these
// launches is L1 wavefronts, and the rows of a block share those loads. Rows alone keep every row's sum (the K split and the reduction are
// per row): bitwise the launch they replace, also on the exact (batch invariant) verify path. Rows per arch from the benches
// (plans/k19_mmvqcols.md): Turing 8 at 2..8 columns (Kaggle T4); Ampere cc 8.0 4 at 2..8 (Colab A100: 8 lost up to 35%, register
// pressure); Ada cc 8.9 4 from IQ2_XXS 4, Q3_K 5, IQ3_S 7 columns (Colab L4: below that within +-5%); other archs unchanged until measured.
// GGML_CUDA_MMVQ_COLS_CFG: unset / AUTO = that table, 0 = off, R4 / R8 = those rows for every listed type, width and NVIDIA table.
// Returns the rows (0: not taken)
// k19-q38next: the same lever for the K-quant / IQ4_XS products of a dense model's verify windows (Qwen3.8-27B UD-Q4_K_XL: Q5_K / IQ4_XS
// / Q4_K / Q6_K, 85% of an L4 MTP pass at 5 rows, each 1.5-1.9x its 1-row launch). Their lanes read 16 bytes of a q8_1 block per load, 8
// such groups per warp load (~5 cache lines), 6 loads per 16 weights per column: the activation loads of a 5-column launch are ~3x its
// weight loads in L1 wavefronts, and the rows of a block share them. Rows only: bitwise the launch it replaces (plain, verify-wide,
// multi-matrix and exact launches). GENERIC table only (Turing sends these widths to MMQ). GGML_CUDA_MMVQ_KQ_ROWS: unset / AUTO = the
// per-arch table (mmvq_kq_rows_auto), 0 = off, R4 = 4 rows at every 2..8 column width (8 rows lost to register pressure on Ampere / Ada
// for the IQ types, k19-mmvqcols, and are not compiled for these types); GGML_CUDA_MMVQ_KQ_COLS_MIN (default 2)
static constexpr bool mmvq_kq_type(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS;
}
// AUTO: Ada (cc 8.9) 4 rows at 2..8 columns for all four types: Colab L4, Qwen3.8-27B UD-Q4_K_XL + MTP drafter (5-row verify windows),
// 3 interleaved launches, chat 34.3 -> 39.5 tok/s, 8k 30.4 -> 35.1 (+15%), text unchanged (default and exact verify). Ampere cc 8.0
// stays off: on a Colab A100 the 4-row launch of Q4_K / Q5_K / Q6_K at 5 columns ran 1-16% slower (test-mmvqcols-exact --bench, register
// pressure; plans/k19_q38next.md). Other archs off until measured
static int mmvq_kq_rows_auto(const ggml_type type, const int ncols_dst, const int cc) {
    GGML_UNUSED(type); GGML_UNUSED(ncols_dst);
    return cc == GGML_CUDA_CC_ADA_LOVELACE ? 4 : 0;
}
static int mmvq_kq_rows(const ggml_type type, const int ncols_dst, const mmvq_parameter_table_id table_id) {
    static const int cfg = [] {
        const char * e = getenv("GGML_CUDA_MMVQ_KQ_ROWS");
        if (e == nullptr) {
            return -1;
        }
        const std::string v = e;
        return v == "R4" || v == "4" ? 4 : v == "AUTO" ? -1 : 0;
    }();
    static const int cols_min = [] {
        const char * e = getenv("GGML_CUDA_MMVQ_KQ_COLS_MIN");
        return e != nullptr ? std::max(2, atoi(e)) : 2;
    }();
    if (!mmvq_kq_type(type) || ncols_dst < cols_min || ncols_dst > 8 || table_id != MMVQ_PARAMETERS_GENERIC) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return 0;
    }
    return cfg >= 0 ? cfg : mmvq_kq_rows_auto(type, ncols_dst, cc);
}
static constexpr bool mmvq_cols_type(const ggml_type type) {
    return type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_Q3_K || mmvq_kq_type(type);
}
static int mmvq_cols_rows(const ggml_type type, const int ncols_dst, const mmvq_parameter_table_id table_id) {
    if (mmvq_kq_type(type)) {
        return mmvq_kq_rows(type, ncols_dst, table_id);
    }
    static const int cfg = [] {
        const char * e = getenv("GGML_CUDA_MMVQ_COLS_CFG");
        if (e == nullptr) {
            return -1;
        }
        const std::string v = e;
        return v == "R4" ? 4 : v == "R8" ? 8 : v == "AUTO" ? -1 : 0;
    }();
    if (!mmvq_cols_type(type) || ncols_dst < 2 || ncols_dst > 8) {
        return 0;
    }
    if (table_id != MMVQ_PARAMETERS_GENERIC && table_id != MMVQ_PARAMETERS_TURING) {
        return 0;
    }
    if (cfg >= 0) {
        return cfg;
    }
    const int  cc  = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool nv  = GGML_CUDA_CC_IS_NVIDIA(cc);
    const int  ada = type == GGML_TYPE_IQ2_XXS ? 4 : type == GGML_TYPE_Q3_K ? 5 : 7;
    return table_id == MMVQ_PARAMETERS_TURING ? 8 :
           nv && cc == GGML_CUDA_CC_AMPERE ? 4 :
           nv && cc == GGML_CUDA_CC_ADA_LOVELACE && ncols_dst >= ada ? 4 : 0;
}
// GGML_CUDA_MMVQ_COLS_LOG=1: one line per launch the rows lever takes (tests/test-mmvqcols-exact counts them)
static void mmvq_cols_log(const ggml_type type, const int ncols_dst, const int rpb, const int nwarps, const char * where) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_MMVQ_COLS_LOG"); return e != nullptr && atoi(e) != 0; }();
    if (on) {
        GGML_LOG_INFO("k19-mmvqcols: %s ncols %d rpb %d nwarps %d %s\n", ggml_type_name(type), ncols_dst, rpb, nwarps, where);
    }
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
    // k19-mmvqcols: the rows lever for the exact / verify-wide launches (mmvq_cols_rows)
    if constexpr (mmvq_cols_type(type) && c_ncols_dst > 1) {
        const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
        const int r = mmvq_cols_rows(type, c_ncols_dst, table_id);
        if ((r == 4 || r == 8) && nrows_x % r == 0) {
            const dim3 bn(nrows_x / r, nchannels_dst, nsamples_dst);
            const dim3 bd(warp_size, c_nwarps, 1);
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(bn, bd, 0, stream);
            const ggml_cuda_mm_fusion_args_device nf{};
#define MMVQ_COLS_NW_LAUNCH(R) ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, false, false, c_nwarps, 0, R, 1>, lp, \
                vx, vy, (const int32_t *) nullptr, nf, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst, \
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, \
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, 0u)
            if constexpr (mmvq_kq_type(type)) {
                MMVQ_COLS_NW_LAUNCH(4);
            } else if (r == 4) {
                MMVQ_COLS_NW_LAUNCH(4);
            } else {
                MMVQ_COLS_NW_LAUNCH(8);
            }
#undef MMVQ_COLS_NW_LAUNCH
            mmvq_cols_log(type, c_ncols_dst, r, c_nwarps, "nwarps");
            return;
        }
    }
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

// k19-vrow3: whether the default launch of ncols_dst columns already gives a row the sums of the 4-warp launch of 1..4 columns: when its
// warps cover the K blocks of a row in one pass, lane t of both takes K block t / (qi / vdr) and nothing else, and the 4-warp reduction
// only adds the +0.0 of its idle warps (the hyper-connection up projection, K = 320: 10 Q8_0 blocks, 16 per pass with 2 warps). The
// default launch then stays (it is faster there); the values are the same up to the sign of a row sum that is exactly zero
template <ggml_type type>
static bool mul_mat_vec_q_wide_same_as_default(const int ncols_x, const int ncols_dst, const mmvq_parameter_table_id table_id, const int warp_size) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    return ncols_x / qk <= vdr * calc_nwarps(type, ncols_dst, table_id) * warp_size / qi;
}

// k19-vrow2 / k19-vrow3: a dense 5..8 column launch of a multi-matrix type (no fusion, no ids) as the 4-column launch makes it, only wider: 4 warps,
// the rows per block and K unroll mul_mat_vec_q_batch_rpb / mul_mat_vec_q_unroll pick for 4 columns. A row's sum depends on the warps
// and the unroll (the K split and order), not on the rows per block. false: not this type / table (the caller falls back)
template <ggml_type type>
static bool mul_mat_vec_q_verify_wide(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {
    if constexpr (!mul_mat_vec_q_multi_type(type)) {
        GGML_UNUSED_VARS(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x,
            nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x,
            stride_sample_y, stride_sample_dst, stream);
        return false;
    } else {
        const int device    = ggml_cuda_get_device();
        const int cc        = ggml_cuda_info().devices[device].cc;
        const int warp_size = ggml_cuda_info().devices[device].warp_size;
        const mmvq_parameter_table_id table_id = get_device_table_id(cc);
        const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.shared_up != nullptr;
        if (has_fusion || table_id != MMVQ_PARAMETERS_GENERIC || calc_nwarps(type, 4, table_id) != 4 || ncols_dst < 5 || ncols_dst > 8) {
            return false;
        }
              int  rpb    = mul_mat_vec_q_batch_rpb<type>(ncols_x, nrows_x, 4, cc, warp_size, table_id);
        const bool unroll = mul_mat_vec_q_unroll<type>(ncols_x, (int64_t) nrows_x*nchannels_dst*nsamples_dst, 4, table_id, warp_size);
        // k19-q38next: the K-quant rows lever (mmvq_kq_rows) where the 4-column rule keeps the default rows and the K loop rolled
        const int  kqr    = rpb == 0 && !unroll ? mmvq_kq_rows(type, ncols_dst, table_id) : 0;
        if (kqr == 4 && nrows_x % kqr == 0) {
            rpb = kqr;
            mmvq_cols_log(type, ncols_dst, kqr, 4, "verify-wide");
        }
        const int  rpb_l  = rpb > 0 ? rpb : 2; // calc_rows_per_block for 2..8 columns
        const uint3 nchannels_y_fd   = make_uint3(0, 0, 0);
        const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
        const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);
        const dim3 block_nums((nrows_x + rpb_l - 1) / rpb_l, nchannels_dst, nsamples_dst);
        const dim3 block_dims(warp_size, 4, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
#define MMVQ_VW_LAUNCH(NC, RPB, UNR) ggml_cuda_kernel_launch(mul_mat_vec_q<type, NC, false, false, false, 4, 0, RPB, UNR>, launch_params, \
            vx, vy, (const int32_t *) nullptr, fusion, dst, (uint32_t) ncols_x, nchannels_y_fd, (uint32_t) stride_row_x, \
            (uint32_t) stride_col_y, (uint32_t) stride_col_dst, channel_ratio_fd, (uint32_t) stride_channel_x, (uint32_t) stride_channel_y, \
            (uint32_t) stride_channel_dst, sample_ratio_fd, (uint32_t) stride_sample_x, (uint32_t) stride_sample_y, \
            (uint32_t) stride_sample_dst, 0u)
#define MMVQ_VW_NC(NC) \
        if (rpb == 4) { MMVQ_VW_LAUNCH(NC, 4, 1); } else if (rpb == 1) { if (unroll) { MMVQ_VW_LAUNCH(NC, 1, 8); } else { MMVQ_VW_LAUNCH(NC, 1, 1); } } \
        else if (unroll) { MMVQ_VW_LAUNCH(NC, 0, 8); } else { MMVQ_VW_LAUNCH(NC, 0, 1); }
        switch (ncols_dst) {
            case 5: { MMVQ_VW_NC(5) } break;
            case 6: { MMVQ_VW_NC(6) } break;
            case 7: { MMVQ_VW_NC(7) } break;
            default: { MMVQ_VW_NC(8) } break;
        }
#undef MMVQ_VW_NC
#undef MMVQ_VW_LAUNCH
        return true;
    }
}

// k19-dsv4p: the single column launch of an mmvq_tab_type with the codebook in shared memory (mul_mat_vec_q c_tab). The warps and the
// K split are those of the launch it replaces, so each row's sum keeps its bytes; the rows per block (which do not change a row's sum)
// are the small-K rows (= warps) for a small K, GGML_CUDA_MMVQ_TURING_IDS_ROWS (default 2; 1 / 4) for a MUL_MAT_ID launch, which also
// shares the shared-memory load between more rows, and k16-turing's 2 rows (turing_2rows) or the table's rows for a dense launch.
// GGML_CUDA_MMVQ_IQ_TAB: 1 (default) = sm_75 binaries on the TURING table (as k16-turing), 0 = off (the launches as before),
// 2 = every NVIDIA table (a byte check elsewhere, not a tuned setting). No shared-expert fusion, unrolled or wider (halve_iters) launch.
// k19-iqsmem:
//  - IQ2_XS / IQ2_S take the same launch under GGML_CUDA_MMVQ_IQ2_TAB (same values 0 / 1 / 2, default 1); a dense launch takes
//    GGML_CUDA_MMVQ_IQ2_TAB_ROWS rows per block (2 = k16-turing's, default; 4 / 8: the 4 / 8 KiB table load shared by more rows).
//  - IQ3_S / IQ3_XXS single token MUL_MAT_ID (MoE experts): the grid kernel (k19-q38t, grid in shared memory unless
//    GGML_CUDA_MMVQ_GRID_SMEM=0) with GGML_CUDA_MMVQ_GRID_IDS rows per block instead of one: unset = 4 (k19-q4xt; 2 before) on sm_75 binaries on the TURING
//    table, off elsewhere; 0 / 1 = off (one row per block, as before); 2 / 4 / 8 = that many rows on every NVIDIA table. A Qwen3.6-35B-A3B
//    expert row (K 2048) is 8 IQ3_XXS blocks, which half of a block's 4 warps cover; the 1 KiB grid copy is then larger than the row.
//    Same 4 warps and K split as the one-row launch: bitwise the same outputs.
template <ggml_type type>
static constexpr bool mmvq_tab_launch_type() {
    return mmvq_tab_type(type) || mmvq_grid_smem_n(type) > 0;
}

template <ggml_type type>
static bool mul_mat_vec_q_tab_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const uint3 nchannels_y_fd, const int nrows_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const uint3 channel_ratio_fd, const int stride_channel_x, const int stride_channel_y,
        const int stride_channel_dst, const uint3 sample_ratio_fd, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst, const int nchannels_dst, const int nsamples_dst, const int ids_stride,
        const bool small_k, const bool other_launch, const bool turing_2rows, const int cc, const mmvq_parameter_table_id table_id,
        const int warp_size, cudaStream_t stream) {
    if constexpr (!mmvq_tab_launch_type<type>()) {
        GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, stride_sample_x,
            stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, ids_stride, small_k, other_launch, turing_2rows, cc,
            table_id, warp_size, stream);
        return false;
    } else {
        static const int tab_env = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_IQ_TAB");
            return e != nullptr ? atoi(e) : 1;
        }();
        static const int tab2_env = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_IQ2_TAB");
            return e != nullptr ? atoi(e) : 1;
        }();
        static const int tab2_rows_env = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_IQ2_TAB_ROWS");
            const int v = e != nullptr ? atoi(e) : 2;
            return v == 4 || v == 8 ? v : 2;
        }();
        // -1 = unset (sm_75 default), 0 = off, else the rows per block
        static const int grid_ids_env = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_GRID_IDS");
            if (e == nullptr) {
                return -1;
            }
            const int v = atoi(e);
            return v == 2 || v == 4 || v == 8 ? v : 0;
        }();
        static const int ids_rows_env = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_TURING_IDS_ROWS");
            const int v = e != nullptr ? atoi(e) : 2;
            return v == 1 || v == 4 ? v : 2;
        }();
        if (other_launch || fusion.shared_up != nullptr || !GGML_CUDA_CC_IS_NVIDIA(cc) || table_id == MMVQ_PARAMETERS_GB10) {
            return false;
        }
        const bool sm75 = table_id == MMVQ_PARAMETERS_TURING && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING;
        constexpr bool grid_type = !mmvq_tab_type(type);
        int rpb_force = 0;
        if constexpr (grid_type) {
            if (ids == nullptr || small_k || grid_ids_env == 0 || (grid_ids_env < 0 && !sm75)) {
                return false;
            }
            // k19-q4xt: the sm_75 default 4 rows (was 2): q36x on a Kaggle T4, 3 launches, md5 equal, full card chat / 8k +2.3% / +3.3%
            // (A/A +0.3% / +0.4%), 12 GB +0.5% / +1.0%, 8 GB +0.3% / +0.4% (inside the A/A there); 8 rows below 4 in 5 of 6 cells
            rpb_force = grid_ids_env > 0 ? grid_ids_env : 4;
        } else {
            const int env = mmvq_tab_iq2_type(type) ? tab2_env : tab_env;
            if (env == 0 || (env == 1 && !sm75)) {
                return false;
            }
        }
        const int nwarps = calc_nwarps(type, 1, table_id);
        if (nwarps != 4) {   // the instances below are 4-warp launches (TURING / GENERIC tables at one column)
            return false;
        }
        int rpb = small_k ? nwarps : calc_rows_per_block(type, 1, table_id);
        if (rpb_force > 0) {
            rpb = rpb_force;
        } else if (!small_k) {
            if (ids != nullptr) {
                rpb = std::max(rpb, ids_rows_env);
            } else if (turing_2rows) {
                rpb = std::max(rpb, mmvq_tab_iq2_type(type) ? tab2_rows_env : 2);
            }
        }
        if (rpb > 1 && nrows_x % rpb != 0) {   // a block reads the weights of all its rows
            return false;
        }
        // k19-iqsmem (GGML_CUDA_MMVQ_GRID_IDS_WARPS, default 1; 0 = always 4): an IQ3 MUL_MAT_ID row whose K blocks 2 warps cover in one
        // pass (K <= 2048 for IQ3_XXS / IQ3_S: Qwen3.6-35B-A3B's experts) runs with 2 warps. Lane t of both launches then takes K block
        // t / (qi / vdr) and nothing else, and the 4-warp launch only adds the +0.0 of its idle warps (every partial sum starts at +0.0,
        // so none is -0.0 and x + 0.0 == x): bitwise the same row sums from half the threads
        static const bool grid_warps_env = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_GRID_IDS_WARPS");
            return e == nullptr || atoi(e) != 0;
        }();
        int nwarps_l = nwarps;
        if constexpr (grid_type) {
            constexpr int qk  = ggml_cuda_type_traits<type>::qk;
            constexpr int qi  = ggml_cuda_type_traits<type>::qi;
            constexpr int vdr = get_vdr_mmvq(type);
            if (grid_warps_env && warp_size == 32 && ncols_x / qk <= 2 * (vdr * warp_size / qi)) {
                nwarps_l = 2;
            }
        }
        const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
        const dim3 block_nums(nrows_x / rpb, nchannels_dst, nsamples_dst);
        const dim3 block_dims(warp_size, nwarps_l, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        // c_tab only for the codebook types: the grid types' kernel (grid_n) is the same with either value
#define MMVQ_TAB_LAUNCH_W(FUS, RPB, NW) ggml_cuda_kernel_launch(mul_mat_vec_q<type, 1, FUS, false, false, NW, 0, RPB, 1, false, !grid_type>, launch_params, \
            vx, vy, ids, fusion, dst, (uint32_t) ncols_x, nchannels_y_fd, (uint32_t) stride_row_x, (uint32_t) stride_col_y, (uint32_t) stride_col_dst, \
            channel_ratio_fd, (uint32_t) stride_channel_x, (uint32_t) stride_channel_y, (uint32_t) stride_channel_dst, \
            sample_ratio_fd, (uint32_t) stride_sample_x, (uint32_t) stride_sample_y, (uint32_t) stride_sample_dst, (uint32_t) ids_stride)
#define MMVQ_TAB_LAUNCH(FUS, RPB) MMVQ_TAB_LAUNCH_W(FUS, RPB, 4)
        constexpr bool rpb8_type = grid_type || mmvq_tab_iq2_type(type);
        bool launched = false;
        if constexpr (grid_type) {
            if (nwarps_l == 2) {
                switch (rpb) {
                    case 2: if (has_fusion) { MMVQ_TAB_LAUNCH_W(true, 2, 2); } else { MMVQ_TAB_LAUNCH_W(false, 2, 2); } break;
                    case 4: if (has_fusion) { MMVQ_TAB_LAUNCH_W(true, 4, 2); } else { MMVQ_TAB_LAUNCH_W(false, 4, 2); } break;
                    case 8: if (has_fusion) { MMVQ_TAB_LAUNCH_W(true, 8, 2); } else { MMVQ_TAB_LAUNCH_W(false, 8, 2); } break;
                    default: return false;
                }
                launched = true;
            }
        }
        if (!launched) {
        switch (rpb) {
            case 1: if (has_fusion) { MMVQ_TAB_LAUNCH(true, 1); } else { MMVQ_TAB_LAUNCH(false, 1); } break;
            case 2: if (has_fusion) { MMVQ_TAB_LAUNCH(true, 2); } else { MMVQ_TAB_LAUNCH(false, 2); } break;
            case 4: if (has_fusion) { MMVQ_TAB_LAUNCH(true, 4); } else { MMVQ_TAB_LAUNCH(false, 4); } break;
            case 8:
                if constexpr (rpb8_type) {
                    if (has_fusion) { MMVQ_TAB_LAUNCH(true, 8); } else { MMVQ_TAB_LAUNCH(false, 8); }
                    break;
                } else {
                    return false;
                }
            default: return false;
        }
        }
#undef MMVQ_TAB_LAUNCH
#undef MMVQ_TAB_LAUNCH_W
        static bool logged = false;
        if (!logged && getenv("GGML_CUDA_MMVQ_IQ_TAB_LOG") != nullptr) {
            logged = true;
            GGML_LOG_INFO("%s: k19-%s %s in shared memory: %s, %d rows per block, ids %d, small_k %d\n", __func__,
                grid_type ? "iqsmem" : mmvq_tab_iq2_type(type) ? "iqsmem" : "dsv4p", grid_type ? "grid (MUL_MAT_ID rows)" : "codebook",
                ggml_type_name(type), rpb, ids != nullptr, (int) small_k);
        }
        // k19-iqsmem: one INFO line per process at the first launch of a new path (engagement, independent of the log switch)
        if constexpr (grid_type || mmvq_tab_iq2_type(type)) {
            static bool engaged = false;
            if (!engaged) {
                engaged = true;
                GGML_LOG_INFO("k19-iqsmem: %s %s engaged (%d rows per block, %d warps, ids %d)\n", ggml_type_name(type),
                    grid_type ? "MUL_MAT_ID grid launch" : "codebook in shared memory", rpb, nwarps_l, ids != nullptr);
            }
        }
        return true;
    }
}

// k19-q4xt: single token MUL_MAT_ID whose expert rows are short enough that one warp covers a row's K blocks in one pass (the K-quant
// / Q4_0 / Q5_0 / IQ4 experts of a Qwen3.6-35B-A3B down projection: K 512 = 2 Q5_K blocks, 16 lanes each). The launch it replaces runs
// 2 (Turing K-quants) or 4 warps per block and only warp 0 has K blocks: the other warps add +0.0 partials (none is -0.0: every partial
// starts at +0.0) and the block's row count is the resident-block limit's. Here every warp is a block of its own (mul_mat_vec_q c_wpr):
// lane t takes the K block t / (qi/vdr) exactly as in warp 0 before, the same warp reduction, without the zeros: bitwise the same rows,
// with every resident warp computing. Rows that 2 warps cover in one pass where the launch had 4 (Q8_0 K 512: the shared expert
// down) run with 2 warps (c_nwarps 2): warp 1's partial is added to warp 0's as before, only the +0.0 of warps 2 and 3 are gone.
// Dense single column launches of these shapes too (the shared expert down is a MUL_MAT).
// GGML_CUDA_MMVQ_IDS_WPR: unset / 1 = sm_75 binaries on the TURING table, 0 = off (the launches as before), 2 = every NVIDIA table (a
// byte check, untuned); GGML_CUDA_MMVQ_IDS_WPR_W = warps per block of the one-warp rows (2 / 4, default 4),
// GGML_CUDA_MMVQ_IDS_WPR_ROWS = rows per warp (1 / 2 / 4, default 2; the 2-warp launch: rows per block).
static constexpr bool mmvq_wpr_type(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
           type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q8_0;
}

template <ggml_type type>
static bool mul_mat_vec_q_wpr_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const uint3 nchannels_y_fd, const int nrows_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const uint3 channel_ratio_fd, const int stride_channel_x, const int stride_channel_y,
        const int stride_channel_dst, const uint3 sample_ratio_fd, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst, const int nchannels_dst, const int nsamples_dst, const int ids_stride,
        const bool small_k, const bool other_launch, const int cc, const mmvq_parameter_table_id table_id,
        const int warp_size, cudaStream_t stream) {
    if constexpr (!mmvq_wpr_type(type) || mmvq_tab_launch_type<type>()) {
        GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, stride_sample_x,
            stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, ids_stride, small_k, other_launch, cc,
            table_id, warp_size, stream);
        return false;
    } else {
        static const int env = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_IDS_WPR");
            return e != nullptr ? atoi(e) : 1;
        }();
        static const int env_w = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_IDS_WPR_W");
            const int v = e != nullptr ? atoi(e) : 4;
            return v == 2 ? 2 : 4;
        }();
        static const int env_rows = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_IDS_WPR_ROWS");
            const int v = e != nullptr ? atoi(e) : 2;
            return v == 1 || v == 4 ? v : 2;
        }();
        if (env == 0 || other_launch || warp_size != 32 || fusion.shared_up != nullptr || !GGML_CUDA_CC_IS_NVIDIA(cc) ||
                (table_id != MMVQ_PARAMETERS_TURING && table_id != MMVQ_PARAMETERS_GENERIC)) {
            return false;
        }
        const bool sm75 = table_id == MMVQ_PARAMETERS_TURING && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING;
        if (env == 1 && !sm75) {
            return false;
        }
        constexpr int qk  = ggml_cuda_type_traits<type>::qk;
        constexpr int qi  = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr = get_vdr_mmvq(type);
        // the launch this replaces: its warps (small_k keeps the table's warps on these tables) and its K split
        const int nwarps_old = calc_nwarps(type, 1, table_id, small_k, false);
        const int bpi1       = vdr * warp_size / qi;   // K blocks one warp covers per pass
        const int needed     = (ncols_x / qk + bpi1 - 1) / bpi1;
        if (nwarps_old < 2 || needed > 2 || needed >= nwarps_old) {
            return false;
        }
        const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
        if (needed == 2) {
            // 2 warps, env_rows rows per block, the shared memory reduction of warp 1 into warp 0 as in the launch it replaces
            if (nrows_x % env_rows != 0) {
                return false;
            }
            const dim3 block_nums(nrows_x / env_rows, nchannels_dst, nsamples_dst);
            const dim3 block_dims(warp_size, 2, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
#define MMVQ_W2_LAUNCH(FUS, RPB) ggml_cuda_kernel_launch(mul_mat_vec_q<type, 1, FUS, false, false, 2, 0, RPB, 1, false, false, 0>, launch_params, \
            vx, vy, ids, fusion, dst, (uint32_t) ncols_x, nchannels_y_fd, (uint32_t) stride_row_x, (uint32_t) stride_col_y, (uint32_t) stride_col_dst, \
            channel_ratio_fd, (uint32_t) stride_channel_x, (uint32_t) stride_channel_y, (uint32_t) stride_channel_dst, \
            sample_ratio_fd, (uint32_t) stride_sample_x, (uint32_t) stride_sample_y, (uint32_t) stride_sample_dst, (uint32_t) ids_stride)
            switch (env_rows) {
                case 1:  if (has_fusion) { MMVQ_W2_LAUNCH(true, 1); } else { MMVQ_W2_LAUNCH(false, 1); } break;
                case 4:  if (has_fusion) { MMVQ_W2_LAUNCH(true, 4); } else { MMVQ_W2_LAUNCH(false, 4); } break;
                default: if (has_fusion) { MMVQ_W2_LAUNCH(true, 2); } else { MMVQ_W2_LAUNCH(false, 2); } break;
            }
#undef MMVQ_W2_LAUNCH
            static bool engaged2 = false;
            if (!engaged2) {
                engaged2 = true;
                GGML_LOG_INFO("k19-q4xt: %s %s 2-warp launch engaged (K %d, %d rows per block, was %d warps)\n", ggml_type_name(type),
                    ids != nullptr ? "MUL_MAT_ID" : "dense", ncols_x, env_rows, nwarps_old);
            }
            return true;
        }
        const int rows_block = env_w * env_rows;
        if (nrows_x % rows_block != 0) {
            return false;
        }
        const dim3 block_nums(nrows_x / rows_block, nchannels_dst, nsamples_dst);
        const dim3 block_dims(warp_size, env_w, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
#define MMVQ_WPR_LAUNCH(FUS, W, RPW) ggml_cuda_kernel_launch(mul_mat_vec_q<type, 1, FUS, false, false, W, 0, RPW, 1, false, false, W>, launch_params, \
            vx, vy, ids, fusion, dst, (uint32_t) ncols_x, nchannels_y_fd, (uint32_t) stride_row_x, (uint32_t) stride_col_y, (uint32_t) stride_col_dst, \
            channel_ratio_fd, (uint32_t) stride_channel_x, (uint32_t) stride_channel_y, (uint32_t) stride_channel_dst, \
            sample_ratio_fd, (uint32_t) stride_sample_x, (uint32_t) stride_sample_y, (uint32_t) stride_sample_dst, (uint32_t) ids_stride)
#define MMVQ_WPR_ROWS(W) \
        switch (env_rows) { \
            case 1:  if (has_fusion) { MMVQ_WPR_LAUNCH(true, W, 1); } else { MMVQ_WPR_LAUNCH(false, W, 1); } break; \
            case 4:  if (has_fusion) { MMVQ_WPR_LAUNCH(true, W, 4); } else { MMVQ_WPR_LAUNCH(false, W, 4); } break; \
            default: if (has_fusion) { MMVQ_WPR_LAUNCH(true, W, 2); } else { MMVQ_WPR_LAUNCH(false, W, 2); } break; \
        }
        if (env_w == 2) {
            MMVQ_WPR_ROWS(2)
        } else {
            MMVQ_WPR_ROWS(4)
        }
#undef MMVQ_WPR_ROWS
#undef MMVQ_WPR_LAUNCH
        static bool engaged = false;
        if (!engaged) {
            engaged = true;
            GGML_LOG_INFO("k19-q4xt: %s %s warp per row launch engaged (K %d, %d warps x %d rows per block, was %d warps)\n",
                ggml_type_name(type), ids != nullptr ? "MUL_MAT_ID" : "dense", ncols_x, env_w, env_rows, nwarps_old);
        }
        return true;
    }
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

    // k19-vrow2 / k19-vrow3 (GGML_CUDA_VERIFY_WIDE bit 2, see ggml_cuda_verify_wide_cols): a dense product without fusion over 5..8 columns of a
    // verify window keeps the 4 warps of 1..4 columns, so its K split is the 3-row window's: the multi-matrix types with the 4-column
    // launch's rows per block and K unroll (mul_mat_vec_q_verify_wide), the others through the batch invariant 4-warp launch
    if (!exact && ids == nullptr && ncols_dst >= 5 &&
            ncols_dst <= ggml_cuda_verify_wide_cols(ggml_cuda_info().devices[ggml_cuda_get_device()].cc, 2) &&
            !mul_mat_vec_q_wide_same_as_default<type>(ncols_x, ncols_dst, get_device_table_id(ggml_cuda_info().devices[ggml_cuda_get_device()].cc),
                ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size)) {
        if (mul_mat_vec_q_verify_wide<type>(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                stride_sample_x, stride_sample_y, stride_sample_dst, stream)) {
            return;
        }
        const int rc = mul_mat_vec_q_exact_batched<type>(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        if (rc > 0) {
            return;
        }
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
        // k19-moegrp (GGML_CUDA_MOE_VL): the same values, c_tpr threads per row, an expert's entries c_g per weight decode
        if (ncols_dst > 1 && nsamples_x == 1 && nsamples_dst == 1 && mul_mat_vec_q_moe_vl_launch<type>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, nchannels_dst, cc, warp_size, stream)) {
            return;
        }
        // k19-moebw (Turing by default, GGML_CUDA_MOE_BW): the same values from each warp's weight rows staged in shared memory
        if (ncols_dst > 1 && nsamples_x == 1 && nsamples_dst == 1 && mul_mat_vec_q_moe_bw_launch<type>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, nchannels_dst, cc, warp_size, stream)) {
            return;
        }
        // k19-vrow: the same values, each expert's rows read once for all the tokens that chose it
        if (ncols_dst > 1 && mul_mat_vec_q_moe_group_launch<type>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, warp_size, nchannels_dst, stream)) {
            return;
        }
        // k19-ampere: more tokens than this device's mul_mat_vec_q_moe instances take (their __launch_bounds__ follow the per-architecture
        // table: a verify window kept on mmvq by ggml_cuda_mmvq_mmid_max_batch) run in chunks of that size; each warp owns one token, so
        // every column keeps its bytes
        const int moe_max = get_mmvq_mmid_max_batch(type, cc);
        if (ncols_dst > moe_max) {
            for (int j0 = 0; j0 < ncols_dst; j0 += moe_max) {
                ggml_cuda_mm_fusion_args_device fusion_j = fusion;
                if (fusion_j.shared_dst != nullptr) {
                    fusion_j.shared_dst = fusion.shared_dst + (size_t) j0*fusion.shared_stride_col_dst;
                }
                mul_mat_vec_q_moe_launch<type>(
                    vx, (const char *) vy + (size_t) j0*stride_col_y*sizeof(block_q8_1), ids + (size_t) j0*ids_stride, fusion_j,
                    dst + (size_t) j0*stride_col_dst, ncols_x, nchannels_y_fd, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst,
                    stride_channel_x, stride_channel_y, stride_channel_dst,
                    std::min(moe_max, ncols_dst - j0), ids_stride, warp_size, nchannels_dst, stream);
            }
            return;
        }
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

    // k19-mmvqcols: the rows lever (mmvq_cols_rows) for a dense 2..8 column launch without ids or fusion
    const auto cols_cfg_launch = [&](auto nc_tag) -> bool {
        constexpr int c_ncols_dst = decltype(nc_tag)::value;
        if constexpr (!mmvq_cols_type(type)) {
            GGML_UNUSED(c_ncols_dst);
            return false;
        } else {
            const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                    fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.shared_up != nullptr;
            const int rpb = mmvq_cols_rows(type, c_ncols_dst, table_id);
            if ((rpb != 4 && rpb != 8) || ids != nullptr || has_fusion || nrows_x % rpb != 0) {
                return false;
            }
            const int nwarps = calc_nwarps(type, c_ncols_dst, table_id);
            const dim3 block_nums(nrows_x / rpb, nchannels_dst, nsamples_dst);
            const dim3 block_dims(warp_size, nwarps, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
#define MMVQ_COLS_LAUNCH(RPB) ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, false, false, 0, 0, RPB, 1>, launch_params, \
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst, \
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, \
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride)
            if constexpr (mmvq_kq_type(type)) {
                MMVQ_COLS_LAUNCH(4);
            } else if (rpb == 4) {
                MMVQ_COLS_LAUNCH(4);
            } else {
                MMVQ_COLS_LAUNCH(8);
            }
#undef MMVQ_COLS_LAUNCH
            mmvq_cols_log(type, c_ncols_dst, rpb, nwarps, "batched");
            return true;
        }
    };

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

            // k16-turing (GGML_CUDA_MMVQ_TURING_2ROWS, default on): on sm_75 a single column launch takes 2 rows per block, which shares
            // the q8_1 loads of the activation between the rows and adds weight loads in flight. A row's sum does not depend on the rows
            // per block (the K split and the reduction are per row), so the output keeps its bytes. sm_75 binaries only (the TURING table
            // also covers Volta, where it did not gain), not Q3_K, not the small-K / unrolled / MoE (ids) / shared-expert launches, and
            // only an even number of rows (a block reads the weights of both its rows). Idea from ik_llama.cpp by ikawrakow (MIT).
            // =0 off; =2 also on other NVIDIA tables (a check that the bytes do not change, not a tuned setting)
            static const int turing_2rows_env = [] {
                const char * e = getenv("GGML_CUDA_MMVQ_TURING_2ROWS");
                return e != nullptr ? atoi(e) : 1;
            }();
            const bool turing_2rows = type != GGML_TYPE_Q3_K && !has_ids && !unroll && fusion.shared_up == nullptr &&
                nrows_x % 2 == 0 && GGML_CUDA_CC_IS_NVIDIA(cc) && !should_use_small_k(c_ncols_dst) && !should_halve_iters() &&
                calc_rows_per_block(type, 1, table_id) == 1 &&
                ((turing_2rows_env == 1 && table_id == MMVQ_PARAMETERS_TURING && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
                 (turing_2rows_env == 2 && (table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GENERIC)));
            // k19-dsv4p: IQ1_S / IQ2_XXS with the codebook in shared memory (mul_mat_vec_q_tab_launch); false = the launches below
            if (mul_mat_vec_q_tab_launch<type>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y,
                    stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, ids_stride,
                    should_use_small_k(c_ncols_dst), unroll || should_halve_iters(), turing_2rows, cc, table_id, warp_size, stream)) {
                break;
            }
            // k19-q4xt: short expert rows, every warp its own rows (mul_mat_vec_q_wpr_launch); false = the launches below
            if (mul_mat_vec_q_wpr_launch<type>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y,
                    stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, ids_stride,
                    should_use_small_k(c_ncols_dst), unroll || should_halve_iters(), cc, table_id, warp_size, stream)) {
                break;
            }
            if constexpr (type != GGML_TYPE_Q3_K) {
                if (turing_2rows) {
                    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
                    const dim3 block_nums(nrows_x / 2, nchannels_dst, nsamples_dst);
                    const dim3 block_dims(warp_size, calc_nwarps(type, c_ncols_dst, table_id), 1);
                    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
#define MMVQ_2ROWS_LAUNCH(FUS) ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, FUS, false, false, 0, 0, 2, 1>, launch_params, \
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst, \
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, \
                        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride)
                    if (has_fusion) {
                        MMVQ_2ROWS_LAUNCH(true);
                    } else {
                        MMVQ_2ROWS_LAUNCH(false);
                    }
#undef MMVQ_2ROWS_LAUNCH
                    break;
                }
            }

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
            if (cols_cfg_launch(std::integral_constant<int, c_ncols_dst>{}) || batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
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
            if (cols_cfg_launch(std::integral_constant<int, c_ncols_dst>{}) || batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
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
            if (cols_cfg_launch(std::integral_constant<int, c_ncols_dst>{}) || batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
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
            if (cols_cfg_launch(std::integral_constant<int, c_ncols_dst>{}) || batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
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
            if (cols_cfg_launch(std::integral_constant<int, c_ncols_dst>{}) || batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
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
            if (cols_cfg_launch(std::integral_constant<int, c_ncols_dst>{}) || batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
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
            if (cols_cfg_launch(std::integral_constant<int, c_ncols_dst>{}) || batch_rpb_launch(std::integral_constant<int, c_ncols_dst>{})) {
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
// c_nwarps > 0: that many warps (k19-vrow3: 4 for a 5..8 column verify window, as mul_mat_vec_q_verify_wide)
template <ggml_type type, int c_ncols_dst, int n_multi, bool c_small_k = false, bool c_halve_iters = false, int c_rpb = 0, int c_unroll = 1,
          int c_nwarps = 0>
static void mul_mat_vec_q_multi_launch(
        ggml_cuda_mm_fusion_args_device fusion, const int * nrows, const void * vy, const int ncols_x,
        const int stride_row_x, const int stride_col_y, const int warp_size, const mmvq_parameter_table_id table_id, cudaStream_t stream) {
    const int nwarps = c_nwarps > 0 ? c_nwarps : calc_nwarps(type, c_ncols_dst, table_id, c_small_k, c_halve_iters);
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
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, c_small_k, c_halve_iters, c_nwarps, n_multi, c_rpb, c_unroll>, launch_params,
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
    // k19-q38next: the K-quant rows lever (mmvq_kq_rows) for the Q / K / V and gate / up launches of 2..8 columns: rows only, so every
    // matrix keeps the bytes of its own launch; only where the rule above keeps the default rows, the K loop rolled and every matrix's
    // rows divide evenly (a block reads all its rows)
    const auto kq_rows = [&](const int nc, const int r0, const bool unr) {
        const int r = r0 == 0 && !unr ? mmvq_kq_rows(type, nc, table_id) : 0;
        if (r != 4) {
            return r0;
        }
        for (int m = 0; m < n_multi; ++m) {
            if (nrows[m] % r != 0) {
                return r0;
            }
        }
        mmvq_cols_log(type, nc, r, nc >= 5 ? 4 : calc_nwarps(type, nc, table_id), "multi");
        return r;
    };
    if (ncols_dst > 1 && ncols_dst < 5) {
        brpb = kq_rows(ncols_dst, brpb, unroll);
    }
#define MMVQ_MULTI_L(NC, RPB, UNR) mul_mat_vec_q_multi_launch<type, NC, n_multi, false, false, RPB, UNR>(fusion, nrows, vy, ncols_x, stride_row_x, stride_col_y, warp_size, table_id, stream)
#define MMVQ_MULTI(NC) (brpb == 1 ? (unroll ? MMVQ_MULTI_L(NC, 1, 8) : MMVQ_MULTI_L(NC, 1, 1)) : brpb == 4 ? MMVQ_MULTI_L(NC, 4, 1) : \
                        unroll ? MMVQ_MULTI_L(NC, 0, 8) : MMVQ_MULTI_L(NC, 0, 1))
    // k19-vrow3 (GGML_CUDA_VERIFY_WIDE bit 4, see ggml_cuda_verify_wide_cols): 5..8 columns of a verify window keep the 4 warps of 1..4
    // columns, with the rows per block and K unroll of the 4-column launch (a row's sum depends on the warps only), so the Q / K / V and
    // gated delta net in-projections of a deep window get the bytes of a 1..4-row window's, as mul_mat_vec_q_verify_wide for single products
    if (ncols_dst >= 5 && table_id == MMVQ_PARAMETERS_GENERIC && calc_nwarps(type, 4, table_id) == 4 &&
            ncols_dst <= ggml_cuda_verify_wide_cols(cc, 4) && !mul_mat_vec_q_wide_same_as_default<type>(ncols_x, ncols_dst, table_id, warp_size)) {
        int brpb4 = mul_mat_vec_q_batch_rpb<type>(ncols_x, total_rows, 4, cc, warp_size, table_id);
        for (int m = 0; m < n_multi; ++m) {
            if (brpb4 == 4 && nrows[m] % 4 != 0) {
                brpb4 = 0;
            }
        }
        const bool unroll4 = mul_mat_vec_q_unroll<type>(ncols_x, total_rows, 4, table_id, warp_size);
        brpb4 = kq_rows(ncols_dst, brpb4, unroll4);
#define MMVQ_MULTI_W(NC, RPB, UNR) mul_mat_vec_q_multi_launch<type, NC, n_multi, false, false, RPB, UNR, 4>(fusion, nrows, vy, ncols_x, stride_row_x, stride_col_y, warp_size, table_id, stream)
#define MMVQ_MULTI_WNC(NC) (brpb4 == 1 ? (unroll4 ? MMVQ_MULTI_W(NC, 1, 8) : MMVQ_MULTI_W(NC, 1, 1)) : brpb4 == 4 ? MMVQ_MULTI_W(NC, 4, 1) : \
                            unroll4 ? MMVQ_MULTI_W(NC, 0, 8) : MMVQ_MULTI_W(NC, 0, 1))
        switch (ncols_dst) {
            case 5: MMVQ_MULTI_WNC(5); break;
            case 6: MMVQ_MULTI_WNC(6); break;
            case 7: MMVQ_MULTI_WNC(7); break;
            default: MMVQ_MULTI_WNC(8); break;
        }
#undef MMVQ_MULTI_WNC
#undef MMVQ_MULTI_W
        return;
    }
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

    // k19-ampere: engagement line (once) of a verify MUL_MAT_ID kept here past the per-architecture table (GGML_CUDA_VERIFY_WIDE bit 8)
    if (ids && dst->ne[2] > 1 && !(dst->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) &&
            dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        static std::once_flag once;
        std::call_once(once, [&] {
            GGML_LOG_INFO("k19-ampere: a verify MUL_MAT_ID of %d tokens (%s%s) on the multi-token mmvq kernels instead of MMQ "
                "(GGML_CUDA_VERIFY_WIDE bit 8)\n", (int) dst->ne[2], ggml_type_name(src0->type), fusion && fusion->gate ? ", fused gate" : "");
        });
    }

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= ggml_cuda_mmvq_mmid_max_batch(src0->type, cc));
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
        if (ctx.q8_1_stream_ok()) {
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

    // k19-kqtc (GGML_CUDA_MMVQ_KQTC, opt-in): a dense product of 2..8 columns without fusion on int8 tensor cores (mmvq-kqtc.cu); not
    // on the bit-exact verify (batch invariant with GGML_CUDA_VERIFY_REASSOC=0)
    if (ids == nullptr && fusion == nullptr && ncols_dst >= 2 && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            (!exact || ggml_cuda_mmvq_kqtc_reassoc_ok()) &&
            ggml_cuda_mmvq_kqtc_ok(src0->type, ne00, ncols_dst, ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        const void * kx[1] = { src0->data };
        float      * kd[1] = { dst_d };
        const int64_t kr[1] = { ne01 };
        const int64_t ks[1] = { stride_col_dst };
        if (ggml_cuda_mmvq_kqtc_launch(src0->type, 1, kx, kd, kr, nb01, src1_q8_1, stride_col_y, ks, ne00, ncols_dst, stream)) {
            return;
        }
    }

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
    if (ctx.q8_1_stream_ok()) {
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

    // k19-kqtc (GGML_CUDA_MMVQ_KQTC, opt-in): the matrices of a 2..8 column launch on int8 tensor cores, as for a single product
    const bool exact = (dst[0]->flags & GGML_TENSOR_FLAG_BATCH_INVARIANT) != 0;
    if (ne11 >= 2 && (!exact || ggml_cuda_mmvq_kqtc_reassoc_ok()) &&
            ggml_cuda_mmvq_kqtc_ok(type, ne10, ne11, ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        const void * kx[GGML_CUDA_MMVQ_MULTI_MAX] = {};
        float      * kd[GGML_CUDA_MMVQ_MULTI_MAX] = {};
        int64_t      kr[GGML_CUDA_MMVQ_MULTI_MAX] = {};
        int64_t      ks[GGML_CUDA_MMVQ_MULTI_MAX] = {};
        for (int m = 0; m < n; ++m) {
            kx[m] = src0[m]->data;
            kd[m] = (float *) dst[m]->data;
            kr[m] = src0[m]->ne[1];
            ks[m] = dst[m]->nb[1] / sizeof(float);
        }
        if (ggml_cuda_mmvq_kqtc_launch(type, n, kx, kd, kr, src0[0]->nb[1], src1_q8_1, stride_col_y, ks, ne10, ne11,
                stream)) {
            return;
        }
    }

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

// k15-mm2: the hyper-connection mat-vecs of qwen4exp (Flash-Next) next to their neighbours, in one launch each. A Q8_0 row's sum
// depends only on the warps of its block (4 here, as the unfused launch has on the generic and Turing tables at 1..4 columns, also
// batch invariant, where a column is computed with the warps of a single column launch), not on the rows per block: the kernels
// below compute every Q8_0 row with the 4-warp K split and reduction of mul_mat_vec_q, so each output keeps its bytes.
static bool mmvq_q8_0_four_warps(const int cc, const int ncols_dst) {
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == 32 &&
        (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_TURING) && ncols_dst >= 1 && ncols_dst <= 4 &&
        calc_nwarps(GGML_TYPE_Q8_0, ncols_dst, table_id) == 4 && calc_nwarps(GGML_TYPE_Q8_0, 1, table_id, true) == 4 &&
        calc_nwarps(GGML_TYPE_Q8_0, 1, table_id, false) == 4;
}

// the q8_1 copy of a contiguous src1, as ggml_cuda_mul_mat_vec_q makes it (the q8_1 cache entry it would use)
const char * ggml_cuda_mmvq_src1_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_cuda_pool_alloc<char> & alloc) {
    const int64_t ne10        = src1->ne[0];
    const int64_t ne11        = src1->ne[1];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_1_size   = ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    char * q = nullptr;
    bool quantized = false;
    if (ctx.q8_1_stream_ok()) {
        int64_t key[ggml_cuda_q8_1_cache::n_key];
        ggml_cuda_q8_1_cache::key_contiguous(ne10, ne11, key);
        q = ctx.q8_1_cache.get(src1->data, ggml_nbytes(src1), key, q8_1_size, quantized);
    }
    if (q == nullptr) {
        q = alloc.alloc(q8_1_size);
    }
    if (!quantized) {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q, GGML_TYPE_Q8_0, ne10, s11, s11*ne11, s11*ne11, ne10_padded, ne11, 1, 1,
            ctx.stream());
    }
    return q;
}

// one Q8_0 row r of rpb rows of a 4-warp block, as mul_mat_vec_q computes it (tid = 32*warp + lane)
#define MMVQ_HC_NWARPS 4

// lever 1, GGML_CUDA_HC_INJECT_DOWN: the hc down projection (Q8_0, K = hc*n_embd -> a few hundred rows) and the inject product (F32,
// the same activation -> hc rows, split-K) in one grid: blocks [0, nblocks_q) are the down rows, the others the split-K blocks of the
// inject (mmvf-splitk.cuh, the code of its own launch)
template <int ncols_dst, int rpb, int c_unroll>
__launch_bounds__(MMVQ_HC_NWARPS*32, 1)
static __global__ void mul_mat_vec_q8_0_inject(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const int ncols_x, const int stride_row_x,
        const int stride_col_y, const int stride_col_dst, const int nblocks_inj, const ggml_cuda_mmvf_splitk_args inj) {
    static_assert(MMVF_SPLITK_BLOCK == MMVQ_HC_NWARPS*32, "the inject blocks have the block size of their own launch");
    // the inject blocks first: they start with the first down blocks instead of after them
    if (blockIdx.x < (unsigned int) nblocks_inj) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        const int b = blockIdx.x;
        mul_mat_vec_f_splitk_block<ncols_dst>(inj.x, inj.y, inj.dst, inj.part, inj.cnt, inj.ncols2, inj.chunk2, inj.stride_row,
            inj.stride_col_y2, inj.stride_col_dst, b % inj.nrows, b / inj.nrows, inj.nsplit, 32*threadIdx.y + threadIdx.x);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        return;
    }
    constexpr ggml_type type = GGML_TYPE_Q8_0;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int nwarps    = MMVQ_HC_NWARPS;
    constexpr int warp_size = 32;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid              = warp_size*threadIdx.y + threadIdx.x;
    const     int row0             = rpb*(blockIdx.x - nblocks_inj);
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter  = vdr * nwarps*warp_size / qi;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    float tmp[ncols_dst][rpb] = {{0.0f}};
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0*stride_row_x;
    // unrolled: a thread has the weight loads of several iterations in flight; it adds its blocks in the same order
#pragma unroll c_unroll
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1][ncols_dst][rpb][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
    dst += row0;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if (threadIdx.x == i) {
                dst[j*stride_col_dst + i] = tmp[j][i];
            }
        }
    }
}

bool ggml_cuda_mul_mat_vec_q_inject_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    return src0->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(src0) && ggml_is_matrix(src0) && src1->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src1) && ggml_is_matrix(src1) && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
        dst->ne[2] == 1 && dst->ne[3] == 1 && dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] && src0->ne[0] == src1->ne[0] &&
        src0->ne[0] % MATRIX_ROW_PADDING == 0 && src0->ne[1] % 2 == 0 && src0->ne[1] <= INT_MAX/2 && mmvq_q8_0_four_warps(cc, (int) dst->ne[1]);
}

void ggml_cuda_mul_mat_vec_q_inject(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mmvf_splitk_args & inj) {
    GGML_ASSERT(ggml_cuda_mul_mat_vec_q_inject_ok(src0, src1, dst, ggml_cuda_info().devices[ctx.device].cc));
    GGML_ASSERT(inj.ncols_dst == (int) dst->ne[1] && inj.nrows > 0 && inj.nsplit > 0);
    ggml_cuda_pool_alloc<char> q8_1_alloc(ctx.pool());
    const char * vy = ggml_cuda_mmvq_src1_q8_1(ctx, src1, q8_1_alloc);
    const int ncols_x        = (int) src0->ne[0];
    const int nrows_x        = (int) src0->ne[1];
    const int ncols_dst      = (int) dst->ne[1];
    const int stride_row_x   = (int) (src0->nb[1] / ggml_type_size(src0->type));
    const int stride_col_y   = (int) (GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING) / QK8_1);
    const int stride_col_dst = (int) (dst->nb[1] / sizeof(float));
    // one row per block and the K loop unrolled 8 times (neither changes a row's sum): B200 Flash-Next verify (3 columns, PDL off)
    // 5.1 us for both products vs 5.9 us for the down projection alone; 2 rows per block without unrolling took 8.4 us
    constexpr int rpb = 1;
    const int nblocks_q   = nrows_x / rpb;
    const int nblocks_inj = inj.nrows*inj.nsplit;
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(dim3(nblocks_q + nblocks_inj, 1, 1), dim3(32, MMVQ_HC_NWARPS, 1), 0, ctx.stream());
#define MMVQ_INJ_LAUNCH(NC) ggml_cuda_kernel_launch(mul_mat_vec_q8_0_inject<NC, rpb, 8>, launch_params, src0->data, (const void *) vy, \
        (float *) dst->data, ncols_x, stride_row_x, stride_col_y, stride_col_dst, nblocks_inj, inj)
    switch (ncols_dst) {
        case 1: MMVQ_INJ_LAUNCH(1); break;
        case 2: MMVQ_INJ_LAUNCH(2); break;
        case 3: MMVQ_INJ_LAUNCH(3); break;
        case 4: MMVQ_INJ_LAUNCH(4); break;
        default: GGML_ABORT("fatal error");
    }
#undef MMVQ_INJ_LAUNCH
}

// lever 2, GGML_CUDA_HC_UP_PRE: the hc up projection (Q8_0, short K -> hc*n_embd rows, gate) and the gated DSV4_HC_PRE that reads it
// (mixed[i] = scale * sum_h x[i, h] * sigmoid(gate[i, h])) plus its q8_1 copy, in one launch. The grid is the up projection's own
// (4 warps, rpb rows per block, the mul_mat_vec_q K split and reduction, so the gate keeps its bytes). The rows of 32 channels of the
// hc = 4 streams belong to 4*32/rpb blocks; the last of them to finish (counted in cnt[group], which it resets) computes, stores and
// quantizes the 32 mixed values of every column as dsv4_hc_pre_q8_1_f32 does it (one warp per column, one q8_1 block).
#define MMVQ_HC_PRE_HC     4
#define MMVQ_HC_PRE_CH     32
template <int ncols_dst, int rpb>
__launch_bounds__(MMVQ_HC_NWARPS*32, 1)
static __global__ void mul_mat_vec_q8_0_hc_pre(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ gate, const float * __restrict__ x,
        float * __restrict__ mixed, void * __restrict__ vq, unsigned int * __restrict__ cnt, const int ncols_x, const int stride_row_x,
        const int stride_col_y, const int stride_col_gate, const int n_embd, const int64_t sx1, const int64_t sx2, const float scale) {
    static_assert(ncols_dst <= MMVQ_HC_NWARPS, "one warp per column in the epilogue");
    constexpr ggml_type type = GGML_TYPE_Q8_0;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int nwarps    = MMVQ_HC_NWARPS;
    constexpr int warp_size = 32;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    static_assert(MMVQ_HC_PRE_CH % rpb == 0, "the rows of a block in one group");

    const     int tid              = warp_size*threadIdx.y + threadIdx.x;
    const     int row0             = rpb*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter  = vdr * nwarps*warp_size / qi;

    __shared__ float tmp_shared[nwarps-1][ncols_dst][rpb][warp_size];
    __shared__ bool  last;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    float tmp[ncols_dst][rpb] = {{0.0f}};
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0*stride_row_x;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                }
                tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
                if (threadIdx.x == i) {
                    gate[j*stride_col_gate + row0 + i] = tmp[j][i];
                }
            }
        }
        if (threadIdx.x < rpb) {
            __threadfence();
        }
    }
    __syncthreads();

    // the rows of a block lie in one stream and one group of 32 channels (rpb divides 32, n_embd % 32 == 0)
    const int c0    = (row0 % n_embd) / MMVQ_HC_PRE_CH * MMVQ_HC_PRE_CH;
    const int group = c0 / MMVQ_HC_PRE_CH;
    if (tid == 0) {
        constexpr unsigned int nblocks_group = MMVQ_HC_PRE_HC*MMVQ_HC_PRE_CH/rpb;
        last = atomicAdd(cnt + group, 1u) == nblocks_group - 1;
        if (last) {
            cnt[group] = 0;
        }
    }
    __syncthreads();
    if (!last || threadIdx.y >= ncols_dst) {
        return;
    }
    __threadfence();

    // dsv4_hc_pre_q8_1_f32 (gated) for token it = warp, channel i0 = c0 + lane
    const int     lane = threadIdx.x;
    const int     i0   = c0 + lane;
    const int64_t it   = threadIdx.y;
    float sum = 0.0f;
    for (int64_t ih = 0; ih < MMVQ_HC_PRE_HC; ++ih) {
        const float xv = x[i0 + ih*sx1 + it*sx2];
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        const float gv = __ldcg(gate + it*stride_col_gate + ih*n_embd + i0);
#else
        // no __ldcg on HIP / MUSA: a volatile load also skips the non-coherent L1 for the other blocks' gate values
        const float gv = *(const volatile float *) (gate + it*stride_col_gate + ih*n_embd + i0);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        const float wv = 1.0f / (1.0f + expf(-gv));
        sum += xv * wv;
    }
    const float xi = scale * sum;
    mixed[it*n_embd + i0] = xi;
    if (vq == nullptr) {
        return;
    }
    float amax = fabsf(xi);
    float qsum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    qsum = warp_reduce_sum<QK8_1>(qsum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    block_q8_1 * yq = (block_q8_1 *) vq;
    const int64_t ir  = it*n_embd + i0;
    const int64_t ib  = ir / QK8_1;
    const int     iqs = ir % QK8_1;
    yq[ib].qs[iqs] = q;
    if (iqs == 0) {
        yq[ib].ds = make_half2(d, qsum);
    }
}

bool ggml_cuda_mul_mat_vec_q_hc_pre_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * gate, const int64_t n_embd,
        const int64_t hc, const int cc) {
    return src0->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(src0) && ggml_is_matrix(src0) && src1->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src1) && ggml_is_matrix(src1) && gate->type == GGML_TYPE_F32 && ggml_is_contiguous(gate) && ggml_is_matrix(gate) &&
        gate->ne[0] == src0->ne[1] && gate->ne[1] == src1->ne[1] && src0->ne[0] == src1->ne[0] && src0->ne[0] % QK8_0 == 0 &&
        n_embd / MMVQ_HC_PRE_CH <= GGML_CUDA_HC_PRE_MAX_GROUPS &&
        hc == MMVQ_HC_PRE_HC && src0->ne[1] == hc*n_embd && n_embd % MATRIX_ROW_PADDING == 0 && src0->ne[1] <= INT_MAX/2 &&
        mmvq_q8_0_four_warps(cc, (int) gate->ne[1]);
}

void ggml_cuda_mul_mat_vec_q_hc_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const char * vy,
        ggml_tensor * gate, const float * x, const int64_t sx1, const int64_t sx2, float * mixed, void * q8_1_out, unsigned int * cnt,
        const int64_t n_embd, const float scale) {
    GGML_ASSERT(ggml_cuda_mul_mat_vec_q_hc_pre_ok(src0, src1, gate, n_embd, MMVQ_HC_PRE_HC, ggml_cuda_info().devices[ctx.device].cc));
    const int ncols_x         = (int) src0->ne[0];
    const int ncols_dst       = (int) gate->ne[1];
    const int stride_row_x    = (int) (src0->nb[1] / ggml_type_size(src0->type));
    const int stride_col_y    = (int) (GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING) / QK8_1);
    const int stride_col_gate = (int) (gate->nb[1] / sizeof(float));
    GGML_ASSERT(cnt != nullptr);
    // 4 rows per block, as the unfused launch (16 rows: 14.9 us instead of 10.1 us at 3 columns on B200)
    constexpr int rpb = 4;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
        dim3((unsigned int) (src0->ne[1] / rpb), 1, 1), dim3(32, MMVQ_HC_NWARPS, 1), 0, ctx.stream());
#define MMVQ_HC_PRE_LAUNCH(NC) ggml_cuda_kernel_launch(mul_mat_vec_q8_0_hc_pre<NC, rpb>, launch_params, src0->data, (const void *) vy, \
        (float *) gate->data, x, mixed, q8_1_out, cnt, ncols_x, stride_row_x, stride_col_y, stride_col_gate, (int) n_embd, sx1, sx2, scale)
    switch (ncols_dst) {
        case 1: MMVQ_HC_PRE_LAUNCH(1); break;
        case 2: MMVQ_HC_PRE_LAUNCH(2); break;
        case 3: MMVQ_HC_PRE_LAUNCH(3); break;
        case 4: MMVQ_HC_PRE_LAUNCH(4); break;
        default: GGML_ABORT("fatal error");
    }
#undef MMVQ_HC_PRE_LAUNCH_R
#undef MMVQ_HC_PRE_LAUNCH
}
