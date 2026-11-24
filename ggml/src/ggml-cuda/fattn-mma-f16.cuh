// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ggml_cuda_mma;

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 4,  64,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 4,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  32,  96,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 2,  64, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1,  64, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);

    // TODO tune specifically for Volta
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  64,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  64,  96,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 2,  64, 128, 128,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 256, 2,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 128, 2,  32, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2,  32, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

// RDNA3 (incl. RDNA3.5): compile-time opt-in GGML_HIP_FA_RDNA3_D256_FLOAT_VKQ gives head size 256 float VKQ accumulators
// (mma_tile_sizes<256, ncols> below) and tile sizes for that register budget. Off by default: on gfx1151 it was 5-25% slower
// in head-256 prefill and ~8% slower in MTP decode after a 2k prompt, with no prefill gain. RDNA4 is unaffected.
static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna3(const int DKQ, const int DV, const int ncols) {
#ifdef GGML_HIP_FA_RDNA3_D256_FLOAT_VKQ
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 256, 1,  32, 128,  64,  64, 1, false);
#endif // GGML_HIP_FA_RDNA3_D256_FLOAT_VKQ

    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_cdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 1,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 256, 3,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 256, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 256, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 256, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 256, 2,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 512, 1,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 512, 1,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  64, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  64, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  64, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (ampere_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    if (turing_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    }
    if (amd_mfma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
    }
    if (amd_wmma_available(cc)) {
        return GGML_CUDA_CC_IS_RDNA3(cc) ? ggml_cuda_fattn_mma_get_config_rdna3(DKQ, DV, ncols) :
                                           ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
    }
    GGML_ASSERT(volta_mma_available(cc));
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols) {
#if defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
#elif defined(AMD_MFMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
#elif defined(VOLTA_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    return ggml_cuda_fattn_mma_get_config_rdna3(DKQ, DV, ncols);
#elif defined(AMD_WMMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
}

static constexpr __device__ int get_cols_per_thread() {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    return 1; // AMD has a single column per thread.
#else
    return 2; // This is specifically KQ columns, Volta only has a single VKQ column.
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static __host__ int get_cols_per_warp(const int cc) {
    if (turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc)) {
        return 16;
    } else {
        // Volta
        return 32;
    }
}

static __host__ bool ggml_cuda_fattn_mma_get_swizzled(const int DKQ, const int DV, const int ncols, const int cc) {
    return turing_mma_available(cc) &&
        ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, cc) % 32 == 0 && ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols, cc) % 32 == 0;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_swizzled(const int DKQ, const int DV, const int ncols) {
#ifdef TURING_MMA_AVAILABLE
    return ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols) % 32 == 0 && ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols) % 32 == 0;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return false;
#endif // TURING_MMA_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, cc) : 0;
}

// the in-place code q8_kv of the MMA kernel. 0: f16 K / V (or a staged f16 copy). Else the low 2 bits are
// the load mode (1: synchronous loads converting on the way, 2: the raw rows through the cp.async pipeline, converted in shared
// memory), bit 2 set: K is q4_0 (else q8_0), bit 3 set: V is q4_0 (else q8_0). 1 / 2 are q8_0 K and V as before
static constexpr __host__ __device__ int  fattn_kvq_mode(const int q8_kv) { return q8_kv & 3; }
static constexpr __host__ __device__ bool fattn_kvq_k4  (const int q8_kv) { return (q8_kv & 4) != 0; }
static constexpr __host__ __device__ bool fattn_kvq_v4  (const int q8_kv) { return (q8_kv & 8) != 0; }
// bytes of a row of D values of the K (or V) type of a code
static constexpr __host__ __device__ int  fattn_kvq_row_bytes(const int D, const bool is_q4) {
    return is_q4 ? (D/QK4_0)*int(sizeof(block_q4_0)) : (D/QK8_0)*int(sizeof(block_q8_0));
}
// bit 4 (16) of the code: INT8 Q*K^T. K is the q8_0 cache read in place (its quants are the int8 operands, its block
// scales the K scales), Q is quantised per 32 values in the prologue (q8_1 style, as the vector kernel), the MMA is s8 x s8 -> s32 per
// 32-value block with the block's d_q * d_k applied in f32. Load mode bits 0: V is f16 (the staged copy) and P*V is unchanged
static constexpr __host__ __device__ bool fattn_kvq_i8   (const int q8_kv) { return (q8_kv & 16) != 0; }
// bit 5 (32) of the code, with bit 4: INT8 Q*K^T with one scale per row over an f16 K cache. Q and K come as rows of
// fattn_i8_stride(D) bytes staged per launch (ggml_cuda_flash_attn_ext_i8_rows: D int8 values, then the row's f32 scale; Q's scale
// carries the softmax scale), so a staged row is the int8 tile row as it is (no conversion in shared memory); the MMAs of all blocks of a
// row accumulate in one int32 sum and the scales are applied once per score. V is the f16 cache, P*V is unchanged
static constexpr __host__ __device__ bool fattn_kvq_i8r  (const int q8_kv) { return (q8_kv & 32) != 0; }
// K given as raw q8_0 / q4_0 rows (byte strides) rather than f16
static constexpr __host__ __device__ bool fattn_kvq_k_raw(const int q8_kv) { return fattn_kvq_mode(q8_kv) != 0 || fattn_kvq_i8(q8_kv); }
// V given as raw q8_0 / q4_0 rows (byte strides) rather than f16
static constexpr __host__ __device__ bool fattn_kvq_v_raw(const int q8_kv) { return fattn_kvq_mode(q8_kv) != 0; }
// the int8 K (and Q) tile in shared memory: rows of D bytes + 16 (16 byte aligned rows, ldmatrix without bank conflicts)
static constexpr __host__ __device__ int  fattn_i8_stride(const int D) { return D + 16; }
// raw q8_0 K rows in shared memory (pipelined loads), padded to 16 bytes
static constexpr __host__ __device__ int  fattn_i8_raw_row(const int D) { return ((D/QK8_0)*int(sizeof(block_q8_0)) + 15) / 16 * 16; }
// the Q scales of a tile in shared memory (behind the mask and the raw K rows): ncols/2 rows of 2*(D/32) + 2 floats
static constexpr __host__ __device__ int  fattn_i8_qd_bytes(const int D, const int ncols) { return (ncols/2)*(2*(D/QK8_1) + 2)*int(sizeof(float)); }
// the kernels compiled with an INT8 variant: head sizes 128 / 256 (K = V), configs whose KQ tiles are 16 columns wide per warp
static constexpr __host__ __device__ bool ggml_cuda_flash_attn_ext_mma_i8_compiled(const int DKQ, const int DV, const int ncols1, const int ncols2) {
    return DKQ == DV && (DKQ == 128 || DKQ == 256) && ncols1*ncols2 >= (DKQ == 256 ? 32 : 16);
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const bool use_sparse, const int q8_kv = 0) {
#ifdef CP_ASYNC_AVAILABLE
    // q8_0 K / V (q8_kv 1) loaded synchronously, converted on the way into shared memory; q8_kv 2: the raw q8_0 rows
    // through the cp.async pipeline of the f16 kernel, converted in shared memory (see flash_attn_ext_q8_load_raw)
    const int nstages_target = ncols2 >= 2 && fattn_kvq_mode(q8_kv) != 1 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2) : 0;
    // sparse gather is not implemented for multi-stage loading
    return use_sparse && nstages_target > 1 ? 1 : nstages_target;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2, use_sparse, q8_kv);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

template<int stride_tile, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity is 16 bytes.
    constexpr int chunk_size = 16;
    constexpr int h2_per_chunk = chunk_size / sizeof(half2);
    const int chunks_per_row = D2 / h2_per_chunk;
    if constexpr (use_cp_async) {
        static_assert(warp_size == 32, "bad warp_size");
        static_assert(!oob_check || use_sparse, "OOB check not compatible with cp_async");
        constexpr int preload = 64;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) {
            const int stride_k = warp_size >> n;
            const int k0_start = stride_k == warp_size ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

                int64_t i_KV;
                if constexpr (use_sparse) {
                    // padded slots gather row 0, the -inf mask removes their contribution
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : 0;
                    i_KV = index >= 0 ? index : 0;
                } else {
                    i_KV = k_VKQ_0 + i;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    cp_async_cg_16<preload>(tile_KV_32 + swizzle<stride_tile*sizeof(half2), char>(i*stride_tile*sizeof(half2) + k*chunk_size, i), KV + i_KV*stride_KV + k*h2_per_chunk);
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    } else {
        const half2 zero[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
        auto load = [&] __device__ (const int n) {
            const int stride_k = 32 >> n;
            const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const half2 * src;
                    if constexpr (use_sparse) {
                        const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                        src = index >= 0 ? KV + int64_t(index)*stride_KV + k*h2_per_chunk : zero;
                    } else {
                        src = !oob_check || i < i_sup ? KV + int64_t(k_VKQ_0 + i)*stride_KV + k*h2_per_chunk : zero;
                    }
                    ggml_cuda_memcpy_1<16>(swizzle<stride_tile>(tile_KV, i*stride_tile + k*h2_per_chunk, i), src);
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    }
}

// a K / V tile read from a q8_0 cache in place (or from its raw rows in shared memory, q8_kv 2). KV points at the first
// q8_0 block of the head's row, stride_KV is the row stride in bytes, D2 the half2 per row (the whole head: q8_0 tiles are never split
// along D). Each thread takes a pair of blocks (68 bytes = 17 aligned 32 bit words: rows start 4 byte aligned) and writes their 64
// values as 8 chunks of 16 bytes into the f16 tile, in the swizzled layout of flash_attn_ext_f16_load_tile (zeros for skipped rows).
// The value is f16(q) * d in one f16 multiply, the staging conversion's (convert.cu dequantize_block_q8_0_f16: __hmul2 of exact f16
// operands, rounded once): f16(q) is built exactly as (1024 + (q ^ 0x80)) - 1152, so the tile holds the bits the f16 kernel reads
// from the staged copy.
// (is_q4): the same for a q4_0 cache. A pair of q4_0 blocks is 36 bytes = 9 aligned 32 bit words (d of block 0 in the
// low half of w[0], its 16 quant bytes at bytes 2..17; d of block 1 in the high half of w[4], its quants in w[5..8]); value j < 16 of a
// block is the low nibble of quant byte j, value j + 16 its high nibble. f16(n - 8) is built exactly as (1024 + n) - 1032 and the value
// is fma(f16(n - 8), d, +0): the exact product rounded once, as the staging conversion (convert.cu dequantize_block_q4_0: d*n - 8*d in
// f32, exact, then rounded to f16), including +0 for n == 8 with d < 0 (a plain multiply would give -0 there)
template<int stride_tile, int nwarps, int nbatch_fa, int D2, bool oob_check, bool use_sparse, bool is_q4 = false>
static __device__ __forceinline__ void flash_attn_ext_q8_load_tile(
        const char * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(D2 % QK8_0 == 0, "q8_0 / q4_0 rows must hold whole pairs of blocks");
    constexpr int npair = D2/QK8_0; // pairs of q8_0 / q4_0 blocks per row (64 values each)
    constexpr int ntot  = nbatch_fa*npair;
    constexpr int nthr  = nwarps*warp_size;
    if constexpr (is_q4) {
        const half2 off4 = make_half2(1032.0f, 1032.0f);
        const half2 zero = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int ip0 = 0; ip0 < ntot; ip0 += nthr) {
            const int ip = ip0 + threadIdx.y*warp_size + threadIdx.x;
            if (ntot % nthr != 0 && ip >= ntot) {
                break;
            }
            const int i = ip / npair;
            const int p = ip % npair;

            int64_t row;
            bool    skip;
            if constexpr (use_sparse) {
                const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                skip = index < 0;
                row  = skip ? 0 : index;
            } else {
                skip = oob_check && i >= i_sup;
                row  = skip ? 0 : k_VKQ_0 + i;
            }

            uint32_t w[9];
            if (!skip) {
                const uint32_t * src = (const uint32_t *) (KV + row*stride_KV + p*int(2*sizeof(block_q4_0)));
#pragma unroll
                for (int l = 0; l < 9; ++l) {
                    w[l] = src[l];
                }
            } else {
#pragma unroll
                for (int l = 0; l < 9; ++l) {
                    w[l] = 0;
                }
            }
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const uint16_t d_bits = blk == 0 ? (uint16_t) (w[0] & 0xFFFF) : (uint16_t) (w[4] >> 16);
                const half2    d2     = __half2half2(__ushort_as_half(d_bits));
                uint32_t qw[4]; // quant bytes 0..15 of the block
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    qw[m] = blk == 0 ? __funnelshift_r(w[m], w[m + 1], 16) : w[5 + m];
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    // chunk c: values 8c .. 8c+7 = the low (c < 2) or high (c >= 2) nibbles of quant bytes 8*(c%2) .. 8*(c%2)+7
                    int v[4];
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint32_t x  = qw[2*(c % 2) + h];
                        const uint32_t n  = c < 2 ? x & 0x0F0F0F0Fu : (x >> 4) & 0x0F0F0F0Fu;
                        const uint32_t lo = __byte_perm(n, 0x64646464u, 0x5140);
                        const uint32_t hi = __byte_perm(n, 0x64646464u, 0x7362);
                        const half2    r0 = __hfma2(__hsub2(*(const half2 *) &lo, off4), d2, zero);
                        const half2    r1 = __hfma2(__hsub2(*(const half2 *) &hi, off4), d2, zero);
                        v[2*h + 0] = skip ? 0 : *(const int *) &r0;
                        v[2*h + 1] = skip ? 0 : *(const int *) &r1;
                    }
                    const int k = p*8 + blk*4 + c; // 16 byte chunk of the row
                    *(int4 *) swizzle<stride_tile>(tile_KV, i*stride_tile + k*4, i) = make_int4(v[0], v[1], v[2], v[3]);
                }
            }
        }
        return;
    }
    const half2 off = make_half2(1152.0f, 1152.0f);
#pragma unroll
    for (int ip0 = 0; ip0 < ntot; ip0 += nthr) {
        const int ip = ip0 + threadIdx.y*warp_size + threadIdx.x;
        if (ntot % nthr != 0 && ip >= ntot) {
            break;
        }
        const int i = ip / npair;
        const int p = ip % npair;

        int64_t row;
        bool    zero;
        if constexpr (use_sparse) {
            const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
            zero = index < 0;
            row  = zero ? 0 : index;
        } else {
            zero = oob_check && i >= i_sup;
            row  = zero ? 0 : k_VKQ_0 + i;
        }

        uint32_t w[17];
        if (!zero) {
            const uint32_t * src = (const uint32_t *) (KV + row*stride_KV + p*int(2*sizeof(block_q8_0)));
#pragma unroll
            for (int l = 0; l < 17; ++l) {
                w[l] = src[l];
            }
        } else {
#pragma unroll
            for (int l = 0; l < 17; ++l) {
                w[l] = 0;
            }
        }
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            // block 0: d in the low half of w[0], quants at bytes 2..33; block 1: d in the high half of w[8], quants in w[9..16]
            const uint16_t d_bits = blk == 0 ? (uint16_t) (w[0] & 0xFFFF) : (uint16_t) (w[8] >> 16);
            const half2    d2     = __half2half2(__ushort_as_half(d_bits));
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                int v[4];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int      j  = 2*c + h;
                    const uint32_t qs = blk == 0 ? __funnelshift_r(w[j], w[j + 1], 16) : w[9 + j];
                    const uint32_t u  = qs ^ 0x80808080u;
                    const uint32_t lo = __byte_perm(u, 0x64646464u, 0x5140);
                    const uint32_t hi = __byte_perm(u, 0x64646464u, 0x7362);
                    const half2    r0 = __hmul2(__hsub2(*(const half2 *) &lo, off), d2);
                    const half2    r1 = __hmul2(__hsub2(*(const half2 *) &hi, off), d2);
                    v[2*h + 0] = zero ? 0 : *(const int *) &r0;
                    v[2*h + 1] = zero ? 0 : *(const int *) &r1;
                }
                const int k = p*8 + blk*4 + c; // 16 byte chunk of the row
                *(int4 *) swizzle<stride_tile>(tile_KV, i*stride_tile + k*4, i) = make_int4(v[0], v[1], v[2], v[3]);
            }
        }
    }
}

// (q8_kv 2): the raw q8_0 rows of a K / V tile (nbatch_fa rows of D2/16 blocks, 16 byte aligned: D 256) into shared
// memory with cp.async, 16 bytes per copy, rows packed at their size; flash_attn_ext_q8_load_tile converts them into the f16 tile.
// (row_bytes is the size of a row of the type, q8_0 or q4_0)
template<int nwarps, int nbatch_fa, int row_bytes>
static __device__ __forceinline__ void flash_attn_ext_q8_load_raw(
        const char * const __restrict__ KV, char * const __restrict__ raw, const int stride_KV, const int k_VKQ_0) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int warp_size      = ggml_cuda_get_physical_warp_size();
    static_assert(row_bytes % 16 == 0, "raw q8_0 / q4_0 rows must be 16 byte multiples");
    constexpr int chunks_per_row = row_bytes/16;
    constexpr int nchunks        = nbatch_fa*chunks_per_row;
    constexpr int preload        = 64;
    const unsigned int raw_32 = ggml_cuda_cvta_generic_to_shared(raw);
#pragma unroll
    for (int ic0 = 0; ic0 < nchunks; ic0 += nwarps*warp_size) {
        const int ic = ic0 + threadIdx.y*warp_size + threadIdx.x;
        if (nchunks % (nwarps*warp_size) != 0 && ic >= nchunks) {
            break;
        }
        const int i = ic / chunks_per_row;
        const int c = ic % chunks_per_row;
        cp_async_cg_16<preload>(raw_32 + i*row_bytes + c*16, KV + int64_t(k_VKQ_0 + i)*stride_KV + c*16);
    }
#else
    GGML_UNUSED_VARS(KV, raw, stride_KV, k_VKQ_0);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// a K tile of a q8_0 cache as int8 operands: the quants of row i to tile_K8 + i*fattn_i8_stride(D) (32 bytes per block,
// in place: no conversion), the block scales as f32 to tile_dk[b*nbatch_fa + i]. src_smem: from the raw rows in shared memory
// (flash_attn_ext_i8_load_raw, rows fattn_i8_raw_row(D) apart), else from the cache rows k_VKQ_0 + i (stride_KV bytes apart). Each
// thread takes a pair of blocks (68 bytes = 17 aligned 32 bit words: d of block 0 in the low half of w[0], its quants at bytes 2..33;
// d of block 1 in the high half of w[8], its quants in w[9..16]). Rows past i_sup (oob_check) get zero quants and scales
template<int nwarps, int nbatch_fa, int D, bool oob_check, bool src_smem>
static __device__ __forceinline__ void flash_attn_ext_i8_load_tile(
        const char * const __restrict__ KV, const int stride_KV, const int k_VKQ_0, const int i_sup,
        char * const __restrict__ tile_K8, float * const __restrict__ tile_dk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(D % (2*QK8_0) == 0, "q8_0 rows must hold whole pairs of blocks");
    constexpr int npair  = D/(2*QK8_0);
    constexpr int ntot   = nbatch_fa*npair;
    constexpr int nthr   = nwarps*warp_size;
    constexpr int stride = fattn_i8_stride(D);
#pragma unroll
    for (int ip0 = 0; ip0 < ntot; ip0 += nthr) {
        const int ip = ip0 + threadIdx.y*warp_size + threadIdx.x;
        if (ntot % nthr != 0 && ip >= ntot) {
            break;
        }
        const int  i    = ip / npair;
        const int  p    = ip % npair;
        const bool zero = oob_check && i >= i_sup;

        const uint32_t * src = src_smem ?
            (const uint32_t *) (KV + i*fattn_i8_raw_row(D) + p*int(2*sizeof(block_q8_0))) :
            (const uint32_t *) (KV + int64_t(zero ? 0 : k_VKQ_0 + i)*stride_KV + p*int(2*sizeof(block_q8_0)));
        int4 * dst = (int4 *) (tile_K8 + i*stride + p*(2*QK8_0));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
        // Blackwell: all 17 words at once (measured faster there)
        {
            uint32_t w[17];
#pragma unroll
            for (int l = 0; l < 17; ++l) {
                w[l] = zero ? 0 : src[l];
            }
            int q[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                q[j]     = __funnelshift_r(w[j], w[j + 1], 16);
                q[8 + j] = w[9 + j];
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                dst[c] = make_int4(q[4*c + 0], q[4*c + 1], q[4*c + 2], q[4*c + 3]);
            }
            tile_dk[(2*p + 0)*nbatch_fa + i] = __half2float(__ushort_as_half((uint16_t) (w[0] & 0xFFFF)));
            tile_dk[(2*p + 1)*nbatch_fa + i] = __half2float(__ushort_as_half((uint16_t) (w[8] >> 16)));
        }
#else
        // one block at a time (9 words each, w[8] read twice): fewer live registers than all 17 words (sm_75 / 80 / 89 spilled)
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            uint32_t w[9];
#pragma unroll
            for (int l = 0; l < 9; ++l) {
                w[l] = zero ? 0 : src[8*blk + l];
            }
            int q[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                q[j] = blk == 0 ? __funnelshift_r(w[j], w[j + 1], 16) : w[1 + j];
            }
            dst[2*blk + 0] = make_int4(q[0], q[1], q[2], q[3]);
            dst[2*blk + 1] = make_int4(q[4], q[5], q[6], q[7]);
            const uint16_t d_bits = blk == 0 ? (uint16_t) (w[0] & 0xFFFF) : (uint16_t) (w[0] >> 16);
            tile_dk[(2*p + blk)*nbatch_fa + i] = __half2float(__ushort_as_half(d_bits));
        }
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
    }
}

// the raw q8_0 rows of a K tile into shared memory with cp.async (rows fattn_i8_raw_row(D) apart): 16 byte copies at
// D 256 (272 byte rows, 16 byte aligned), 8 byte copies at D 128 (136 byte rows, 8 byte aligned); flash_attn_ext_i8_load_tile
// takes them from there
template<int nwarps, int nbatch_fa, int D>
static __device__ __forceinline__ void flash_attn_ext_i8_load_raw(
        const char * const __restrict__ KV, char * const __restrict__ raw, const int stride_KV, const int k_VKQ_0) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int row_bytes = (D/QK8_0)*int(sizeof(block_q8_0));
    constexpr int chunk     = row_bytes % 16 == 0 ? 16 : 8;
    static_assert(row_bytes % chunk == 0, "bad q8_0 row");
    constexpr int cpr       = row_bytes/chunk;
    constexpr int nchunks   = nbatch_fa*cpr;
    const unsigned int raw_32 = ggml_cuda_cvta_generic_to_shared(raw);
#pragma unroll
    for (int ic0 = 0; ic0 < nchunks; ic0 += nwarps*warp_size) {
        const int ic = ic0 + threadIdx.y*warp_size + threadIdx.x;
        if (nchunks % (nwarps*warp_size) != 0 && ic >= nchunks) {
            break;
        }
        const int i = ic / cpr;
        const int c = ic % cpr;
        const unsigned int dst = raw_32 + i*fattn_i8_raw_row(D) + c*chunk;
        const char *       src = KV + int64_t(k_VKQ_0 + i)*stride_KV + c*chunk;
        if constexpr (chunk == 16) {
            cp_async_cg_16<64>(dst, src);
        } else {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" : : "r"(dst), "l"(src));
        }
    }
#else
    GGML_UNUSED_VARS(KV, raw, stride_KV, k_VKQ_0);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// the KQ tile of a warp (16 query columns, the keys i_KQ_0 .. of its share) from the int8 Q fragments (Q8, one per 32-value
// block; Qd[b] = d_q of its rows lane/4 and 8 + lane/4, in shared memory) and the int8 K tile: per block one s8 MMA per 8 keys (m16n8k32; 4 x m8n8k16 on
// Turing), then KQ += f32(isum) * (d_q * d_k). The int32 sum is converted with the 2^23 + 2^22 trick (the MMA's accumulator starts at
// the bits of 12582912.0f, |isum| <= 32*127*128 < 2^22, so the float bits of the result are 12582912 + isum exactly) or, on Blackwell,
// with I2F (faster there: integer and f32 pipes are unified)
template<int DKQ, int nbatch_fa, int np, typename T_C_KQ>
static __device__ __forceinline__ void flash_attn_ext_i8_kq(
        T_C_KQ * const __restrict__ KQ_C, const tile<16, 8, int> * const __restrict__ Q8, const float2 * const __restrict__ Qd,
        const char * const __restrict__ tile_K8, const float * const __restrict__ tile_dk) {
#ifdef TURING_MMA_AVAILABLE
    static_assert(T_C_KQ::I == 16 && T_C_KQ::J == 16, "INT8 KQ needs the 16 column KQ tiles");
    constexpr int stride_i = fattn_i8_stride(DKQ)/int(sizeof(int));
#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
    constexpr int   acc0  = 0;
#else
    constexpr int   acc0  = 0x4B400000; // 12582912.0f
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
#pragma unroll
    for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*16) {
        const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*16;
        T_C_KQ & C = KQ_C[i_KQ_00/(np*16)];
#pragma unroll
        for (int b = 0; b < DKQ/QK8_0; ++b) {
            // 16 keys x 32 int8: x[0] / x[2] = bytes 0..15 / 16..31 of keys 0..7, x[1] / x[3] the same of keys 8..15
            tile<16, 8, int> K_A;
            load_ldmatrix(K_A, (const int *) tile_K8 + i_KQ_0*stride_i + b*(QK8_0/int(sizeof(int))), stride_i);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                tile<8, 8, int> K_B;
                K_B.x[0] = K_A.x[h];
                K_B.x[1] = K_A.x[2 + h];
                tile<16, 8, int> S;
#pragma unroll
                for (int l = 0; l < S.ne; ++l) {
                    S.x[l] = acc0;
                }
                mma(S, Q8[b], K_B);
                const float2 dk = *(const float2 *) (tile_dk + b*nbatch_fa + i_KQ_0 + 8*h + 2*(threadIdx.x % 4));
#pragma unroll
                for (int l = 0; l < S.ne; ++l) {
                    // S: row (query column) (l/2)*8 + lane/4, column (key) 2*(lane%4) + l%2; C: keys 8h .. 8h + 7 at x[4h .. 4h + 3]
                    const float s = (l/2 == 0 ? Qd[b].x : Qd[b].y) * (l % 2 == 0 ? dk.x : dk.y);
#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
                    const float f = __int2float_rn(S.x[l]);
#else
                    const float f = __int_as_float(S.x[l]) - 12582912.0f;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
                    C.x[4*h + l] += f*s;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(KQ_C, Q8, Qd, tile_K8, tile_dk);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

// (fattn_kvq_i8r) a K tile of staged int8 rows (fattn_i8_stride(D) bytes, the same layout as the int8 tile) in 16 byte chunks:
// with cp.async (pipelined loads) or synchronously, rows past i_sup (oob_check) as zeros
template<int nwarps, int nbatch_fa, int D, bool oob_check, bool use_cp_async>
static __device__ __forceinline__ void flash_attn_ext_i8r_load_tile(
        const char * const __restrict__ K, char * const __restrict__ tile_K8, const int stride_K, const int k_VKQ_0, const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int row       = fattn_i8_stride(D);
    static_assert(row % 16 == 0, "bad int8 row");
    constexpr int cpr       = row/16;
    constexpr int nchunks   = nbatch_fa*cpr;
#pragma unroll
    for (int ic0 = 0; ic0 < nchunks; ic0 += nwarps*warp_size) {
        const int ic = ic0 + threadIdx.y*warp_size + threadIdx.x;
        if (nchunks % (nwarps*warp_size) != 0 && ic >= nchunks) {
            break;
        }
        const int i = ic / cpr;
        const int c = ic % cpr;
        char * const dst = tile_K8 + i*row + c*16;
        if constexpr (use_cp_async) {
#ifdef CP_ASYNC_AVAILABLE
            cp_async_cg_16<64>(ggml_cuda_cvta_generic_to_shared(dst), K + int64_t(k_VKQ_0 + i)*stride_K + c*16);
#else
            GGML_UNUSED_VARS(dst, K, stride_K, k_VKQ_0);
            NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
        } else {
            int4 v = make_int4(0, 0, 0, 0);
            if (!oob_check || i < i_sup) {
                v = *(const int4 *) (K + int64_t(k_VKQ_0 + i)*stride_K + c*16);
            }
            *(int4 *) dst = v;
        }
    }
    GGML_UNUSED(i_sup);
}

// (fattn_kvq_i8r) the KQ tile of a warp from the int8 Q fragments (Q8, one per 32-value block) and the int8 K tile: the s8 MMAs of all
// blocks of a row accumulate in one int32 sum per score (|sum| <= D*127*127 < 2^22 at D <= 256, so the 2^23 + 2^22 conversion of
// flash_attn_ext_i8_kq stays exact), then KQ += f32(sum) * (d_q * d_k) once. Qd: d_q of the rows lane/4 and 8 + lane/4; d_k is the f32
// behind the D quants of each K row
template<int DKQ, int nbatch_fa, int np, typename T_C_KQ>
static __device__ __forceinline__ void flash_attn_ext_i8r_kq(
        T_C_KQ * const __restrict__ KQ_C, const tile<16, 8, int> * const __restrict__ Q8, const float2 Qd,
        const char * const __restrict__ tile_K8) {
#ifdef TURING_MMA_AVAILABLE
    static_assert(T_C_KQ::I == 16 && T_C_KQ::J == 16, "INT8 KQ needs the 16 column KQ tiles");
    static_assert(DKQ*127*127 < (1 << 22), "the int32 sum of a row must stay below 2^22");
    constexpr int row      = fattn_i8_stride(DKQ);
    constexpr int stride_i = row/int(sizeof(int));
#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
    constexpr int   acc0  = 0;
#else
    constexpr int   acc0  = 0x4B400000; // 12582912.0f
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
#pragma unroll
    for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*16) {
        const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*16;
        T_C_KQ & C = KQ_C[i_KQ_00/(np*16)];
        tile<16, 8, int> S[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int l = 0; l < S[h].ne; ++l) {
                S[h].x[l] = acc0;
            }
        }
#pragma unroll
        for (int b = 0; b < DKQ/QK8_0; ++b) {
            // 16 keys x 32 int8: x[0] / x[2] = bytes 0..15 / 16..31 of keys 0..7, x[1] / x[3] the same of keys 8..15
            tile<16, 8, int> K_A;
            load_ldmatrix(K_A, (const int *) tile_K8 + i_KQ_0*stride_i + b*(QK8_0/int(sizeof(int))), stride_i);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                tile<8, 8, int> K_B;
                K_B.x[0] = K_A.x[h];
                K_B.x[1] = K_A.x[2 + h];
                mma(S[h], Q8[b], K_B);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int   k   = i_KQ_0 + 8*h + 2*(threadIdx.x % 4);
            const float dk0 = *(const float *) (tile_K8 + (k + 0)*row + DKQ);
            const float dk1 = *(const float *) (tile_K8 + (k + 1)*row + DKQ);
#pragma unroll
            for (int l = 0; l < S[h].ne; ++l) {
                // S: row (query column) (l/2)*8 + lane/4, column (key) 2*(lane%4) + l%2; C: keys 8h .. 8h + 7 at x[4h .. 4h + 3]
                const float s = (l/2 == 0 ? Qd.x : Qd.y) * (l % 2 == 0 ? dk0 : dk1);
#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
                const float f = __int2float_rn(S[h].x[l]);
#else
                const float f = __int_as_float(S[h].x[l]) - 12582912.0f;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
                C.x[4*h + l] += f*s;
            }
        }
    }
#else
    GGML_UNUSED_VARS(KQ_C, Q8, Qd, tile_K8);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int k_VKQ_0, const int i_sup, const int j0, const uint3 ne01,
        const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*warp_size && nbatch_fa % 8 == 0, "bad nbatch_fa");
        static_assert(!oob_check, "OOB check incompatible with cp_async");
        static_assert(!use_sparse, "sparse gather incompatible with cp_async");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = 8 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half), mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
        }
    } else if constexpr (oob_check || use_sparse) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                if constexpr (use_sparse) {
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = index >= 0 ? mask_h[int64_t(j_vram)*stride_mask + index] : half(-INFINITY);
                } else {
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[int64_t(j_vram)*stride_mask + k_VKQ_0 + i] : half(0.0f);
                }
            }
        }
    } else if constexpr (nbatch_fa < 2*warp_size) {
        constexpr int cols_per_warp = 2*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (warp_size/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*warp_size) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
            }
        }
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check, int q8_kv,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        char         * const __restrict__ raw_K,
        char         * const __restrict__ raw_V,
        T_B_KQ       * const __restrict__ Q_B,
        const tile<16, 8, int> * const __restrict__ Q8, // (fattn_kvq_i8) Q as int8 per 32-value block and its scales
        const float2 * const __restrict__ Qd,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int kb0_next, // the KV tile the multi-stage pipeline preloads for the next iteration
        const int k_VKQ_sup) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages  (DKQ, DV, ncols1, ncols2, use_sparse, q8_kv);

    constexpr bool swz = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols);
    constexpr int stride_tile_K = swz ? nbatch_K2 : nbatch_K2 + 4;
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : (swz ? nbatch_V2 : nbatch_V2 + 4);

    const int k_VKQ_0 = kb0 * nbatch_fa;
    // the int8 K tile and its scales over the f16 K tile (smaller: D + 16 + D/8 bytes per row instead of >= 2 D)
    constexpr bool i8  = fattn_kvq_i8(q8_kv);
    constexpr bool i8r = fattn_kvq_i8r(q8_kv); // staged int8 rows: loaded straight into the int8 tile, scales inside the rows
    char  * const tile_K8 = (char *) tile_K;
    float * const tile_dk = (float *) (tile_K8 + nbatch_fa*fattn_i8_stride(DKQ));
    GGML_UNUSED_VARS(tile_K8, tile_dk);
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    if constexpr (nstages > 1) {
        static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if constexpr (fattn_kvq_mode(q8_kv) == 2) {
            // the K tile's raw rows (preloaded) to f16 in the one f16 tile, then this tile's V rows in flight during KQ
            flash_attn_ext_q8_load_tile<stride_tile_K, nwarps, nbatch_fa, DKQ/2, false, false, fattn_kvq_k4(q8_kv)>
                (raw_K, tile_K, fattn_kvq_row_bytes(DKQ, fattn_kvq_k4(q8_kv)), 0, nbatch_fa, nullptr);
            flash_attn_ext_q8_load_raw<nwarps, nbatch_fa, fattn_kvq_row_bytes(DV, fattn_kvq_v4(q8_kv))>((const char *) V_h2, raw_V, stride_V, k_VKQ_0);
            __syncthreads();
        } else {
            if constexpr (i8 && !i8r) {
                // this tile's raw K rows (preloaded) to the int8 tile, this tile's V rows in flight during KQ
                flash_attn_ext_i8_load_tile<nwarps, nbatch_fa, DKQ, false, true>(raw_K, 0, 0, nbatch_fa, tile_K8, tile_dk);
            }
            flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (V_h2, tile_V, nbatch_V2, stride_V, k_VKQ_0, k_VKQ_sup, nullptr);
            if constexpr (i8 && !i8r) {
                __syncthreads();
            }
        }
    } else {
        // the sparse mask values are gathered per element, always load them synchronously
        constexpr bool use_cp_async = nstages == 1 && !use_sparse;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt*ncols1, ne01, indices);
        }
    }

    if constexpr (i8) {
#if defined(TURING_MMA_AVAILABLE)
        static_assert(cols_per_warp == 16 && Q_in_reg && nbatch_K2 == DKQ/2 && !V_is_K_view && !use_sparse, "bad INT8 KQ config");
        if constexpr (nstages <= 1) {
            if constexpr (i8r) {
                flash_attn_ext_i8r_load_tile<nwarps, nbatch_fa, DKQ, oob_check, false>
                    ((const char *) K_h2, tile_K8, stride_K, k_VKQ_0, k_VKQ_sup);
            } else {
                flash_attn_ext_i8_load_tile<nwarps, nbatch_fa, DKQ, oob_check, false>
                    ((const char *) K_h2, stride_K, k_VKQ_0, k_VKQ_sup, tile_K8, tile_dk);
            }
            __syncthreads();
        }
        if constexpr (i8r) {
            flash_attn_ext_i8r_kq<DKQ, nbatch_fa, np>(KQ_C, Q8, Qd[0], tile_K8);
        } else {
            flash_attn_ext_i8_kq<DKQ, nbatch_fa, np>(KQ_C, Q8, Qd, tile_K8, tile_dk);
        }
        if constexpr (nstages <= 1) {
            __syncthreads(); // tile_V == tile_K
        }
#else
        NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
    } else {
    // For MLA K and V have the same data.
    // Therefore, iterate over K in reverse and later re-use the data if possible.
#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;

        if constexpr (fattn_kvq_mode(q8_kv) == 1) {
            static_assert(nbatch_K2 == DKQ/2 && nstages == 0, "q8_0 K tiles are loaded whole and synchronously");
            flash_attn_ext_q8_load_tile<stride_tile_K, nwarps, nbatch_fa, DKQ/2, oob_check, use_sparse, fattn_kvq_k4(q8_kv)>
                ((const char *) K_h2, tile_K, stride_K, k_VKQ_0, k_VKQ_sup, indices);
            __syncthreads();
        } else if constexpr (nstages <= 1) {
            const int k0_diff = k0_stop - k0_start;
            constexpr bool use_cp_async = nstages == 1;
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (K_h2 + k0_start, tile_K, k0_diff, stride_K, k_VKQ_0, k_VKQ_sup, indices);
            if (use_cp_async) {
                cp_async_wait_all();
            }
            __syncthreads();
        }

        // Calculate tile of KQ:
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
                    load_ldmatrix_swizzled<stride_tile_K>(K_A, tile_K, i_KQ_0*stride_tile_K + k_KQ_0-k0_start);
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        } else {
            constexpr int stride_tile_Q = DKQ/2 + 4;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;

                    T_A_KQ K_A;
                    load_ldmatrix_swizzled<stride_tile_K>(K_A, tile_K, i_KQ_0*stride_tile_K + k_KQ_0-k0_start);

                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        }

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (ncols2 > 1 || mask_h) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        if (ncols2 > 1 || mask_h) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;

                // The mask is stored as 16 bit half values, loading them as 32 bit half2 values is preferred in terms of speed.
                // However, this is not possible for RDNA3 where 2 consecutive l indices are not consecutive in the mask memory layout.
#ifdef RDNA3
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_j(l);
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::J)].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
#else
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                    const float2 tmp = __half22float2(((const half2 *)tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
#endif // RDNA3
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#elif defined(AMD_MFMA_AVAILABLE)
            // MFMA: 4 threads per Q column (threadIdx.x % 16 == col, spaced by 16).
            constexpr int offset_first = 32;
            constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 16;
            constexpr int offset_last  = 16;
#else // Volta
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    if constexpr (nstages > 1) {
        static_assert(!use_sparse, "sparse gather not implemented for multi-stage loading");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if constexpr (fattn_kvq_mode(q8_kv) == 2) {
            // every warp is past KQ / softmax: the V rows to f16 over the K tile (one f16 tile), the next K rows in flight during VKQ
            flash_attn_ext_q8_load_tile<stride_tile_V, nwarps, nbatch_fa, DV/2, false, false, fattn_kvq_v4(q8_kv)>
                (raw_V, tile_V, fattn_kvq_row_bytes(DV, fattn_kvq_v4(q8_kv)), 0, nbatch_fa, nullptr);
        }
        if (!last_iter) {
            if (ncols2 > 1 || mask_h) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                    (mask_h, tile_mask, stride_mask, kb0_next*nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr);
            }
            if constexpr (fattn_kvq_mode(q8_kv) == 2) {
                flash_attn_ext_q8_load_raw<nwarps, nbatch_fa, fattn_kvq_row_bytes(DKQ, fattn_kvq_k4(q8_kv))>((const char *) K_h2, raw_K, stride_K, kb0_next*nbatch_fa);
            } else if constexpr (i8r) {
                flash_attn_ext_i8r_load_tile<nwarps, nbatch_fa, DKQ, false, true>
                    ((const char *) K_h2, tile_K8, stride_K, kb0_next*nbatch_fa, nbatch_fa);
            } else if constexpr (i8) {
                flash_attn_ext_i8_load_raw<nwarps, nbatch_fa, DKQ>((const char *) K_h2, raw_K, stride_K, kb0_next*nbatch_fa);
            } else {
                flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                    (K_h2, tile_K, nbatch_K2, stride_K, kb0_next*nbatch_fa, k_VKQ_sup, nullptr);
            }
        }
        if constexpr (fattn_kvq_mode(q8_kv) == 2) {
            __syncthreads();
        }
    }


    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;

        if constexpr (fattn_kvq_mode(q8_kv) == 1) {
            static_assert(2*nbatch_V2 == DV && !V_is_K_view && nstages == 0, "q8_0 V tiles are loaded whole and synchronously");
            flash_attn_ext_q8_load_tile<stride_tile_V, nwarps, nbatch_fa, DV/2, oob_check, use_sparse, fattn_kvq_v4(q8_kv)>
                ((const char *) V_h2, tile_V, stride_V, k_VKQ_0, k_VKQ_sup, indices);
            __syncthreads();
        } else if constexpr (nstages <= 1) {
            const int i0_diff = i0_stop - i0_start;
            if (!V_is_K_view || i0_stop > 2*nbatch_K2) {
                constexpr bool use_cp_async = nstages == 1;
                flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                    (V_h2 + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_0, k_VKQ_sup, indices);
                if (use_cp_async) {
                    cp_async_wait_all();
                }
                __syncthreads();
            }
        }
        const int tile_V_offset_i = !V_is_K_view || i0_stop > 2*nbatch_K2 ? 0 : i0_start/2;

#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                load_ldmatrix_trans_swizzled<stride_tile_V>(A, tile_V, tile_V_offset_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major.
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
#else
                    // swap A and B for CUDA.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                load_ldmatrix_swizzled<stride_tile_V>(A, tile_V, tile_V_offset_i + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, Q8, Qd, VKQ_C, KQ_max, KQ_rowsum, kb0, kb0_next);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

#if defined(TURING_MMA_AVAILABLE)
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
// If there are only 8 columns, use thinner B tiles to avoid wasting compute:
template<int DKQ> struct mma_tile_sizes<DKQ, 8> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
// For very large head sizes, use thinner B tiles to reduce register pressure:
template<> struct mma_tile_sizes<256, 16> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
template<> struct mma_tile_sizes<512, 16> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
template<> struct mma_tile_sizes<512, 32> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#ifdef GGML_HIP_FA_RDNA3_D256_FLOAT_VKQ
// head size 256: float VKQ accumulators (more precise than half2 over long rows), see ggml_cuda_fattn_mma_get_config_rdna3
template<int ncols> struct mma_tile_sizes<256, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // GGML_HIP_FA_RDNA3_D256_FLOAT_VKQ
#else
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR_SCRAMBLED>; // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16, 16, float>; // column-major
};
#else // Volta
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_KQ  = tile<32,  8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile< 8,  4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
    using T_B_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

// the KV loop of flash_attn_ext_f16_process_tile over KV tiles kb0_start .. kb0_stop - 1 of one source (K / V / mask of
// ne11 cells, in-place code q8_kv), with the softmax state (KQ_max, KQ_rowsum, VKQ_C) carried in and out: a two-source node runs it
// once per source, so its cells are processed exactly as those of one source of the concatenated cells (raw rows packed at the row
// size of this source's type; the raw buffers of the shared memory layout fit the larger of the two)
template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup,
    bool is_fixup, int q8_kv, typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_f16_kv_loop(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int ne11,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        char         * const __restrict__ raw_K,
        char         * const __restrict__ raw_V,
        T_B_KQ       * const __restrict__ Q_B,
        const tile<16, 8, int> * const __restrict__ Q8, // (fattn_kvq_i8) Q as int8 per 32-value block and its scales
        const float2 * const __restrict__ Qd,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0_start,
        const int kb0_stop,
        const int kb0_max = 1 << 30, // (rotate) the KV tiles >= kb0_max are fully masked and skipped
        const int iter_k  = 1 << 30, // (rotate) the KV tiles of the whole loop: tiles kb0_stop - iter_k .. wrap around to 0
        const bool rotate = false) {
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, use_sparse, q8_kv);
    constexpr bool swz             = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols);
    constexpr int  stride_tile_K   = swz ? nbatch_K2 : nbatch_K2 + 4;
    GGML_UNUSED(stride_tile_K);

    // rotated K/V tile order (multi-stage loading only, see flash_attn_ext_f16): the tiles kb0_start .. kb0_stop - 1 taken modulo
    // iter_k, the fully masked tiles >= kb0_max skipped; at least one tile is processed, as without the rotation
    const bool rot      = nstages > 1 && rotate;
    const int  nkb_pre  = rot ? max(0, min(kb0_stop, kb0_max) - kb0_start) : 0;
    const int  nkb_post = rot ? max(0, min(kb0_stop - iter_k, kb0_max))    : 0;
    const int  nkb      = max(1, nkb_pre + nkb_post);

    int kb0 = rot && nkb_pre == 0 && nkb_post > 0 ? 0 : kb0_start;

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        static_assert(!use_sparse, "sparse gather not implemented for multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline");
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        constexpr int  k_VKQ_sup    = nbatch_fa;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (mask_h, tile_mask, stride_mask, kb0*nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr);
        }
        if constexpr (fattn_kvq_mode(q8_kv) == 2) {
            flash_attn_ext_q8_load_raw<nwarps, nbatch_fa, fattn_kvq_row_bytes(DKQ, fattn_kvq_k4(q8_kv))>((const char *) K_h2, raw_K, stride_K, kb0*nbatch_fa);
        } else if constexpr (fattn_kvq_i8r(q8_kv)) {
            flash_attn_ext_i8r_load_tile<nwarps, nbatch_fa, DKQ, false, true>
                ((const char *) K_h2, (char *) tile_K, stride_K, kb0*nbatch_fa, nbatch_fa);
        } else if constexpr (fattn_kvq_i8(q8_kv)) {
            flash_attn_ext_i8_load_raw<nwarps, nbatch_fa, DKQ>((const char *) K_h2, raw_K, stride_K, kb0*nbatch_fa);
        } else {
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (K_h2, tile_K, nbatch_K2, stride_K, kb0*nbatch_fa, k_VKQ_sup, nullptr);
        }
    }

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
    if constexpr (nstages > 1) {
      if (rot) {
        constexpr bool oob_check = false;
        constexpr int  k_VKQ_sup = nbatch_fa;
        for (int ikb = 1; ikb < nkb; ++ikb) {
            const int kb0_next = kb0 + 1 == kb0_max ? 0 : kb0 + 1;
            constexpr bool last_iter = false;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check, q8_kv,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, kb0_next, k_VKQ_sup);
            kb0 = kb0_next;
        }
        constexpr bool last_iter = true;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check, q8_kv,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, kb0, k_VKQ_sup);
        return;
      }
    }
    if constexpr (ncols2 == 1 || use_sparse) {
        constexpr bool oob_check = true;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check, q8_kv,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, kb0 + 1, k_VKQ_sup);
        }
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check, q8_kv,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, kb0, k_VKQ_sup);
    } else {
        constexpr bool oob_check = false;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check, q8_kv,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, kb0 + 1, k_VKQ_sup);
        }
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check, q8_kv,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, kb0, k_VKQ_sup);
    }

}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup, int q8_kv, int q8_kv2>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop,
        // the second source of a two-source node (q8_kv2 >= 0, see ggml_cuda_fattn_dev_len), its first tile is it1
        const half2  * const __restrict__ K2_h2,
        const half2  * const __restrict__ V2_h2,
        const half   * const __restrict__ mask2_h,
        const int ne11_2,
        const int stride_K2,
        const int stride_V2,
        const int stride_mask2,
        const int it1,
        const uint32_t * const __restrict__ ring_b, // bit t: the second source's tile t holds a visible cell of this query tile (nullptr: all)
        const int2 body_r, // [x, y): the first source's tiles that can hold a visible cell of this query tile
        // one source: the KV tiles kb0_start .. kb0_stop - 1 wrap around at iter_k, the tiles >= kb0_max are skipped
        const int kb0_max, const int iter_k, const bool rotate) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, use_sparse, q8_kv);

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr bool swz = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols);
    constexpr int stride_tile_Q = DKQ/2 + 4;
    constexpr int stride_tile_K = swz ? nbatch_K2 : nbatch_K2 + 4;
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : (swz ? nbatch_V2 : nbatch_V2 + 4);
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
    constexpr bool raw_kv = fattn_kvq_mode(q8_kv) == 2;
    half2 * tile_V    =           nstages > 1 && !raw_kv ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (nstages > 1 && !raw_kv ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);
    // (q8_kv 2): one f16 tile (K, then V), the raw q8_0 (or q4_0) rows of K and V behind it, the mask behind them
    // (a two-source node's sources share the mode; the raw rows are laid out at the larger row of the two)
    constexpr int q8_kv2_l  = q8_kv2 < 0 ? q8_kv : q8_kv2;
    static_assert(fattn_kvq_mode(q8_kv2_l) == fattn_kvq_mode(q8_kv), "the sources of a two-source node share the load mode");
    constexpr int raw_row_K1 = fattn_kvq_row_bytes(DKQ, fattn_kvq_k4(q8_kv)), raw_row_K2 = fattn_kvq_row_bytes(DKQ, fattn_kvq_k4(q8_kv2_l));
    constexpr int raw_row_V1 = fattn_kvq_row_bytes(DV,  fattn_kvq_v4(q8_kv)), raw_row_V2 = fattn_kvq_row_bytes(DV,  fattn_kvq_v4(q8_kv2_l));
    constexpr int raw_row_K  = raw_row_K1 > raw_row_K2 ? raw_row_K1 : raw_row_K2;
    constexpr int raw_row_V  = raw_row_V1 > raw_row_V2 ? raw_row_V1 : raw_row_V2;
    char * raw_K = (char *) tile_mask;
    char * raw_V = raw_K + (raw_kv ? nbatch_fa*raw_row_K : 0);
    if constexpr (raw_kv) {
        tile_mask = (half *) (raw_V + nbatch_fa*raw_row_V);
    }
    // the int8 K tile and its scales lie over the f16 K tile (flash_attn_ext_f16_iter), V and the mask where the f16
    // kernel has them; the raw K rows of the pipelined loads behind the mask
    constexpr bool i8 = fattn_kvq_i8(q8_kv);
    if constexpr (i8 && nstages > 1) {
        raw_K = (char *) tile_mask + ncols1*(nbatch_fa + 8)*int(sizeof(half));
    }
    // the Q scales behind them for the whole tile (fattn_i8_qd_bytes): per warp column group g and row r < 8 the scales of rows r and
    // r + 8 of each block as a float2, rows 2*nblk + 2 floats apart (one 64 bit load, no bank conflicts)
    float * const tile_Qd = (float *) ((char *) tile_mask + ncols1*(nbatch_fa + 8)*int(sizeof(half)) +
        (i8 && nstages > 1 ? nbatch_fa*fattn_i8_raw_row(DKQ) : 0));
    GGML_UNUSED(tile_Qd);

    T_B_KQ    Q_B[(Q_in_reg && !i8 ? DKQ/(2*T_B_KQ::J) : 1)];
    tile<16, 8, int> Q8[i8 ? DKQ/QK8_0 : 1];
    const float2 *   Qd = nullptr;
    // the Q scales in registers on Blackwell (fast there: no spill), read from shared memory elsewhere (sm_75 / 80 / 89 spill with them in
    // registers)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
    constexpr bool qd_regs = i8;
#else
    constexpr bool qd_regs = false;
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
    float2 Qd_r[qd_regs ? DKQ/QK8_0 : 1];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    T_C_VKQ VKQ_C[std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]> ? DV/T_C_VKQ::J : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_MFMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/T_C_VKQ::J];
#elif defined(AMD_WMMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    if constexpr (fattn_kvq_i8r(q8_kv)) {
        // the staged int8 Q rows (fattn_i8_stride(DKQ) bytes, their scale behind the quants) as they are into the int8 tile, rows
        // outside the tile's queries / heads as zeros; then each warp keeps the A fragments of its 16 query columns and the scales of
        // its rows lane/4 and 8 + lane/4 in registers
#if defined(TURING_MMA_AVAILABLE)
        static_assert(Q_in_reg && cols_per_warp == 16, "bad INT8 KQ config");
#else
        NO_DEVICE_CODE; // the INT8 kernels exit at entry below Turing (flash_attn_ext_f16)
#endif // defined(TURING_MMA_AVAILABLE)
        constexpr int nblk = DKQ/QK8_1;
        constexpr int row  = fattn_i8_stride(DKQ);
        constexpr int cpr  = row/16;
        char       * tile_Q8 = (char *) tile_Q;
        const char * Q8g     = (const char *) Q_f2;
        for (int ic0 = 0; ic0 < ncols*cpr; ic0 += nwarps*warp_size) {
            const int ic = ic0 + threadIdx.y*warp_size + threadIdx.x;
            if ((ncols*cpr) % (nwarps*warp_size) != 0 && ic >= ncols*cpr) {
                break;
            }
            const int jc  = ic / cpr;
            const int c16 = ic % cpr;
            const int j   = jc / ncols2;
            const int c   = jc % ncols2;
            int4 v = make_int4(0, 0, 0, 0);
            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
                v = *(const int4 *) (Q8g + (int64_t(jt*ncols1 + j)*stride_Q1 + int64_t(c)*stride_Q2)*int(sizeof(float2)) + c16*16);
            }
            *(int4 *) (tile_Q8 + jc*row + c16*16) = v;
        }
        __syncthreads();
        const int j0 = (threadIdx.y / np) * cols_per_warp;
#pragma unroll
        for (int b = 0; b < nblk; ++b) {
            load_ldmatrix(Q8[b], (const int *) tile_Q8 + j0*(row/4) + b*(QK8_1/4), row/4);
        }
        Qd_r[0] = make_float2(*(const float *) (tile_Q8 + (j0 +     threadIdx.x/4)*row + DKQ),
                              *(const float *) (tile_Q8 + (j0 + 8 + threadIdx.x/4)*row + DKQ));
        Qd = Qd_r;
        __syncthreads();
    } else if constexpr (i8) {
        // Q * scale quantised per 32 values as the vector kernel does (quantize_q8_1_to_shared: d = amax/127 in f32,
        // q = roundf(v/d)), 8 lanes per block, into an int8 tile (rows fattn_i8_stride(DKQ) bytes) + the scales behind it; then each
        // warp keeps the A fragments of its 16 query columns (one per block) and their scales in registers
#if defined(TURING_MMA_AVAILABLE)
        static_assert(Q_in_reg && cols_per_warp == 16, "bad INT8 KQ config");
#else
        NO_DEVICE_CODE; // the INT8 kernels exit at entry below Turing (flash_attn_ext_f16)
#endif // defined(TURING_MMA_AVAILABLE)
        constexpr int nblk      = DKQ/QK8_1;
        constexpr int stride_q8 = fattn_i8_stride(DKQ);
        char  * tile_Q8 = (char *) tile_Q;
        static_assert((ncols*nblk) % (nwarps*(warp_size/8)) == 0, "bad Q quantisation loop");
        // unrolled, all its Q loads are hoisted and the kernel spills on sm_75 / 80 / 89 (444 bytes); on Blackwell unrolled is faster
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
#pragma unroll
#else
#pragma unroll 1
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
        for (int jb0 = 0; jb0 < ncols*nblk; jb0 += nwarps*(warp_size/8)) {
            const int jb = jb0 + threadIdx.y*(warp_size/8) + threadIdx.x/8;
            const int jc = jb / nblk;
            const int b  = jb % nblk;
            const int j  = jc / ncols2;
            const int c  = jc % ncols2;

            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
                const float2 * q = Q_f2 + (jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + b*(QK8_1/2) + (threadIdx.x % 8)*2;
                const float2 t0 = q[0];
                const float2 t1 = q[1];
                v[0] = scale*t0.x;
                v[1] = scale*t0.y;
                v[2] = scale*t1.x;
                v[3] = scale*t1.y;
            }
            float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
            for (int mask = 4; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, warp_size));
            }
            const float d = amax / 127;
            uint32_t q32 = 0;
            if (d != 0.0f) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    q32 |= (uint32_t(int(roundf(v[l] / d))) & 0xFFu) << (8*l);
                }
            }
            *(uint32_t *) (tile_Q8 + jc*stride_q8 + b*QK8_1 + (threadIdx.x % 8)*4) = q32;
            if (threadIdx.x % 8 == 0) {
                tile_Qd[((jc/16)*8 + jc % 8)*(2*nblk + 2) + 2*b + (jc % 16)/8] = d;
            }
        }
        __syncthreads();
        const int j0 = (threadIdx.y / np) * cols_per_warp;
#pragma unroll
        for (int b = 0; b < nblk; ++b) {
            load_ldmatrix(Q8[b], (const int *) tile_Q8 + j0*(stride_q8/4) + b*(QK8_1/4), stride_q8/4);
        }
        Qd = (const float2 *) (tile_Qd + ((j0/16)*8 + threadIdx.x/4)*(2*nblk + 2));
        if constexpr (qd_regs) {
#pragma unroll
            for (int b = 0; b < nblk; ++b) {
                Qd_r[b] = Qd[b];
            }
            Qd = Qd_r;
        }
        __syncthreads();
    } else {
    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();
    }

    // a two-source node (q8_kv2 >= 0): tiles 0 .. it1 - 1 from the first source, it1 .. from the second
    if constexpr (q8_kv2 < 0) {
        flash_attn_ext_f16_kv_loop
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, q8_kv,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap, ne01, ne02, ne11, stride_K, stride_V,
             stride_mask, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C, KQ_max, KQ_rowsum, jt, kb0_start, kb0_stop,
             kb0_max, iter_k, rotate);
    } else {
        const int kb1_start = max(kb0_start, body_r.x);
        const int kb1_stop  = min(min(kb0_stop, it1), body_r.y);
        const int kb2_start = max(kb0_start, it1) - it1;
        const int kb2_stop  = kb0_stop - it1;
        if (kb1_start < kb1_stop) {
            flash_attn_ext_f16_kv_loop
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, q8_kv,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap, ne01, ne02, ne11, stride_K, stride_V,
                 stride_mask, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C, KQ_max, KQ_rowsum, jt, kb1_start, kb1_stop);
        }
        // the second source in runs of visible tiles (the ring's slots are not in position order, so the visible tiles of a query
        // tile need not be one interval); the bits are the same for the whole block, so every thread walks the same runs
        bool synced = !(kb1_start < kb1_stop);
        int k = kb2_start;
        while (k < kb2_stop) {
            const int rs = ring_b ? flash_attn_src2_next_bit<true> (ring_b, k,  kb2_stop) : k;
            if (rs >= kb2_stop) {
                break;
            }
            const int re = ring_b ? flash_attn_src2_next_bit<false>(ring_b, rs, kb2_stop) : kb2_stop;
            if (!synced) {
                __syncthreads(); // every warp is done with the previous run's last tiles before the next run's preload
            }
            synced = false;
            flash_attn_ext_f16_kv_loop
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, q8_kv2 < 0 ? 0 : q8_kv2,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K2_h2, V2_h2, mask2_h, nullptr, dstk, dstk_fixup, scale, slope, logit_softcap, ne01, ne02, ne11_2, stride_K2, stride_V2,
                 stride_mask2, tile_Q, tile_K, tile_V, tile_mask, raw_K, raw_V, Q_B, Q8, Qd, VKQ_C, KQ_max, KQ_rowsum, jt, rs, re);
            k = re;
        }
    }

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if constexpr (nstages > 1 && nwarps*cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#elif defined(AMD_MFMA_AVAILABLE)
        // The partial sums are spread across 4 threads (wavefront64, 16 cols).
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchronization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = (threadIdx.y/np)*cols_per_warp + (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col));
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if constexpr (swz) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if constexpr (swz) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1) {
        constexpr int nmeta = np*cols_per_warp >= warp_size ? np*cols_per_warp/warp_size : 1;

        float KQ_cmn;
        float KQ_cms[nmeta];
        float KQ_crs;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < warp_size ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;

        if (threadIdx.y % np == 0) {
            // Combine the meta data for parallel warps via shared memory.
            float2 meta[nmeta];
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
            }

            KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, warp_size));
                }
            }

#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
            }

            KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_crs += KQ_cms[imeta]*meta[imeta].y;
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, warp_size);
                }
            }
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // Write back combined meta data:
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                if (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp) {
                    // Combined KQ max scale + rowsum.
                    meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
                }
            }

            // Combined KQ max + rowsum.
            static_assert(cols_per_warp <= warp_size);
            if (needs_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
            if (is_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>, "bad VKQ type");
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
            if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
                if constexpr (T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR) {
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                        for (int l = 0; l < T_C_VKQ::ne; ++l) {
                            const int j = j0 + T_C_VKQ::get_i(l);
                            const int k = k1 + T_C_VKQ::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                        }
                    }
                } else {
                    static_assert(T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR_SCRAMBLED, "bad T_C_VKQ data layout");
                    using T_C_VKQ_us = tile<T_C_VKQ::I, T_C_VKQ::J, half2, DATA_LAYOUT_I_MAJOR>; // us == unscrambled
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
                        const T_C_VKQ_us VKQ_C_us = unscramble(VKQ_C[(k00 + k1)/T_C_VKQ::J]);
#pragma unroll
                        for (int l = 0; l < T_C_VKQ_us::ne; ++l) {
                            const int j = j0 + T_C_VKQ_us::get_i(l);
                            const int k = k1 + T_C_VKQ_us::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C_us.x[l];
                        }
                    }
                }
            } else {
                static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
                half * tile_Q_h = (half *) tile_Q;
#pragma unroll
                for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J/2) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        const int j = j0 + T_C_VKQ::get_i(l);
                        const int k = 2*k1 + T_C_VKQ::get_j(l);

                        tile_Q_h[j*(2*tile_stride) + k] = VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && ((ncols1 > 1 && jt*ncols1 + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1 || nbatch_combine != DKQ/2) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop, K2_h2, V2_h2, mask2_h, ne11_2, stride_K2, stride_V2, stride_mask2, it1, ring_b, body_r,
        kb0_max, iter_k, rotate);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static constexpr __host__ __device__ bool ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(
        const int DKQ, const int DV, const int ncols1, const int ncols2) {
    return (DKQ == 512 && DV == 512 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 576 && DV == 512 && ncols1 == 1 && ncols2 == 16) ||
           (DKQ == 256 && DV == 256 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 256 && DV == 256 && ncols1 == 8 && ncols2 == 8);
}

// head sizes with a q8_0 kernel (the tiles of these hold whole rows: nbatch_K2 = DKQ/2, nbatch_V2 = DV/2 in every config)
// (and head size 64, for q8_0 and q4_0)
static constexpr __host__ __device__ bool ggml_cuda_flash_attn_ext_mma_q8_may_use(
        const int DKQ, const int DV, const bool use_logit_softcap, const bool V_is_K_view) {
    return DKQ == DV && (DKQ == 64 || DKQ == 128 || DKQ == 256) && !use_logit_softcap && !V_is_K_view;
}

// the K / V type bits (fattn_kvq_k4 / fattn_kvq_v4) of the in-place kernels compiled for a tile config: q8_0 K / V at
// every config, the pairs with a q4_0 K or V only where ncols1 <= 16 (at most 16 query rows: the configs a verify window
// can pick, ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1; wider ones never take the in-place path by default)
static constexpr __host__ __device__ bool ggml_cuda_flash_attn_ext_mma_kvq_compiled(const int ncols1, const int kvq_types) {
    return kvq_types == 0 || ncols1 <= 16;
}

// q8_kv != 0: K and V are q8_0 (the cache itself, read in place, see flash_attn_ext_q8_load_tile), 1 = synchronous
// tile loads, 2 = the raw rows through the cp.async pipeline (16 byte aligned rows, f16 kernel with 2 stages, no sparse gather);
// nb11 / nb21 are then the byte strides of the q8_0 rows
// q8_kv2 >= 0: a two-source node (ggml_flash_attn_ext_set_src2), the second source (dev_len.K2 / V2 / mask2, in-place code
// q8_kv2, the load mode of q8_kv) continues the KV tiles of the first at tile dev_len.it1
template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool use_sparse, int q8_kv, int q8_kv2 = -1>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_f16(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const ggml_cuda_fattn_dev_len dev_len) {
    ggml_cuda_pdl_sync(); // TODO optimize placement
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    const char * GGML_CUDA_RESTRICT Q              = Q_ptr;
    const char * GGML_CUDA_RESTRICT K              = K_ptr;
    const char * GGML_CUDA_RESTRICT V              = V_ptr;
    const char * GGML_CUDA_RESTRICT mask           = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks          = sinks_ptr;
    // sparse: one index list per (sequence, query tile), the live count of each list follows the lists
    const int  * GGML_CUDA_RESTRICT sparse_indices = use_sparse ? KV_max_ptr : nullptr;
    const int  * GGML_CUDA_RESTRICT KV_max         = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst            = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta       = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512)) {
        NO_DEVICE_CODE;
        return;
    }
    if (DKQ == 192 && ncols2 != 8 && ncols2 != 16) {
        NO_DEVICE_CODE;
        return;
    }

    if (!ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2) && use_sparse) {
        NO_DEVICE_CODE;
        return;
    }
    if (q8_kv && !ggml_cuda_flash_attn_ext_mma_q8_may_use(DKQ, DV, use_logit_softcap, V_is_K_view)) {
        NO_DEVICE_CODE;
        return;
    }
    if (fattn_kvq_mode(q8_kv) == 2 && (use_sparse || DKQ != 256 || ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, use_sparse, q8_kv) != 2)) {
        NO_DEVICE_CODE;
        return;
    }
    // INT8 Q*K^T kernels exist for the configs of ggml_cuda_flash_attn_ext_mma_i8_compiled, plain (one source, no
    // sparse gather, no logit softcap) only
    if (fattn_kvq_i8(q8_kv) && (fattn_kvq_mode(q8_kv) != 0 || !ggml_cuda_flash_attn_ext_mma_i8_compiled(DKQ, DV, ncols1, ncols2) ||
            use_sparse || use_logit_softcap || V_is_K_view || q8_kv2 >= 0)) {
        NO_DEVICE_CODE;
        return;
    }
#if !defined(TURING_MMA_AVAILABLE) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    if (q8_kv) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(TURING_MMA_AVAILABLE) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 256) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

#if defined(AMD_MFMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || (DKQ > 256 && ncols1*ncols2 < 32)) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_MFMA_AVAILABLE)

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / warp_size;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = fattn_kvq_k_raw(q8_kv) ? nb11 : nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = V_is_K_view ? stride_K : (fattn_kvq_v_raw(q8_kv) ? nb21 : nb21 / sizeof(half2));

    // a padded node with its live KV length on the device (see ggml_cuda_fattn_dev_len): the loop bound and the stream-k
    // blocks of the launch at that length; the blocks past them exit. The launched grid (gridDim.x) still lays out the fixup buffer
    int ne11_live = ne11;
    int nblocks   = gridDim.x;
    if (!use_sparse && dev_len.kv_len != nullptr) {
        ne11_live = *dev_len.kv_len;
        nblocks   = dev_len.stream_k ? ggml_cuda_fattn_stream_k_blocks((ne11_live + dev_len.nbatch_fa - 1) / dev_len.nbatch_fa,
            dev_len.ntiles_dst, dev_len.max_blocks, dev_len.ntiles_dst_ref) : dev_len.ntiles_dst;
        if ((int) blockIdx.x >= nblocks) {
            return;
        }
    }

    // a two-source node loops over the tiles of both sources (no device length, host-checked)
    const int iter_k     = q8_kv2 >= 0 ? dev_len.it1 + dev_len.n_kv2 / nbatch_fa : (ne11_live + (nbatch_fa - 1)) / nbatch_fa;
    const int stride_K2    = q8_kv2 > 0 ? dev_len.nb11_2 : dev_len.nb11_2 / int(sizeof(half2));
    const int stride_V2    = V_is_K_view ? stride_K2 : (q8_kv2 > 0 ? dev_len.nb21_2 : dev_len.nb21_2 / int(sizeof(half2)));
    const int stride_mask2 = dev_len.nb31_2 / int(sizeof(half));
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    if (use_sparse) {
        KV_max = KV_max_ptr + int64_t(iter_j)*ne33*ne11;
    }

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / nblocks;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / nblocks;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    // Rotated K/V tile order (GGML_CUDA_FATTN_KV_ROTATE, dev_len.kv_rotate): position kbc of the continuous index takes KV tile
    // (kbc - kbc_tile) % iter_k instead of kbc % iter_k, kbc_tile = the start of the block that started the output tile, so the
    // blocks that run at the same time read the same KV tiles and share them in L2. Only with multi-stage loading (else every block
    // waits for the same KV data) and one KV source. Each block still covers a contiguous (cyclic) run of a tile's KV tiles, so the
    // stream-k fixup is unchanged; the partial sums are grouped differently (not bit-identical to the plain order).
    constexpr bool rotate_cfg = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, use_sparse, q8_kv) > 1 && !use_sparse && q8_kv2 < 0;
    const bool rotate = rotate_cfg && dev_len.kv_rotate != 0;
    const int kbc0     = kbc;
    int       kbc_tile = kbc;
    if (rotate && kb0_start != 0) {
        // the block that started the tile: the inverse of the kbc formula above
        const int64_t nkbc    = int64_t(iter_k)*iter_j*iter_z_gqa*ne12*ne03;
        const int64_t b_first = ((kbc - kb0_start + 1)*int64_t(nblocks) - 1) / nkbc;
        kbc_tile = b_first*nkbc / nblocks;
    }

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

        const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
        const half2 * K2_h2   = q8_kv2 >= 0 ? (const half2 *) (dev_len.K2 + dev_len.nb13_2*sequence + dev_len.nb12_2*z_KV) : nullptr;
        const half2 * V2_h2   = q8_kv2 >= 0 ? (V_is_K_view ? K2_h2 : (const half2 *) (dev_len.V2 + dev_len.nb23_2*sequence + dev_len.nb22_2*z_KV)) : nullptr;
        const half  * mask2_h = q8_kv2 >= 0 ? (const half *) (dev_len.mask2 + dev_len.nb33_2*(sequence % ne33)) : nullptr;
        const uint32_t * ring_b = q8_kv2 >= 0 && dev_len.ring_bits ? dev_len.ring_bits + size_t((sequence % ne33)*iter_j + jt)*dev_len.ring_nw : nullptr; const int2 body_r = q8_kv2 >= 0 && dev_len.body_range ? dev_len.body_range[(sequence % ne33)*iter_j + jt] : make_int2(0, 1 << 30);
        const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
        const int32_t * indices = use_sparse ? sparse_indices + (int64_t(sequence % ne33)*iter_j + jt)*ne11 : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (use_sparse) {
            kb0_stop = min(kb0_stop, (KV_max[(sequence % ne33)*iter_j + jt] + nbatch_fa - 1) / nbatch_fa);
        } else if (KV_max && !rotate) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
        // rotated: kb0_stop is not cut, the kv loop skips the fully masked KV tiles >= kb0_max
        const int kb0_max = rotate && KV_max ? KV_max[sequence*iter_j + jt] / nbatch_fa : iter_k;
        const int kb0_rot = rotate ? (kbc - kbc_tile) % iter_k : kb0_start;
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, q8_kv, q8_kv2>
                (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11_live, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_rot, kb0_rot + kb0_stop - kb0_start,
                 K2_h2, V2_h2, mask2_h, dev_len.n_kv2, stride_K2, stride_V2, stride_mask2, dev_len.it1, ring_b, body_r, kb0_max, iter_k, rotate);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, q8_kv, q8_kv2>
                (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11_live, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_rot, kb0_rot + kb0_stop - kb0_start,
                 K2_h2, V2_h2, mask2_h, dev_len.n_kv2, stride_K2, stride_V2, stride_mask2, dev_len.it1, ring_b, body_r, kb0_max, iter_k, rotate);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;
        kbc_tile = kbc0;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

    const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
    const half2 * K2_h2   = q8_kv2 >= 0 ? (const half2 *) (dev_len.K2 + dev_len.nb13_2*sequence + dev_len.nb12_2*z_KV) : nullptr;
    const half2 * V2_h2   = q8_kv2 >= 0 ? (V_is_K_view ? K2_h2 : (const half2 *) (dev_len.V2 + dev_len.nb23_2*sequence + dev_len.nb22_2*z_KV)) : nullptr;
    const half  * mask2_h = q8_kv2 >= 0 ? (const half *) (dev_len.mask2 + dev_len.nb33_2*(sequence % ne33)) : nullptr;
    const uint32_t * ring_b = q8_kv2 >= 0 && dev_len.ring_bits ? dev_len.ring_bits + size_t((sequence % ne33)*iter_j + jt)*dev_len.ring_nw : nullptr; const int2 body_r = q8_kv2 >= 0 && dev_len.body_range ? dev_len.body_range[(sequence % ne33)*iter_j + jt] : make_int2(0, 1 << 30);
    const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
    const int32_t * indices = use_sparse ? sparse_indices + (int64_t(sequence % ne33)*iter_j + jt)*ne11 : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (use_sparse) {
        kb0_stop = min(kb0_stop, (KV_max[(sequence % ne33)*iter_j + jt] + nbatch_fa - 1) / nbatch_fa);
    } else if (KV_max && !rotate) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }
    const int kb0_max = rotate && KV_max ? KV_max[sequence*iter_j + jt] / nbatch_fa : iter_k;
    const int kb0_rot = rotate ? (kbc - kbc_tile) % iter_k : kb0_start;

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, q8_kv, q8_kv2>
        (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11_live, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_rot, kb0_rot + kb0_stop - kb0_start,
                 K2_h2, V2_h2, mask2_h, dev_len.n_kv2, stride_K2, stride_V2, stride_mask2, dev_len.it1, ring_b, body_r, kb0_max, iter_k, rotate);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33, dev_len);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2);
// the MMA kernel reads this node's q8_0 K / V in place (no staged f16 copy); fattn.cu
bool ggml_cuda_flash_attn_ext_mma_q8_inplace(const int device, const ggml_tensor * dst);
// 0 = staged, else the in-place code (mode bits 1, K / V type bits fattn_kvq_k4 / fattn_kvq_v4) for q8_0 / q4_0 K and V;
// for_alloc: the answer for the graph allocation (the f16 copy is reserved unless every launch of the node reads in place); fattn.cu
int ggml_cuda_flash_attn_ext_mma_kvq_inplace(const int device, const ggml_tensor * dst, const bool for_alloc);
// a two-source node's in-place codes: 0 = both sources staged to f16, else code1 | code2 << 8 (fattn.cu)
int ggml_cuda_flash_attn_ext_src2_codes(const int device, const ggml_tensor * dst, const bool for_alloc);
// the MMA kernel computes this node's Q*K^T in INT8 over its q8_0 K cache (GGML_CUDA_FATTN_I8); fattn.cu
bool ggml_cuda_flash_attn_ext_mma_i8(const int device, const ggml_tensor * dst);
// the MMA kernel computes this node's Q*K^T in INT8 over its f16 K cache, Q and K staged as int8 rows with one scale each
// (GGML_CUDA_FATTN_I8_F16): 0 no, else 1 + the rotation mode of Q and K (Walsh-Hadamard) before the quantisation (0 never, 1 always,
// 2 per KV head whose rows have outliers); fattn.cu
int ggml_cuda_flash_attn_ext_mma_i8_f16(const int device, const ggml_tensor * dst);
static inline bool ggml_cuda_fattn_q8_no_data(const ggml_tensor * t) {
    return ggml_cuda_fattn_probe_cur != nullptr || t->data == nullptr;
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = DKQ == 576; // Guaranteed by the kernel selection logic in fattn.cu

    // KV tile strides must match flash_attn_ext_f16_iter / _process_tile.
    const bool swz = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols, cc);
    const int stride_tile_K = swz ? nbatch_K2 : nbatch_K2 + 4;
    const int stride_tile_V = V_is_K_view ? stride_tile_K : (swz ? nbatch_V2 : nbatch_V2 + 4);
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K,  stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (stride_tile_K + stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);
    // the shared memory whose occupancy the sparse launch keeps (0: its own), see GGML_CUDA_FA_SPARSE_SMEM_FIX below
    size_t sparse_occ_nbytes = 0;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    bool use_sparse = false;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2)) {
            if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, ncols1, ncols2)) {
                constexpr bool use_sparse_kernel = true;
                fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, 0>;
                use_sparse = true;

                static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
                if (!shared_memory_limit_raised[id]) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                    shared_memory_limit_raised[id] = true;
                }

                // the sparse kernel loads K / V in one stage (device ggml_cuda_fattn_mma_get_nstages), but the size above is the
                // two-stage one. GGML_CUDA_FA_SPARSE_SMEM_FIX (default 1): launch it with the one-stage size, its occupancy (so the stream-k
                // split) still that of the two-stage size, so every output is bitwise the same; 2: the occupancy of the one-stage size too
                // (more blocks per SM where shared memory bound them: another KV split, rounding-level); 0: as before
                static const int smem_fix = getenv("GGML_CUDA_FA_SPARSE_SMEM_FIX") ? atoi(getenv("GGML_CUDA_FA_SPARSE_SMEM_FIX")) : 1;
                if (smem_fix != 0 && nstages > 1) {
                    const size_t nbytes_shared_sparse = std::max(nbytes_shared_combine, Q_in_reg ?
                        std::max(nbytes_shared_Q,  nbytes_shared_KV_1stage + nbytes_shared_mask) :
                                 nbytes_shared_Q + nbytes_shared_KV_1stage + nbytes_shared_mask);
                    static std::atomic<int> n_logged { 0 };
                    if (ggml_cuda_fattn_probe_cur == nullptr && n_logged++ == 0) {
                        int occ_old = 0, occ_new = 0;
                        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_old, fattn_kernel, nthreads, nbytes_shared_total));
                        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_new, fattn_kernel, nthreads, nbytes_shared_sparse));
                        GGML_LOG_INFO("%s: QSA: sparse flash attention with one-stage shared memory, first D %d ncols %d x %d: %zu -> %zu bytes, "
                            "%d -> %d blocks/SM, occupancy %s (GGML_CUDA_FA_SPARSE_SMEM_FIX=0: off)\n", __func__, DKQ, ncols1, ncols2,
                            nbytes_shared_total, nbytes_shared_sparse, occ_old, occ_new, smem_fix == 2 ? "free" : "pinned");
                    }
                    if (smem_fix == 2) {
                        nbytes_shared_total = nbytes_shared_sparse;
                    } else {
                        sparse_occ_nbytes   = nbytes_shared_total;
                        nbytes_shared_total = nbytes_shared_sparse;
                    }
                }
            } else {
                constexpr bool use_sparse_kernel = false;
                fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, 0>;

                static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
                if (!shared_memory_limit_raised[id]) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                    shared_memory_limit_raised[id] = true;
                }
            }
        } else
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
            constexpr bool use_sparse_kernel = false;
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, 0>;

            static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
            if (!shared_memory_limit_raised[id]) {
                CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                shared_memory_limit_raised[id] = true;
            }
        }
    } else {
        constexpr bool use_logit_softcap = true;
        constexpr bool use_sparse_kernel = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, 0>;

        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
    }

    const bool async_kv_preload = nstages == 2 && !use_sparse;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // a two-source node (ggml_flash_attn_ext_set_src2: a q4_0 body and a q8_0 recent-window ring). One
    // kernel loops over the KV tiles of both sources with one online softmax, so it computes what the same kernel computes over the
    // concatenated cells (same tiles, stream-k split and fixup). In place where both sources are q4_0 / q8_0 pairs of a compiled
    // combination (body q4_0 / q4_0 or q8_0 / q8_0, ring q8_0 / q8_0) and the rows allow it (ggml_cuda_flash_attn_ext_src2_codes), else
    // both staged to f16 (the body behind the output as before, the ring from the pool). Occupancy reference: the plain f16 kernel
    if (dst->src[6] != nullptr) {
        GGML_ASSERT(logit_softcap == 0.0f && !use_sparse);
        const int codes = ggml_cuda_flash_attn_ext_src2_codes(id, dst, false);
        const int code1 = codes & 0xFF, code2 = (codes >> 8) & 0xFF;
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        bool aligned16 = true;
        for (const ggml_tensor * t : {K, V, (const ggml_tensor *) dst->src[6], (const ggml_tensor *) dst->src[7]}) {
            aligned16 = aligned16 && (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
        }
        static const int q8_mode_env = getenv("GGML_CUDA_FATTN_Q8_MODE") ? atoi(getenv("GGML_CUDA_FATTN_Q8_MODE")) : 0;
        const int mode = codes == 0 ? 0 : (DKQ == 256 && nstages == 2 && aligned16 && q8_mode_env != 1 ? 2 : 1);
        fattn_kernel_t k2 = nullptr;
        size_t nbytes_k2 = nbytes_shared_total;
        if constexpr ((DKQ == DV && (DKQ == 64 || DKQ == 128 || DKQ == 256)) || DKQ == 576) {
            if (codes == 0) {
                k2 = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, 0, 0>;
            } else if constexpr (ggml_cuda_flash_attn_ext_mma_q8_may_use(DKQ, DV, false, V_is_K_view) && ncols1 <= 16) {
                const int t1 = code1 & 12, t2 = code2 & 12;
                if (mode == 1) {
                    if (t1 == 12 && t2 == 0) { k2 = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, 13, 1>; }
                    if (t1 ==  0 && t2 == 0) { k2 = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false,  1, 1>; }
                } else if constexpr (DKQ == 256) {
                    if (t1 == 12 && t2 == 0) { k2 = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, 14, 2>; }
                    if (t1 ==  0 && t2 == 0) { k2 = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false,  2, 2>; }
                }
                const size_t rowK = std::max(fattn_kvq_row_bytes(DKQ, fattn_kvq_k4(code1)), fattn_kvq_row_bytes(DKQ, fattn_kvq_k4(code2)));
                const size_t rowV = std::max(fattn_kvq_row_bytes(DV,  fattn_kvq_v4(code1)), fattn_kvq_row_bytes(DV,  fattn_kvq_v4(code2)));
                const size_t raw_bytes = mode == 2 ? size_t(nbatch_fa) * (rowK + rowV) : 0;
                nbytes_k2 = std::max(nbytes_shared_combine, Q_in_reg ?
                    std::max(nbytes_shared_Q,  nbytes_shared_KV_1stage + raw_bytes + nbytes_shared_mask) :
                             nbytes_shared_Q + nbytes_shared_KV_1stage + raw_bytes + nbytes_shared_mask);
            }
        }
        GGML_ASSERT(k2 != nullptr && "two-source flash attention: no kernel for this combination (ggml_cuda_flash_attn_ext_src2_codes)");
        {
            static std::mutex mtx;
            static std::vector<std::pair<const void *, int>> raised;
            std::lock_guard<std::mutex> lock(mtx);
            const std::pair<const void *, int> key((const void *) k2, id);
            if (std::find(raised.begin(), raised.end(), key) == raised.end()) {
                CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(k2), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_k2));
                raised.push_back(key);
            }
        }
        static std::atomic<int> n_logged { 0 };
        if (ggml_cuda_fattn_probe_cur == nullptr && n_logged++ == 0) {
            GGML_LOG_INFO("%s: quantized-KV flash attention: two-source flash attention (body %s / %s %lld cells, ring %s / %s %lld cells), first D %d ncols %d x %d, %s "
                "(GGML_CUDA_FATTN_SRC2=0: unsupported, the graph's generic form)\n", __func__, ggml_type_name(K->type), ggml_type_name(V->type),
                (long long) K->ne[1], ggml_type_name(dst->src[6]->type), ggml_type_name(dst->src[7]->type), (long long) dst->src[6]->ne[1],
                DKQ, ncols1, ncols2, codes == 0 ? "staged f16" : (mode == 2 ? "in place, raw rows pipelined" : "in place, synchronous loads"));
        }
        launch_fattn<DV, ncols1, ncols2>
            (ctx, dst, k2, nwarps, nbytes_k2, nbatch_fa, codes == 0, codes == 0, true, false, warp_size_host, async_kv_preload,
             nullptr, 1, 1, false, fattn_kernel, nbytes_shared_total);
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // q8_0 K / V read in place by a q8_kv instance of the same kernel instead of converting the whole K and V to f16
    // first: q8_kv 2 where the f16 kernel pipelines its loads (cp.async, 2 stages; head size 256, 16 byte aligned rows, not sparse):
    // the raw q8_0 rows through the same pipeline and converted to one f16 tile in shared memory; else q8_kv 1: synchronous loads that
    // convert on the way. Both write the f16 values of the staging conversion. The launch geometry (stream-k blocks, fixup) is the f16
    // kernel's: its occupancy at its shared memory, so the KV split, and with the same f16 tiles every MMA and sum, is the staged
    // launch's. GGML_CUDA_FATTN_Q8_INPLACE=0: staged as before
    // the same for q4_0 K and / or V (q4_0 / q4_0, q4_0 / q8_0, q8_0 / q4_0), kvq = the type bits of the code. Kernels with a
    // q4_0 side exist for ncols1 <= 16 (ggml_cuda_flash_attn_ext_mma_kvq_compiled); a wider config (GGML_CUDA_FATTN_Q4_INPLACE_MAX_ROWS
    // raised above 16) runs staged, the graph reserved the f16 copy for it (ggml_cuda_flash_attn_ext_get_alloc_size)
    if constexpr (ggml_cuda_flash_attn_ext_mma_q8_may_use(DKQ, DV, false, V_is_K_view)) {
        const int kvq_code = logit_softcap == 0.0f ? ggml_cuda_flash_attn_ext_mma_kvq_inplace(id, dst, false) : 0;
        if (kvq_code != 0 && ggml_cuda_flash_attn_ext_mma_kvq_compiled(ncols1, kvq_code & 12)) {
            const ggml_tensor * K = dst->src[1];
            const ggml_tensor * V = dst->src[2];
            // rows start 4 byte aligned (ggml_cuda_flash_attn_ext_mma_kvq_inplace checked the strides; buffers and cache views keep it)
            GGML_ASSERT(ggml_cuda_fattn_q8_no_data(K) || (uintptr_t) K->data % 4 == 0);
            GGML_ASSERT(ggml_cuda_fattn_q8_no_data(V) || (uintptr_t) V->data % 4 == 0);
            bool aligned16 = true;
            for (const ggml_tensor * t : {K, V}) {
                aligned16 = aligned16 && (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
            }
            static const int q8_mode_env = getenv("GGML_CUDA_FATTN_Q8_MODE") ? atoi(getenv("GGML_CUDA_FATTN_Q8_MODE")) : 0; // 1: never 2
            const int kvq     = kvq_code & 12;
            const int q8_mode = DKQ == 256 && nstages == 2 && !use_sparse && aligned16 && q8_mode_env != 1 ? 2 : 1;
            const size_t raw_bytes = q8_mode == 2 ?
                size_t(nbatch_fa) * (fattn_kvq_row_bytes(DKQ, fattn_kvq_k4(kvq)) + fattn_kvq_row_bytes(DV, fattn_kvq_v4(kvq))) : 0;
            const size_t nbytes_shared_q8 = std::max(nbytes_shared_combine, Q_in_reg ?
                std::max(nbytes_shared_Q,  nbytes_shared_KV_1stage + raw_bytes + nbytes_shared_mask) :
                         nbytes_shared_Q + nbytes_shared_KV_1stage + raw_bytes + nbytes_shared_mask);
            fattn_kernel_t q8_kernel = nullptr;
            // the in-place kernel of a type pair: [sparse sync, sync, raw pipelined]
            auto pick = [&](auto kvq_c) {
                constexpr int c = decltype(kvq_c)::value;
                if constexpr (ggml_cuda_flash_attn_ext_mma_kvq_compiled(ncols1, c)) {
                    if (use_sparse) {
                        if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2)) {
                            q8_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, true, c | 1>;
                        }
                    } else if (q8_mode == 1) {
                        q8_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, c | 1>;
                    } else {
                        if constexpr (DKQ == 256) {
                            q8_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, c | 2>;
                        }
                    }
                }
            };
            switch (kvq) {
                case  0: pick(std::integral_constant<int,  0>{}); break;
                case  4: pick(std::integral_constant<int,  4>{}); break;
                case  8: pick(std::integral_constant<int,  8>{}); break;
                case 12: pick(std::integral_constant<int, 12>{}); break;
                default: break;
            }
            GGML_ASSERT(q8_kernel != nullptr);
            {
                // the dynamic shared memory limit, raised once per (kernel, device)
                static std::mutex mtx;
                static std::vector<std::pair<const void *, int>> raised;
                std::lock_guard<std::mutex> lock(mtx);
                const std::pair<const void *, int> key((const void *) q8_kernel, id);
                if (std::find(raised.begin(), raised.end(), key) == raised.end()) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(q8_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_q8));
                    raised.push_back(key);
                }
            }
            if (kvq == 0) {
                static std::atomic<int> n_logged { 0 };
                if (ggml_cuda_fattn_probe_cur == nullptr && n_logged++ == 0) {
                    GGML_LOG_INFO("%s: q8_0-KV flash attention: tensor-core flash attention reads the q8_0 K / V cache in place, first D %d ncols %d x %d, "
                        "%s, %lld cells (GGML_CUDA_FATTN_Q8_INPLACE=0: staged f16 copy)\n", __func__, DKQ, ncols1, ncols2,
                        q8_mode == 2 ? "raw rows pipelined" : "synchronous loads", (long long) K->ne[1]);
                }
            } else {
                static std::atomic<int> n_logged { 0 };
                if (ggml_cuda_fattn_probe_cur == nullptr && n_logged++ == 0) {
                    GGML_LOG_INFO("%s: quantized-KV flash attention: tensor-core flash attention reads the %s K / %s V cache in place, first D %d ncols %d x %d, "
                        "%s, %lld cells (GGML_CUDA_FATTN_Q4_INPLACE=0: staged f16 copy)\n", __func__, ggml_type_name(K->type),
                        ggml_type_name(V->type), DKQ, ncols1, ncols2, q8_mode == 2 ? "raw rows pipelined" : "synchronous loads",
                        (long long) K->ne[1]);
                }
            }
            launch_fattn<DV, ncols1, ncols2>
                (ctx, dst, q8_kernel, nwarps, nbytes_shared_q8, nbatch_fa, false, false, true, use_sparse, warp_size_host, async_kv_preload,
                 nullptr, 1, 1, false, fattn_kernel, sparse_occ_nbytes ? sparse_occ_nbytes : nbytes_shared_total);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // INT8 Q*K^T over the q8_0 K cache in place (GGML_CUDA_FATTN_I8, ggml_cuda_flash_attn_ext_mma_i8 decides: prompt /
    // verify nodes the in-place q8_0 / q4_0 kernels above do not take). The quants of the cache are the int8 operands (loaded into
    // shared memory as they are; through the cp.async pipeline as raw rows where the f16 kernel pipelines), Q is quantised per 32 values
    // in the prologue, V is the staged f16 copy and P*V is unchanged. Not bit-equal to the f16 kernel: the numerics of the vector
    // kernel's Q*K^T for a q8_0 K
    if constexpr (ggml_cuda_flash_attn_ext_mma_i8_compiled(DKQ, DV, ncols1, ncols2)) {
        // over an f16 K cache (GGML_CUDA_FATTN_I8_F16, ggml_cuda_flash_attn_ext_mma_i8_f16 decides: prompt / verify nodes past 16
        // rows): launch_fattn stages Q and K as int8 rows with one scale each behind the output (optionally rotated first), the kernel
        // loads the K rows into its int8 tile as they are and applies the scales once per score; V is the f16 cache
        const int i8_f16 = logit_softcap == 0.0f && !use_sparse && Q_in_reg && cols_per_warp == 16 ?
            ggml_cuda_flash_attn_ext_mma_i8_f16(id, dst) : 0;
        if (i8_f16 != 0) {
            fattn_kernel_t i8_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, false, false, 16 | 32>;
            const size_t nbytes_i8 = std::max(nbytes_shared_combine, std::max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask));
            {
                static std::mutex mtx;
                static std::vector<std::pair<const void *, int>> raised;
                std::lock_guard<std::mutex> lock(mtx);
                const std::pair<const void *, int> key((const void *) i8_kernel, id);
                if (std::find(raised.begin(), raised.end(), key) == raised.end()) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(i8_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_i8));
                    raised.push_back(key);
                }
            }
            static std::atomic<int> n_logged { 0 };
            if (ggml_cuda_fattn_probe_cur == nullptr && n_logged++ == 0) {
                GGML_LOG_INFO("%s: INT8 Q*K^T flash attention over the f16 K cache (Q and K int8 per row, %s), first D %d ncols %d x %d, "
                    "%s K loads, %lld cells (GGML_CUDA_FATTN_I8_F16=0: f16 kernel)\n", __func__,
                    i8_f16 == 1 ? "not rotated" : i8_f16 == 2 ? "rotated" : "rotated per KV head with outliers", DKQ,
                    ncols1, ncols2, nstages > 1 ? "pipelined" : "synchronous", (long long) dst->src[1]->ne[1]);
            }
            launch_fattn<DV, ncols1, ncols2>
                (ctx, dst, i8_kernel, nwarps, nbytes_i8, nbatch_fa, false, true, true, false, warp_size_host, async_kv_preload,
                 nullptr, 1, 1, false, nullptr, 0, i8_f16);
            return;
        }
        if (logit_softcap == 0.0f && !use_sparse && Q_in_reg && cols_per_warp == 16 && ggml_cuda_flash_attn_ext_mma_i8(id, dst)) {
            const ggml_tensor * K = dst->src[1];
            const int align = nstages > 1 ? (DKQ == 256 ? 16 : 8) : 4; // rows of raw cp.async copies: 16 / 8 byte chunks
            GGML_ASSERT(ggml_cuda_fattn_q8_no_data(K) || (uintptr_t) K->data % align == 0);
            fattn_kernel_t i8_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, false, false, 16>;
            const size_t raw_bytes = (nstages > 1 ? size_t(nbatch_fa)*fattn_i8_raw_row(DKQ) : 0) + fattn_i8_qd_bytes(DKQ, ncols);
            const size_t nbytes_i8 = std::max(nbytes_shared_combine,
                std::max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask + raw_bytes));
            {
                static std::mutex mtx;
                static std::vector<std::pair<const void *, int>> raised;
                std::lock_guard<std::mutex> lock(mtx);
                const std::pair<const void *, int> key((const void *) i8_kernel, id);
                if (std::find(raised.begin(), raised.end(), key) == raised.end()) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(i8_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_i8));
                    raised.push_back(key);
                }
            }
            static std::atomic<int> n_logged { 0 };
            if (ggml_cuda_fattn_probe_cur == nullptr && n_logged++ == 0) {
                GGML_LOG_INFO("%s: INT8 Q*K^T flash attention over the q8_0 K cache in place (Q int8 per 32 values), first D %d "
                    "ncols %d x %d, %s K loads, V %s, %lld cells (GGML_CUDA_FATTN_I8=0: f16 kernel)\n", __func__, DKQ, ncols1, ncols2,
                    nstages > 1 ? "pipelined" : "synchronous", ggml_type_name(dst->src[2]->type), (long long) K->ne[1]);
            }
            launch_fattn<DV, ncols1, ncols2>
                (ctx, dst, i8_kernel, nwarps, nbytes_i8, nbatch_fa, false, true, true, false, warp_size_host, async_kv_preload);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if (sparse_occ_nbytes) {
        // the launch geometry of the two-stage size (GGML_CUDA_FA_SPARSE_SMEM_FIX=1)
        launch_fattn<DV, ncols1, ncols2>
            (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true, use_sparse, warp_size_host, async_kv_preload,
             nullptr, 1, 1, false, fattn_kernel, sparse_occ_nbytes);
        return;
    }
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true, use_sparse, warp_size_host, async_kv_preload);
}


#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  64)

extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 32,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  2,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  1,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  2,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  8);

// The number of viable configurations for Deepseek is very limited:
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 16);

// Mistral Small 4 (DKQ=320, DV=256), GQA=32-only build:
extern DECL_FATTN_MMA_F16_CASE(320, 256,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(320, 256,  2, 32);

// For GLM 4.7 Flash
extern DECL_FATTN_MMA_F16_CASE(576, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  2, 32);
