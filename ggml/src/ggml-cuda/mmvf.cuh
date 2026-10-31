// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// k19-tpp: set while ggml-cuda plans or evaluates a graph whose nodes carry GGML_TENSOR_FLAG_VERIFY (a verify window, not batch
// invariant): F32 weights then keep mmvf at 4 columns on NVIDIA Ampere and newer, as at 3, so a 4-row window's rows get the sums of a
// 3-row window's (mmvf's per-column sums do not depend on the column count: the block size follows K only)
struct ggml_cuda_verify_graph_scope {
    explicit ggml_cuda_verify_graph_scope(const struct ggml_cgraph * cgraph);
    ~ggml_cuda_verify_graph_scope();
    bool prev;
};
bool ggml_cuda_verify_graph_active();
// k19-vrow2: the most columns of a verify window that keep a 3-row window's kernels for GGML_CUDA_VERIFY_WIDE bit `bit` (1: F32 mmvf,
// 2: mmvq warps) on this device, 0 outside a verify graph / off / not NVIDIA Ampere+ / HIP / MUSA
int ggml_cuda_verify_wide_cols(int cc, int bit);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);

// k15-mm2: whether ggml_cuda_mul_mat_vec_f_multi has a kernel for this type, K and number of columns (the single launch's block size)
bool ggml_cuda_mul_mat_vec_f_multi_ok(ggml_type type, int64_t ncols, int64_t ncols_dst, int device);

// k15-mm2: n (2..GGML_CUDA_MMVQ_MULTI_MAX) F32 / BF16 products of one type and K over src1 in one launch, each output with the bytes of
// its own ggml_cuda_mul_mat_vec_f launch
void ggml_cuda_mul_mat_vec_f_multi(ggml_backend_cuda_context & ctx, int n, const ggml_tensor * const * src0, const ggml_tensor * src1,
    ggml_tensor * const * dst);

// k15-mm2: the split-K launch (mmvf-splitk.cuh) ggml_cuda_mul_mat_vec_f would make for a dense F32 product without fusion, without
// allocating its buffers; false if it would not take it
struct ggml_cuda_mmvf_splitk_args;
bool ggml_cuda_mul_mat_vec_f_splitk_plan(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    ggml_tensor * dst, ggml_cuda_mmvf_splitk_args & a);
