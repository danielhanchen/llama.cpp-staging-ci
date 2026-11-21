// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// get_mmvq_mmid_max_batch, raised to the columns of a verify window (5..8 rows) inside a verify-tagged graph where the
// per-architecture table would send the MUL_MAT_ID to MMQ (GGML_CUDA_VERIFY_WIDE bit 8, see ggml_cuda_verify_wide_cols)
int ggml_cuda_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// whether ggml_cuda_mul_mat_vec_q_multi has a kernel for this type, K and number of columns that gives every output the bytes
// of its own ggml_cuda_mul_mat_vec_q (exact = batch invariant verify columns)
bool ggml_cuda_mul_mat_vec_q_multi_ok(ggml_type type, int64_t ncols_x, int64_t ncols_dst, bool exact, int cc);

// n (2..GGML_CUDA_MMVQ_MULTI_MAX) dense quantized matrices of one type and K over the same contiguous src1, one launch
void ggml_cuda_mul_mat_vec_q_multi(ggml_backend_cuda_context & ctx, int n, const ggml_tensor * const * src0, const ggml_tensor * src1,
    ggml_tensor * const * dst);

// whether ggml_cuda_mul_mat_vec_q_ew gives dst the bytes of the unfused chain + ggml_cuda_mul_mat_vec_q
bool ggml_cuda_mul_mat_vec_q_ew_ok(const ggml_tensor * src0, const ggml_tensor * dst, int cc);

// dst = src0 * chain(ew_src), chain = n_ops SCALE (ops[o] == 0: p0 * x + p1) / SILU (ops[o] == 1) ops, quantized to q8_1 in the kernel
void ggml_cuda_mul_mat_vec_q_ew(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * ew_src, int n_ops,
    const int * ops, const float * p0, const float * p1, ggml_tensor * dst);

// hyper-connection mat-vecs with their neighbour in one launch (same bytes as the separate launches), see mmvq.cu.
// down (Q8_0) + inject (F32 split-K, args from ggml_cuda_mul_mat_vec_f_splitk_plan) over the same activation
struct ggml_cuda_mmvf_splitk_args;
bool ggml_cuda_mul_mat_vec_q_inject_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_vec_q_inject(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const ggml_cuda_mmvf_splitk_args & inj);
// up (Q8_0, gate = src0 * src1) + gated DSV4_HC_PRE (mixed, and its q8_1 copy if q8_1_out != nullptr) over x [n_embd, 4, ncols]
bool ggml_cuda_mul_mat_vec_q_hc_pre_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * gate, int64_t n_embd,
    int64_t hc, int cc);
// vy: the q8_1 copy of src1 (ggml_cuda_mmvq_src1_q8_1)
// cnt: GGML_CUDA_HC_PRE_MAX_GROUPS zeroed self-resetting counters (ggml_backend_cuda_context::hc_pre_cnt)
void ggml_cuda_mul_mat_vec_q_hc_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const char * vy,
    ggml_tensor * gate, const float * x, int64_t sx1, int64_t sx2, float * mixed, void * q8_1_out, unsigned int * cnt, int64_t n_embd,
    float scale);
// the q8_1 copy of a contiguous F32 src1 that ggml_cuda_mul_mat_vec_q would read (from the q8_1 cache, else quantized into alloc)
const char * ggml_cuda_mmvq_src1_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_cuda_pool_alloc<char> & alloc);
