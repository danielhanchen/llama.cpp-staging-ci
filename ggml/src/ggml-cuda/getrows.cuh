// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"

#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BACK_BLOCK_SIZE 256

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream);

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// k15-hostcut: a recurrent state clear (SCALE by 0, in place, of one row of the states) and the GET_ROWS of the states after it in one
// launch (llm_graph_context::build_rs at a new sequence). The cleared row gets scale*x + bias as the scale kernel writes it, and every
// gathered copy of that row the same value (x*0 + 0 again gives the same bits, so a gather that reads the row after the clear agrees).
// ggml_cuda_rs_zero_get_rows_ok: whether the pair qualifies. GGML_CUDA_RS_ZERO_FUSE=0: two launches
bool ggml_cuda_rs_zero_get_rows_ok(const ggml_tensor * scale, const ggml_tensor * get_rows);
void ggml_cuda_op_rs_zero_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * scale, ggml_tensor * get_rows);
