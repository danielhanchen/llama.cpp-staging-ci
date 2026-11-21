#pragma once

#include "common.cuh"

// k15-pp: GGML_OP_KQ_MASK (ggml_kq_mask), the KQ mask of a KV-cached attention from its cells and tokens
void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
