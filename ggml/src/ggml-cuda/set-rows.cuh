// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the SET_ROWS nodes a and b (f32 rows, same index and dst types) in one launch; false (nothing launched) if not supported
bool ggml_cuda_set_rows_pair_supported(const ggml_tensor * a, const ggml_tensor * b);
bool ggml_cuda_set_rows_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b);
