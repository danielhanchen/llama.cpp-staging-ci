#pragma once

#include "common.cuh"

// Grouped F16/BF16 GEMM for MUL_MAT_ID at batch sizes that the vector and mmf kernels do not cover.
// The expert sort, the tile schedule and the GEMM all run on the device, so no stream synchronization is needed
// and the op can be captured in a CUDA graph.

bool ggml_cuda_should_use_mmid_gemm(const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_id_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
