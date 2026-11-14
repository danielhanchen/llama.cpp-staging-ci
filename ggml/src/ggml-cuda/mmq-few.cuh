// k19-kqtc: few-row MMQ ported from upstream llama.cpp (the llama.cpp authors, MIT), routing in ggml-cuda.cu behind GGML_CUDA_MMQ_FEW.
#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, int cc);

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// GGML_CUDA_VERIFY_REASSOC (default 1): batch invariant verify products may reassociate (0 = the bit-exact per-token verify)
bool ggml_cuda_verify_reassoc_ok();
