// few-row MMQ ported from upstream llama.cpp (the llama.cpp authors, MIT), routing in ggml-cuda.cu behind GGML_CUDA_MMQ_FEW.
#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, int cc);

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// one launch for n (<= 4) dense products that each take the few-row MMQ alone, over one src1 (see mmq-few.cu, GGML_CUDA_MMQ_FEW_MULTI)
bool ggml_cuda_mmq_few_multi_on(int cc);
bool ggml_cuda_mmq_few_multi_pair(const ggml_tensor * a, const ggml_tensor * b);
void ggml_cuda_mul_mat_q_few_multi(ggml_backend_cuda_context & ctx, int n, const ggml_tensor * const * src0, const ggml_tensor * src1,
    ggml_tensor * const * dst);

// MUL_MAT_ID (ids = dst->src[2]) at a few tokens per expert, behind GGML_CUDA_MMQ_FEW_MOE (see mmq-few.cu)
bool ggml_cuda_should_use_mmq_few_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, int cc);

void ggml_cuda_mul_mat_q_few_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst);

// GGML_CUDA_VERIFY_REASSOC (default 1): batch invariant verify products may reassociate (0 = the bit-exact per-token verify)
bool ggml_cuda_verify_reassoc_ok();
