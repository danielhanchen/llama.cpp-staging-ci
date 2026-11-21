// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "common.cuh"

bool ggml_cuda_mm_ids_helper_supported(int n_experts, int n_tokens, int n_expert_used);

void ggml_cuda_launch_mm_ids_helper(
        ggml_cuda_pool & pool,
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// inverts ids_dst of ggml_cuda_launch_mm_ids_helper (compact row -> token slot) into token slot -> compact row,
// n = number of filled compact rows, n_slots = number of token slots
void ggml_cuda_launch_mm_ids_invert(
        const int32_t * ids_dst, int32_t * ids_inverse, int n, int n_slots, cudaStream_t stream);

// counts the skipped slots (id -1) into *n_skipped
void ggml_cuda_launch_mm_ids_count_skipped(
        const int32_t * ids, int32_t * n_skipped, int n_tokens, int n_expert_used, int si1, cudaStream_t stream);

// false if ids cannot hold -1 (no skipped slot), so the zeroing launch below is not needed: ids is a view chain (VIEW,
// RESHAPE, PERMUTE, TRANSPOSE) of an ARGSORT / TOP_K result, whose values are column indices. GGML_CUDA_MMID_ZERO_SKIP=0: always true
bool ggml_cuda_mm_ids_may_skip(const ggml_tensor * ids);

// zero the dst rows of the skipped slots, which no expert owns
void ggml_cuda_launch_mm_ids_zero_dst(
        const int32_t * ids, float * dst,
        int n_tokens, int n_expert_used, int ne0, int si1, int64_t s1, int64_t s2, cudaStream_t stream);
