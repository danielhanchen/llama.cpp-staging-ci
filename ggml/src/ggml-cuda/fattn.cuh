// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// k19-graph: the launch signature of a node with a KV length (ggml_flash_attn_ext_set_kv_len) at its current length, 0 without one:
// a CUDA graph holding the node is replayed only while it stays the same (see fattn.cu)
int64_t ggml_cuda_flash_attn_ext_kv_len_sig(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// k15-mm2 QSA maskless, GGML_CUDA_QSA_MASKLESS (see fattn-common.cuh)
int ggml_cuda_qsa_maskless_mode();
