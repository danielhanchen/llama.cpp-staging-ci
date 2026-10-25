#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// k15-mm2 QSA maskless, GGML_CUDA_QSA_MASKLESS (see fattn-common.cuh)
int ggml_cuda_qsa_maskless_mode();
