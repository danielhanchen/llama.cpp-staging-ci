#include "common.cuh"

// more new tokens than this take the long-token kernel (k19-fuse F5: and the prompt concat elide)
static constexpr int64_t ssm_conv_long_token_min_t = 32;

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
