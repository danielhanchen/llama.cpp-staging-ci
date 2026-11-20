#include "common.cuh"

// more new tokens than this take the long-token kernel (k19-fuse F5: and the prompt concat elide)
static constexpr int64_t ssm_conv_long_token_min_t = 32;

// conv-state copies one short-token convolution writes (ssm_conv_state_copies): a verify window of T rows with a rollback of T - 1
// states copies up to T of them (DFlash2 n_max 6: 7 rows), 4 left the concat of a 5..8-row window unelided
static constexpr int ssm_conv_state_max_copies = 8;

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
