#pragma once

#include "common.cuh"

// LFM2 short-conv fusions (shortconv-fused.cu); each returns the number of following nodes it consumed (0: not fused)
int ggml_cuda_try_fuse_shortconv_window(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
int ggml_cuda_try_fuse_shortconv_conv_mul(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
