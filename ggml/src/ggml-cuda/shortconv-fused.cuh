#pragma once

#include "common.cuh"

// k11-decode: LFM2 short-conv fusions (shortconv-fused.cu); each returns the number of following nodes it consumed (0: not fused)
int ggml_cuda_try_fuse_shortconv_window(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
int ggml_cuda_try_fuse_shortconv_conv_mul(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// a run of same-layout F32 copies in one launch; returns the number of nodes consumed after i (0: none)
int ggml_cuda_try_fuse_cpy_batch(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
