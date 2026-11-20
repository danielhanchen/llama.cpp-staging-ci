// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

#include "common.cuh"

// k11-decode: LFM2 short-conv fusions (shortconv-fused.cu); each returns the number of following nodes it consumed (0: not fused)
int ggml_cuda_try_fuse_shortconv_window(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
int ggml_cuda_try_fuse_shortconv_conv_mul(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
