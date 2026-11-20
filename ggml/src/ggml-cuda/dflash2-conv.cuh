// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#include "common.cuh"

// k19-handoff: GGML_OP_DFLASH2_CONV (see ggml_dflash2_conv)
void ggml_cuda_op_dflash2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
