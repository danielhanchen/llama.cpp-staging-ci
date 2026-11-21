// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_moe_cache_lookup_supported(const ggml_tensor * op);

// asynchronous refill of the lent slots of a MoE cache group (see moe_rf_wait in moe-cache.cu): -1 / false where not available
int  ggml_cuda_moe_refill_register(ggml_backend_cuda_context & ctx, const void * state_data);
bool ggml_cuda_moe_refill_need(ggml_backend_cuda_context & ctx, const void * state_data, const int32_t * il, int n, int32_t epoch);
bool ggml_cuda_moe_refill_done(ggml_backend_cuda_context & ctx, const void * state_data, int il, int32_t epoch);
