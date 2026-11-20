// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

// k19-kvlong: the dense selection mask of a sparse (top-k selected cells) attention as one node, GGML_OP_KQ_MASK of kind GGML_KQ_MASK_KIND_SEL (ggml-impl.h: the kind sits in op_params[8], as
// ggml_kq_mask_win's window kinds 1 / 2; op_params[0..5] stay free for the host inputs llama keeps there)

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

    // idx: I32 [n_sel, n_tokens], the selected cells of every token (values >= n_kv are dead slots, ignored); kq_mask: F16 [n_kv, ...]
    // contiguous, n_tokens rows. Result F16, the shape of kq_mask: kq_mask + 0 at the selected cells of each row, kq_mask - INFINITY
    // elsewhere (each value rounded as the f16 add of the chain add(view(set_rows(fill(-inf), fill(0), idx)), kq_mask) it replaces),
    // without that chain's [n_kv + n_sel, n_tokens] scatter buffer
    GGML_API struct ggml_tensor * ggml_sel_mask(
            struct ggml_context * ctx,
            struct ggml_tensor  * idx,
            struct ggml_tensor  * kq_mask);

#ifdef __cplusplus
}
#endif
