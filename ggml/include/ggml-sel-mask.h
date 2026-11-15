#pragma once

// k19-kvlong: the dense selection mask of a sparse (top-k selected cells) attention as one node, GGML_OP_KQ_MASK mode 1

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
