// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

// k19-kvwin: two-source flash attention (kept out of ggml.h so that the backends do not rebuild for it)

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

// k19-kvwin: two-source flash attention. The node attends over the union of (k, v, mask) and (k2, v2, mask2):
//   k2:    [n_embd_k, n_kv2, n_head_kv, ne3]  any flash attention K type, may differ from k's type
//   v2:    [n_embd_v, n_kv2, n_head_kv, ne3]  !! not transposed !!, any flash attention V type
//   mask2: [n_kv2,    n_batch_pad, ne32, ne33] F16, contiguous, the same ne[1..3] as mask
// scale, ALiBi, logit softcap and sinks apply as for the first source (sinks once); kv_len (src[5]) and n_kv_max apply to the
// first source only. Stored as src[6] = k2, src[7] = v2, src[8] = mask2. A backend supports it only if its registry exports the
// proc address "ggml_backend_fa_src2" (ggml_backend_dev_supports_op refuses the node elsewhere)
GGML_API void ggml_flash_attn_ext_set_src2(
        struct ggml_tensor * a,
        struct ggml_tensor * k2,
        struct ggml_tensor * v2,
        struct ggml_tensor * mask2);

#ifdef  __cplusplus
}
#endif
