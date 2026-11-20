// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

// as ggml_gallocr_set_external with up to GGML_GALLOCR_MAX_EXTERNAL regions of one buffer: the allocation is planned in chunks of at most
// the size of each region (in order), a tensor never spans two regions
#define GGML_GALLOCR_MAX_EXTERNAL 8
void ggml_gallocr_set_external_n(ggml_gallocr_t galloc, int buffer_id, ggml_backend_buffer_t buffer, const size_t * offsets, const size_t * sizes, int n);
// k15-fn: as ggml_gallocr_set_external_n with a buffer per region (e.g. the allocator's own buffer first, then memory lent by the user)
void ggml_gallocr_set_external_bufs(ggml_gallocr_t galloc, int buffer_id, const ggml_backend_buffer_t * bufs, const size_t * offsets, const size_t * sizes, int n);
// k15-fn: the own buffer of a buffer id when it is one backend buffer that no other id shares: false otherwise
bool ggml_gallocr_get_own_region(ggml_gallocr_t galloc, int buffer_id, ggml_backend_buffer_t * buffer, size_t * size);
// k15-fn: plan the graph with the first chunk of buffer_id limited to limit0 (the own buffer): size0 = that chunk, rest = the others
// (SIZE_MAX when they are more than one chunk); a lent region of at least rest after the own buffer holds this plan
void ggml_gallocr_measure_limited(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids,
                                  int buffer_id, size_t limit0, size_t * size0, size_t * rest);
// k15-fn: the last external region of buffer_id does not limit the plan (it is only checked against it): with the own buffer first and one
// lent region the plan is the one ggml_gallocr_measure_limited measured
void ggml_gallocr_set_external_last_free(ggml_gallocr_t galloc, int buffer_id, bool on);
// k13-small: frees the buffer of a buffer id (and of the ids sharing it): the next allocation plans again and allocates it at the size it needs
void ggml_gallocr_free_buffer(ggml_gallocr_t galloc, int buffer_id);

#ifdef  __cplusplus
}
#endif
