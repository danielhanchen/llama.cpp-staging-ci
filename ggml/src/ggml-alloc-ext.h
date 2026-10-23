#pragma once

#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

// as ggml_gallocr_set_external with up to GGML_GALLOCR_MAX_EXTERNAL regions of one buffer: the allocation is planned in chunks of at most
// the size of each region (in order), a tensor never spans two regions
#define GGML_GALLOCR_MAX_EXTERNAL 8
void ggml_gallocr_set_external_n(ggml_gallocr_t galloc, int buffer_id, ggml_backend_buffer_t buffer, const size_t * offsets, const size_t * sizes, int n);
// k13-small: frees the buffer of a buffer id (and of the ids sharing it): the next allocation plans again and allocates it at the size it needs
void ggml_gallocr_free_buffer(ggml_gallocr_t galloc, int buffer_id);

#ifdef  __cplusplus
}
#endif
