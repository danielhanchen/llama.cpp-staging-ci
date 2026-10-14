#pragma once

#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

// as ggml_gallocr_set_external with up to GGML_GALLOCR_MAX_EXTERNAL regions of one buffer: the allocation is planned in chunks of at most
// the size of each region (in order), a tensor never spans two regions
#define GGML_GALLOCR_MAX_EXTERNAL 8
void ggml_gallocr_set_external_n(ggml_gallocr_t galloc, int buffer_id, ggml_backend_buffer_t buffer, const size_t * offsets, const size_t * sizes, int n);

#ifdef  __cplusplus
}
#endif
