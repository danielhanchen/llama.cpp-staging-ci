#pragma once

#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

// k19-kvsel: K/V cache chunks backed by pinned host memory inside a CUDA lazy buffer (see ggml_backend_cuda_buffer_commit_host)

// the host chunk map of a lazy buffer with host chunks and a device copy of its map (LLAMA_KV_SEL_STATS); false otherwise
bool ggml_cuda_buffer_host_chunks(const ggml_backend_buffer * buf, const char ** base, size_t * gran, const uint8_t ** map_dev,
        const uint8_t ** map_host, size_t * n);

// LLAMA_KV_SEL_STATS=1: bytes flash attention read from host chunks, summed over the devices (the devices are synchronized first):
// out[0] = K + V bytes of the cells the rows of decode / verify nodes select (per row: exact for one row, an upper bound of the union a
// query tile reads for several), out[1] = K + V bytes of the host chunks a prompt node reads whole (its staging copy), out[2] = nodes
// counted, out[3] = nodes with any host chunk in their K / V range
void ggml_cuda_kv_sel_stats(uint64_t * out);

static inline bool ggml_cuda_kv_sel_stats_on() {
    static const bool on = getenv("LLAMA_KV_SEL_STATS") != nullptr && atoi(getenv("LLAMA_KV_SEL_STATS")) != 0;
    return on;
}
