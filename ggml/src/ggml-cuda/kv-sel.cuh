#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

// K/V cache chunks backed by pinned host memory inside a CUDA lazy buffer (see ggml_backend_cuda_buffer_commit_host)

// the host chunk map of a lazy buffer with host chunks and a device copy of its map (LLAMA_KV_SEL_STATS); false otherwise
bool ggml_cuda_buffer_host_chunks(const ggml_backend_buffer * buf, const char ** base, size_t * gran, const uint8_t ** map_dev,
        const uint8_t ** map_host, size_t * n);

// LLAMA_KV_SEL_STATS=1: bytes flash attention read from host chunks, summed over the devices (the devices are synchronized first):
// out[0] = K + V bytes of the cells the rows of decode / verify nodes select (per row: exact for one row, an upper bound of the union a
// query tile reads for several), out[1] = K + V bytes of the host chunks a prompt node reads whole (its staging copy), out[2] = nodes
// counted, out[3] = nodes with any host chunk in their K / V range, out[4] = sparse nodes staged (ggml_cuda_kv_sel_stage; eager
// launches, a CUDA graph replay is not counted)
void ggml_cuda_kv_sel_stats(uint64_t * out);

// true if the buffer is a lazy buffer with any chunk backed by host memory
bool ggml_cuda_buffer_has_host_chunks(const ggml_backend_buffer * buf);

// true if the buffer is a lazy buffer (its chunks may be backed by host memory later)
bool ggml_cuda_buffer_is_lazy(const ggml_backend_buffer * buf);

// sparse flash attention over K / V cells partly in host memory (LLAMA_KV_SEL_STAGE=0: off): copy the K and V rows and the mask values
// of every listed cell (indices: per query tile, n_kv_max entries, counts[tile] of them live) into compact device staging in list order
// and rewrite the lists to the staging rows, so the kernel loads the same values in the same order from device memory. A gather kernel
// with many reads in flight instead of the attention kernel's latency-bound loads over the bus. mask / M_c: halves, row stride s31, n_rows rows (the mask's, padding included).
// K_c / V_c: ntiles * n_kv_max rows of nbk / nbv bytes; M_c: as the mask (ntiles * n_kv_max <= the mask's columns)
void ggml_cuda_kv_sel_stage(const char * K, size_t nbk, const char * V, size_t nbv, const half * mask, int64_t s31,
        int32_t * indices, const int32_t * counts, int ntiles, int ncols1, int n_rows, int n_kv_max,
        char * K_c, char * V_c, half * M_c, cudaStream_t stream);

static inline bool ggml_cuda_kv_sel_stage_on() {
    static const bool on = getenv("LLAMA_KV_SEL_STAGE") == nullptr || atoi(getenv("LLAMA_KV_SEL_STAGE")) != 0;
    return on;
}

static inline bool ggml_cuda_kv_sel_stats_on() {
    static const bool on = getenv("LLAMA_KV_SEL_STATS") != nullptr && atoi(getenv("LLAMA_KV_SEL_STATS")) != 0;
    return on;
}
