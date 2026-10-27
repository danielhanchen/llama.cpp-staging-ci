// Ported from ikawrakow/ik_llama.cpp ggml/src/ggml-moe-prefetch.h at e34d163a (PR #2101, author dmaivel), MIT.
// Copyright (c) 2024-2025 The ik_llama.cpp authors. See LICENSE.
// k16-disk changes: public header (GGML_API), the registry holds expert tensor ranges with a reference count (not whole mappings),
// a lookahead byte budget, MADV_RANDOM on the registered ranges, counters (ggml_moe_prefetch_stats).
#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// mapping registry; a registered range marks tensors as mmap-backed and
// eligible for prefetch. Nothing is opened or retained beyond (base, size).
// k16-disk: ranges are reference counted (several contexts register the same model); with rand_read the range gets MADV_RANDOM
// while it is registered (no read-around on a demand fault: the engine reads the exact ranges), MADV_NORMAL when the last
// reference goes
GGML_API void ggml_moe_prefetch_register_mapping  (const void * base, size_t size, bool rand_read);
GGML_API void ggml_moe_prefetch_unregister_mapping(const void * base);

// worker pool control; n_threads > 0 (re)creates the pool, <= 0 shuts it down
GGML_API void ggml_moe_prefetch_set_n_threads(int n_threads);
GGML_API bool ggml_moe_prefetch_enabled(void);

// k16-disk: lookahead (ggml_moe_prefetch_tensor) budget in bytes queued or being read; beyond it lookahead is dropped (0 = no limit)
GGML_API void ggml_moe_prefetch_set_budget(size_t bytes);

// k16-disk: scheduler lookahead depth in MoE splits (ggml-backend.cpp; default GGML_MOE_PREFETCH_AHEAD or 3, 0 = kernel hook only)
GGML_API void ggml_moe_prefetch_set_ahead(int n);

// the epoch is bumped once per scheduler pass so per-tensor enqueues are
// idempotent within one graph execution but re-issued on the next.
GGML_API void ggml_moe_prefetch_new_epoch(void);

// selective enqueue for one MoE node (GGML_OP_MUL_MAT_ID); reads the ids tensor bytes (must be final and
// host-visible) and enqueues the selected expert ranges of src weights.
GGML_API void ggml_moe_prefetch_node(const struct ggml_tensor * node);

// full-tensor lookahead enqueue (low priority), e.g. for prompt processing
// where the next layer's experts are needed in bulk.
GGML_API void ggml_moe_prefetch_tensor(const struct ggml_tensor * w);

// true when w lies in a registered range (the engine would stream it)
GGML_API bool ggml_moe_prefetch_is_registered(const struct ggml_tensor * w);

// block until all pending prefetch jobs for tensor w are complete.
// Returns immediately when nothing is pending.
GGML_API void ggml_moe_prefetch_wait(const struct ggml_tensor * w);

// MADV_COLD the pages the streamer had to read from storage for tensor w,
// so prompt-processing streaming traffic is reclaimed ahead of the decode
// working set. Pages already resident before the sweep are left alone.
// Only applies to tensors enqueued via ggml_moe_prefetch_tensor.
GGML_API void ggml_moe_prefetch_cold(const struct ggml_tensor * w);

// kernel-entry hook, called at the start of the CPU MUL_MAT_ID. Thread 0 fire-and-forget enqueues the routed experts of src0
// (urgent). k16-disk: a no-op while the engine is off (ik falls back to madvise(MADV_WILLNEED) there; here off = k15 behaviour)
GGML_API void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor * node, int ith);

// k16-disk: cumulative counters (engagement checks, logs)
struct ggml_moe_prefetch_counters {
    uint64_t n_jobs;            // chunks taken by the workers
    uint64_t n_skipped;         // chunks found resident (mincore)
    uint64_t bytes_populated;   // bytes of chunks the workers read
    uint64_t n_colded;          // chunks given MADV_COLD
    uint64_t n_node;            // kernel-hook / node enqueues (MUL_MAT_ID weights)
    uint64_t n_tensor;          // whole-tensor lookahead enqueues
    uint64_t n_dropped;         // lookahead enqueues dropped by the budget or the queue cap
    uint64_t n_ranges;          // registered ranges
    uint64_t bytes_registered;  // bytes in the registered ranges
};
GGML_API void ggml_moe_prefetch_stats(struct ggml_moe_prefetch_counters * out);

#ifdef __cplusplus
}
#endif
