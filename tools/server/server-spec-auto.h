#pragma once

// k11-decode: n-gram (suffix) drafting on by default in the server.
//
// With no --spec-type, no draft model and no lookup cache, the server drafts from repeats of the prompt + output
// (ngram-suffix) and verifies the drafts batch invariant (llama_set_verify_exact), so greedy output is byte equal to plain
// decoding. Its windows are cost-priced (common/spec-cost-policy.h): measured round times and acceptance decide each round, so it
// stops drafting where drafts lose. It stays off for architectures whose rollback needs a large state snapshot per drafted token
// (Mamba, GDN hybrids such as Qwen3.5/3.6 and Qwen3.8-Flash-Next, DeepSeek-V4): there the snapshots take VRAM from the weights.
//
// LLAMA_SPEC_AUTO=0: off (the k10 behaviour). =2: also for those architectures. LLAMA_SPEC_AUTO_EXACT=0: drafts verified as
// ordinary batches (faster verify kernels, greedy text may differ from plain decoding in the last bits of the logits).

#include "common.h"
#include "gguf.h"
#include "log.h"

#include "../../src/llama-ext.h"

#include <cstdlib>
#include <algorithm>
#include <string>

// k13: default spill drafting modes per model and host path (see server_spec_spill_mode), from the KLD gate of plans/k11.md and the G4
// timing of plans/k13_lfm.md section 10:
//   - LFM2.5 with the hybrid gate (small caches, G4 4 GB): mode 1, repeats of at least 12 tokens (chat inside its spread, 2k +50%);
//   - LFM2.5 with the CPU-miss rule (G4 8 GB): mode 3 (chat +28%, 2k +45%, 8k +42%);
//   - gemma-4: off (mode 1 loses chat by 1% on G4 8 GB, LLAMA_SPEC_SPILL=1 for its 2k +30% / 8k +43%).
#ifndef SPEC_SPILL_LFM_GATE
#define SPEC_SPILL_LFM_GATE   1
#define SPEC_SPILL_LFM_CM     3
#define SPEC_SPILL_GEMMA4_GATE 0
#define SPEC_SPILL_GEMMA4_CM  0
#endif
#ifndef SPEC_SPILL_MIN_MATCH_GATE
#define SPEC_SPILL_MIN_MATCH_GATE 12
#define SPEC_SPILL_MIN_MATCH_CM   3
#endif

// k13: the shortest repeat the default spill drafts take (LLAMA_SPEC_SPILL_MIN_MATCH; 3 = every match of the index). host_mode 1 (the
// hybrid gate) verifies a window with the fixed host share of every row, so short repeats lose there (G4 LFM2.5 4 GB chat -2%)
static int server_spec_spill_min_match(int32_t host_mode) {
    const char * e = getenv("LLAMA_SPEC_SPILL_MIN_MATCH");
    return std::max(3, e != nullptr ? atoi(e) : host_mode == 1 ? SPEC_SPILL_MIN_MATCH_GATE : SPEC_SPILL_MIN_MATCH_CM);
}

static int server_spec_auto_env() {
    const char * e = getenv("LLAMA_SPEC_AUTO");
    return e != nullptr ? atoi(e) : 1;
}

static std::string server_spec_auto_arch(const std::string & path) {
    gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        return "";
    }
    std::string arch;
    const int64_t k = gguf_find_key(g, "general.architecture");
    if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_STRING) {
        arch = gguf_get_val_str(g, k);
    }
    gguf_free(g);
    return arch;
}

// turns ngram-suffix on in params when nothing else drafts; returns true when it did
static bool server_spec_auto_apply(common_params & params) {
    const int mode = server_spec_auto_env();
    if (mode == 0) {
        return false;
    }
    auto & sp = params.speculative;
    const bool none = sp.types.empty() || (sp.types.size() == 1 && sp.types[0] == COMMON_SPECULATIVE_TYPE_NONE);
    if (!none || sp.has_dft() || sp.has_synth() || !sp.ngram_cache.lookup_cache_static.empty() ||
            !sp.ngram_cache.lookup_cache_dynamic.empty() || params.model.path.empty() || params.embedding || params.n_parallel < 1) {
        return false;
    }
    const std::string arch = server_spec_auto_arch(params.model.path);
    const int32_t     rs   = llama_arch_rollback_state(arch.c_str());
    if (rs < 0 || (rs == 2 && mode != 2)) {
        LOG_INF("%s: n-gram drafting stays off for '%s' (%s; LLAMA_SPEC_AUTO=2 turns it on)\n", __func__, arch.c_str(),
                rs < 0 ? "unknown architecture" : "its rollback snapshots a large recurrent state per drafted token");
        return false;
    }
    sp.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX };
    sp.ngram_suffix.auto_mode = true;
    LOG_INF("%s: n-gram drafting on by default for '%s' (cost-priced ngram-suffix windows, verified exact; LLAMA_SPEC_AUTO=0 turns it off)\n",
            __func__, arch.c_str());
    return true;
}

static bool server_spec_auto_exact() {
    const char * e = getenv("LLAMA_SPEC_AUTO_EXACT");
    return e == nullptr || atoi(e) != 0;
}

// k13: the default drafts freeze the small batches of a MoE cache with CPU misses so that they verify exactly (llama_moe_freeze_small);
// opt-in (LLAMA_SPEC_EXACT_FROZEN=1): exact, but the frozen cache stops following the decode, a loss at long prompts (plans/k13_lfm.md)
static bool server_spec_exact_frozen() {
    const char * e = getenv("LLAMA_SPEC_EXACT_FROZEN");
    return e != nullptr && atoi(e) != 0;
}

// k13: default n-gram drafts where a MoE cache's host computes experts (text changes, so a model gets it only after the KLD gate of
// plans/k11.md passed for it, plans/k13_lfm.md). host_mode: llama_moe_host_mode. Returns 0 = no drafts (the k12 behaviour), 1 = drafts
// verified as ordinary batches with the host path, 2 = frozen small batches with drafts verified exactly against them, 3 = as 1 with the
// small-batch host path of the CPU-miss rule pinned on (llama_moe_pin_small; 1 where the cache has no CPU-miss rule).
// LLAMA_SPEC_SPILL=0 off, 1 / 2 / 3 force that mode for any model.
static int server_spec_spill_mode(const std::string & arch, int32_t host_mode) {
    if (const char * e = getenv("LLAMA_SPEC_SPILL")) {
        return std::max(0, std::min(3, atoi(e)));
    }
    if (host_mode == 0) {
        return 0;
    }
    // per (model, host path) after the KLD gate and the G4 timing (plans/k13_lfm.md section 10)
    if (arch == "lfm2moe") {
        return host_mode == 1 ? SPEC_SPILL_LFM_GATE : SPEC_SPILL_LFM_CM;
    }
    if (arch == "gemma4") {
        return host_mode == 1 ? SPEC_SPILL_GEMMA4_GATE : SPEC_SPILL_GEMMA4_CM;
    }
    return 0;
}
