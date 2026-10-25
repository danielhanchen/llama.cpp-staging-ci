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

// k13: default spill drafting modes per model and host path (see server_spec_spill_mode), gated by the KLD gate of plans/k11.md and the
// G4 timing of plans/k13_lfm.md section 10 (both must pass):
//   - LFM2.5 with the hybrid gate (small caches, G4 4 GB): mode 1 from repeats of at least 16 tokens (chat = kill switch, 2k +35%,
//     8k +5%; decode KLD inside the -fa off band, first token unchanged);
//   - LFM2.5 with the CPU-miss rule (G4 8 GB): mode 3 (k15: its prompts stay off the pinned host path, LLAMA_MOE_PIN_PROMPT_DEV, so
//     the first token is the all-device one: first-token KLD 0 on 192 + 12 prompts, decode inside the calibrated band; G4 k15d chat
//     +24%, 2k +62%, 8k +16% against mode 0). k13's mode 3 without that failed the first-token top-1 (0.896 vs band 0.906);
//   - gemma-4: off (mode 1 loses chat by 1-1.7% on G4 8 GB; LLAMA_SPEC_SPILL=1 opts in).
#ifndef SPEC_SPILL_LFM_GATE
#define SPEC_SPILL_LFM_GATE   1
#define SPEC_SPILL_LFM_CM     3
#define SPEC_SPILL_GEMMA4_GATE 0
#define SPEC_SPILL_GEMMA4_CM  0
#endif
#ifndef SPEC_SPILL_MIN_MATCH_GATE
#define SPEC_SPILL_MIN_MATCH_GATE 16
#define SPEC_SPILL_MIN_MATCH_CM   3
#endif

// k13: the shortest repeat the default spill drafts take (LLAMA_SPEC_SPILL_MIN_MATCH; 3 = every match of the index). host_mode 1 (the
// hybrid gate) verifies a window with the fixed host share of every row, so short repeats lose there (G4 LFM2.5 4 GB chat: -2% from 3, -0.5..-0.9% from 12, = kill switch from 16)
static int server_spec_spill_min_match(int32_t host_mode) {
    const char * e = getenv("LLAMA_SPEC_SPILL_MIN_MATCH");
    return std::max(3, e != nullptr ? atoi(e) : host_mode == 1 ? SPEC_SPILL_MIN_MATCH_GATE : SPEC_SPILL_MIN_MATCH_CM);
}

// k15-spec: the shortest repeat the default spill drafts take on a long prompt (LLAMA_SPEC_SPILL_LONG_PROMPT tokens, default 4096). A long
// prompt (a document, code to edit) is copied in long runs, where a repeat of a few tokens already predicts the next ones; the cost
// policy still prices each window. host_mode 1 (the hybrid gate, G4 LFM2.5 4 GB) only: the CPU-miss rule's mode 3 (8 GB) already takes
// every match; chat prompts keep server_spec_spill_min_match (16). Changes text, KLD-gated (plans/k15_models.md, G4 k15spec kld4:
// short192 + long48 inside the calibrated band, = the 16+ default) and timed (G4 4 GB 8k 122-124 -> 164-166 tok/s, chat unchanged).
// LLAMA_SPEC_SPILL_MIN_MATCH_LONG=16: off.
#ifndef SPEC_SPILL_MIN_MATCH_LONG_GATE
#define SPEC_SPILL_MIN_MATCH_LONG_GATE 4
#endif
static int server_spec_spill_long_prompt() {
    const char * e = getenv("LLAMA_SPEC_SPILL_LONG_PROMPT");
    return std::max(1, e != nullptr ? atoi(e) : 4096);
}

// k15-spec: with the CPU-miss rule (mode 3, G4 LFM2.5 8 GB) a long prompt takes no drafts: their verify windows read host-spilled experts
// (4 rows 13.4 ms, 8 rows 33 ms against a 4.5 ms plain round), and the pinned host path alone is what wins (G4 8 GB 8k: pinned without
// drafts 221.3 tok/s, with drafts 196.1, mode 0 206.0; chat 170.7 / 170.0). A repeat no longer than the longest match (64) never drafts.
// The text is that of the pinned path without drafts (KLD-gated, plans/k15_models.md). LLAMA_SPEC_SPILL_LONG_CM=3: drafts from 3+ (as before)
#ifndef SPEC_SPILL_MIN_MATCH_LONG_CM
#define SPEC_SPILL_MIN_MATCH_LONG_CM 65
#endif
static int server_spec_spill_min_match_long(int32_t host_mode, int min_match) {
    const char * e = getenv("LLAMA_SPEC_SPILL_MIN_MATCH_LONG");
    if (e != nullptr) {
        return std::max(3, atoi(e));
    }
    if (getenv("LLAMA_SPEC_SPILL_MIN_MATCH") != nullptr) {
        return min_match;
    }
    if (host_mode == 1) {
        return SPEC_SPILL_MIN_MATCH_LONG_GATE;
    }
    const char * c = getenv("LLAMA_SPEC_SPILL_LONG_CM");
    return c != nullptr ? std::max(3, atoi(c)) : SPEC_SPILL_MIN_MATCH_LONG_CM;
}

// k15-lfm2: see server_context::spill_pinned (LLAMA_SPEC_SPILL_UNPIN_LONG=0: keep the pin for requests that cannot draft)
static bool server_spec_spill_unpin_long() {
    const char * e = getenv("LLAMA_SPEC_SPILL_UNPIN_LONG");
    return e == nullptr || atoi(e) != 0;
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
// k15: the CPU-miss rule's spill mode needs host threads: it pins the small-batch host path on (no timing trials), so every drafted
// verify window computes its missing experts on the host. On a Kaggle T4 half (2 threads; the CPU-miss calibration measured 680-800 us per
// host expert) LFM2.5 mode 3 decoded 10-35% slower than mode 0 (k14) at 8 / 12 / 16 GB in chat / 8k / 16k (k15r kgB, LLAMA_SPEC_SPILL=0
// arm = k14), while G4 (24 threads) and L4 (6) gain from it. Below LLAMA_SPEC_SPILL_CM_MIN_THREADS (4) the default is mode 0, the k14
// behaviour (no spill drafts); 0 turns the guard off. An explicit LLAMA_SPEC_SPILL keeps its meaning.
static int server_spec_spill_cm_min_threads() {
    const char * e = getenv("LLAMA_SPEC_SPILL_CM_MIN_THREADS");
    return e != nullptr ? std::max(0, atoi(e)) : 4;
}

static int server_spec_spill_mode(const std::string & arch, int32_t host_mode, int32_t n_threads = -1) {
    if (const char * e = getenv("LLAMA_SPEC_SPILL")) {
        return std::max(0, std::min(3, atoi(e)));
    }
    if (host_mode == 0) {
        return 0;
    }
    // per (model, host path) after the KLD gate and the G4 timing (plans/k13_lfm.md section 10)
    if (arch == "lfm2moe") {
        if (host_mode != 1 && n_threads > 0 && n_threads < server_spec_spill_cm_min_threads()) {
            return 0;
        }
        return host_mode == 1 ? SPEC_SPILL_LFM_GATE : SPEC_SPILL_LFM_CM;
    }
    if (arch == "gemma4") {
        return host_mode == 1 ? SPEC_SPILL_GEMMA4_GATE : SPEC_SPILL_GEMMA4_CM;
    }
    return 0;
}
