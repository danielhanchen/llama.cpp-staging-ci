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
#include <string>

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
// opt-in (LLAMA_SPEC_EXACT_FROZEN=1) until the host path of a verify window is shown to give the bytes of single token decodes
static bool server_spec_exact_frozen() {
    const char * e = getenv("LLAMA_SPEC_EXACT_FROZEN");
    return e != nullptr && atoi(e) != 0;
}
