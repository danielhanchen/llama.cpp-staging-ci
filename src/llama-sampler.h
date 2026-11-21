// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
// a chain whose backend_init was called (false for any other sampler)
bool llama_sampler_chain_is_init(const llama_sampler * sampler);
// k of a backend sampler whose first graph stage is a top-k (a chain whose first sampler is a backend top-k), else 0
int32_t llama_sampler_backend_top_k_first(const llama_sampler * sampler);
// the backend stages of such a chain after its leading top-k (build_sampling ran that top-k over several rows at once and
// passes this row's candidates and logits, as the top-k stage leaves them)
void llama_sampler_backend_apply_after_top_k(llama_sampler * sampler, ggml_context * ctx, ggml_cgraph * gf, llama_sampler_data * data);
// the sampler keeps graph inputs bound to the graph built last (dist, penalties, logit_bias): such a graph is not reused once
// another graph was built with the sampler (llm_graph_input_sampling::can_reuse)
bool llama_sampler_graph_bound(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
