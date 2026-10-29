// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool verify;                  // the caller marks the next batches as speculative verify batches (llama_set_verify)
    bool verify_exact;            // k11-decode: verify batches are computed batch invariant without LLAMA_MTP_EXACT (llama_set_verify_exact)
    bool batch_invariant;         // the current ubatch is a verify batch computed batch invariant (LLAMA_MTP_EXACT=1)
    uint32_t exact_split;         // batch invariant ubatch: rows whose attention sees fewer KV cells (llm_graph_exact_split)
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    bool training;           // set by llama_opt_init()

    size_t moe_cache_size;

    int32_t  weight_stream;
    bool     spec_draft; // llama_context_params::spec_draft
    int32_t  n_state_capture; // llama_context_params::n_state_capture (-1: default)
    int32_t  weight_stream_io;
    int32_t  weight_stream_threads;
    uint32_t weight_stream_ram_mib;
    int32_t  moe_host_cache_mib;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
