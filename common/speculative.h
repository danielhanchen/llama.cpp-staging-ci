// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// true if any of the types drafts with a draft context (draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark)
bool common_speculative_uses_draft_ctx(const std::vector<enum common_speculative_type> & types);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

// the largest ubatch of the context of a separate MTP drafter (LLAMA_DRAFT_UBATCH_MAX, default 512, 0 = no cap)
uint32_t common_draft_ubatch_max();

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // candidate distribution per drafted token; set it to make draft-simple and draft-mtp sample
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    // the target's truncation, mirrored by the probabilistic draft sampler (LLAMA_SPEC_DRAFT_MIRROR=0: top-k 10 only)
    int32_t top_k = 40;
    float   top_p = 1.0f;
    float   min_p = 0.0f;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// k11 (LLAMA_LM_CKPT_MERGE): process() of a segment of the target batch whose first entry is target batch row row_offset (the target's nextn
// rows are read from there); rows_ok: every impl of spec supports it (eagle3 / dflash do not)
bool common_speculative_process_rows(common_speculative * spec, const common_batch & batch, int32_t row_offset);
bool common_speculative_rows_ok(const common_speculative * spec);

// k15-pp3 (LLAMA_MTP_PROMPT_OVERLAP): process_rows on target rows the caller waited for with llama_nextn_rows_wait, while the target's
// decode may still compute later ubatches; rows_early_ok says whether every impl reads only those rows of the target
bool common_speculative_rows_early_ok(const common_speculative * spec);
bool common_speculative_process_rows_early(common_speculative * spec, const common_batch & batch, int32_t row_offset);

// k19-host (LLAMA_MTP_PROMPT_BG): process_rows() reads the target's rows through h (row i at h + i * floats), rows [0, n_lo) from a copy made
// here, and calls nothing of the target context, so it can run on a worker beside the target's next decode (which may overwrite rows
// [0, n_lo)). rows_ext_floats: floats per row where every impl supports it, else 0. Set and clear on the thread that owns the target
int64_t common_speculative_rows_ext_floats(const common_speculative * spec);
bool common_speculative_rows_ext_set(common_speculative * spec, const float * h, int32_t n_rows, int32_t n_lo);
void common_speculative_rows_ext_clear(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
