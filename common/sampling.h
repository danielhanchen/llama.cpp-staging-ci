// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// k15-fn: the chain picks the argmax of the raw logits (greedy, no logit bias, no active penalties, no probabilities requested)
bool common_sampler_device_greedy(const struct common_sampler * gsmpl);
// k15-fn: the request picks the argmax with a token set excluded (every bias -inf, e.g. ignore_eos): true and mask = those tokens, ascending
bool common_sampler_device_greedy_masked(const struct common_sampler * gsmpl, std::vector<llama_token> & mask);
// k15-fn: the chain gives the same token and candidates from the k_dev highest raw logits of a row (a device top-k, see
// common_sampler_device_topk in sampling.cpp for the conditions): k_dev, or 0 when the request needs the full row
int32_t common_sampler_device_topk(const struct common_sampler * gsmpl, int32_t k_dev);
// k15-fn: the rows' logits are device top-k candidates (put in id order before the chain)
void common_sampler_set_device_topk(struct common_sampler * gsmpl, bool on);
// k15-fn: the sampler takes the token a device greedy chain picked for the row (the server sets it for device greedy requests)
void common_sampler_set_device_token(struct common_sampler * gsmpl, bool on);
// k19-gramdev: a device top-k request with a grammar or a reasoning budget also needs its raw rows on the host
// (llama_set_sampler_keep_logits) for the positions the candidates cannot decide
bool common_sampler_device_topk_needs_raw(const struct common_sampler * gsmpl);
// k19-gramdev: LLAMA_DEVICE_TOPK_GRAMMAR (default 1); 0: grammar / reasoning budget requests sample the full rows on the host as before
bool common_sampler_device_topk_grammar_enabled();
// k19-gramdev: rows sampled from device top-k candidates with a grammar / reasoning budget; rejected = the grammar rejected the chain's
// token, fallback = the candidates could not decide the resample, forced = the reasoning budget was forcing; full = full rows sampled
struct common_sampler_dev_stats {
    int64_t n_rows     = 0;
    int64_t n_rejected = 0;
    int64_t n_fallback = 0;
    int64_t n_forced   = 0;
    int64_t n_full     = 0;
};
common_sampler_dev_stats common_sampler_get_dev_stats(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// as above, but verifies by rejection sampling; draft_q holds the draft's candidates per token
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// k19-sharednoise: shared-noise (Gumbel-max) sampling, LLAMA_SHARED_NOISE=1 (default off; 0 = off, the kill switch).
// The final draw of a sampled chain (the one `dist` makes) becomes argmax_i(l_i + g(seed, n, i)) over the candidates the chain left
// (after penalties, truncation and temperature), with g a Gumbel(0, 1) variate computed from a counter-based hash of the request
// seed, the number n of tokens the sampler has accepted so far, and the token id. That is an exact sample of softmax(l): the
// target's distribution does not change, only how the randomness maps to a token. A drafter that picks argmax(draft logit / T + the
// same g) for the token it drafts at n, n+1, .. then agrees with the target almost as often as under greedy decoding, and a plain
// sample-and-match verify keeps the output an exact sample of the target (it is the token the target draws with that noise).
bool  common_shared_noise_enabled();
// the Gumbel(0, 1) variate of (seed, pos, token)
float common_shared_noise_gumbel(uint32_t seed, uint64_t pos, llama_token token);
// the shared-noise final stage alone (tests): argmax(logit + g(seed, n_accepted, id)), p = softmax(logit)
struct llama_sampler * common_sampler_init_shared_noise(uint32_t seed);

// what a drafter needs to draw its tokens with the target's noise: the key of the target's next draw and a copy of the target
// chain's penalties / truncation / temperature, applied to the draft's own candidates (an approximation of the target chain: it only
// moves acceptance, never the output distribution)
struct common_shared_noise_draft {
    bool     on   = false;
    uint32_t seed = 0;
    uint64_t pos  = 0;                            // the key of the target's next draw: the first drafted token
    std::vector<enum common_sampler_type> samplers;
    float    temp        = 1.0f;
    int32_t  top_k       = 0;
    float    top_p       = 1.0f;
    float    min_p       = 0.0f;
    int32_t  pen_last_n  = 0;
    float    pen_repeat  = 1.0f;
    float    pen_freq    = 0.0f;
    float    pen_present = 0.0f;
    std::vector<llama_token> window;              // the target's last pen_last_n accepted tokens
    std::vector<std::pair<llama_token, float>> scratch;
};

// fills sn from the target sampler: false (sn.on = false) when its chain does not end in the shared-noise stage
bool common_sampler_shared_noise_draft_init(const struct common_sampler * gsmpl, common_shared_noise_draft & sn);

// the token a drafter proposes at draft index i (0 = the token after the target's last one) from its candidates cand[0..n) (raw
// logits); drafted = the i tokens drafted before it. p_pick (optional): the token's probability under the mirrored chain
llama_token common_shared_noise_draft_pick(common_shared_noise_draft & sn, int32_t i, const llama_token * drafted,
        const llama_token_data * cand, size_t n, float * p_pick);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

// add the strings that are a single token in the vocab to the preserved tokens
void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens);

// add grammar triggers, a trigger word that is a single token becomes a token trigger and must be a preserved token
void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
