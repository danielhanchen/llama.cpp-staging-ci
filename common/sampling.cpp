// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"
#include "../src/llama-ext.h" // llama_get_raw_logits_ith

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    // greedy_fast: the chain can only pick the highest biased logit (see common_sampler_greedy_fast), so common_sampler_sample
    // takes the argmax of the logits + bias_dense instead of building and sorting n_vocab candidates
    bool greedy_fast = false;
    std::vector<float> bias_dense; // empty: no bias
    std::vector<std::pair<llama_token, float>> bias_sparse; // the tokens with a bias (summed), ascending ids
    std::vector<float> bias_saved;                          // scratch: their raw logits during the scan
    std::vector<float> row;                                 // scratch: a copy of the logits row (mode 3)
    bool pen_fast = false;                                  // k15: the fast path applies the repetition penalties (common_sampler_penalty_fast)
    std::vector<std::pair<llama_token, int>> pen_cnt;       // scratch: the window's tokens and their counts
    std::vector<std::pair<llama_token, float>> sp;          // scratch: biased and penalized tokens with their bias, ascending ids
    bool dev_token = false;                                 // k15-fn: take the token of a device greedy sampler (common_sampler_set_device_token)
    bool dev_topk = false;                                  // k15-fn: the row's logits are the device top-k candidates (common_sampler_set_device_topk)
    int32_t dev_n_bias = 0;                                 // k15-fn: tokens with a bias (logit_bias + suppress tokens), for common_sampler_device_topk
    bool dev_bias_pos  = false;                             //         one of those biases can raise a logit
    // k19-gramdev: the device top-k serves a request with a grammar or a reasoning budget (common_sampler_device_topk); the raw rows are
    // on the host too (llama_set_sampler_keep_logits) for the positions the candidates cannot decide
    bool    dev_gram  = false;
    int32_t dev_kneed = 0;                                  //         k + m of common_sampler_device_topk
    common_sampler_dev_stats dev_stats = {};

    // k19-gramdev: the full raw row of idx (the candidates of a device top-k are in cur otherwise)
    void set_logits_full(struct llama_context * ctx, int idx) {
        const float * logits = llama_get_raw_logits_ith(ctx, idx);
        GGML_ASSERT(logits != nullptr && "device top-k with a grammar / reasoning budget needs the raw rows (llama_set_sampler_keep_logits)");
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        cur.resize(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
        cur_p = { cur.data(), cur.size(), -1, false };
    }

    // the token a backend sampler picked for row idx, if this sampler takes it (backend sampling, or the device greedy chain of the server)
    llama_token sampled_token(struct llama_context * ctx, int idx) const {
        return params.backend_sampling || dev_token ? llama_get_sampled_token_ith(ctx, idx) : LLAMA_TOKEN_NULL;
    }

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
            if (dev_topk) {
                // k15-fn: in id order, as the full row, for the samplers that depend on the order (greedy, dist before any sort)
                std::sort(cur.begin(), cur.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.id < b.id; });
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

// LLAMA_SAMPLE_GREEDY_FAST: 0 = off (the chain), 1 = one running max over logits + bias, 2 (default) = the max in 16 independent
// lanes over the raw logits with the biased tokens handled from the sparse bias list, 3 = 2 on a memcpy copy of the row
static int common_sampler_greedy_fast_mode() {
    static const int mode = [] {
        const char * e = getenv("LLAMA_SAMPLE_GREEDY_FAST");
        return e == nullptr ? 2 : atoi(e);
    }();
    return mode;
}

static bool common_sampler_penalty_active(const common_params_sampling & params) {
    return params.penalty_last_n != 0 && (params.penalty_repeat != 1.0f || params.penalty_freq != 0.0f || params.penalty_present != 0.0f);
}

// k15: repetition penalties keep greedy sampling an argmax: the penalties sampler changes only the logits of the tokens of its window
// (the last penalty_last_n accepted tokens, the same ones common_sampler::prev holds when the window fits its ring), and the samplers
// after it keep the highest logit; that needs the penalties before every truncating sampler (top-k / top-p / min-p / top-n-sigma would
// otherwise pick among the unpenalized top). G4 LFM2.5 (its GGUF sets repeat penalty 1.05): the full chain built and sorted 128k
// candidates per sampled row. LLAMA_SAMPLE_GREEDY_PENALTY=0: the full chain, as before
static bool common_sampler_penalty_fast(const common_params_sampling & params) {
    static const bool on = [] {
        const char * e = getenv("LLAMA_SAMPLE_GREEDY_PENALTY");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!on || common_sampler_greedy_fast_mode() < 2 || params.penalty_last_n <= 0 || params.penalty_last_n > std::max(32, params.n_prev)) {
        return false;
    }
    // exactly one penalties stage (a second one penalizes again), and only samplers that leave the logits unchanged before it: a
    // temperature other than 1 scales the logits the frequency / presence penalties are subtracted from (temperature 0 keeps only
    // the max), and the fast path penalizes the raw logits
    if (std::count(params.samplers.begin(), params.samplers.end(), COMMON_SAMPLER_TYPE_PENALTIES) != 1) {
        return false;
    }
    for (const auto & t : params.samplers) {
        if (t == COMMON_SAMPLER_TYPE_PENALTIES) {
            return true;
        }
        if (t == COMMON_SAMPLER_TYPE_TOP_K || t == COMMON_SAMPLER_TYPE_TOP_P || t == COMMON_SAMPLER_TYPE_MIN_P || t == COMMON_SAMPLER_TYPE_TOP_N_SIGMA) {
            return false;
        }
        if (t == COMMON_SAMPLER_TYPE_TEMPERATURE && (params.temp != 1.0f || params.dynatemp_range != 0.0f)) {
            return false;
        }
    }
    return false;
}

// whether every sampler of the chain built from params keeps the token with the highest (biased) logit and the chain ends in
// greedy: then the sampled token is the argmax of the biased logits. LLAMA_SAMPLE_GREEDY_FAST=0 disables it.
static bool common_sampler_greedy_fast(const common_params_sampling & params, const llama_sampler * grmr, const llama_sampler * rbudget) {
    if (common_sampler_greedy_fast_mode() == 0 || grmr != nullptr || rbudget != nullptr || params.mirostat != 0 || params.n_probs != 0 || params.samplers.empty()) {
        return false;
    }
    const bool greedy =
        (params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
        (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1);
    if (!greedy) {
        return false;
    }
    for (const auto & t : params.samplers) {
        switch (t) {
            case COMMON_SAMPLER_TYPE_TOP_K:       // keep the top k
            case COMMON_SAMPLER_TYPE_TOP_P:       // keep the most probable tokens up to p
            case COMMON_SAMPLER_TYPE_MIN_P:       // keep p >= min_p * p_max
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: // keep logits within n sigma of the max
            case COMMON_SAMPLER_TYPE_TEMPERATURE: // temp 0 (checked above for the last one): keeps the max
                break;
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (common_sampler_penalty_active(params) && !common_sampler_penalty_fast(params)) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (params.dry_multiplier != 0.0f && params.dry_base >= 1.0f && params.dry_penalty_last_n != 0) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P: // can drop the most probable token
                if (params.typ_p < 1.0f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_XTC: // can drop the most probable tokens
                if (params.xtc_probability > 0.0f) {
                    return false;
                }
                break;
            default: // infill, adaptive-p, ...
                return false;
        }
    }
    return true;
}

// k19-sharednoise (see sampling.h): the shared-noise final stage
bool common_shared_noise_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_SHARED_NOISE");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static inline uint64_t sn_mix64(uint64_t x) {
    // splitmix64 finalizer (a bijection with full avalanche)
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

float common_shared_noise_gumbel(uint32_t seed, uint64_t pos, llama_token token) {
    uint64_t x = sn_mix64(((uint64_t) seed << 32) ^ 0x9e3779b97f4a7c15ull ^ sn_mix64(pos + 0x632be59bd9b4e019ull));
    x = sn_mix64(x ^ ((uint64_t) (uint32_t) token * 0xc2b2ae3d27d4eb4full));
    // u in (0, 1): 53 bits, never 0 or 1, so g is finite (|g| < 38)
    const double u = ((double) (x >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return (float) -std::log(-std::log(u));
}

// g never exceeds -log(-log(1 - 2^-54)) = 37.43: a candidate whose logit is that far below the best perturbed one cannot win
static constexpr float SN_G_MAX = 37.5f;

struct sn_sampler_ctx {
    uint32_t seed;     // as requested (LLAMA_DEFAULT_SEED: a random one per reset)
    uint32_t seed_cur; // in use
    uint64_t n_acc;    // tokens accepted since the last reset: the key of the next draw
};

static uint32_t sn_resolve_seed(uint32_t seed) {
    if (seed != LLAMA_DEFAULT_SEED) {
        return seed;
    }
    std::random_device rd;
    return rd();
}

static const char * sn_name(const struct llama_sampler * /*smpl*/) {
    return "shared-noise";
}

static void sn_accept(struct llama_sampler * smpl, llama_token /*token*/) {
    ((sn_sampler_ctx *) smpl->ctx)->n_acc++;
}

static void sn_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    const auto * ctx = (const sn_sampler_ctx *) smpl->ctx;
    if (cur_p->size == 0) {
        cur_p->selected = -1;
        return;
    }
    if (cur_p->size == 1) {
        cur_p->selected = 0;
        cur_p->data[0].p = 1.0f;
        return;
    }
    // probabilities as dist leaves them (softmax of the candidates' logits, in place, any order)
    size_t i_max = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[i_max].logit) {
            i_max = i;
        }
    }
    const float max_l = cur_p->data[i_max].logit;
    double sum = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float p = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        sum += p;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p = (float) (cur_p->data[i].p / sum);
    }
    // Gumbel-max: argmax(logit + g); -inf logits never win, ties go to the lower index
    const uint64_t pos = ctx->n_acc;
    size_t best   = i_max;
    float  best_v = max_l + common_shared_noise_gumbel(ctx->seed_cur, pos, cur_p->data[i_max].id);
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float l = cur_p->data[i].logit;
        if (i == i_max || !(l + SN_G_MAX >= best_v)) {
            continue;
        }
        const float v = l + common_shared_noise_gumbel(ctx->seed_cur, pos, cur_p->data[i].id);
        if (v > best_v || (v == best_v && i < best)) {
            best_v = v;
            best   = i;
        }
    }
    cur_p->selected = (int64_t) best;
}

static void sn_reset(struct llama_sampler * smpl) {
    auto * ctx = (sn_sampler_ctx *) smpl->ctx;
    ctx->seed_cur = sn_resolve_seed(ctx->seed);
    ctx->n_acc    = 0;
}

static struct llama_sampler * sn_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const sn_sampler_ctx *) smpl->ctx;
    auto * res = common_sampler_init_shared_noise(ctx->seed);
    *(sn_sampler_ctx *) res->ctx = *ctx;
    return res;
}

static void sn_free(struct llama_sampler * smpl) {
    delete (sn_sampler_ctx *) smpl->ctx;
}

static void sn_copy_state(const struct llama_sampler * src, struct llama_sampler * dst) {
    *(sn_sampler_ctx *) dst->ctx = *(const sn_sampler_ctx *) src->ctx;
}

static struct llama_sampler_i sn_sampler_i = {
    /* .name              = */ sn_name,
    /* .accept            = */ sn_accept,
    /* .apply             = */ sn_apply,
    /* .reset             = */ sn_reset,
    /* .clone             = */ sn_clone,
    /* .free              = */ sn_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ sn_copy_state,
};

struct llama_sampler * common_sampler_init_shared_noise(uint32_t seed) {
    return llama_sampler_init(&sn_sampler_i, new sn_sampler_ctx { seed, sn_resolve_seed(seed), 0 });
}

// the shared-noise stage of a chain (its last sampler), or nullptr
static const sn_sampler_ctx * sn_of_chain(const struct llama_sampler * chain) {
    if (chain == nullptr) {
        return nullptr;
    }
    const int32_t n = llama_sampler_chain_n(chain);
    if (n <= 0) {
        return nullptr;
    }
    const llama_sampler * last = llama_sampler_chain_get(const_cast<llama_sampler *>(chain), n - 1);
    return last != nullptr && last->iface == &sn_sampler_i ? (const sn_sampler_ctx *) last->ctx : nullptr;
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;
    std::vector<llama_logit_bias> merged_bias;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
        merged_bias = std::move(merged);
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // Keep distribution sampling when callers request probabilities.
            const bool greedy = params.n_probs == 0 && !params.samplers.empty() &&
                ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
                 (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));
            // k19-sharednoise: LLAMA_SHARED_NOISE=1: the sampled draw is the counter-based Gumbel-max stage (the same distribution as
            // dist). Not with backend sampling (the chain runs on the device there)
            const bool shared_noise = !greedy && common_shared_noise_enabled() && !params.backend_sampling;
            if (shared_noise) {
                static bool logged = false;
                if (!logged) {
                    logged = true;
                    LOG_INF("%s: shared noise: sampled draws are counter-based Gumbel-max (LLAMA_SHARED_NOISE=1)\n", __func__);
                }
            }
            samplers.push_back(greedy ? llama_sampler_init_greedy() :
                               shared_noise ? common_sampler_init_shared_noise(params.seed) : llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        // mix it, the chain and the draft are seeded from this one too
        /* .rng     = */ std::mt19937((sn_of_chain(chain) ? sn_of_chain(chain)->seed_cur : llama_sampler_get_seed(chain)) ^ 0x9e3779b9u),
        /* .greedy_fast = */ false,
        /* .bias_dense  = */ {},
        /* .bias_sparse = */ {},
        /* .bias_saved  = */ {},
        /* .row         = */ {},
        /* .pen_fast    = */ false,
        /* .pen_cnt     = */ {},
        /* .sp          = */ {},
        /* .dev_token   = */ false,
        /* .dev_topk    = */ false,
        /* .dev_n_bias  = */ 0,
        /* .dev_bias_pos = */ false,
    };

    {
        std::vector<llama_token> bt;
        for (const auto & lb : merged_bias) {
            bt.push_back(lb.token);
            result->dev_bias_pos = result->dev_bias_pos || !(lb.bias <= 0.0f);
        }
        std::sort(bt.begin(), bt.end());
        result->dev_n_bias = (int32_t) (std::unique(bt.begin(), bt.end()) - bt.begin());
    }

    result->greedy_fast = common_sampler_greedy_fast(params, grmr, rbudget);
    // the fast path applies the penalties only when the chain has a penalties stage (common_sampler_penalty_fast): a chain without one
    // never penalizes, whatever the penalty parameters
    result->pen_fast    = result->greedy_fast && common_sampler_penalty_active(params) && common_sampler_penalty_fast(params);
    // only LLAMA_SAMPLE_GREEDY_FAST=1 reads bias_dense: else the sparse list is built from the biases alone (the same sums, in the same
    // order), not from a fresh n_vocab array and a scan of it (0.5 ms per request with ignore_eos on a 248k vocab). LLAMA_SAMPLE_BIAS_SPARSE=0: as before
    static const bool bias_sparse_only = [] {
        const char * e = getenv("LLAMA_SAMPLE_BIAS_SPARSE");
        return e == nullptr || atoi(e) != 0;
    }();
    if (result->greedy_fast && !merged_bias.empty() && bias_sparse_only && common_sampler_greedy_fast_mode() != 1) {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<std::pair<llama_token, float>> lbs;
        lbs.reserve(merged_bias.size());
        for (const auto & lb : merged_bias) {
            if (lb.token >= 0 && lb.token < n_vocab) {
                lbs.emplace_back(lb.token, lb.bias);
            }
        }
        std::stable_sort(lbs.begin(), lbs.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        for (size_t k = 0; k < lbs.size();) {
            float sum = 0.0f;
            size_t e = k;
            for (; e < lbs.size() && lbs[e].first == lbs[k].first; ++e) {
                sum += lbs[e].second;
            }
            if (sum != 0.0f) {
                result->bias_sparse.emplace_back(lbs[k].first, sum);
            }
            k = e;
        }
    } else if (result->greedy_fast && !merged_bias.empty()) {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        result->bias_dense.assign(n_vocab, 0.0f);
        for (const auto & lb : merged_bias) {
            if (lb.token >= 0 && lb.token < n_vocab) {
                result->bias_dense[lb.token] += lb.bias;
            }
        }
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (result->bias_dense[i] != 0.0f) {
                result->bias_sparse.emplace_back(i, result->bias_dense[i]);
            }
        }
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    auto * res = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .rng     = */ gsmpl->rng,
        /* .greedy_fast = */ gsmpl->greedy_fast,
        /* .bias_dense  = */ gsmpl->bias_dense,
        /* .bias_sparse = */ gsmpl->bias_sparse,
        /* .bias_saved  = */ {},
        /* .row         = */ {},
        /* .pen_fast    = */ gsmpl->pen_fast,
        /* .pen_cnt     = */ {},
        /* .sp          = */ {},
        /* .dev_token   = */ gsmpl->dev_token,
        /* .dev_topk    = */ gsmpl->dev_topk,
        /* .dev_n_bias  = */ gsmpl->dev_n_bias,
        /* .dev_bias_pos = */ gsmpl->dev_bias_pos,
    };
    res->dev_gram  = gsmpl->dev_gram;
    res->dev_kneed = gsmpl->dev_kneed;
    return res;
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->rng        = src->rng;
    dst->t_total_us = src->t_total_us;
    dst->greedy_fast = src->greedy_fast;
    dst->bias_dense  = src->bias_dense;
    dst->bias_sparse = src->bias_sparse;
    dst->pen_fast    = src->pen_fast;
    dst->dev_token   = src->dev_token;
    dst->dev_topk    = src->dev_topk;
    dst->dev_n_bias  = src->dev_n_bias;
    dst->dev_bias_pos = src->dev_bias_pos;
    dst->dev_gram     = src->dev_gram;
    dst->dev_kneed    = src->dev_kneed;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

static int64_t common_sampler_device_topk_need(const struct common_sampler * gsmpl);

void common_sampler_set_device_topk(struct common_sampler * gsmpl, bool on) {
    if (gsmpl != nullptr) {
        gsmpl->dev_topk  = on;
        gsmpl->dev_gram  = on && (gsmpl->grmr != nullptr || gsmpl->rbudget != nullptr);
        gsmpl->dev_kneed = on ? (int32_t) std::max<int64_t>(0, common_sampler_device_topk_need(gsmpl)) : 0;
    }
}

bool common_sampler_device_topk_needs_raw(const struct common_sampler * gsmpl) {
    return gsmpl != nullptr && (gsmpl->grmr != nullptr || gsmpl->rbudget != nullptr);
}

common_sampler_dev_stats common_sampler_get_dev_stats(const struct common_sampler * gsmpl) {
    return gsmpl != nullptr ? gsmpl->dev_stats : common_sampler_dev_stats {};
}

// k19-gramdev: LLAMA_DEVICE_TOPK_GRAMMAR=0: requests with a grammar or a reasoning budget copy the full rows and sample on the host, as before
bool common_sampler_device_topk_grammar_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_DEVICE_TOPK_GRAMMAR");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

void common_sampler_set_device_token(struct common_sampler * gsmpl, bool on) {
    if (gsmpl != nullptr) {
        gsmpl->dev_token = on;
    }
}

bool common_sampler_device_greedy(const struct common_sampler * gsmpl) {
    // k15-fn: the sampled token is the argmax of the raw logits (no bias, no active penalties): a device argmax that returns the lowest id
    // on a tie gives the same token without copying the logits
    const bool ok = gsmpl != nullptr && gsmpl->greedy_fast && !gsmpl->pen_fast && gsmpl->bias_dense.empty() && gsmpl->bias_sparse.empty();
    return ok;
}

bool common_sampler_device_greedy_masked(const struct common_sampler * gsmpl, std::vector<llama_token> & mask) {
    // k15-fn: the argmax of the logits with some tokens excluded (every bias is -inf, e.g. ignore_eos): the device greedy_masked chain
    // gives the same token
    mask.clear();
    if (gsmpl == nullptr || !gsmpl->greedy_fast || gsmpl->pen_fast || gsmpl->bias_sparse.empty()) {
        return false;
    }
    for (const auto & [tok, bias] : gsmpl->bias_sparse) {
        if (!(std::isinf(bias) && bias < 0.0f)) {
            mask.clear();
            return false;
        }
        mask.push_back(tok);
    }
    return true;
}

// k + m of common_sampler_device_topk (the candidates the chain needs), or 0 when the chain cannot finish from a device top-k
static int64_t common_sampler_device_topk_need(const struct common_sampler * gsmpl) {
    // k15-fn: the chain runs on the k_dev highest raw logits of the row (a device top-k) and still gives the token, the
    // candidates and the probabilities of the full row, when
    //   - everything before the chain's top_k (or before its final greedy) only lowers logits or does nothing: penalties with
    //     repeat >= 1, frequency >= 0, presence >= 0; biases <= 0 (logit_bias, suppress tokens, ignore_eos); inactive dry,
    //     top_n_sigma, typical, top_p, min_p, xtc;
    //   - k + m <= k_dev, k the top_k (1 for greedy), m the tokens whose logit can change before it (the penalty window + the
    //     biased tokens): the k highest lowered logits are then among the k + m highest raw ones;
    //   - no mirostat, adaptive-p, infill, dynamic temperature or probabilities output.
    // The host top_k breaks ties by the lower id, so the full row and its device top-k give the same sorted candidates.
    // k19-gramdev: a grammar or a reasoning budget qualifies too (LLAMA_DEVICE_TOPK_GRAMMAR=0: not), with the raw rows also copied to
    // the host: common_sampler_sample_dev_gram decides each position from the candidates when that is provably the full row's result
    // and samples the full row otherwise.
    // Returns k_dev when the request qualifies, 0 otherwise (the raw logits are copied as before)
    if (gsmpl == nullptr || gsmpl->dev_bias_pos) {
        return 0;
    }
    if ((gsmpl->grmr != nullptr || gsmpl->rbudget != nullptr) && !common_sampler_device_topk_grammar_enabled()) {
        return 0;
    }
    const auto & p = gsmpl->params;
    if (p.mirostat != 0 || p.n_probs != 0 || p.samplers.empty()) {
        return 0;
    }
    const bool pen_active = p.penalty_last_n != 0 && (p.penalty_repeat != 1.0f || p.penalty_freq != 0.0f || p.penalty_present != 0.0f);
    if (pen_active && (p.penalty_last_n < 0 || !(p.penalty_repeat >= 1.0f) || !(p.penalty_freq >= 0.0f) || !(p.penalty_present >= 0.0f))) {
        return 0;
    }
    int32_t k = 0;
    for (size_t i = 0; i < p.samplers.size(); ++i) {
        const auto t = p.samplers[i];
        if (t == COMMON_SAMPLER_TYPE_TOP_K && p.top_k > 0) {
            k = p.top_k;
            break;
        }
        switch (t) {
            case COMMON_SAMPLER_TYPE_PENALTIES:   break;
            case COMMON_SAMPLER_TYPE_TOP_K:       break; // top_k <= 0: does nothing
            case COMMON_SAMPLER_TYPE_DRY:         if (p.dry_multiplier != 0.0f && p.dry_penalty_last_n != 0) { return 0; } break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: if (p.top_n_sigma > 0.0f) { return 0; } break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:   if (p.typ_p < 1.0f) { return 0; } break;
            case COMMON_SAMPLER_TYPE_TOP_P:       if (p.top_p < 1.0f) { return 0; } break;
            case COMMON_SAMPLER_TYPE_MIN_P:       if (p.min_p > 0.0f) { return 0; } break;
            case COMMON_SAMPLER_TYPE_XTC:         if (p.xtc_probability > 0.0f) { return 0; } break;
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                // a greedy chain ends with temperature 0 (common_sampler_init picks greedy for it)
                if (i + 1 == p.samplers.size() && p.temp == 0.0f && p.dynatemp_range == 0.0f) { k = 1; break; }
                return 0;
            default: return 0;
        }
        if (k > 0) {
            break;
        }
    }
    if (k <= 0) {
        return 0;
    }
    for (const auto t : p.samplers) {
        if (t == COMMON_SAMPLER_TYPE_ADAPTIVE_P || t == COMMON_SAMPLER_TYPE_INFILL ||
                (t == COMMON_SAMPLER_TYPE_TEMPERATURE && p.dynatemp_range != 0.0f)) {
            return 0;
        }
    }
    const int64_t m = (pen_active ? (int64_t) p.penalty_last_n : 0) + gsmpl->dev_n_bias;
    return (int64_t) k + m;
}

int32_t common_sampler_device_topk(const struct common_sampler * gsmpl, int32_t k_dev) {
    if (k_dev <= 0) {
        return 0;
    }
    const int64_t need = common_sampler_device_topk_need(gsmpl);
    return need > 0 && need <= (int64_t) k_dev ? k_dev : 0;
}

// k19-gramdev: one row of a device top-k request with a grammar or a reasoning budget (the candidates C, the k_dev highest raw logits,
// are in cur, in id order; the full raw row is on the host). Today's host path, step by step, with the reason each step on C gives the
// result of the same step on the full row:
//   - the reasoning budget passes the logits through unless it is forcing; forcing (a few tokens per request): the full row
//   - the chain on C picks the full row's token and leaves its candidates (common_sampler_device_topk, k + m <= k_dev): the grammar
//     plays no part in it (grammar_first = false), and the RNG of the chain advances alike
//   - the token is tested against the grammar alone: same token, same grammar state, same verdict (an untriggered lazy grammar
//     accepts every token, as before)
//   - rejected: the resample runs the chain on the grammar-masked row. Let tau = the lowest raw logit of C; every token outside C has
//     a raw logit <= tau, and the grammar and every stage before the chain's top_k only lower logits. If at least k + m tokens of C
//     are valid with a raw logit > tau, at least k of them keep it (m bounds the tokens a penalty or a bias can move), so every token
//     outside C has k tokens strictly above it and is not in the full row's top k: the top k of the masked row are in C, in the same
//     order (same values, ties by the lower id), and the chain after them sees the same candidates. Otherwise the full row is sampled
//     (same RNG state: one chain draw happened on each side)
//   - grammar_first = true: the same count on the masked C before the chain, else the full row
static llama_token common_sampler_sample_dev_gram(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    auto * grmr    = gsmpl->grmr;
    auto * rbudget = gsmpl->rbudget;
    auto * chain   = gsmpl->chain;
    auto & cur_p   = gsmpl->cur_p;
    auto & st      = gsmpl->dev_stats;

    st.n_rows++;

    // the host path of common_sampler_sample on the full row, from the start
    auto full_row = [&]() {
        st.n_full++;
        gsmpl->set_logits_full(ctx, idx);
        llama_sampler_apply(rbudget, &cur_p);
        if (grammar_first && grammar_should_apply(gsmpl)) {
            llama_sampler_apply(grmr, &cur_p);
        }
        llama_sampler_apply(chain, &cur_p);
        llama_token id = cur_p.data[cur_p.selected].id;
        if (grammar_first || !grammar_should_apply(gsmpl)) {
            return id;
        }
        {
            llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
            llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };
            llama_sampler_apply(grmr, &single_token_data_array);
            if (single_token_data_array.data[0].logit != -INFINITY) {
                return id;
            }
        }
        gsmpl->set_logits_full(ctx, idx);
        llama_sampler_apply(rbudget, &cur_p);
        if (grammar_should_apply(gsmpl)) {
            llama_sampler_apply(grmr, &cur_p);
        }
        llama_sampler_apply(chain, &cur_p);
        GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");
        return cur_p.data[cur_p.selected].id;
    };

    if (rbudget != nullptr && common_reasoning_budget_get_state(rbudget) == REASONING_BUDGET_FORCING) {
        st.n_forced++;
        return full_row();
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    // tokens of the masked C (grammar applied to cur_p) that are valid with a raw logit above tau: enough when >= k + m
    // LLAMA_DEVICE_TOPK_GRAMMAR_NOFALLBACK=1: test only, deliberately inexact (the candidates decide every position): shows that
    // test-dtopk-grammar can fail
    static const bool no_fallback = [] {
        const char * e = getenv("LLAMA_DEVICE_TOPK_GRAMMAR_NOFALLBACK");
        return e != nullptr && atoi(e) != 0;
    }();
    auto enough = [&](float tau, size_t n_c) {
        if ((int64_t) n_c >= (int64_t) n_vocab || no_fallback) {
            return true; // C is the row
        }
        int64_t n_ok = 0;
        for (size_t i = 0; i < cur_p.size; ++i) {
            n_ok += cur_p.data[i].logit > tau;
        }
        return gsmpl->dev_kneed > 0 && n_ok >= (int64_t) gsmpl->dev_kneed;
    };
    auto tau_of = [&]() {
        float tau = INFINITY;
        for (size_t i = 0; i < cur_p.size; ++i) {
            tau = std::min(tau, cur_p.data[i].logit);
        }
        return tau;
    };

    // cur_p holds C (set_logits)
    llama_sampler_apply(rbudget, &cur_p); // not forcing: passes through

    if (grammar_first && grammar_should_apply(gsmpl)) {
        const float  tau = tau_of();
        const size_t n_c = cur_p.size;
        llama_sampler_apply(grmr, &cur_p);
        if (!enough(tau, n_c)) {
            st.n_fallback++;
            return full_row();
        }
        llama_sampler_apply(chain, &cur_p);
        return cur_p.data[cur_p.selected].id;
    }

    llama_sampler_apply(chain, &cur_p);

    llama_token id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        if (single_token_data_array.data[0].logit != -INFINITY) {
            return id;
        }
    }

    st.n_rejected++;

    // resample on the masked C if it decides the full row's result
    gsmpl->set_logits(ctx, idx);
    const float  tau = tau_of();
    const size_t n_c = cur_p.size;
    llama_sampler_apply(rbudget, &cur_p);
    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }
    if (!enough(tau, n_c)) {
        st.n_fallback++;
        // the second half of the host path on the full row (the chain drew once on each side)
        st.n_full++;
        gsmpl->set_logits_full(ctx, idx);
        llama_sampler_apply(rbudget, &cur_p);
        if (grammar_should_apply(gsmpl)) {
            llama_sampler_apply(grmr, &cur_p);
        }
    }
    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    return cur_p.data[cur_p.selected].id;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    // k19-graph: only while work is queued (the second and later rows of a verify window found the stream idle: 2-4 syncs per pass)
    llama_synchronize_pending(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // k15-fn: a device greedy sampler (common_sampler_device_greedy, set by the server) picked the token and no logits were copied
    if (gsmpl->dev_token && gsmpl->greedy_fast && llama_get_sampled_probs_ith(ctx, idx) == nullptr && llama_get_sampled_logits_ith(ctx, idx) == nullptr) {
        const llama_token tok = gsmpl->sampled_token(ctx, idx);
        if (tok != LLAMA_TOKEN_NULL) {
            gsmpl->cur.resize(1);
            gsmpl->cur[0] = llama_token_data{tok, 0.0f, 1.0f};
            cur_p = { gsmpl->cur.data(), 1, 0, false };
            return tok;
        }
    }

    // greedy chain on the full logits (no backend sampler output): the argmax of the biased logits, the lowest id on a tie
    if (gsmpl->greedy_fast && gsmpl->sampled_token(ctx, idx) == LLAMA_TOKEN_NULL && llama_get_sampled_probs_ith(ctx, idx) == nullptr &&
            llama_get_sampled_logits_ith(ctx, idx) == nullptr) {
        const float * logits  = llama_get_logits_ith(ctx, idx);
        GGML_ASSERT(logits != nullptr);
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        int32_t best   = 0;
        float   best_v = -INFINITY;
        if (common_sampler_greedy_fast_mode() == 1) {
            const float * bias = gsmpl->bias_dense.empty() ? nullptr : gsmpl->bias_dense.data();
            best_v = bias ? logits[0] + bias[0] : logits[0];
            for (int32_t i = 1; i < n_vocab; ++i) {
                const float v = bias ? logits[i] + bias[i] : logits[i];
                if (v > best_v) {
                    best_v = v;
                    best   = i;
                }
            }
        } else {
            // the biased tokens leave the raw scan (their logit is set to -inf and restored after it) and compete with their biased
            // logit; the scan keeps 16 running maxima (one running max is a dependency chain as long as the vocab), then the lowest
            // id holding the max is searched
            float * lg = const_cast<float *>(logits);
            if (common_sampler_greedy_fast_mode() >= 3) {
                gsmpl->row.resize(n_vocab);
                memcpy(gsmpl->row.data(), logits, (size_t) n_vocab*sizeof(float));
                lg = gsmpl->row.data();
            }
            const std::vector<std::pair<llama_token, float>> * bsp = &gsmpl->bias_sparse;
            if (gsmpl->pen_fast) {
                // the penalties window: the last penalty_last_n accepted tokens (common_sampler_accept feeds the chain and prev alike)
                auto & pc = gsmpl->pen_cnt;
                pc.clear();
                const size_t n_win = std::min<size_t>((size_t) gsmpl->params.penalty_last_n, gsmpl->prev.size());
                for (size_t i = 0; i < n_win; ++i) {
                    const llama_token t = gsmpl->prev.rat(i);
                    bool found = false;
                    for (auto & e : pc) {
                        if (e.first == t) {
                            e.second++;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        pc.emplace_back(t, 1);
                    }
                }
                // the window tokens without a bias join the biased ones: membership is looked up in bias_sparse (ascending ids, filled in
                // every mode), not in bias_dense, which the default sparse mode leaves empty: a biased token added twice was saved after
                // its first -inf write, so the row kept -inf and the bias could be lost (a -inf biased token sampled)
                auto & sp = gsmpl->sp;
                const auto & bsl = gsmpl->bias_sparse;
                sp = bsl;
                for (const auto & e : pc) {
                    if (e.first < 0 || e.first >= n_vocab) {
                        continue;
                    }
                    const auto it = std::lower_bound(bsl.begin(), bsl.end(), e.first, [](const auto & b, llama_token t) { return b.first < t; });
                    if (it == bsl.end() || it->first != e.first) {
                        sp.emplace_back(e.first, 0.0f);
                    }
                }
                std::sort(sp.begin(), sp.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
                bsp = &sp;
            }
            const auto & bs = *bsp;
            gsmpl->bias_saved.resize(bs.size());
            for (size_t k = 0; k < bs.size(); ++k) {
                gsmpl->bias_saved[k] = lg[bs[k].first];
                lg[bs[k].first] = -INFINITY;
            }
            float lane[16];
            for (int j = 0; j < 16; ++j) {
                lane[j] = -INFINITY;
            }
            const int32_t n16 = n_vocab - n_vocab % 16;
            for (int32_t i = 0; i < n16; i += 16) {
                for (int j = 0; j < 16; ++j) {
                    lane[j] = lg[i + j] > lane[j] ? lg[i + j] : lane[j];
                }
            }
            float raw_v = -INFINITY;
            for (int j = 0; j < 16; ++j) {
                raw_v = lane[j] > raw_v ? lane[j] : raw_v;
            }
            for (int32_t i = n16; i < n_vocab; ++i) {
                raw_v = lg[i] > raw_v ? lg[i] : raw_v;
            }
            int32_t raw_i = -1;
            for (int32_t i = 0; i < n_vocab; ++i) {
                if (lg[i] == raw_v) {
                    raw_i = i;
                    break;
                }
            }
            for (size_t k = 0; k < bs.size(); ++k) {
                lg[bs[k].first] = gsmpl->bias_saved[k];
            }
            best   = raw_i < 0 ? 0 : raw_i;
            best_v = raw_v;
            // a biased token wins if its biased logit is larger, or equal with a lower id (the lowest id on a tie); a token of the
            // penalties window competes with its biased logit penalized as llama_sampler_penalties_apply does it
            for (size_t k = 0; k < bs.size(); ++k) {
                float v = gsmpl->bias_saved[k] + bs[k].second;
                if (gsmpl->pen_fast) {
                    int count = 0;
                    for (const auto & e : gsmpl->pen_cnt) {
                        if (e.first == bs[k].first) {
                            count = e.second;
                            break;
                        }
                    }
                    if (count > 0) {
                        const auto & pp = gsmpl->params;
                        if (v <= 0) {
                            v *= pp.penalty_repeat;
                        } else {
                            v /= pp.penalty_repeat;
                        }
                        v -= float(count) * pp.penalty_freq + float(count > 0) * pp.penalty_present;
                    }
                }
                if (v > best_v || (v == best_v && (raw_i < 0 || bs[k].first < best))) {
                    best_v = v;
                    best   = bs[k].first;
                    raw_i  = best;
                }
            }
        }
        gsmpl->cur.resize(1);
        gsmpl->cur[0] = llama_token_data{best, best_v, 1.0f};
        cur_p = { gsmpl->cur.data(), 1, 0, false };
        return best;
    }

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = gsmpl->sampled_token(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // k19-gramdev: device top-k candidates of a request with a grammar or a reasoning budget
    if (gsmpl->dev_topk && gsmpl->dev_gram && llama_get_sampled_logits_ith(ctx, idx) != nullptr && llama_get_sampled_probs_ith(ctx, idx) == nullptr) {
        return common_sampler_sample_dev_gram(gsmpl, ctx, idx, grammar_first);
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        // do not accept draft tokens after an EOG - they are not output but would stay in the context
        // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    if (const auto * sn = sn_of_chain(gsmpl->chain)) {
        return sn->seed_cur;
    }
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_shared_noise_draft_init(const struct common_sampler * gsmpl, common_shared_noise_draft & sn) {
    sn.on = false;
    const sn_sampler_ctx * c = gsmpl != nullptr ? sn_of_chain(gsmpl->chain) : nullptr;
    if (c == nullptr) {
        return false;
    }
    const auto & p = gsmpl->params;
    sn.on          = true;
    sn.seed        = c->seed_cur;
    sn.pos         = c->n_acc;
    sn.samplers    = p.samplers;
    sn.temp        = p.temp;
    sn.top_k       = p.top_k;
    sn.top_p       = p.top_p;
    sn.min_p       = p.min_p;
    sn.pen_last_n  = p.penalty_last_n;
    sn.pen_repeat  = p.penalty_repeat;
    sn.pen_freq    = p.penalty_freq;
    sn.pen_present = p.penalty_present;
    sn.window.clear();
    const bool pen = p.penalty_last_n != 0 && (p.penalty_repeat != 1.0f || p.penalty_freq != 0.0f || p.penalty_present != 0.0f);
    if (pen) {
        // prev holds the most recent max(32, n_prev) accepted tokens; a longer window is cut there (an approximation: acceptance only)
        const size_t n_win = p.penalty_last_n < 0 ? gsmpl->prev.size() : std::min<size_t>((size_t) p.penalty_last_n, gsmpl->prev.size());
        for (size_t i = n_win; i-- > 0; ) {
            sn.window.push_back(gsmpl->prev.rat(i));
        }
    }
    return true;
}

llama_token common_shared_noise_draft_pick(common_shared_noise_draft & sn, int32_t i, const llama_token * drafted,
        const llama_token_data * cand, size_t n, float * p_pick) {
    GGML_ASSERT(sn.on && n > 0);
    auto & c = sn.scratch; // (id, logit), kept sorted by logit, descending
    c.clear();
    for (size_t k = 0; k < n; ++k) {
        c.emplace_back(cand[k].id, cand[k].logit);
    }
    auto sort_desc = [&]() {
        std::sort(c.begin(), c.end(), [](const auto & a, const auto & b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
    };
    // softmax of c (sorted) at the logits as they are, written into probs
    std::vector<float> probs;
    auto softmax = [&]() {
        probs.resize(c.size());
        double s = 0.0;
        for (size_t k = 0; k < c.size(); ++k) {
            probs[k] = expf(c[k].second - c[0].second);
            s += probs[k];
        }
        for (auto & q : probs) {
            q = (float) (q / s);
        }
    };
    sort_desc();
    for (const auto t : sn.samplers) {
        if (c.size() <= 1) {
            break;
        }
        switch (t) {
            case COMMON_SAMPLER_TYPE_PENALTIES: {
                if (sn.pen_last_n == 0 || (sn.pen_repeat == 1.0f && sn.pen_freq == 0.0f && sn.pen_present == 0.0f)) {
                    break;
                }
                for (auto & e : c) {
                    int cnt = 0;
                    for (const llama_token w : sn.window) {
                        cnt += w == e.first;
                    }
                    for (int32_t j = 0; j < i; ++j) {
                        cnt += drafted[j] == e.first;
                    }
                    if (cnt == 0) {
                        continue;
                    }
                    e.second = e.second <= 0.0f ? e.second * sn.pen_repeat : e.second / sn.pen_repeat;
                    e.second -= (float) cnt * sn.pen_freq + sn.pen_present;
                }
                sort_desc();
                break;
            }
            case COMMON_SAMPLER_TYPE_TOP_K:
                if (sn.top_k > 0 && (size_t) sn.top_k < c.size()) {
                    c.resize(sn.top_k);
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_P:
                if (sn.top_p < 1.0f) {
                    softmax();
                    double cum = 0.0;
                    size_t keep = c.size();
                    for (size_t k = 0; k < c.size(); ++k) {
                        cum += probs[k];
                        if (cum >= sn.top_p) {
                            keep = k + 1;
                            break;
                        }
                    }
                    c.resize(keep);
                }
                break;
            case COMMON_SAMPLER_TYPE_MIN_P:
                if (sn.min_p > 0.0f) {
                    softmax();
                    size_t keep = 1;
                    while (keep < c.size() && probs[keep] >= sn.min_p * probs[0]) {
                        keep++;
                    }
                    c.resize(keep);
                }
                break;
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                if (sn.temp > 0.0f) {
                    for (auto & e : c) {
                        e.second /= sn.temp;
                    }
                }
                break;
            default: // dry, xtc, typical, top-n-sigma, ...: not mirrored
                break;
        }
    }
    const uint64_t pos = sn.pos + (uint64_t) i;
    size_t best = 0;
    float best_v = -INFINITY;
    for (size_t k = 0; k < c.size(); ++k) {
        if (!(c[k].second + SN_G_MAX >= best_v)) {
            continue;
        }
        const float v = c[k].second + common_shared_noise_gumbel(sn.seed, pos, c[k].first);
        if (v > best_v) {
            best_v = v;
            best   = k;
        }
    }
    if (p_pick != nullptr) {
        softmax();
        *p_pick = probs[best];
    }
    return c[best].first;
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
