// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "spec-cost-policy.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// Rebuild seq_id's draft sampler at the target's temperature: rejection weighs q against p, so
// both have to sample alike. Only temp and seed carry over; the draft keeps its own top_k.
// k9-draft: OPT-IN, LLAMA_SPEC_DRAFT_MIRROR=1: the probabilistic draft samples from the target's own truncation (top-k, top-p, min-p,
// then the temperature; top-k capped at LLAMA_SPEC_DRAFT_TOPK_MAX, 40). The draft stays exact for any q (rejection sampling), but the
// mirrored q accepted less than the default top-k 10 + temperature (Qwen3.6-35B-A3B + MTP, Studio's presence penalty 1.5 moves p and
// not q: B200 acceptance over 4-6 x 256 tokens 0.61 vs 0.64 at T 1.0, 0.62 vs 0.64 at T 0.6, 0.61 vs 0.63 at T 0.7; G4 chat_s / chat_t
// decode below the default), so the default keeps the earlier sampler.
static void spec_retune(
        std::vector<common_sampler_ptr> & smpls,
        std::vector<common_params_sampling> & cfg,
        const llama_model * model,
        llama_seq_id seq_id,
        const common_speculative_draft_params & dp) {
    static const bool mirror = getenv("LLAMA_SPEC_DRAFT_MIRROR") != nullptr && atoi(getenv("LLAMA_SPEC_DRAFT_MIRROR")) != 0;
    static const int  k_max  = getenv("LLAMA_SPEC_DRAFT_TOPK_MAX") != nullptr ? std::max(1, atoi(getenv("LLAMA_SPEC_DRAFT_TOPK_MAX"))) : 40;

    const float temp  = dp.temp;
    const uint32_t seed = dp.seed;
    const int32_t top_k = mirror ? (dp.top_k > 0 ? std::min<int32_t>(dp.top_k, k_max) : k_max) : 10;
    const float   top_p = mirror ? dp.top_p : 1.0f;
    const float   min_p = mirror ? dp.min_p : 0.0f;

    if (cfg.size() != smpls.size()) {
        const size_t n_old = cfg.size();
        cfg.resize(smpls.size());

        // the initial sampler has no temperature, so no request may match the cache and skip a rebuild
        for (size_t i = n_old; i < cfg.size(); ++i) {
            cfg[i].temp = NAN;
        }
    }

    auto & cur = cfg[seq_id];

    if (cur.temp == temp && cur.seed == seed && cur.top_k == top_k && cur.top_p == top_p && cur.min_p == min_p) {
        return;
    }

    cur.temp  = temp;
    cur.seed  = seed;
    cur.top_k = top_k;
    cur.top_p = top_p;
    cur.min_p = min_p;

    common_params_sampling sparams;
    sparams.no_perf  = false;
    sparams.top_k    = top_k;
    sparams.top_p    = top_p;
    sparams.min_p    = min_p;
    sparams.temp     = cur.temp;
    // must be explicit, the default reseeds at random; mixed so it differs from the target's
    sparams.seed     = cur.seed == LLAMA_DEFAULT_SEED ? cur.seed : cur.seed ^ 0x85ebca6bu;
    if (mirror) {
        sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
    } else {
        sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    }

    smpls[seq_id].reset(common_sampler_init(model, sparams));
}

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"ngram-suffix",  COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const common_batch & batch) = 0;

    // k11 (LLAMA_LM_CKPT_MERGE): process() reads the target's rows from this batch index on (a segment of a merged prompt batch);
    // impls that read target features without honouring it say so
    int32_t nextn_row_offset = 0;
    virtual bool row_offset_ok() const { return true; }
    // k15-pp3 (LLAMA_MTP_PROMPT_OVERLAP): process() reads the target's rows without a synchronization (llama_get_embeddings_nextn_early),
    // the caller waited for them (llama_nextn_rows_wait); only impls that read nothing else of the target say yes
    bool rows_early = false;
    virtual bool rows_early_ok() const { return false; }

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    common_batch batch;

    // zero row at the draft input width, stands in for target embeddings the draft cannot read
    std::vector<float> zeros;
    bool zeros_warned = false; // the substitution is reported once

    std::vector<common_sampler_ptr> smpls;

    std::vector<common_params_sampling> smpls_cfg;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        zeros.assign(llama_model_n_embd_inp(llama_get_model(ctx_dft)), 0.0f);

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = common_batch(ctx_dft);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers.assign(1, COMMON_SAMPLER_TYPE_TOP_K);

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        // reset here rather than per round, or two identical requests differ
        common_sampler_reset(smpls[seq_id].get());
    }

    bool process(const common_batch & batch_in) override {
        auto * ctx_dft = params.ctx_dft;

        // copy the entries to a batch owned by the draft context, only the last token is output
        batch.clear();
        const int32_t n_tokens = batch_in.size();
        for (int32_t k = 0; k < n_tokens; ++k) {
            const auto & t = batch_in.tokens[k];
            const bool output = k == n_tokens - 1;
            if (t.id != LLAMA_TOKEN_NULL) {
                const int32_t idx = batch.add(t.id, t.pos[0], t.seq_id, output);
                if (t.embd.data) {
                    batch.set_embd(idx, t.embd);
                }
            } else {
                // mtmd input is projected by the target encoder, a draft with a different width cannot read it
                // it gets zeros instead, keeping its positions contiguous
                // ref: https://github.com/ggml-org/llama.cpp/pull/29385#discussion_r4124743243
                const size_t n_embd = t.embd.n_rows * t.embd.n_embd;
                const bool same_width = n_embd == zeros.size();
                if (!same_width && !zeros_warned) {
                    SPC_WRN("target embeddings of size %zu do not fit the draft input width %zu, "
                            "the draft receives zero rows for them and drafts after multimodal input will be poor\n",
                            n_embd, zeros.size());
                    zeros_warned = true;
                }
                const llama_embd embd = same_width ? t.embd : llama_embd{ zeros.data(), 1, zeros.size() };
                batch.add_embd(embd, t.pos.data(), t.seq_id, output);
            }
        }

        if (batch.size() == 0) {
            return true;
        }

        const int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
            if (!params.probabilistic) {
                dp.result_q = nullptr;
            }

            // result_q is only set when the caller wants rejection, so it also gates the retune
            if (dp.result_q) {
                spec_retune(smpls, smpls_cfg, llama_get_model(ctx_dft), seq_id, dp);
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            if (!dp.result_q) {
                common_sampler_reset(smpls[seq_id].get());
            }

            batch.add(dp.id_last, dp.pos0, seq_id, true);
        }

        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            SPC_ERR("llama_process returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            batch.clear();

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                const llama_token id_sampled = common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = dparams.at(seq_id).result_q ? id_sampled : cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (dp.result_q) {
                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                }

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                batch.add(id, dp.pos0 + i + 1, seq_id, true);
            }

            if (batch.size() == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    bool row_offset_ok() const override { return false; }

    common_params_speculative_draft params;
    common_batch batch;     // decoder input, (token, g_embd) pairs
    common_batch batch_enc; // encoder input, built from the extracted target features

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        batch     = common_batch(ctx_dft);
        batch_enc = common_batch(ctx_dft);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        if (!batch_in.has_token() || batch_in.has_embd()) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_process() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            // the per-token encoder does not use positions, generate placeholder ones from the memory state
            batch_enc.clear();
            llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) + 1;
            for (int32_t j = 0; j < n_chunk; ++j) {
                batch_enc.add_embd({ features_buf.data() + (size_t) (i + j) * n_embd_enc, 1, (size_t) n_embd_enc }, &pos, 0, true);
                pos++;
            }

            const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_ENCODE, batch_enc.get());
            if (rc != 0) {
                SPC_ERR("llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        batch.clear();

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.tokens[beg].pos[0]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    const int32_t idx = batch.add(batch_in.tokens[beg].id, pending_pos, seq_id, /*output=*/ false);
                    batch.set_embd(idx, { pending_g_last[seq_id].data(), 1, (size_t) n_embd_dec });
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                const int32_t idx = batch.add(batch_in.tokens[k + 1].id, batch_in.tokens[k].pos[0], seq_id, /*output=*/ false);
                batch.set_embd(idx, { g_embd + (size_t) k * n_embd_dec, 1, (size_t) n_embd_dec });
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.tokens[beg].pos[0];
            pending_pos_last[seq_id] = batch_in.tokens[end].pos[0];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.size() > 0) {
            const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0) {
                SPC_ERR("llama_process(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.size(), (int) batch_in.tokens[0].pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            const int32_t idx = batch.add(dp.id_last, pending_pos_last[seq_id], seq_id, true);
            batch.set_embd(idx, { pending_g_last[seq_id].data(), 1, (size_t) n_embd_dec });
        }

        if (batch.size() == 0) {
            return;
        }

        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            SPC_ERR("llama_process returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            batch.clear();

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                const int32_t idx = batch.add(id, pending_pos_last[seq_id] + (i + 1), seq_id, true);
                batch.set_embd(idx, { prenorm, 1, (size_t) n_embd_dec });
            }

            if (batch.size() == 0) {
                break;
            }

            ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    bool row_offset_ok() const override { return false; }

    common_params_speculative_draft params;

    common_batch batch;        // noise tokens
    common_batch batch_inject; // target features for KV cache injection

    std::vector<float> features_buf; // [n_chunk, n_embd_enc] gathered target features

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_lilicorr    = false;
    int32_t lilicorr_top_k = 0;
    size_t lilicorr_rows = 0;
    std::vector<float> lilicorr_embd;
    std::vector<llama_pos> lilicorr_pos;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        char lilicorr_meta[32] = {};
        lilicorr_top_k = llama_model_meta_val_str(model_dft, "lilicorr.candidate_topk", lilicorr_meta, sizeof(lilicorr_meta)) >= 0
                ? std::atoi(lilicorr_meta) : 0;
        is_lilicorr = lilicorr_top_k > 0;
        if (is_lilicorr) {
            lilicorr_rows = (size_t) block_size + 1;
            const size_t lilicorr_slots = (size_t) n_seq * lilicorr_rows;
            lilicorr_embd.resize(lilicorr_slots * (size_t) n_embd_dec);
            lilicorr_pos.assign(lilicorr_slots, -1);
        }
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        this->n_max = this->params.n_max;

        batch        = common_batch(ctx_dft);
        batch_inject = common_batch(ctx_dft);

        // embd batches on an M-RoPE draft carry 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2 && !is_lilicorr) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2 && !is_lilicorr);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // embeddings are injected too, except the pinned ones skipped below.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.has_token();
        const bool has_embeddings = batch_in.has_embd();
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // an M-RoPE image pins all its rows to one position, so a windowed draft
            // cache cannot free cells for it - skip it, the draft can jump over the gap
            const bool pos_pinned = batch_in.tokens[i_batch_beg[seq_id]].pos[0] == batch_in.tokens[i_batch_end[seq_id]].pos[0];
            if (has_embeddings && n_rows > 1 && pos_pinned) {
                continue;
            }

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                batch_inject.clear();
                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = batch_in.tokens[i_batch_beg[seq_id] + offset + i].pos[0];
                    const llama_pos pos_arr[4] = { p, p, p, 0 };
                    batch_inject.add_embd({ features_buf.data() + (size_t) i * n_embd_enc, 1, (size_t) n_embd_enc }, pos_arr, seq_id, false);
                }
                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_inject.get());
                if (rc != 0) {
                    LOG_ERR("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
                if (is_lilicorr) {
                    // Ordinary DFlash can write to the KV cache, but LiliCorr needs to store embeddings across sequences.
                    const float * projected = llama_get_embeddings_nextn(ctx_dft);
                    if (projected) {
                        const size_t seq_offset = (size_t) seq_id * lilicorr_rows;
                        const llama_pos pos_min = std::max(0, batch_inject.tokens.back().pos[0] - block_size);
                        for (int32_t i = 0; i < n_chunk; ++i) {
                            const llama_pos pos = batch_inject.tokens[i].pos[0];
                            if (pos < pos_min) {
                                continue;
                            }
                            const size_t slot = seq_offset + (size_t) pos % lilicorr_rows;
                            std::memcpy(lilicorr_embd.data() + slot * (size_t) n_embd_dec,
                                    projected + (size_t) i * n_embd_dec, (size_t) n_embd_dec * sizeof(float));
                            lilicorr_pos[slot] = pos;
                        }
                    }
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.pos0;

            const int32_t n_draft = params.n_max;
            const int32_t n_block_tokens = is_lilicorr ? block_size : n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.size();
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                const int32_t idx = batch.add(i == 0 ? dp.id_last : mask_token_id, n + i, seq_id, !is_dflash2 && !is_lilicorr);
                if (is_lilicorr) {
                    // For LiliCorr drafting batch, set state embeddings for use in the associated transformer head.
                    const llama_pos pos = dp.pos0 - 1;
                    const size_t slot = (size_t) seq_id * lilicorr_rows + (size_t) pos % lilicorr_rows;
                    batch.set_embd_state(idx, { lilicorr_embd.data() + slot * (size_t) n_embd_dec, 1, (size_t) n_embd_dec });
                }
            }
        }

        if (batch.size() == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (is_dflash2 || is_lilicorr) {
                // Both DFlash and LiliCorr greedily walk the candidate lattice.
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                int32_t predecessor = 0;
                for (int32_t i = 1; i < (is_dflash2 ? n_block_tokens : params.n_max + 1); ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const int32_t top_k = is_dflash2 ? selector_top_k : lilicorr_top_k;
                    const float * scores = row + top_k + (size_t) predecessor * top_k;

                    predecessor = (int32_t) std::distance(scores,
                            std::max_element(scores, scores + top_k));
                    if (params.p_min > 0.0f) {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                        float sum = 0.0f;
                        for (int32_t k = 0; k < top_k; ++k) {
                            sum += std::exp(scores[k] - scores[predecessor]);
                        }
                        if (1.0f / sum < params.p_min) {
                            break;
                        }
                    }
                    result.push_back((llama_token) row[predecessor]);
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                }
                continue;
            }

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    common_batch batch;

    std::vector<common_sampler_ptr> smpls;

    std::vector<common_params_sampling> smpls_cfg;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;
    std::vector<int32_t> verify_h_kept; // rows of verify_h that hold data (the last row is also in pending_h)

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // LLAMA_MTP_HANDOFF_COPY=0: the verify rows are copied row by row, as before
    const bool handoff_copy = common_mtp_handoff_on("COPY");
    // LLAMA_MTP_HANDOFF_VH=0: verify_h keeps every row of a prompt ubatch, as before
    const bool verify_h_trim = handoff_copy && common_mtp_handoff_on("VH");

    // LLAMA_MTP_DEV_H=1 (opt-in: decode -0.7..-1.0% on Qwen3.6 at 32 GB, Flash-Next 8k prompt -8% on G4; default = host rows, as before):
    // (single-sequence batches of LLAMA_MTP_DEV_H_MIN = 64 tokens or more) they stay on the device (llama_mtp_dev_h_attach) and the
    // drafter's pass is queued without waiting on the target from the host.
    bool dev_h = false;
    // k11: segments of a merged prompt batch (LLAMA_LM_CKPT_MERGE) read host rows at an offset, the device sink holds whole batches
    bool row_offset_ok() const override { return !dev_h; }
    bool rows_early_ok() const override { return !dev_h && !is_mem_shared; }
    llama_seq_id dev_seq = -1;            // the sequence whose pending row is in sink row 0
    std::vector<uint8_t> pending_dev;     // per seq: its pending row is sink row 0 (pending_h is stale)
    std::vector<uint8_t> verify_dev;      // per seq: the verify rows are sink rows 1 .. verify_h_rows (verify_h is unused)

    // k15-fn: LLAMA_SPEC_COST=1 (opt-in): the cost-aware MTP window (common_spec_mtp_cost, Strata's STRATA_SPEC_COST): the drafts
    // kept per round are the ones with the best expected tokens per ms, from the drafts' probabilities and measured round times
    const bool spec_cost = [] {
        const char * e = getenv("LLAMA_SPEC_COST");
        return e != nullptr && atoi(e) != 0;
    }();
    common_spec_mtp_cost cost_pol;
    struct cost_seq {
        int64_t t_us = 0;            // the previous draft() call (0: none since the request began)
        int     rows = 1;            // the window drafted then
        std::vector<float> probs;    // its drafts' probabilities
        common_spec_mtp_cost::chain_state chain;
    };
    std::vector<cost_seq> cost_st;

    // the pending row of seq back on the host (a blocking copy of one row)
    void pending_to_host(llama_seq_id seq_id) {
        if (dev_h && pending_dev[seq_id]) {
            GGML_ASSERT(dev_seq == seq_id);
            GGML_ASSERT(llama_mtp_dev_h_get_row(params.ctx_dft, 0, pending_h[seq_id].data()));
            pending_dev[seq_id] = 0;
        }
    }

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = common_batch(ctx_dft);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
        verify_h_kept.assign(n_seq, 0);

        pending_dev.assign(n_seq, 0);
        verify_dev.assign(n_seq, 0);
        cost_st.assign(n_seq, {});
        if (spec_cost) {
            LOG_INF("%s: MTP window by expected tokens per ms (LLAMA_SPEC_COST=1), up to %d drafts\n", __func__, this->params.n_max);
        }
        // k15-pp: on by default beside built-in MTP heads (common sets LLAMA_SPEC_MTP_HEADS=1): the target's next-n rows of a prompt
        // batch then go to the device sink and the host keeps none of them (llama_context::nextn_store_ensure), so the host rows no
        // longer grow with n_batch (q4x 16384-token groups: 128 MiB pinned). A -md draft model keeps the host path (G4 Flash-Next 8k
        // prompt +8% with it, plans/impl_mtp.md). LLAMA_MTP_DEV_H=0 / =1 forces it off / on
        const char * dev_h_env = getenv("LLAMA_MTP_DEV_H");
        // (G4 k15g4q1 measured it together with 16384 groups beside the heads, which cost decode and stay opt-in; at 8192 the device
        // path saves the 64 MiB of host rows for a 64 MiB sink)
        const bool dev_h_on = dev_h_env != nullptr ? atoi(dev_h_env) != 0 :
            getenv("LLAMA_SPEC_MTP_HEADS") != nullptr && atoi(getenv("LLAMA_SPEC_MTP_HEADS")) != 0;
        if (!is_mem_shared && !chain_heads && common_mtp_handoff_on("DEV_H") && dev_h_on) {
            dev_h = llama_mtp_dev_h_attach(ctx_dft, ctx_tgt, (int32_t) llama_n_batch(ctx_tgt));
        }
        // k17: the device handoff reads no host state rows, so the batch's prefaulted ones (n_batch x n_embd floats: 64 MiB at 8192,
        // 128 MiB at 16384 for Qwen3.6 + MTP heads) are given back; a host-path batch grows them again (LLAMA_MTP_DEV_H_KEEP_STATE=1 keeps them)
        if (dev_h && getenv("LLAMA_MTP_DEV_H_KEEP_STATE") == nullptr) {
            llama_batch_ext_release_state(batch.batch.get());
        }
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        {
            // the restricted draft head (LLAMA_MTP_DRAFT_VOCAB) gets rows for the prompt's tokens it lacks
            const int32_t n_add = llama_model_draft_head_add(llama_get_model(params.ctx_dft), prompt.data(), (int32_t) prompt.size());
            if (n_add > 0) {
                SPC_DBG("%s: draft head: %d prompt tokens added\n", __func__, n_add);
            }
        }
        // reset here rather than per round, or two identical requests differ
        common_sampler_reset(smpls[seq_id].get());
        if (spec_cost && seq_id >= 0 && seq_id < (llama_seq_id) cost_st.size()) {
            if (cost_st[seq_id].t_us > 0) {
                LOG_INF("%s: MTP window cost table: %s\n", __func__, cost_pol.text().c_str());
            }
            cost_st[seq_id].t_us = 0;
            cost_st[seq_id].probs.clear();
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / output flag on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        if (!batch_in.has_token() && !batch_in.has_embd()) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // remember the first and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (batch_in.tokens[k].seq_id == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // k15-pp3: rows the caller already waited for (unmasked: row i at i*n_embd), else the synchronizing lookups
        const float * h_early = rows_early ? llama_get_embeddings_nextn_early(ctx_tgt) : nullptr;
        GGML_ASSERT(!rows_early || h_early);
        auto tgt_row = [&](int32_t i) -> const float * {
            return h_early ? h_early + (size_t) i * n_embd : llama_get_embeddings_nextn_ith(ctx_tgt, i);
        };

        const int64_t t_tr0 = common_mtp_handoff_trace() && n_tokens >= 64 ? ggml_time_us() : 0;
        int64_t t_tr_sync  = 0;
        int64_t t_tr_build = 0;
        int64_t t_tr_dft   = 0;

        // LLAMA_MTP_DEV_H: the target wrote this batch's hidden rows to the device sink (one sequence)
        const int32_t dev_rows = dev_h ? llama_mtp_dev_h_last_rows(ctx_tgt) : 0;
        const bool dev = dev_rows > 0;
        llama_seq_id dev_s = -1;
        if (dev) {
            GGML_ASSERT(dev_rows == n_tokens && "MTP device handoff: the target wrote part of the batch");
            dev_s = batch_in.tokens[0].seq_id;
            GGML_ASSERT(i_batch_beg[dev_s] == 0 && i_batch_end[dev_s] == n_tokens - 1);
        }
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            verify_dev[seq_id] = 0;
            if (seq_id != dev_s) {
                pending_to_host(seq_id); // a host-path batch pairs its first row with pending_h
            }
        }
        if (dev && !(dev_seq == dev_s && pending_dev[dev_s])) {
            // the first row of this batch pairs with the host's pending row (another sequence held sink row 0, or a host-path batch ran)
            if (dev_seq >= 0 && dev_seq != dev_s) {
                pending_to_host(dev_seq);
            }
            GGML_ASSERT(llama_mtp_dev_h_set_row(ctx_dft, 0, pending_h[dev_s].data()));
        }

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            batch.clear();

            // pair each token with the tgt embedding shifted right by one position, and
            // the first token of each sequence with the pending embedding from a previous run
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            // the device path reads no host rows: no sync on the target here
            const float * h_tgt = dev ? nullptr : (h_early ? h_early : llama_get_embeddings_nextn(ctx_tgt)) + (size_t) nextn_row_offset * n_embd;
            if (t_tr0) {
                t_tr_sync = ggml_time_us();
            }

            for (int k = 0; k < n_tokens; ++k) {
                const auto & t = batch_in.tokens[k];

                const llama_seq_id seq_id = t.seq_id;

                // vision tokens carry an embedding instead of an id
                const int32_t idx = t.id != LLAMA_TOKEN_NULL
                    ? batch.add(t.id, t.pos[0], seq_id, false)
                    : batch.add_embd(t.embd, t.pos.data(), seq_id, false);

                if (dev) {
                    continue; // row k of the sink
                }

                const float * h_row = k == i_batch_beg[seq_id]
                    ? pending_h[seq_id].data()
                    : h_tgt + (size_t) (k - 1) * n_embd;

                batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd });
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            if (t_tr0) {
                t_tr_build = ggml_time_us();
            }

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                if (dev) {
                    GGML_ASSERT(llama_mtp_dev_h_begin(ctx_dft));
                }
                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (dev) {
                    // sink row n_tokens = the target's last row: the next batch's first row
                    llama_mtp_dev_h_end(ctx_dft, n_tokens);
                    pending_dev[dev_s] = 1;
                    dev_seq = dev_s;
                }
                if (rc != 0) {
                    SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.tokens[0].pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
            if (t_tr0) {
                t_tr_dft = ggml_time_us();
            }
        }

        if (dev && is_mem_shared) {
            GGML_ABORT("MTP device handoff with a shared KV draft");
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;

            if (dev) {
                // the verify rows stay in the sink (rows 1 .. n_rows), the last one is already the pending row 0
                verify_dev[seq_id]    = 1;
                verify_h_kept[seq_id] = n_rows;
                continue;
            }

            // the rows of one sequence are consecutive in the target's nextn buffer: one copy into a buffer that only grows (a
            // prompt ubatch is n_rows*n_embd floats) instead of a zero-filled resize and a synchronizing lookup per row
            const float * h_beg = handoff_copy ? tgt_row(nextn_row_offset + i_batch_beg[seq_id]) : nullptr;
            const float * h_end = handoff_copy ? tgt_row(nextn_row_offset + i_batch_end[seq_id]) : nullptr;
            if (h_beg && h_end == h_beg + (size_t) (n_rows - 1) * n_embd) {
                // accept() reads row min(n_accepted, n_rows - 1) and n_accepted <= n_max: a prompt ubatch (thousands of rows,
                // 160 MiB at 4096 x 10240 floats) keeps its first n_keep rows and the last one (in pending_h), not all of them;
                // the copy ran on the main thread after the draft was queued, with the GPU idle (LLAMA_MTP_HANDOFF_VH=0: all rows)
                const int32_t n_keep = verify_h_trim ? std::min<int32_t>(n_rows, std::max<int32_t>(256, n_max + 2)) : n_rows;
                verify_h_kept[seq_id] = n_keep;
                if (verify_h[seq_id].size() < (size_t) n_keep * n_embd) {
                    verify_h[seq_id].resize((size_t) n_keep * n_embd);
                }
                std::memcpy(verify_h[seq_id].data(), h_beg, (size_t) n_keep * row_bytes);
                std::memcpy(pending_h[seq_id].data(), h_end, row_bytes);
                continue;
            } else {
                verify_h_kept[seq_id] = n_rows;
                verify_h[seq_id].resize((size_t) n_rows * n_embd);

                for (int32_t i = 0; i < n_rows; ++i) {
                    const float * h = tgt_row(nextn_row_offset + i_batch_beg[seq_id] + i);
                    std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
                }
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        if (t_tr0 && t_tr_dft) {
            const int64_t t_tr_end = ggml_time_us();
            SPC_INF("mtp_trace process n=%d sync=%.2f build=%.2f dft_decode=%.2f verify_h=%.2f ms\n", n_tokens, (t_tr_sync - t_tr0)/1e3,
                    (t_tr_build - t_tr_sync)/1e3, (t_tr_dft - t_tr_build)/1e3, (t_tr_end - t_tr_dft)/1e3);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
            if (!params.probabilistic) {
                dp.result_q = nullptr;
            }

            if (spec_cost) {
                // the time since the previous draft() call is one round (verify + sampling + this drafting) of the window drafted then
                auto & cs = cost_st[seq_id];
                const int64_t now = ggml_time_us();
                cost_pol.set_context(dp.pos0);
                if (cs.t_us > 0) {
                    cost_pol.observe(cs.rows, (now - cs.t_us)/1000.0);
                }
                cs.t_us  = now;
                cs.rows  = 1;
                cs.chain = {};
                cs.probs.clear();
            }

            // result_q is only set when the caller wants rejection, so it also gates the retune
            if (dp.result_q) {
                spec_retune(smpls, smpls_cfg, llama_get_model(ctx_dft), seq_id, dp);
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            if (!dp.result_q) {
                common_sampler_reset(smpls[seq_id].get());
            }

            pending_to_host(seq_id);
            const int32_t idx = batch.add(dp.id_last, dp.pos0, seq_id, true);
            batch.set_embd_state(idx, { pending_h[seq_id].data(), 1, (size_t) n_embd });

            i_last[seq_id] = idx;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < pos0
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            batch.clear();

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                const llama_token id_sampled = common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = dparams.at(seq_id).result_q ? id_sampled : cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (dp.result_q) {
                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                }

                if (spec_cost && !dp.result_q) {
                    auto & cs = cost_st[seq_id];
                    cs.probs.push_back(cur_p->data[0].p);
                    const int cap = 1 + (dp.n_max > 0 ? std::min(params.n_max, dp.n_max) : params.n_max);
                    if (!cost_pol.keep_draft(cs.chain, (int) result.size() - 1, cur_p->data[0].p, cap)) {
                        drafting[seq_id] = false;
                        n_drafting--;
                        continue;
                    }
                }

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        const int32_t idx = batch.add(tok, dp.pos0 + t, seq_id, t == n_rows - 1);
                        batch.set_embd_state(idx, { chain_h[seq_id].data() + (size_t) t * n_embd, 1, (size_t) n_embd });
                        i_last[seq_id] = idx;
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    const int32_t idx = batch.add(id, dp.pos0, seq_id, true);
                    batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd });
                    i_last[seq_id] = idx;
                } else {
                    const int32_t idx = batch.add(id, dp.pos0 + i + 1, seq_id, true);
                    batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd });
                    i_last[seq_id] = idx;
                }
            }

            if (batch.size() == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (spec_cost && !dp.result_q) {
                // the window keeps the drafts of the best expected tokens per ms
                const size_t kept = (size_t) std::max(0, cost_st[seq_id].chain.best_rows - 1);
                if (dp.result->size() > kept) {
                    dp.result->resize(kept);
                }
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            if (spec_cost) {
                auto & cs = cost_st[seq_id];
                cs.rows = 1 + (int) dp.result->size();
                cs.probs.resize(std::min(cs.probs.size(), dp.result->size()));
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        if (spec_cost) {
            auto & cs = cost_st[seq_id];
            if (!cs.probs.empty()) {
                cost_pol.observe_probs(cs.probs.data(), (int) cs.probs.size(), n_accepted);
                cs.probs.clear();
            }
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        if (verify_dev[seq_id]) {
            // the last row is the pending row 0 already; another row comes from the sink
            if (i_h < n_rows - 1) {
                if (dev_seq != seq_id) {
                    // another sequence's prompt batch has overwritten the sink since (not seen in the server's flow: a verify batch
                    // runs process() before its accept())
                    SPC_WRN("MTP device handoff: seq %d accepted row %d of a prompt batch the sink no longer holds\n", (int) seq_id, (int) i_h);
                    return;
                }
                GGML_ASSERT(llama_mtp_dev_h_get_row(params.ctx_dft, 1 + i_h, pending_h[seq_id].data()));
                pending_dev[seq_id] = 0;
            }
            return;
        }
        if (i_h >= verify_h_kept[seq_id]) {
            // only the last row is past the kept ones (n_accepted <= n_max < n_keep), and process() left it in pending_h
            GGML_ASSERT(i_h == n_rows - 1 && "MTP verify_h: accepted row was not kept");
            return;
        }
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const common_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const common_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const common_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

// k9-draft: ngram-suffix. The draft is what followed the longest earlier repeat (at least n_min_match, at most 64 tokens) of the last
// tokens of the prompt + output, found through a hash of trigrams that keeps the last 4 places of each (the suffix lookup of
// PolyStrata: src/spec/suffix_drafter.cpp:47-87, --lookup 3 by default in src/poly/main.cpp:1127). Its length follows the decayed
// acceptance of earlier suffix drafts of the same match-length class, q per token, with PolyStrata's priors (src/spec/draft_policy.cpp:
// 13-19, 26-33): k tokens while q^k >= p_cut, at most n_max_short below a match of n_long_match tokens and n_max from there (its
// window of 4 for short repeats and the full window for a copy of 8 or more, src/poly/main.cpp:68-73, 489). Beside a model drafter
// (later in the priority list: MTP, -md, eagle3, dflash) it drafts only a full window from a match class right at least p_take of the
// time and otherwise leaves the round to the model (main.cpp:491-494); the model's rounds still see the verify rows (process()), so
// both compose. The verifier accepts a suffix token only if the target samples it, so the output distribution is unchanged.
struct common_speculative_impl_ngram_suffix : public common_speculative_impl {
    static constexpr int    WAYS      = 4;
    static constexpr int    MAX_MATCH = 64;
    static constexpr int    N_BUCKET  = 4;
    static constexpr double PRIOR_N   = 4.0;
    static constexpr double DECAY     = 0.97;  // per verified suffix draft
    static constexpr double DRIFT     = 0.99;  // per match not drafted: the class drifts back to its prior, so a low class is retried

    common_params_speculative_ngram_suffix params;
    bool with_model = false;
    // k11-decode: cost-priced windows alone (common/spec-cost-policy.h); one cost table for every sequence
    bool policy_on  = false;
    common_spec_cost_policy policy;

    struct slot_t {
        uint64_t key = 0;
        uint32_t pos[WAYS] = {};
        uint8_t  n = 0;
    };

    struct seq_info {
        std::vector<llama_token> hist;
        std::vector<slot_t>      table;
        size_t mask   = 0;
        size_t n_used = 0;
        double ok[N_BUCKET]  = {};
        double bad[N_BUCKET] = {};
        int bucket_last = -1;
        int k_last      = 0;
        int match_last  = 0;
        // per request counters (logged when the next request begins)
        int64_t n_lookups = 0, n_matches = 0, n_drafts = 0, n_drafted = 0, n_accepted = 0, n_yield = 0;
        common_spec_round_timer timer;   // k11-decode: round times of this sequence (policy_on)
        int64_t n_probe = 0;
        int min_match = 3;               // k15-spec: this request's shortest repeat (n_min_match, n_min_match_long for a long prompt)
        int n_short   = 3;               // k15-lfm2: this request's short window and full-window match (n_max_short / n_long_match, or
        int n_long    = 8;               //           n_max_short_long / n_long_match_long for a long prompt)
    };

    std::vector<seq_info> sinfos;

    static int env_i(const char * name, int def) {
        const char * e = getenv(name);
        return e ? atoi(e) : def;
    }
    static float env_f(const char * name, float def) {
        const char * e = getenv(name);
        return e ? (float) atof(e) : def;
    }

    common_speculative_impl_ngram_suffix(const common_params_speculative & params_all, uint32_t n_seq, bool with_model)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX, n_seq, params_all.suffix_n_max())
        , params(params_all.ngram_suffix)
        , with_model(with_model) {
        params.n_max        = params_all.suffix_n_max();
        params.n_min_match  = std::max(3, env_i("LLAMA_SPEC_SUFFIX_MIN_MATCH", params.n_min_match));
        params.n_min_draft  = std::max(params.n_min_match, env_i("LLAMA_SPEC_SUFFIX_MIN_DRAFT", params.n_min_draft));
        params.n_max_short  = std::max(1, env_i("LLAMA_SPEC_SUFFIX_N_MAX_SHORT", params.n_max_short));
        params.n_long_match = std::max(params.n_min_match, env_i("LLAMA_SPEC_SUFFIX_LONG_MATCH", params.n_long_match));
        params.p_cut        = env_f("LLAMA_SPEC_SUFFIX_P_CUT", params.p_cut);
        params.p_take       = env_f("LLAMA_SPEC_SUFFIX_P_TAKE", params.p_take);
        params.n_max        = std::max(1, params.n_max);
        params.n_max_short  = std::min(params.n_max_short, params.n_max);
        this->n_max = params.n_max;
        sinfos.resize(n_seq);
        if (params.n_min_match_long > 0) {
            params.n_min_match_long = std::max(3, params.n_min_match_long);
        }
        for (auto & si : sinfos) {
            si.min_match = params.n_min_match;
            si.n_short   = params.n_max_short;
            si.n_long    = params.n_long_match;
        }
        if (params.win_long_prompt > 0) {
            params.n_max_short_long  = std::clamp(params.n_max_short_long > 0 ? params.n_max_short_long : params.n_max_short, 1, params.n_max);
            params.n_long_match_long = std::max(params.n_min_match, params.n_long_match_long > 0 ? params.n_long_match_long : params.n_long_match);
            SPC_INF("ngram-suffix: prompts of %d+ tokens: window of %d below a match of %d, else %d\n", params.win_long_prompt,
                    params.n_max_short_long, params.n_long_match_long, params.n_max);
        }
        {
            const char * e = getenv("LLAMA_SPEC_SUFFIX_POLICY");
            policy_on = !with_model && (e != nullptr ? atoi(e) != 0 : params.auto_mode);
            policy.short_k = params.n_max_short;
        }
        if (policy_on) {
            SPC_INF("ngram-suffix: cost-priced windows of %d / %d drafts (margin %.2f)%s; LLAMA_SPEC_SUFFIX_POLICY=0 for the fixed gate\n",
                    params.n_max_short, params.n_max, policy.margin, params.auto_mode ? ", on by default (LLAMA_SPEC_AUTO=0: off)" : "");
        }
        if (params.long_prompt > 0 && params.n_min_match_long > 0 && params.n_min_match_long != params.n_min_match) {
            SPC_INF("ngram-suffix: prompts of %d+ tokens draft from repeats of at least %d tokens\n", params.long_prompt, params.n_min_match_long);
        }
        SPC_INF("ngram-suffix: min match %d (%d to draft alone), n_max %d (%d below a match of %d), p_cut %.2f, %s %.2f\n", params.n_min_match,
                params.n_min_draft, params.n_max, params.n_max_short, params.n_long_match, params.p_cut,
                with_model ? "beside a model drafter: drafts a full window when q >=" : "alone, p_take unused", params.p_take);
    }

    static int bucket(int match) {
        return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
    }

    double rate(const seq_info & si, int b) const {
        static constexpr double prior[N_BUCKET] = { 0.75, 0.88, 0.93, 0.96 };
        return (si.ok[b] + PRIOR_N*prior[b]) / (si.ok[b] + si.bad[b] + PRIOR_N);
    }

    static uint64_t mix(uint64_t x) {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; return x ^ (x >> 33);
    }

    static uint64_t key_at(const std::vector<llama_token> & h, size_t end) {
        const uint64_t a = (uint32_t) h[end - 2], b = (uint32_t) h[end - 1], c = (uint32_t) h[end];
        return mix(a*0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;
    }

    static slot_t * find(seq_info & si, uint64_t key, bool insert) {
        if (si.table.empty()) {
            return nullptr;
        }
        for (size_t i = key & si.mask, probes = 0; probes <= si.mask; i = (i + 1) & si.mask, ++probes) {
            slot_t & s = si.table[i];
            if (s.key == key) {
                return &s;
            }
            if (s.key == 0) {
                if (!insert) {
                    return nullptr;
                }
                s.key = key;
                si.n_used++;
                return &s;
            }
        }
        return nullptr;
    }

    static bool indexable(const seq_info & si, size_t end) {
        const auto & h = si.hist;
        return end >= 2 && h[end] != LLAMA_TOKEN_NULL && h[end - 1] != LLAMA_TOKEN_NULL && h[end - 2] != LLAMA_TOKEN_NULL;
    }

    static void insert_at(seq_info & si, size_t end) {
        slot_t * s = find(si, key_at(si.hist, end), true);
        if (s == nullptr) {
            return;
        }
        for (int w = WAYS - 1; w > 0; --w) {
            s->pos[w] = s->pos[w - 1];
        }
        s->pos[0] = (uint32_t) end;
        if (s->n < WAYS) {
            s->n++;
        }
    }

    static void index_at(seq_info & si, size_t end) {
        if (!indexable(si, end)) {
            return;
        }
        if (2*(si.n_used + 1) > si.table.size()) {
            // grow: every key again at twice the size (load factor <= 0.5)
            const size_t cap = std::max<size_t>(8192, si.table.size()*2);
            si.table.assign(cap, slot_t{});
            si.mask   = cap - 1;
            si.n_used = 0;
            for (size_t e = 2; e < end; ++e) {
                if (indexable(si, e)) {
                    insert_at(si, e);
                }
            }
        }
        insert_at(si, end);
    }

    static void clear(seq_info & si) {
        si.hist.clear();
        si.table.clear();
        si.mask   = 0;
        si.n_used = 0;
    }

    // bring the history up to the prompt (the slot's tokens: accepted tokens are appended; a new request or a truncation rebuilds)
    static void sync(seq_info & si, const llama_tokens & prompt) {
        const size_t n = si.hist.size();
        bool same = n <= prompt.size();
        if (same && n > 0) {
            same = si.hist[0] == prompt[0] && si.hist[n - 1] == prompt[n - 1] && si.hist[n/2] == prompt[n/2];
        }
        if (!same) {
            clear(si);
        }
        for (size_t i = si.hist.size(); i < prompt.size(); ++i) {
            si.hist.push_back(prompt[i]);
            index_at(si, si.hist.size() - 1);
        }
    }

    // the continuation of the longest earlier repeat of the history + id_last, at most max_k tokens; sets match
    int propose(seq_info & si, llama_token id_last, int max_k, llama_tokens & out, int & match) const {
        match = 0;
        si.hist.push_back(id_last);
        const size_t n = si.hist.size();
        int k = 0;
        if (n >= 4 && max_k > 0 && indexable(si, n - 1)) {
            const size_t cur = n - 1;
            const slot_t * s = find(si, key_at(si.hist, cur), false);
            if (s != nullptr) {
                size_t best_end = 0;
                int    best_len = 0;
                for (int w = 0; w < s->n; ++w) {
                    const size_t p = s->pos[w];
                    if (p >= cur) {
                        continue;
                    }
                    int len = 0;
                    while (len < MAX_MATCH && len <= (int) p && si.hist[p - len] == si.hist[cur - len]) {
                        ++len;
                    }
                    if (len > best_len) {   // most recent first, so a tie keeps the newer place
                        best_len = len;
                        best_end = p;
                    }
                }
                if (best_len >= si.min_match) {
                    match = best_len;
                    // the continuation may run into the current suffix (periodic text): reading up to cur is valid
                    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) {
                        if (si.hist[q] == LLAMA_TOKEN_NULL) {
                            break;
                        }
                        out.push_back(si.hist[q]);
                        ++k;
                    }
                }
            }
        }
        si.hist.pop_back();
        return k;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & si = sinfos[seq_id];
        if (si.n_lookups > 0) {
            SPC_INF("ngram-suffix seq %d (last request): %" PRId64 " lookups, %" PRId64 " matches, %" PRId64 " drafts of %" PRId64 " tokens, %"
                    PRId64 " accepted, %" PRId64 " left to the model drafter; q by match class %.2f %.2f %.2f %.2f\n", seq_id, si.n_lookups,
                    si.n_matches, si.n_drafts, si.n_drafted, si.n_accepted, si.n_yield, rate(si, 0), rate(si, 1), rate(si, 2), rate(si, 3));
        }
        if (policy_on && si.n_lookups > 0) {
            const int ts = std::min(si.n_short + 1, common_spec_cost_policy::MAX_T);
            const int tl = std::min(params.n_max + 1, common_spec_cost_policy::MAX_T);
            SPC_INF("ngram-suffix seq %d: round ms by window 1 / %d / %d: %.2f (%.0f) / %.2f (%.0f) / %.2f (%.0f), realized gain %.2f (%.0f) / %.2f (%.0f), %"
                    PRId64 " probes\n", seq_id, ts, tl, policy.cost_ms(1), policy.cost_n[1], policy.cost_ms(ts), policy.cost_n[ts], policy.cost_ms(tl),
                    policy.cost_n[tl], policy.gain[ts], policy.gain_n[ts], policy.gain[tl], policy.gain_n[tl], si.n_probe);
        }
        si.n_lookups = si.n_matches = si.n_drafts = si.n_drafted = si.n_accepted = si.n_yield = 0;
        si.n_probe = 0;
        si.k_last = 0;
        si.timer.reset();
        si.min_match = (params.long_prompt > 0 && params.n_min_match_long > 0 && (int64_t) prompt.size() >= params.long_prompt) ?
                       params.n_min_match_long : params.n_min_match;
        {
            const bool lw = params.win_long_prompt > 0 && (int64_t) prompt.size() >= params.win_long_prompt;
            si.n_short = lw ? params.n_max_short_long  : params.n_max_short;
            si.n_long  = lw ? params.n_long_match_long : params.n_long_match;
        }
        if (policy_on && n_seq == 1) {
            policy.reset_gain();
        }
        clear(si);
        sync(si, prompt);
    }

    bool process(const common_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }
            auto & si = sinfos[seq_id];
            si.k_last = 0;
            if (policy_on) {
                si.timer.tick(policy, ggml_time_us());
            }
            sync(si, *dp.prompt);
            si.n_lookups++;

            int cap = params.n_max;
            if (dp.n_max > 0) {
                cap = std::min(cap, dp.n_max);
            }
            llama_tokens cand;
            int match = 0;
            const int offered = propose(si, dp.id_last, cap, cand, match);
            if (offered == 0) {
                continue;
            }
            si.n_matches++;
            const int    b   = bucket(match);
            const double q   = rate(si, b);
            const int    win = match >= si.n_long ? cap : std::min(cap, si.n_short);
            // the draft is the whole window (fewer tokens only where the repeat runs out): every verify length is a graph of its own
            // (a change of length rebuilds the target graph, 10-15 ms on G4, plans/mtp_adapt.md), so the lengths stay few
            int k = 0;
            if (with_model) {
                // a full window from a class that is right most of the time, else the model drafts this round
                k = (offered >= win && q >= params.p_take) ? win : 0;
            } else if (policy_on) {
                // k11-decode: the window that commits the most tokens per measured ms, or none (common/spec-cost-policy.h)
                const int k_full = std::min(offered, win);
                k = policy.choose(k_full, std::min(k_full, si.n_short), q, si.n_short != params.n_max_short);
                if (k > 0 && policy.cost_n[std::min(k + 1, common_spec_cost_policy::MAX_T)] < common_spec_cost_policy::PROBES) {
                    si.n_probe++;
                }
            } else {
                // a short repeat (under n_min_draft tokens) is in any text and drafts alone only rarely pay: a drafted step between
                // plain ones changes the graph twice (LLAMA_SPEC_SUFFIX_MIN_DRAFT=3: every match of the index drafts)
                k = (match >= params.n_min_draft && q >= params.p_cut) ? std::min(offered, win) : 0;
            }
            if (k == 0) {
                si.ok[b]  *= DRIFT;
                si.bad[b] *= DRIFT;
                if (with_model) {
                    si.n_yield++;
                }
                continue;
            }
            dp.result->assign(cand.begin(), cand.begin() + k);
            if (policy_on) {
                si.timer.next(k + 1);
            }
            si.k_last      = k;
            si.bucket_last = b;
            si.match_last  = match;
            si.n_drafts++;
            si.n_drafted += k;
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        auto & si = sinfos[seq_id];
        if (!is_other && si.k_last > 0 && si.bucket_last >= 0) {
            const int b = si.bucket_last;
            const int a = std::min<int>(n_accepted, si.k_last);
            si.ok[b]  = DECAY*si.ok[b] + a;
            si.bad[b] = DECAY*si.bad[b] + (a < si.k_last ? 1.0 : 0.0);
            si.n_accepted += a;
            si.timer.acc_last = a;
        }
        si.k_last = 0;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const common_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    std::vector<double> synth_probs;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX:  return "ngram-suffix";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);

    // a gemma-4 assistant is an MTP head on the target's hidden state (its graph needs the state embedding of the target), it has no
    // nextn tensors: without this, -md with no --spec-type loaded it as a plain draft and the server aborted in its first test decode
    if (arch == "gemma4-assistant") {
        SPC_INF("auto-detected speculative type '%s' from the draft model metadata (%s)\n",
                common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_DRAFT_MTP).c_str(), arch.c_str());
        return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    }

    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

bool common_speculative_uses_draft_ctx(const std::vector<common_speculative_type> & types) {
    for (const auto type : types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                return true;
            default:
                break;
        }
    }
    return false;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX:
                n_max = std::max(n_max, spec->suffix_n_max());
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

// LLAMA_DRAFT_UBATCH_MAX: the largest ubatch of the context of a separate MTP drafter (default 512, 0 = the target's n_ubatch)
uint32_t common_draft_ubatch_max() {
    static const uint32_t v = [] {
        const char * env = getenv("LLAMA_DRAFT_UBATCH_MAX");
        return env != nullptr ? (uint32_t) std::max(0, atoi(env)) : 512u;
    }();
    return v;
}

// k16-draft: the shape of tensor `name` in a GGUF (split-aware: a target's lm head is usually in a later shard), false if absent
static bool common_gguf_tensor_ne(const std::string & path, const char * name, int64_t ne[GGML_MAX_DIMS]) {
    std::string prefix;
    int n_split = 1;
    const size_t at = path.rfind("-00001-of-");
    if (at != std::string::npos && path.size() == at + 20 && path.compare(path.size() - 5, 5, ".gguf") == 0) {
        prefix  = path.substr(0, at);
        n_split = atoi(path.substr(at + 10, 5).c_str());
    }
    for (int i = 0; i < std::max(1, n_split); ++i) {
        std::string fname = path;
        if (n_split > 1) {
            std::vector<char> sp(prefix.size() + 32);
            llama_split_path(sp.data(), sp.size(), prefix.c_str(), i, n_split);
            fname = sp.data();
        }
        ggml_context * meta = nullptr;
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
        gguf_context_ptr ctx(gguf_init_from_file(fname.c_str(), gp));
        if (!ctx) {
            return false;
        }
        ggml_context_ptr meta_ptr(meta);
        if (const ggml_tensor * t = ggml_get_tensor(meta, name)) {
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                ne[d] = t->ne[d];
            }
            return true;
        }
    }
    return false;
}

// k16-draft (idea from ikawrakow/ik_llama.cpp PR #2403): a separate MTP draft model (-md with --spec-type draft-mtp) uses the target's
// token_embd and output when both files hold them with the same shapes (llama_model_loader::borrow_io checks again at load, against
// the loaded target). Decided here so the fit's measurements of the draft (no target loaded yet) count the lm head as the target's.
// LLAMA_MTP_BORROW_IO=0: the draft loads its own copies
static bool common_mtp_borrow_io_ok(const std::string & path_tgt, const std::string & path_dft) {
    const char * e = getenv("LLAMA_MTP_BORROW_IO");
    if ((e != nullptr && atoi(e) == 0) || path_tgt.empty() || path_dft.empty() || path_tgt == path_dft) {
        return false;
    }
    for (const char * name : { "token_embd.weight", "output.weight" }) {
        int64_t a[GGML_MAX_DIMS] = {}, b[GGML_MAX_DIMS] = {};
        if (!common_gguf_tensor_ne(path_tgt, name, a) || !common_gguf_tensor_ne(path_dft, name, b) ||
                !std::equal(a, a + GGML_MAX_DIMS, b)) {
            return false;
        }
    }
    return true;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        // default to global devices value
        if (!params_spec.devices.empty()) {
            result.devices           = params_spec.devices;
        }
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
        // (the device list is null-terminated, so a single device means size 2)
        const size_t n_devs = std::count_if(params_spec.devices.begin(), params_spec.devices.end(),
                [](ggml_backend_dev_t d) { return d != nullptr; });
        if (n_devs == 1) {
            result.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }

        // k18 (LLAMA_DRAFT_REQUANT=q4_k|q3_k|..., opt-in): the draft model's FFN / expert weights are requantized to that type at load
        // (llama_model_loader, KV override general.requant_type), so the fit prices the smaller drafter; drafts are verified, so the
        // output does not change, only the acceptance
        if (const char * rq = getenv("LLAMA_DRAFT_REQUANT")) {
            if (rq[0] != 0) {
                if (!result.kv_overrides.empty() && result.kv_overrides.back().key[0] == 0) {
                    result.kv_overrides.pop_back();
                }
                llama_model_kv_override o = {};
                snprintf(o.key, sizeof(o.key), "%s", "general.requant_type");
                o.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                snprintf(o.val_str, sizeof(o.val_str), "%s", rq);
                result.kv_overrides.push_back(o);
                result.kv_overrides.emplace_back();
                result.kv_overrides.back().key[0] = 0;
            }
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    result.n_ubatch      = (int32_t) common_draft_n_ubatch(params, (uint32_t) params.n_ubatch);
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // the MoE cache of a separate draft model is not supported; an MTP context runs on the weights of the target model and caches its
    // NextN layers with its share of the same budget (see llama_moe_cache::budget, LLAMA_MOE_CACHE_MTP=0 disables it)
    const bool spec_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(),
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    result.moe_cache_size = spec_mtp && !has_draft ? params.moe_cache_size : 0;

    // k16-draft: borrowed token_embd / output (llama_model_params::mtp_borrow_io)
    result.mtp_borrow_io = spec_mtp && has_draft && common_mtp_borrow_io_ok(params.model.path, params_spec.mparams.path);

    // a separate MTP drafter (-md) is one block: its context reads the prompt in ubatches of at most common_draft_ubatch_max()
    // tokens (it splits the target's larger ubatches itself), so its compute buffer stays small and the MoE cache of the target
    // keeps the difference (808 -> 225 MiB on Qwen3.8-Flash-Next at -ub 2048); n_batch is kept, so a target ubatch is one decode
    // the cap only saves MoE cache memory: without a MoE cache (all experts resident) the drafter reads each target ubatch in one
    // pass (8 passes of 512 rows per 4096-row ubatch cost ~1.9x its kernel time plus 8 host round trips per ubatch); an explicit
    // LLAMA_DRAFT_UBATCH_MAX still applies everywhere, LLAMA_MTP_HANDOFF_DUB=0 restores the cap without a MoE cache
    const bool dub_cap = params.moe_cache_size > 0 || getenv("LLAMA_DRAFT_UBATCH_MAX") != nullptr || !common_mtp_handoff_on("DUB");
    if (spec_mtp && has_draft && dub_cap) {
        const uint32_t cap = common_draft_ubatch_max();
        if (cap > 0 && (uint32_t) result.n_ubatch > cap) {
            result.n_ubatch = (int32_t) cap;
            result.n_batch  = std::max(result.n_batch, params.n_ubatch);
        }
    }

    // an explicit draft ubatch (--spec-draft-n-ubatch) wins over the caps above
    if (params_spec.n_ubatch > 0) {
        result.n_ubatch = params_spec.n_ubatch;
    }

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        // k16-draft: the target lends its token_embd / output to an MTP draft model (llama_model_loader::borrow_io)
        if (spec_mtp && mparams.mtp_borrow_io) {
            mparams.model_shared = model_tgt;
        }

        llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 12);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX: {
                // a model drafter later in the list takes the rounds the suffix draft is not sure of
                const bool with_model = std::any_of(configs.begin(), configs.end(), [](const common_speculative_config & c) {
                    return c.type == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE || c.type == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||
                           c.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP    || c.type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
                           c.type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
                });
                impls.push_back(std::make_unique<common_speculative_impl_ngram_suffix>(config.params, n_seq, with_model));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_rows_ok(const common_speculative * spec) {
    if (spec == nullptr) {
        return true;
    }
    for (const auto & impl : spec->impls) {
        if (!impl->row_offset_ok()) {
            return false;
        }
    }
    return true;
}

bool common_speculative_process_rows(common_speculative * spec, const common_batch & batch, int32_t row_offset) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        impl->nextn_row_offset = row_offset;
        result = result && impl->process(batch);
        impl->nextn_row_offset = 0;
    }

    return result;
}

bool common_speculative_rows_early_ok(const common_speculative * spec) {
    if (spec == nullptr || spec->impls.empty()) {
        return false;
    }
    for (const auto & impl : spec->impls) {
        if (!impl->rows_early_ok()) {
            return false;
        }
    }
    return true;
}

bool common_speculative_process_rows_early(common_speculative * spec, const common_batch & batch, int32_t row_offset) {
    GGML_ASSERT(common_speculative_rows_early_ok(spec));
    bool result = true;
    for (auto & impl : spec->impls) {
        impl->nextn_row_offset = row_offset;
        impl->rows_early       = true;
        result = result && impl->process(batch);
        impl->rows_early       = false;
        impl->nextn_row_offset = 0;
    }
    return result;
}

bool common_speculative_process(common_speculative * spec, const common_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);

                        // trim the candidates only if the drafter produced them (n-gram drafters do not)
                        if (dp.result_q && !dp.result_q->empty()) {
                            dp.result_q->resize(dp.n_max);
                        }
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}
