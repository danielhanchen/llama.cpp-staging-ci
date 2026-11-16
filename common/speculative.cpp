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

// k19-df2pp: the DFlash2 window beside a small MoE cache (see the draft window below)
#ifndef K19_DF2PP_TOK_ALL
#define K19_DF2PP_TOK_ALL 5
#endif
#ifndef K19_DF2PP_POS_MIN
#define K19_DF2PP_POS_MIN 16384
#endif

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

    // kdp (LLAMA_MTP_DRAFT_LEND): a prompt batch whose rows lie before the prompt's last LLAMA_MTP_PROMPT_TAIL rows is not run
    // through the drafter; an impl that keeps its own state per position processes it as usual
    virtual bool skip_prompt(const common_batch & batch) { return process(batch); }

    // k11 (LLAMA_LM_CKPT_MERGE): process() reads the target's rows from this batch index on (a segment of a merged prompt batch);
    // impls that read target features without honouring it say so
    int32_t nextn_row_offset = 0;
    virtual bool row_offset_ok() const { return true; }
    // k15-pp3 (LLAMA_MTP_PROMPT_OVERLAP): process() reads the target's rows without a synchronization (llama_get_embeddings_nextn_early),
    // the caller waited for them (llama_nextn_rows_wait); only impls that read nothing else of the target say yes
    bool rows_early = false;
    virtual bool rows_early_ok() const { return false; }
    // k19-host (LLAMA_MTP_PROMPT_BG): process() reads the target's rows through a pointer the caller resolved (no target call: it may run
    // on a worker beside the target's next decode), rows [0, n_lo) from a copy the caller made first; floats per row (0 = not supported)
    virtual int64_t rows_ext_floats() const { return 0; }
    virtual bool rows_ext_set(const float * /*h*/, int32_t /*n_rows*/, int32_t /*n_lo*/) { return false; }
    virtual void rows_ext_clear() {}

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
            // k19-sharednoise: so does a draft drawn with the target's noise (sample-and-match is what keeps it exact)
            if (!params.probabilistic || (dp.sn != nullptr && dp.sn->on)) {
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
                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                // k19-sharednoise: the draft takes the target's noise at its position (the probability fed to the window policies
                // is the picked token's under the mirrored chain; they calibrate it against what held)
                float p_draft = cur_p->data[0].p;
                const bool sn_on = dp.sn != nullptr && dp.sn->on;
                const llama_token id = sn_on ? common_shared_noise_draft_pick(*dp.sn, (int32_t) result.size(), result.data(), cur_p->data, cur_p->size, &p_draft) :
                                       dp.result_q ? id_sampled : cur_p->data[0].id;
                GGML_UNUSED(p_draft);

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

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

// k19-handoff diagnostics of the DFlash2 drafter (all off by default; none changes a draft except LLAMA_DFLASH_CYCLE)
common_dflash_diag_t & common_dflash_diag() {
    static common_dflash_diag_t d = [] {
        common_dflash_diag_t x;
        auto on = [](const char * n) { const char * e = getenv(n); return e != nullptr && atoi(e) != 0; };
        x.trace  = on("LLAMA_DFLASH_TRACE");
        x.cycle  = on("LLAMA_DFLASH_CYCLE");
        x.timing = on("LLAMA_PASS_TIMING");
        x.p.resize(64); x.m.resize(64); x.pos0.assign(64, 0);
        return x;
    }();
    return d;
}

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    // k19-df2pp: process() reads the target's layer inputs at the segment's rows of the decode (nextn_row_offset), so a merged prompt
    // batch (the server's checkpoint break inside one layer-major decode) feeds the drafter as the separate batches did.
    // LLAMA_DFLASH_ROW_OFFSET=0: refused, the server splits the batch at the checkpoint as before
    bool row_offset_ok() const override {
        static const bool on = getenv("LLAMA_DFLASH_ROW_OFFSET") == nullptr || atoi(getenv("LLAMA_DFLASH_ROW_OFFSET")) != 0;
        return on;
    }

    common_params_speculative_draft params;

    common_batch batch;        // noise tokens
    common_batch batch_inject; // target features for KV cache injection

    std::vector<float> features_buf; // [n_chunk, n_embd_enc] gathered target features
    std::vector<uint64_t> feat_hash_v; // k19-df2pp: LLAMA_DFLASH_FEAT_HASH running hash per sequence
    std::vector<int64_t>  feat_hash_n;

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

    // k19-a100chat (LLAMA_DFLASH_EV_AUTO): the DFlash2 window that maximises the expected tokens per ms, with both inputs measured in
    // this run instead of taken from a G4 table (LLAMA_DFLASH_EV): the pass cost by verify width (host time from the end of the draft
    // to the target's accept, plus the draft call itself) and the chance that a step is accepted given its selector probability (20
    // bins, a prior of the G4 calibration worth 10 drafts per bin). Beside a MoE cache whose slots hold most of the host experts (no
    // other window rule runs there: A100 80 GB fn4n full card / 48 GB) every further verify row costs 2.7 / 4.4 ms (tc_n1a: w3 16.2 /
    // 25.6 ms, w7 26.8 / 43.0), more than on G4, so the whole 7-row window verifies rows that are rarely accepted. The window then
    // follows timing, so the default-mode text can differ between launches; under the exact verify contract it does not
    struct eva_t {
        static constexpr int W  = 10; // widths 1..9
        static constexpr int NB = 20;
        double  v_sum[W] = {}, v_n[W] = {};          // decayed sums / counts of the verify ms by width
        double  d_ms = 0.0, d_n = 0.0;               // decayed draft call ms
        double  f_acc[NB] = {}, f_reach[NB] = {};    // accepted / reached steps by selector probability bin
        int64_t n_meas = 0, n_cut = 0, n_rows_cut = 0;
        int64_t picks[W] = {};
        int64_t cnt[W] = {}, last_meas[W] = {};      // k19-g4verify: measured passes by width, n_meas at a width's last measurement
        int64_t n_probe = 0;                         // k19-g4verify: drafts cut to probe a width (see df2_meas)
        double  k_prior = 4.0;                       // weight of the shape prior in eva_verify_ms, in passes
        bool    small   = false;                     // k19-g4verify: the rule runs beside a small MoE cache (df2_meas)
        std::vector<int64_t> t_end;                  // per seq: end of the last draft call (0 = nothing pending)
        std::vector<int32_t> w_last;                 // per seq: verify rows of that draft
        std::vector<std::vector<float>> p_last;      // per seq: its steps' selector probabilities
        bool init = false;
    } eva;

    // 0 = off, 1 = on wherever a DFlash2 draft runs without another window rule, 2 = default: one sequence beside a MoE cache whose
    // slots hold more than LLAMA_DFLASH_CACHED_FRAC_MAX of the host experts and >= LLAMA_DFLASH_CACHED_TOKENS_MAX tokens (A100 48 GB:
    // 70.7%, 36 tokens; chat +6.2%, 8k +20% vs the kill switch, ac_a48b); the cached bands below keep their own rules. Not without a
    // cache: A100 fn4n full card (the 26.8 GiB PLE table is the only host part) has a flat cost curve, chat level (169.7 vs 170.1,
    // ac_a0), and srv_kld long48 put one position above the -fa off band's max there (0.090 vs 0.075).
    // Under the exact verify contract beside a MoE cache it stays off by default: there the cache's host path computes the misses of
    // some batch sizes on the CPU (ac_a48 48 GB 8k: EV-auto on / off gave two exact texts; plans/k19_a100chat.md); =1 forces it
    static int eva_mode() {
        static const int m = [] {
            const char * e = getenv("LLAMA_DFLASH_EV_AUTO");
            return e == nullptr ? 2 : (atoi(e) != 0 ? 1 : 0);
        }();
        return m;
    }

    // verify ms of a pass of w rows: the decayed mean where measured, shrunk towards the shape of A100 / G4 cached passes
    // (v(w) ~ 0.3 + 0.1 w of v(7): tc_n1a v3 / v7 = 0.60 at full card and 48 GB) scaled by the most measured width
    double eva_verify_ms(int w) const {
        int wr = -1;
        for (int i = 1; i < eva_t::W; ++i) {
            if (eva.v_n[i] > 0.0 && (wr < 0 || eva.v_n[i] > eva.v_n[wr])) {
                wr = i;
            }
        }
        if (wr < 0) {
            return 1.0;
        }
        const double m_ref = eva.v_sum[wr] / eva.v_n[wr];
        const double prior = m_ref * (0.3 + 0.1 * w) / (0.3 + 0.1 * wr);
        const double k     = eva.k_prior;
        const double n     = w < eva_t::W ? eva.v_n[w] : 0.0;
        const double m     = n > 0.0 ? eva.v_sum[w] / n : prior;
        return (n * m + k * prior) / (n + k);
    }

    double eva_f(float p) const {
        const int b = std::min(eva_t::NB - 1, std::max(0, (int) (p * eva_t::NB)));
        return eva.f_acc[b] / eva.f_reach[b];
    }

    void eva_init() {
        if (eva.init) {
            return;
        }
        eva.init = true;
        static const float cal[eva_t::NB] = { 0.141f, 0.141f, 0.141f, 0.141f, 0.203f, 0.264f, 0.278f, 0.339f, 0.363f, 0.423f,
                                              0.460f, 0.492f, 0.583f, 0.614f, 0.614f, 0.619f, 0.657f, 0.748f, 0.814f, 0.929f };
        for (int b = 0; b < eva_t::NB; ++b) {
            eva.f_reach[b] = 10.0;
            eva.f_acc[b]   = 10.0 * cal[b];
        }
        eva.t_end.assign(n_seq, 0);
        eva.w_last.assign(n_seq, 0);
        eva.p_last.assign(n_seq, {});
    }

    void eva_log(const char * when) const {
        std::string s;
        char b[64];
        for (int w = 1; w < eva_t::W; ++w) {
            if (eva.v_n[w] > 0.0 || eva.picks[w] > 0) {
                snprintf(b, sizeof(b), " w%d %.2f (%.0f, %" PRId64 ")", w, eva_verify_ms(w), eva.v_n[w], eva.picks[w]);
                s += b;
            }
        }
        LOG_INF("k19-a100chat: DFlash2 EV-auto %s: %" PRId64 " passes measured, draft %.2f ms, %" PRId64 " drafts cut by %" PRId64 " rows, %" PRId64 " probes; verify ms (decayed n, picks):%s\n",
                when, eva.n_meas, eva.d_n > 0.0 ? eva.d_ms / eva.d_n : 0.0, eva.n_cut, eva.n_rows_cut, eva.n_probe, s.c_str());
    }

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
        if (eva.init) {
            eva_log("at exit");
        }
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

        if (is_dflash2) {
            // k19-draftcost: the restricted DFlash2 draft head (LLAMA_DFLASH_DRAFT_VOCAB) gets rows for the prompt's tokens it lacks
            const int32_t n_add = llama_model_draft_head_add(llama_get_model(params.ctx_dft), prompt.data(), N);
            if (n_add > 0) {
                SPC_DBG("%s: draft head: %d prompt tokens added\n", __func__, n_add);
            }
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
                        const float * src = layer + (size_t) (nextn_row_offset + i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                // k19-df2pp (diagnostic, LLAMA_DFLASH_FEAT_HASH=1): a running FNV-1a hash of the features the drafter takes, in position
                // order per sequence, logged after each chunk: equal hashes = the drafter saw bit-identical target features
                static const bool feat_hash = getenv("LLAMA_DFLASH_FEAT_HASH") != nullptr && atoi(getenv("LLAMA_DFLASH_FEAT_HASH")) != 0;
                if (feat_hash) {
                    if (feat_hash_v.size() < (size_t) n_seq) {
                        feat_hash_v.resize(n_seq, 1469598103934665603ULL);
                        feat_hash_n.resize(n_seq, 0);
                    }
                    if (batch_in.tokens[i_batch_beg[seq_id] + offset].pos[0] == 0) {
                        feat_hash_v[seq_id] = 1469598103934665603ULL; // a prompt from position 0: a new hash
                        feat_hash_n[seq_id] = 0;
                    }
                    const auto * b = (const uint8_t *) features_buf.data();
                    for (size_t x = 0; x < (size_t) n_chunk * n_embd_enc * sizeof(float); ++x) {
                        feat_hash_v[seq_id] = (feat_hash_v[seq_id] ^ b[x]) * 1099511628211ULL;
                    }
                    feat_hash_n[seq_id] += n_chunk;
                    LOG_INF("k19-df2pp: dflash features seq %d: %" PRId64 " rows up to pos %d, hash %016" PRIx64 "\n", (int) seq_id,
                            feat_hash_n[seq_id], (int) batch_in.tokens[i_batch_beg[seq_id] + offset + n_chunk - 1].pos[0], feat_hash_v[seq_id]);
                }

                auto inject = [&](int32_t i0, int32_t i1) {
                    batch_inject.clear();
                    for (int32_t i = i0; i < i1; ++i) {
                        const llama_pos p = batch_in.tokens[i_batch_beg[seq_id] + offset + i].pos[0];
                        const llama_pos pos_arr[4] = { p, p, p, 0 };
                        batch_inject.add_embd({ features_buf.data() + (size_t) i * n_embd_enc, 1, (size_t) n_embd_enc }, pos_arr, seq_id, false);
                    }
                    return llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_inject.get());
                };
                // k19-a100chat (LLAMA_DFLASH_INJECT_SPLIT=0: off): a chunk whose graph does not get its compute buffer (rc -2: the
                // allocation failed, the drafter's KV from the chunk's first position on is removed) is injected again from the first
                // row not yet in, in halves down to 32 rows: a smaller graph needs a smaller buffer. Before, the request failed with an
                // HTTP 500 (Colab A100 48 GB fn4n, default cache, -fa on, srv_kld long48: the 512-row graph's 30 MiB did not fit).
                // Only after a failed allocation, so every run that did not fail is unchanged
                static const bool inject_split = getenv("LLAMA_DFLASH_INJECT_SPLIT") == nullptr || atoi(getenv("LLAMA_DFLASH_INJECT_SPLIT")) != 0;
                int32_t done = 0;
                int32_t part = n_chunk;
                int32_t rc   = 0;
                while (done < n_chunk) {
                    const int32_t end = std::min(n_chunk, done + part);
                    rc = inject(done, end);
                    if (rc == 0) {
                        done = end;
                        continue;
                    }
                    if (!inject_split || is_lilicorr || rc != -2 || part <= 32) {
                        break;
                    }
                    part = std::max<int32_t>(32, part/2);
                    LOG_WRN("%s: the drafter's %d-row injection did not get its compute buffer: rows %d.. again in pieces of %d "
                            "(LLAMA_DFLASH_INJECT_SPLIT=0 off)\n", __func__, (int) (end - done), (int) done, (int) part);
                }
                if (rc != 0) {
                    LOG_ERR("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) (offset + done));
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

        const int64_t t_eva0 = is_dflash2 && eva_mode() != 0 ? ggml_time_us() : 0;

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
            // k19-df2acc (LLAMA_DFLASH_FULL_BLOCK=1): a DFlash2 draft decodes the whole trained block (the anchor + block_size - 1 masks)
            // and walks only its first n_max rows. The attention over the block is not causal, so a block cut to n_max + 1 rows changes
            // every row's draft against the block the drafter was trained on
            static const bool full_block = getenv("LLAMA_DFLASH_FULL_BLOCK") != nullptr && atoi(getenv("LLAMA_DFLASH_FULL_BLOCK")) != 0;
            const int32_t n_block_tokens = is_lilicorr ? block_size :
                (is_dflash2 && full_block && n + block_size <= (int32_t) llama_n_ctx(ctx_dft)) ? std::max<int32_t>(block_size, n_draft + 1) : n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
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
        const bool    tm   = common_dflash_diag().timing;
        const int64_t t_d0 = tm ? ggml_time_us() : 0;
        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
            return;
        }
        const int64_t t_d1 = tm ? ggml_time_us() : 0;

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
                const int64_t t_d2 = tm ? ggml_time_us() : 0;
                auto & dg = common_dflash_diag();
                // k19-handoff (diagnostic, LLAMA_DFLASH_TRACE=1): the selector's per-step probability and score margin of each draft,
                // logged with its accepted count in accept(); the draft itself is unchanged
                const bool trace = is_dflash2 && dg.trace && seq_id < (llama_seq_id) dg.p.size();
                if (trace) {
                    dg.p[seq_id].clear();
                    dg.m[seq_id].clear();
                    dg.pos0[seq_id] = dp.pos0;
                }

                // LLAMA_DFLASH_P_CUM=P (default 0 = off): the walk also stops where the product of the steps' probabilities falls below P,
                // the chance that the whole draft so far is accepted if the selector were calibrated: a verify row costs about the same
                // wherever it ends, so a long draft of unlikely tokens costs verify rows for few accepted ones
                static const float p_cum_min = [] {
                    const char * e = getenv("LLAMA_DFLASH_P_CUM");
                    return e != nullptr ? std::max(0.0f, (float) atof(e)) : 0.0f;
                }();
                const float p_cum_thr = is_dflash2 ? p_cum_min : 0.0f;
                float p_cum = 1.0f;
                // k19-handoff: LLAMA_DFLASH_P_CUM_EXP=a (default 1): the product above is of p^a, a temperature for a selector whose
                // probabilities are not calibrated (a < 1 trusts a low p less)
                static const float p_cum_exp = [] {
                    const char * e = getenv("LLAMA_DFLASH_P_CUM_EXP");
                    return e != nullptr && atof(e) > 0.0 ? (float) atof(e) : 1.0f;
                }();
                // k19-handoff: LLAMA_DFLASH_EV="f_0,..,f_(B-1)/c_1,..,c_(n_max+1)" (default off): the window that maximises the expected
                // tokens per ms, from a calibration table f (the chance that step i is accepted given the steps before it were, by the
                // step's selector probability in B equal bins of [0, 1]) and the cost c_w of a pass that verifies w rows
                static const std::pair<std::vector<float>, std::vector<float>> ev_rule = [] {
                    std::pair<std::vector<float>, std::vector<float>> r;
                    const char * e = getenv("LLAMA_DFLASH_EV");
                    if (e == nullptr) {
                        return r;
                    }
                    const std::string v(e);
                    const size_t sl = v.find('/');
                    if (sl == std::string::npos) {
                        return r;
                    }
                    auto parse = [](const std::string & t) {
                        std::vector<float> out;
                        size_t a = 0;
                        while (a < t.size()) {
                            size_t b = t.find(',', a);
                            if (b == std::string::npos) {
                                b = t.size();
                            }
                            out.push_back((float) atof(t.substr(a, b - a).c_str()));
                            a = b + 1;
                        }
                        return out;
                    };
                    r.first  = parse(v.substr(0, sl));
                    r.second = parse(v.substr(sl + 1));
                    if (r.first.empty() || r.second.size() < 2) {
                        r.first.clear();
                        r.second.clear();
                    } else {
                        LOG_INF("DFlash2 window: expected tokens per ms, %zu calibration bins, costs for widths 1..%zu (LLAMA_DFLASH_EV)\n",
                                r.first.size(), r.second.size());
                    }
                    return r;
                }();
                const bool use_ev = is_dflash2 && !ev_rule.first.empty();
                // k19-handoff: the window of a DFlash2 draft beside a target without a MoE cache, on by default under the exact verify
                // contract (LLAMA_MTP_EXACT=1: the text does not depend on the window), else opt-in (LLAMA_DFLASH_WINDOW=1: the default
                // verify's kernels follow the width, so the text changes: srv_kld long48 on G4 mean 0.0016 / p99 0.015 vs the -fa off band
                // 0.0037 / 0.035, but one position at 0.30 above the band's max 0.155); LLAMA_DFLASH_WINDOW=0: off; LLAMA_DFLASH_EV,
                // --spec-draft-p-min and LLAMA_DFLASH_P_CUM replace it. Of two windows, the short one
                // (4 rows below position 4096, 3 above) and the whole draft, the one with more expected tokens per ms: the expected tokens
                // come from the chance that each step is accepted given its selector probability (calibrated on G4 Flash-Next, 48 prompts
                // of 2k / 8k tokens: the selector is near calibrated, a little over-confident at the top), the ms from G4's pass cost by
                // verify width (the drafter and host ~1.25 ms plus the verify, ~0.9 ms per row below 4096 and ~1.1 above). Two widths
                // only: each width is one more CUDA graph capture (~20 ms) at every new KV bucket
                static const bool df2_window = [] {
                    const char * e = getenv("LLAMA_DFLASH_WINDOW");
                    return e == nullptr ? llama_exact_verify_enabled() : atoi(e) != 0;
                }();
                // k19-placemid (LLAMA_DFLASH_WINDOW=2, opt-in, measurement): the rule beside a MoE cache too, with the same pass costs
                static const bool df2_window_cached = getenv("LLAMA_DFLASH_WINDOW") != nullptr && atoi(getenv("LLAMA_DFLASH_WINDOW")) == 2;
                // k19-df2pp: beside a MoE cache each further verify row costs ~3.4 ms (the experts it misses), not the ~1 ms of a full card,
                // so the whole 7-row draft lost to MTP at long contexts (G4 Flash-Next fn4, default verify: 3 rows ~17 / 20 ms, 7 rows ~29 / 35 ms
                // per pass at 16 / 12 GB, the same ratio 1.7 at both). There the rule runs by default with those costs: the short window is
                // 3 rows at every position, LLAMA_DFLASH_CACHED_COST="c_lo,slope" (ms) overrides them; LLAMA_DFLASH_CACHED_WINDOW=0 (or
                // LLAMA_DFLASH_WINDOW=0) off, LLAMA_DFLASH_WINDOW=2 keeps the full-card costs beside the cache. The EV rule over every width
                // (LLAMA_DFLASH_EV) won 30k too but lost chat / 8k: each width is one more CUDA graph capture per KV bucket
                static const bool df2_window_cache_on = [] {
                    const char * w = getenv("LLAMA_DFLASH_WINDOW");
                    const char * e = getenv("LLAMA_DFLASH_CACHED_WINDOW");
                    return (w == nullptr || atoi(w) != 0) && (e == nullptr || atoi(e) != 0);
                }();
                static const std::pair<double, double> df2_cached_cost = [] {
                    std::pair<double, double> c = { 18.5, 3.4 };
                    if (const char * e = getenv("LLAMA_DFLASH_CACHED_COST")) {
                        double lo = 0.0, sl = 0.0;
                        if (sscanf(e, "%lf,%lf", &lo, &sl) == 2 && lo > 0.0 && sl >= 0.0) {
                            c = { lo, sl };
                        }
                    }
                    return c;
                }();
                // only where the cache's device slots hold the experts of fewer than LLAMA_DFLASH_CACHED_TOKENS_MAX (12) tokens: G4 fn4 12 / 16 GB
                // hold 3 / 7, 48 GB 32 (there the full-card costs stay opt-in, LLAMA_DFLASH_WINDOW=2)
                static const int32_t df2_cached_tok_max = getenv("LLAMA_DFLASH_CACHED_TOKENS_MAX") ? atoi(getenv("LLAMA_DFLASH_CACHED_TOKENS_MAX")) : 12;
                // k19-df2cost (LLAMA_DFLASH_CACHED_FRAC_MAX, default K19_DF2COST_FRAC_MAX 0.5; 0 = off; the cache leaves
                // LLAMA_DFLASH_CACHED_RESERVE_MIB for the second width's graphs, see common.cpp: without it the srv_kld gate at 32 GB died
                // of a CUDA out of memory, plans/k19_df2cost.md round 6): also
                // where the cache holds LLAMA_DFLASH_CACHED_FRAC_MIN (0.20) .. LLAMA_DFLASH_CACHED_FRAC_MAX of the
                // host experts (llama_moe_cache_frac; G4 fn4 24 GB 27.5%, 32 GB 42%), at every position, with that band's pass costs
                // (LLAMA_DFLASH_CACHED_COST_MID="c_lo,slope", default 15 ms at 3 rows + 2.3 ms per further row: G4 fn4 fixed-width passes
                // w3 14.8 / 16.1 ms, w7 23.0 / 25.9 ms at 32 / 24 GB, k19-df2cost round 2). The whole 7-row window there cost 8-11 ms more
                // than a 3-row one at 8k (dc_r6, 3 launches: 32 GB 143.6 -> 166.0, 24 GB 105.2 -> 129.8 tok/s; chat +1 / +5%).
                // LLAMA_DFLASH_CACHED_WINDOW=0 turns it off with the band below it
                static const double df2_mid_lo = getenv("LLAMA_DFLASH_CACHED_FRAC_MIN") ? atof(getenv("LLAMA_DFLASH_CACHED_FRAC_MIN")) : 0.20;
#ifndef K19_DF2COST_FRAC_MAX
#define K19_DF2COST_FRAC_MAX 0.5
#endif
                static const double df2_mid_hi = getenv("LLAMA_DFLASH_CACHED_FRAC_MAX") ? atof(getenv("LLAMA_DFLASH_CACHED_FRAC_MAX")) : K19_DF2COST_FRAC_MAX;
                static const std::pair<double, double> df2_mid_cost = [] {
                    std::pair<double, double> c = { 15.0, 2.3 };
                    if (const char * e = getenv("LLAMA_DFLASH_CACHED_COST_MID")) {
                        double lo = 0.0, sl = 0.0;
                        if (sscanf(e, "%lf,%lf", &lo, &sl) == 2 && lo > 0.0 && sl >= 0.0) {
                            c = { lo, sl };
                        }
                    }
                    return c;
                }();
                const bool   df2_cache_on = is_dflash2 && llama_moe_cache_active(params.ctx_tgt);
                const double df2_frac     = df2_cache_on ? llama_moe_cache_frac(params.ctx_tgt) : -1.0;
                const bool   df2_mid      = df2_cache_on && llama_moe_cache_tokens(params.ctx_tgt) >= df2_cached_tok_max &&
                    df2_frac >= df2_mid_lo && df2_frac < df2_mid_hi;
                const bool df2_cached = df2_cache_on && (llama_moe_cache_tokens(params.ctx_tgt) < df2_cached_tok_max || df2_mid);
                // where the slots hold at least LLAMA_DFLASH_CACHED_TOKENS_ALL (K19_DF2PP_TOK_ALL) tokens, only from position
                // LLAMA_DFLASH_CACHED_POS_MIN (K19_DF2PP_POS_MIN) on: there the whole draft paid at chat / 8k (G4 fn4 16 GB, slots of 7 tokens)
                static const int32_t df2_cached_tok_all = getenv("LLAMA_DFLASH_CACHED_TOKENS_ALL") ? atoi(getenv("LLAMA_DFLASH_CACHED_TOKENS_ALL")) : K19_DF2PP_TOK_ALL;
                static const int32_t df2_cached_pos_min = getenv("LLAMA_DFLASH_CACHED_POS_MIN") ? atoi(getenv("LLAMA_DFLASH_CACHED_POS_MIN")) : K19_DF2PP_POS_MIN;
                const bool df2_cached_here = df2_cached && (df2_window_cached || df2_mid ||
                    llama_moe_cache_tokens(params.ctx_tgt) < df2_cached_tok_all || dp.pos0 >= df2_cached_pos_min);
                // k19-g4verify: below the mid band (the slots hold the experts of fewer than LLAMA_DFLASH_CACHED_TOKENS_MAX tokens) the
                // fixed costs above underprice wide windows: a verify ubatch of more tokens than the slots hold
                // (llama_moe_cache_tokens) leaves the device slots and the single-graph host path, so every row's experts run on the
                // host in an eager split. G4 fn4 Flash-Next 10 GB / 131K and 12 GB / 262K (slots of 3-4 tokens): w3 25-28 ms, w4 31-35,
                // w7 56-60 ms per pass, not the 32 ms priced, and 47% of chat passes verified 7 rows. There the window comes from this
                // run's measured costs instead (the EV-auto rule below over every width, each width probed LLAMA_DFLASH_MEAS_PROBES (3)
                // times before its cost is trusted and re-probed after LLAMA_DFLASH_MEAS_STALE (256) passes without a measurement), with
                // one sequence and not under the exact verify contract (there the fixed rule keeps the window independent of timing).
                // LLAMA_DFLASH_CACHED_MEASURED=0: the fixed costs
                static const bool df2_meas_on = [] {
                    const char * e = getenv("LLAMA_DFLASH_CACHED_MEASURED");
                    return e == nullptr || atoi(e) != 0;
                }();
                const bool df2_meas = df2_meas_on && is_dflash2 && !use_ev && params.p_min <= 0.0f && p_cum_thr <= 0.0f &&
                    df2_cached && !df2_mid && df2_cached_here && !df2_window_cached && df2_window_cache_on &&
                    n_seq == 1 && eva_mode() != 0 && !llama_exact_verify_enabled();
                const bool use_df2w = is_dflash2 && !use_ev && !df2_meas && params.p_min <= 0.0f && p_cum_thr <= 0.0f &&
                    (df2_cached ? df2_cached_here && (df2_window_cached || df2_window_cache_on) : df2_window);
                const bool df2w_cached_costs = df2_cached && !df2_window_cached;
                if (use_df2w) {
                    static bool logged = false;
                    if (!logged) {
                        logged = true;
                        if (df2w_cached_costs) {
                            LOG_INF("%s: DFlash2 window beside the MoE cache (its slots hold %d tokens, %.1f%% of the host experts): the 3-row or "
                                    "the whole draft by expected tokens per ms, %.2f ms at 3 rows + %.2f per row (LLAMA_DFLASH_CACHED_WINDOW=0 off)\n",
                                    df2_mid ? "k19-df2cost" : "k19-df2pp", (int) llama_moe_cache_tokens(params.ctx_tgt), 100.0*df2_frac,
                                    df2_mid ? df2_mid_cost.first : df2_cached_cost.first, df2_mid ? df2_mid_cost.second : df2_cached_cost.second);
                        } else {
                            LOG_INF("k19-handoff: DFlash2 window: the short or the whole draft by expected tokens per ms (%s; LLAMA_DFLASH_WINDOW=0 off)\n",
                                    getenv("LLAMA_DFLASH_WINDOW") != nullptr ? "LLAMA_DFLASH_WINDOW=1" : "exact verify");
                        }
                    }
                }
                // k19-a100chat: the measured EV window (see eva_t), where neither LLAMA_DFLASH_EV nor a two-width window runs
                const bool use_eva = is_dflash2 && !use_ev && !use_df2w && eva_mode() != 0 && (df2_meas ||
                    eva_mode() == 1 || (n_seq == 1 && df2_cache_on && !df2_cached && df2_frac >= df2_mid_hi && !llama_exact_verify_enabled()));
                if (use_eva) {
                    eva_init();
                    if (df2_meas && !eva.small) {
                        eva.small   = true;
                        eva.k_prior = 1.0;  // the shape prior (fit to wider caches) is far off here: trust the probes
                    }
                    static bool logged = false;
                    if (!logged) {
                        logged = true;
                        LOG_INF("%s: DFlash2 window by expected tokens per ms from this run's measured pass costs and acceptance "
                                "(MoE cache %s, %.1f%% of the host experts, slots of %d tokens%s; LLAMA_DFLASH_EV_AUTO=0 off)\n",
                                df2_meas ? "k19-g4verify" : "k19-a100chat", df2_cache_on ? "on" : "off", df2_cache_on ? 100.0*df2_frac : 0.0,
                                df2_cache_on ? (int) llama_moe_cache_tokens(params.ctx_tgt) : -1,
                                df2_meas ? ", every width probed; LLAMA_DFLASH_CACHED_MEASURED=0: fixed costs" : "");
                    }
                }
                std::vector<float> step_p;

                int32_t predecessor = 0;
                std::vector<llama_token_data> sn_cand;
                // k19-a80verify (LLAMA_DFLASH_N_MAX_SM80, default 4, 0 = off): a DFlash2 draft walks at most that many steps where every
                // GPU is compute capability 8.0 (A100 / A30). There each further verify row costs 2.6-2.9 ms of a ~27 ms 7-row pass (no
                // MoE cache) and the drafter's 5th / 6th steps are accepted too rarely to pay for theirs: Colab A100 80 GB, Qwen3.8-Flash-Next
                // fn4n full card, --spec-draft-n-max 6: capped chat 182.9 vs 165.9 tok/s, 8k 109.8 vs 93.2, the same text (md5 equal), 5
                // rows (n_max 4) at 21.2 ms vs 7 rows at 26.8 (plans/k19_a80verify.md). The window rules below (EV, P_CUM, p_min) still
                // cut inside it. A user's --spec-draft-n-max below it is kept. Not beside a MoE cache (LLAMA_DFLASH_N_MAX_SM80_CACHED=1: there
                // too): there the measured EV window (eva_t) already picks the width, and the route tail skip makes the text follow the
                // window: A100 80 GB at a 48 GB cap, chat 119.5 vs 115.0 tok/s (same text), 8k 84.0 vs 93.9 (other text, 2.98 vs 3.41
                // tokens per pass at about the same pass time)
                static const int32_t df2_n_max_sm80 = [] {
                    const char * e = getenv("LLAMA_DFLASH_N_MAX_SM80");
                    const int32_t n = e != nullptr ? atoi(e) : 4;
                    if (n <= 0) {
                        return 0;
                    }
                    int n_gpu = 0;
                    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
                        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
                        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                            continue;
                        }
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                        auto cc_fn = reg ? (int (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
                        if (cc_fn == nullptr || cc_fn(dev) != 800) {
                            return 0;
                        }
                        ++n_gpu;
                    }
                    return n_gpu > 0 ? n : 0;
                }();
                static const bool df2_n_max_sm80_cached = [] {
                    const char * e = getenv("LLAMA_DFLASH_N_MAX_SM80_CACHED");
                    return e != nullptr && atoi(e) != 0;
                }();
                const bool df2_cap_here = is_dflash2 && df2_n_max_sm80 > 0 &&
                    (df2_n_max_sm80_cached || params.ctx_tgt == nullptr || !llama_moe_cache_active(params.ctx_tgt));
                const int32_t df2_n_max = df2_cap_here ? std::min(params.n_max, df2_n_max_sm80) : params.n_max;
                if (is_dflash2 && df2_n_max < params.n_max) {
                    static bool logged = false;
                    if (!logged) {
                        logged = true;
                        LOG_INF("k19-a80verify: DFlash2 drafts of at most %d steps (%d asked) on a compute capability 8.0 GPU without a MoE cache "
                                "(LLAMA_DFLASH_N_MAX_SM80=0 off, LLAMA_DFLASH_N_MAX_SM80_CACHED=1 beside a cache too)\n", df2_n_max, params.n_max);
                    }
                }
                for (int32_t i = 1; i < (is_dflash2 ? std::min(n_block_tokens, df2_n_max + 1) : params.n_max + 1); ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const int32_t top_k = is_dflash2 ? selector_top_k : lilicorr_top_k;
                    const float * scores = row + top_k + (size_t) predecessor * top_k;

                    predecessor = (int32_t) std::distance(scores,
                            std::max_element(scores, scores + top_k));
                    // k19-sharednoise: the step takes the target's noise at its position over the selector's candidates
                    if (dp.sn != nullptr && dp.sn->on) {
                        sn_cand.resize(top_k);
                        for (int32_t k = 0; k < top_k; ++k) {
                            sn_cand[k] = llama_token_data{ (llama_token) row[k], scores[k], 0.0f };
                        }
                        const llama_token pick = common_shared_noise_draft_pick(*dp.sn, (int32_t) result.size(), result.data(), sn_cand.data(), sn_cand.size(), nullptr);
                        for (int32_t k = 0; k < top_k; ++k) {
                            if ((llama_token) row[k] == pick) {
                                predecessor = k;
                                break;
                            }
                        }
                    }
                    if (trace) {
                        float sum = 0.0f;
                        float second = -INFINITY;
                        for (int32_t k = 0; k < top_k; ++k) {
                            sum += std::exp(scores[k] - scores[predecessor]);
                            if (k != predecessor) {
                                second = std::max(second, scores[k]);
                            }
                        }
                        dg.p[seq_id].push_back(1.0f / sum);
                        dg.m[seq_id].push_back(scores[predecessor] - second);
                    }
                    if (params.p_min > 0.0f || p_cum_thr > 0.0f || use_ev || use_df2w || use_eva) {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                        float sum = 0.0f;
                        for (int32_t k = 0; k < top_k; ++k) {
                            sum += std::exp(scores[k] - scores[predecessor]);
                        }
                        if (use_ev || use_df2w || use_eva) {
                            step_p.push_back(1.0f / sum);
                        }
                        p_cum *= p_cum_exp == 1.0f ? 1.0f / sum : std::pow(1.0f / sum, p_cum_exp);
                        if (1.0f / sum < params.p_min || p_cum < p_cum_thr) {
                            break;
                        }
                    }
                    result.push_back((llama_token) row[predecessor]);
                }
                if (use_df2w && !result.empty()) {
                    static const float df2_cal[20] = { 0.141f, 0.141f, 0.141f, 0.141f, 0.203f, 0.264f, 0.278f, 0.339f, 0.363f, 0.423f,
                                                       0.460f, 0.492f, 0.583f, 0.614f, 0.614f, 0.619f, 0.657f, 0.748f, 0.814f, 0.929f };
                    const bool   lng   = dp.pos0 >= 4096;
                    const size_t w_lo  = df2w_cached_costs ? 3 : lng ? 3 : 4;  // verify rows of the short window
                    const size_t w_hi  = result.size() + 1;                    // the whole draft
                    const double c_lo  = df2w_cached_costs ? (df2_mid ? df2_mid_cost.first  : df2_cached_cost.first)  : lng ? 11.05 : 11.80; // ms per pass at w_lo
                    const double slope = df2w_cached_costs ? (df2_mid ? df2_mid_cost.second : df2_cached_cost.second) : lng ? 1.125 : 0.933; // ms per further row
                    if (w_lo < w_hi) {
                        double e_tok = 1.0, reach = 1.0, e_lo = 0.0;
                        for (size_t k = 1; k < w_hi; ++k) {
                            reach *= df2_cal[std::min(19, std::max(0, (int) (step_p[k - 1] * 20)))];
                            e_tok += reach;
                            if (k + 1 == w_lo) {
                                e_lo = e_tok;
                            }
                        }
                        const double c_hi = c_lo + slope * (double) (w_hi - w_lo);
                        if (e_lo / c_lo > e_tok / c_hi) {
                            result.resize(w_lo - 1);
                        }
                    }
                }
                if (use_ev && !result.empty()) {
                    const auto & f = ev_rule.first;
                    const auto & c = ev_rule.second;
                    const int32_t nb = (int32_t) f.size();
                    size_t best_k = 0;
                    double best = 1.0 / c[0];
                    double e_tok = 1.0;
                    double reach = 1.0;
                    for (size_t k = 1; k <= result.size() && k < c.size(); ++k) {
                        reach *= f[std::min(nb - 1, std::max(0, (int32_t) (step_p[k - 1] * nb)))];
                        e_tok += reach;
                        const double v = e_tok / c[k];
                        if (v > best) {
                            best   = v;
                            best_k = k;
                        }
                    }
                    result.resize(best_k);
                }
                // k19-a100chat: the cut of the measured EV rule. The first LLAMA_DFLASH_EV_AUTO_WARM (8) passes keep the whole draft so
                // the most used width is measured first; at least one draft token stays (an empty draft skips the verify and its timing)
                if (use_eva && !result.empty()) {
                    static const int64_t warm = getenv("LLAMA_DFLASH_EV_AUTO_WARM") ? atoll(getenv("LLAMA_DFLASH_EV_AUTO_WARM")) : 8;
                    static const int64_t probes = getenv("LLAMA_DFLASH_MEAS_PROBES") ? atoll(getenv("LLAMA_DFLASH_MEAS_PROBES")) : 3;
                    static const int64_t stale  = getenv("LLAMA_DFLASH_MEAS_STALE")  ? atoll(getenv("LLAMA_DFLASH_MEAS_STALE"))  : 256;
                    // k19-g4verify: a width of this draft that is measured fewer than `probes` times, else the one whose last measurement
                    // is the oldest beyond `stale` passes, is verified as is (one pass); the widest such first
                    int probe_w = 0;
                    if (df2_meas) {
                        const int w_max = (int) std::min(result.size() + 1, (size_t) eva_t::W - 1);
                        int64_t oldest = -1;
                        for (int w = w_max; w >= 2; --w) {
                            if (eva.cnt[w] < probes) {
                                probe_w = w;
                                break;
                            }
                            const int64_t age = eva.n_meas - eva.last_meas[w];
                            if (stale > 0 && age > stale && age > oldest) {
                                oldest  = age;
                                probe_w = w;
                            }
                        }
                    }
                    if (probe_w > 0) {
                        eva.n_probe++;
                        result.resize((size_t) probe_w - 1);
                    } else if (df2_meas || eva.n_meas >= warm) {
                        const double d_ms = eva.d_n > 0.0 ? eva.d_ms / eva.d_n : 0.0;
                        size_t best_k = result.size();
                        double best   = -1.0;
                        double e_tok  = 1.0;
                        double reach  = 1.0;
                        for (size_t k = 1; k <= result.size() && k + 1 < (size_t) eva_t::W; ++k) {
                            reach *= eva_f(step_p[k - 1]);
                            e_tok += reach;
                            const double v = e_tok / (d_ms + eva_verify_ms((int) k + 1));
                            if (v > best) {
                                best   = v;
                                best_k = k;
                            }
                        }
                        if (best_k < result.size()) {
                            eva.n_cut++;
                            eva.n_rows_cut += (int64_t) (result.size() - best_k);
                            result.resize(best_k);
                        }
                    }
                }

                // k19-handoff (diagnostic, LLAMA_DFLASH_CYCLE=1): the draft is cut to 0, 1, .., n_max tokens in turn, so one run
                // times every verify width at the same contexts (under the exact verify contract the text does not change)
                if (is_dflash2 && dg.cycle) {
                    const size_t keep = (size_t) (dg.n_cycle++ % (uint64_t) (params.n_max + 1));
                    if (result.size() > keep) {
                        result.resize(keep);
                    }
                }
                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                }
                // k19-a100chat: the pass of this draft is timed until its accept (the verify, sampling and rollback) and the draft call
                // itself; the server's own cut (dp.n_max) is the width it verifies
                if (use_eva) {
                    const int64_t now = ggml_time_us();
                    const double  dms = (now - t_eva0) / 1000.0;
                    if (dms > 0.0 && dms < 200.0) {
                        eva.d_ms = 0.97 * eva.d_ms + dms;
                        eva.d_n  = 0.97 * eva.d_n  + 1.0;
                    }
                    size_t n_ver = result.size();
                    if (dp.n_max > 0) {
                        n_ver = std::min(n_ver, (size_t) dp.n_max);
                    }
                    eva.t_end [seq_id] = n_ver > 0 ? now : 0;
                    eva.w_last[seq_id] = (int32_t) n_ver + 1;
                    eva.p_last[seq_id].assign(step_p.begin(), step_p.begin() + std::min(step_p.size(), n_ver));
                    if (n_ver + 1 < (size_t) eva_t::W) {
                        eva.picks[n_ver + 1]++;
                    }
                }
                if (tm) {
                    const int64_t t_d3 = ggml_time_us();
                    dg.t_launch += t_d1 - t_d0;
                    dg.t_wait   += t_d2 - t_d1;
                    dg.t_walk   += t_d3 - t_d2;
                    dg.n_draft  += 1;
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

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        // k19-a100chat: the measured pass cost of the draft's width and the acceptance of its steps (LLAMA_DFLASH_EV_AUTO)
        if (is_dflash2 && eva.init && seq_id >= 0 && seq_id < (llama_seq_id) eva.t_end.size() && eva.t_end[seq_id] > 0) {
            const double ms = (ggml_time_us() - eva.t_end[seq_id]) / 1000.0;
            const int    w  = eva.w_last[seq_id];
            eva.t_end[seq_id] = 0;
            if (w > 0 && w < eva_t::W) {
                // a pass that also re-captured graphs or waited for something else (> 3x its width's mean once measured) is left out
                const double m = eva.v_n[w] >= 3.0 ? eva.v_sum[w] / eva.v_n[w] : 0.0;
                // k19-g4verify: beside a small cache the first pass of a width also builds its graphs: counted as a probe, not timed
                const bool first = eva.small && eva.cnt[w] == 0;
                eva.cnt[w]++;
                if (!first && ms > 0.0 && ms < 2000.0 && (m == 0.0 || ms < 3.0 * m)) {
                    eva.v_sum[w] = 0.97 * eva.v_sum[w] + ms;
                    eva.v_n[w]   = 0.97 * eva.v_n[w]   + 1.0;
                    eva.n_meas++;
                    eva.last_meas[w] = eva.n_meas;
                    if (eva.n_meas % 512 == 0) {
                        eva_log("so far");
                    }
                }
            }
            const auto & ps = eva.p_last[seq_id];
            for (size_t k = 0; k < ps.size() && k <= (size_t) n_accepted; ++k) {
                const int b = std::min(eva_t::NB - 1, std::max(0, (int) (ps[k] * eva_t::NB)));
                eva.f_reach[b] += 1.0;
                if (k < (size_t) n_accepted) {
                    eva.f_acc[b] += 1.0;
                }
            }
        }
        auto & dg = common_dflash_diag();
        if (is_dflash2 && dg.trace && seq_id >= 0 && seq_id < (llama_seq_id) dg.p.size() && !dg.p[seq_id].empty()) {
            std::string sp, sm;
            char b[32];
            for (size_t i = 0; i < dg.p[seq_id].size(); ++i) {
                snprintf(b, sizeof(b), "%s%.4f", i ? "," : "", dg.p[seq_id][i]); sp += b;
                snprintf(b, sizeof(b), "%s%.3f", i ? "," : "", dg.m[seq_id][i]); sm += b;
            }
            LOG_INF("df2 trace: pos %d acc %d p %s m %s\n", (int) dg.pos0[seq_id], (int) n_accepted, sp.c_str(), sm.c_str());
            dg.p[seq_id].clear();
        }
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

    // k19-host (LLAMA_MTP_PROMPT_BG): see common_speculative_rows_ext_set
    const float *      h_ext = nullptr; // the target's next-n rows, row i at h_ext + i*n_embd
    int32_t            h_ext_lo = 0;    // rows [0, h_ext_lo) are read from h_ext_stash instead (the next decode may overwrite them)
    std::vector<float> h_ext_stash;     // kept between prompts (at most 256 rows)
    int64_t rows_ext_floats() const override { return rows_early_ok() && !chain_heads ? n_embd : 0; }
    bool rows_ext_set(const float * h, int32_t n_rows, int32_t n_lo) override {
        if (!rows_ext_floats() || h == nullptr || n_rows <= 0) {
            return false;
        }
        n_lo = std::max<int32_t>(0, std::min(n_lo, n_rows));
        if (h_ext_stash.size() < (size_t) n_lo * n_embd) {
            h_ext_stash.resize((size_t) n_lo * n_embd);
        }
        if (n_lo > 0) {
            std::memcpy(h_ext_stash.data(), h, (size_t) n_lo * n_embd * sizeof(float));
        }
        h_ext    = h;
        h_ext_lo = n_lo;
        return true;
    }
    void rows_ext_clear() override {
        h_ext    = nullptr;
        h_ext_lo = 0;
    }
    llama_seq_id dev_seq = -1;            // the sequence whose pending row is in sink row 0
    std::vector<uint8_t> pending_dev;     // per seq: its pending row is sink row 0 (pending_h is stale)
    std::vector<uint8_t> verify_dev;      // per seq: the verify rows are sink rows 1 .. verify_h_rows (verify_h is unused)

    // k15-fn: LLAMA_SPEC_COST=1 (opt-in): the cost-aware MTP window (common_spec_mtp_cost, Strata's STRATA_SPEC_COST): the drafts
    // kept per round are the ones with the best expected tokens per ms, from the drafts' probabilities and measured round times
    const bool spec_cost = [] {
        const char * e = getenv("LLAMA_SPEC_COST");
        return e != nullptr && atoi(e) != 0;
    }();
    // k19-tpp: LLAMA_SPEC_TPP (the server sets params.n_floor): the window keeps n_floor drafts and one more only where the expected
    // tokens per ms of the wider window (calibrated draft acceptance, common_spec_mtp_cost::chain_e) beat the floor window's at the
    // measured round times (common_spec_tpp_cost), so the verify sizes are n_floor + 1 and n_floor + 2 only
    int32_t tpp_floor = 0;
    bool    cost_on   = false; // spec_cost || tpp_floor > 0: the round times and draft probabilities are recorded
    common_spec_mtp_cost cost_pol;
    common_spec_tpp_cost tpp_cost;
    // k19-adraft (params.adraft > 0: the greedy window's drafts). Sampled requests, LLAMA_SPEC_ADRAFT=1: confidence gate, the 2nd draft
    // is kept when its probability is at least adraft_p2 (LLAMA_SPEC_ADRAFT_P2, default 0: always), every later one when at least adraft_p
    // (LLAMA_SPEC_ADRAFT_P, default 0.9), the window ending at the first draft below its bar (a function of the text: no timing).
    // =2: priced by cost_pol (expected tokens per ms, live round times; LLAMA_SPEC_ADRAFT_FLOOR: fewest drafts)
    int32_t adraft_keep  = 0;
    int32_t adraft_mode  = 0;
    int32_t adraft_floor = 0;
    float   adraft_p     = 0.9f;
    float   adraft_p2    = 0.0f;
    bool    adraft_prev  = false;   // LLAMA_SPEC_ADRAFT_PREV=1: the bar is checked on the last kept draft, so a draft is made only when kept
    std::array<int64_t, 9> adraft_hist{};   // sampled windows by kept drafts
    // =3: one width per stretch of the request: a sampled request drafts n_max while the running share of its drafts that held (EMA,
    // weight 1/LLAMA_SPEC_ADRAFT_EMA_N, 16) is at least LLAMA_SPEC_ADRAFT_ON (0.75), adraft_keep once it falls below LLAMA_SPEC_ADRAFT_OFF
    // (0.65), switching at most every LLAMA_SPEC_ADRAFT_DWELL (16) rounds (each width switch costs the cached tier a graph change);
    // =4: n_max below position LLAMA_SPEC_ADRAFT_POS (1536), adraft_keep from there. Both functions of the text
    double  adraft_on  = 0.75, adraft_off = 0.65, adraft_ema_n = 16.0;
    int32_t adraft_dwell = 16;
    int64_t adraft_pos   = 1536;
    struct adraft_seq {
        double ema    = 0.0;
        int    rounds = 0;      // rounds since the request began
        int    since  = 0;      // rounds since the last width switch
        bool   deep   = false;
        int    drafted = 0;     // drafts of the window in flight
    };
    std::vector<adraft_seq> adraft_st;
    int64_t adraft_switches = 0;
    static bool sampled(const common_speculative_draft_params & dp) {
        return dp.temp > 0.0f;
    }
    bool priced(const common_speculative_draft_params & dp) const {
        return !dp.result_q && (spec_cost || (adraft_mode == 2 && adraft_keep > 0 && sampled(dp)));
    }
    bool gated(const common_speculative_draft_params & dp) const {
        return !dp.result_q && adraft_mode == 1 && adraft_keep > 0 && sampled(dp);
    }
    // the window cap of a request that is neither gated nor priced (k19-adraft modes 3 / 4: per-stretch width; else adraft_keep)
    int adraft_cap(llama_seq_id seq_id, const common_speculative_draft_params & dp) const {
        if (dp.result_q || !sampled(dp) || adraft_keep <= 0) {
            return adraft_keep;
        }
        if (adraft_mode == 3) {
            return adraft_st[seq_id].deep ? params.n_max : adraft_keep;
        }
        if (adraft_mode == 4) {
            return dp.pos0 < adraft_pos ? params.n_max : adraft_keep;
        }
        return adraft_keep;
    }
    struct cost_seq {
        int64_t t_us = 0;            // the previous draft() call (0: none since the request began)
        int     rows = 1;            // the window drafted then
        int     drafted = 0;         // k19-tpp: the drafts made then (more than rows - 1: a wider window was drafted and not kept)
        bool    cal = true;          // k19-adraft: its drafts calibrate the hold rates (sampled rounds only)
        bool    shared = false;      // k19-small1115: the round drafted then ran beside another active slot (dp.shared)
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
        // k19-tpp: one trained head without shared memory only (chained heads and gemma4's shared KV keep the floor as their length)
        if (this->params.n_floor > 0) {
            // (k19-depth: n_floor + 2 and more, the deep windows below LLAMA_SPEC_DEPTH_POS)
            if (!chain_heads && !is_mem_shared && !spec_cost && this->params.n_max >= this->params.n_floor + 1) {
                tpp_floor = this->params.n_floor;
                tpp_cost.deep_extra = this->params.n_max - this->params.n_floor - 1;
            } else {
                this->params.n_max = std::min(this->params.n_max, this->params.n_floor);
            }
        }
        // k19-adraft: one trained head without shared memory, no other window policy
        if (this->params.adraft > 0) {
            if (!chain_heads && !is_mem_shared && !spec_cost && tpp_floor == 0 && this->params.n_max > this->params.adraft) {
                const char * m = getenv("LLAMA_SPEC_ADRAFT");
                adraft_keep  = this->params.adraft;
                adraft_mode  = m != nullptr && atoi(m) >= 2 && atoi(m) <= 4 ? atoi(m) : 1;
                auto envf = [](const char * n, double & v) { if (const char * e = getenv(n)) { v = atof(e); } };
                envf("LLAMA_SPEC_ADRAFT_ON", adraft_on);
                envf("LLAMA_SPEC_ADRAFT_OFF", adraft_off);
                envf("LLAMA_SPEC_ADRAFT_EMA_N", adraft_ema_n);
                adraft_ema_n = std::max(1.0, adraft_ema_n);
                if (const char * e = getenv("LLAMA_SPEC_ADRAFT_DWELL")) {
                    adraft_dwell = std::max(1, atoi(e));
                }
                if (const char * e = getenv("LLAMA_SPEC_ADRAFT_POS")) {
                    adraft_pos = atoll(e);
                }
                adraft_floor = [] {
                    const char * f = getenv("LLAMA_SPEC_ADRAFT_FLOOR");
                    return f != nullptr ? std::max(0, atoi(f)) : 0;
                }();
                adraft_floor = std::min(adraft_floor, this->params.n_max);
                if (const char * e = getenv("LLAMA_SPEC_ADRAFT_P")) {
                    adraft_p = (float) atof(e);
                }
                if (const char * e = getenv("LLAMA_SPEC_ADRAFT_P2")) {
                    adraft_p2 = (float) atof(e);
                }
                if (const char * e = getenv("LLAMA_SPEC_ADRAFT_PREV")) {
                    adraft_prev = atoi(e) != 0;
                }
                if (adraft_mode == 3) {
                    LOG_INF("%s: k19-adraft: sampled requests: %d drafts while the running hold share is >= %.2f, %d below %.2f, at most one "
                        "switch per %d rounds; greedy requests: %d drafts (LLAMA_SPEC_ADRAFT=0: off)\n", __func__, this->params.n_max, adraft_on,
                        adraft_keep, adraft_off, adraft_dwell, adraft_keep);
                } else if (adraft_mode == 4) {
                    LOG_INF("%s: k19-adraft: sampled requests: %d drafts below position %lld, %d from there; greedy requests: %d drafts "
                        "(LLAMA_SPEC_ADRAFT=0: off)\n", __func__, this->params.n_max, (long long) adraft_pos, adraft_keep, adraft_keep);
                } else if (adraft_mode == 1) {
                    LOG_INF("%s: k19-adraft: sampled requests: MTP windows of 1..%d drafts, the 2nd kept at p >= %.2f, later ones at p >= %.2f, "
                        "bar on %s; greedy requests: %d drafts (LLAMA_SPEC_ADRAFT=0: off)\n", __func__, this->params.n_max, adraft_p2, adraft_p,
                        adraft_prev ? "the draft before (LLAMA_SPEC_ADRAFT_PREV)" : "the draft itself", adraft_keep);
                } else {
                    LOG_INF("%s: k19-adraft: sampled requests: MTP windows by expected tokens per ms, %d..%d drafts; greedy requests: %d drafts "
                        "(LLAMA_SPEC_ADRAFT=2; =0: off)\n", __func__, adraft_floor, this->params.n_max, adraft_keep);
                }
            } else {
                this->params.n_max = std::min(this->params.n_max, this->params.adraft);
            }
        }
        cost_on = spec_cost || tpp_floor > 0 || adraft_mode == 2;
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
        adraft_st.assign(n_seq, {});
        if (spec_cost) {
            LOG_INF("%s: MTP window by expected tokens per ms (LLAMA_SPEC_COST=1), up to %d drafts\n", __func__, this->params.n_max);
        }
        if (tpp_floor > 0) {
            // (LLAMA_SPEC_TPP_FORCE_CACHED=1, a test hook: as if an MoE cache ran)
            const char * fe = getenv("LLAMA_SPEC_TPP_FORCE_CACHED");
            const bool cached = (this->params.ctx_tgt != nullptr && llama_moe_cache_active(this->params.ctx_tgt)) || (fe != nullptr && atoi(fe) != 0);
            const char * ce = getenv("LLAMA_SPEC_TPP_CACHED");
            // (k19-fnmtp: the server's rule for a cache, params.deep_cached, keeps the windows)
            if (cached && !(ce != nullptr && atoi(ce) != 0) && !this->params.deep_cached) {
                // an MoE cache: the policy is off and the window is the floor's, as with LLAMA_SPEC_TPP=0 (a wider window reads more
                // streamed experts and moves which ones stay resident; G4 fn4 16 GB, ktp_g4e: 0.94-0.96 of base at 16k / 30k)
                tpp_floor = 0;
                tpp_cost.deep_extra = 0;
                this->params.n_max = this->params.n_floor;
                this->n_max = this->params.n_max;
                cost_on = spec_cost;
                LOG_INF("%s: k19-tpp: off: 3-draft windows are off because the MoE cache is on; MTP windows of %d drafts, as LLAMA_SPEC_TPP=0 "
                    "(LLAMA_SPEC_TPP_CACHED=1: on)\n", __func__, this->params.n_max);
            }
        }
        if (tpp_floor > 0) {
            const bool cached = this->params.ctx_tgt != nullptr && llama_moe_cache_active(this->params.ctx_tgt);
            tpp_cost.set_cached(cached);
            if (this->params.deep_cached) {
                // k19-fnmtp: every window below deep_pos drafts n_max, also where that is floor + 1 (3 drafts)
                tpp_cost.deep_all = true;
                LOG_INF("%s: k19-fnmtp: MoE cache %s: MTP windows of %d drafts below position %lld, from there %d and a priced %d "
                    "(LLAMA_SPEC_DEPTH_CACHED=0: off)\n", __func__, cached ? "on" : "off (expected by the server)", this->params.n_max,
                    (long long) tpp_cost.deep_pos, tpp_floor, tpp_floor + 1);
            }
            // k19-q38d: a dense target (no <arch>.expert_count) keeps the deep windows at every position. Its verify reads each weight
            // once for all the window's rows, so the deeper window's extra rows cost little at long context too: G4 Qwen3.8-27B
            // UD-Q4_K_XL + its MTP head, full card, 30k prompt: 4 drafts past position 12288 151.1 tok/s (3.12 tokens, 20.7 ms per pass)
            // against 134.5 with 2 (2.44, 18.1 ms); chat / 8k unchanged (deep band already). MoE targets keep 12288 (fn4 full card: 4
            // drafts -13% at 30k, plans/k19_depth.md). LLAMA_SPEC_DEPTH_POS set: that position; LLAMA_SPEC_DEPTH_DENSE=0: 12288 as before.
            // Not on the HBM data-centre parts (compute capability 8.0 / 9.0: A100, A30, H100, H200), whose bandwidth per dp4a op makes the
            // extra verify rows dear: A100 80 GB, the same model and prompt, 30k 86.6 tok/s with every position deep against 90.1 at 12288
            // (-3.9%; chat / 8k equal). L4 (8.9) +10.6% and G4 (12.0) +12.5% at 30k. LLAMA_SPEC_DEPTH_DENSE=1 forces it on any GPU
            const char * e_dd = getenv("LLAMA_SPEC_DEPTH_DENSE");
            const bool hbm_dc = [] {
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                        continue;
                    }
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    auto cc_fn = reg ? (int (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
                    const int cc = cc_fn != nullptr ? cc_fn(dev) : -1;
                    if (cc == 800 || cc == 900) {
                        return true;
                    }
                }
                return false;
            }();
            if (tpp_cost.deep_extra > 0 && getenv("LLAMA_SPEC_DEPTH_POS") == nullptr && this->params.ctx_tgt != nullptr &&
                    (e_dd != nullptr ? atoi(e_dd) != 0 : !hbm_dc)) {
                const llama_model * mt = llama_get_model(this->params.ctx_tgt);
                char arch[128] = {0};
                char nexp[32]  = {0};
                const bool dense = mt != nullptr && llama_model_meta_val_str(mt, "general.architecture", arch, sizeof(arch)) > 0 &&
                    (llama_model_meta_val_str(mt, (std::string(arch) + ".expert_count").c_str(), nexp, sizeof(nexp)) < 0 || atoi(nexp) == 0);
                if (dense) {
                    tpp_cost.deep_pos = INT64_MAX;
                    LOG_INF("%s: k19-q38d: dense target: the deep MTP windows at every position (LLAMA_SPEC_DEPTH_DENSE=0: below 12288)\n", __func__);
                }
            } else if (tpp_cost.deep_extra > 0 && hbm_dc && e_dd == nullptr && getenv("LLAMA_SPEC_DEPTH_POS") == nullptr) {
                LOG_INF("%s: k19-q38d: an A100 / H100-class GPU (compute capability 8.0 / 9.0): the deep MTP windows stay below 12288 (LLAMA_SPEC_DEPTH_DENSE=1: every position for a dense target)\n", __func__);
            }
            if (tpp_cost.ctx_on) {
                LOG_INF("%s: k19-tppctx: windows fixed per context bucket: drafts past %d per bucket <1536 / <6144 / <12288 / <24576 / longer "
                    "%d / %d / %d / %d / %d (-1 priced; LLAMA_SPEC_TPP_CTX=0: off)\n", __func__, tpp_floor, tpp_cost.ctx_extra[0],
                    tpp_cost.ctx_extra[1], tpp_cost.ctx_extra[2], tpp_cost.ctx_extra[3], tpp_cost.ctx_extra[4]);
            }
            if (tpp_cost.deep_extra > 0) {
                LOG_INF("%s: k19-depth: MTP windows of %d drafts below position %lld (LLAMA_SPEC_DEPTH_POS), the windows below from there "
                    "(LLAMA_SPEC_DEPTH=0: off)\n", __func__, this->params.n_max, (long long) tpp_cost.deep_pos);
            }
            LOG_INF("%s: k19-tpp: MTP windows of %d drafts, %d where the expected tokens per ms pay for the wider window (LLAMA_SPEC_TPP=0: "
                "off); round times %s (cost ratio %.2f at 16k, MoE cache %s; LLAMA_SPEC_TPP_LIVE=1: live medians)\n", __func__, tpp_floor,
                tpp_floor + 1, tpp_cost.live ? "live" : "from the fixed table", tpp_cost.cost_ratio[3], cached ? "on" : "off");
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
        if (adraft_mode == 3 && seq_id >= 0 && seq_id < (llama_seq_id) adraft_st.size()) {
            adraft_st[seq_id] = {};
        }
        if (adraft_mode == 1 || adraft_mode >= 3) {
            int64_t n = 0;
            std::string h;
            for (size_t k = 0; k <= (size_t) params.n_max && k < adraft_hist.size(); ++k) {
                n += adraft_hist[k];
                h += string_format(" %zu:%lld", k, (long long) adraft_hist[k]);
            }
            if (n > 0) {
                LOG_INF("%s: k19-adraft: sampled windows by drafts kept (so far):%s, width switches %lld\n", __func__, h.c_str(),
                    (long long) adraft_switches);
            }
        }
        if (cost_on && seq_id >= 0 && seq_id < (llama_seq_id) cost_st.size()) {
            if (cost_st[seq_id].t_us > 0) {
                if (tpp_floor > 0) {
                    const int n_deep = tpp_floor + 1 + tpp_cost.deep_extra;
                    LOG_INF("%s: MTP window cost table (k19-tpp): %s | windows: %d drafts %.0f, %d drafts %.0f%s\n", __func__, tpp_cost.text().c_str(),
                        tpp_floor, cost_pol.n_rounds[tpp_floor + 1], tpp_floor + 1, cost_pol.n_rounds[tpp_floor + 2],
                        tpp_cost.deep_extra > 0 && n_deep + 1 <= common_spec_mtp_cost::MAX_T ?
                            string_format(", %d drafts %.0f (k19-depth)", n_deep, cost_pol.n_rounds[n_deep + 1]).c_str() : "");
                } else {
                    LOG_INF("%s: MTP window cost table: %s\n", __func__, cost_pol.text().c_str());
                }
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

    // kdp: the drafter keeps no cells of a skipped batch's sequences (the k18 window's drop: it attends to the rows it gets later,
    // RoPE positions as in the target) and pairs the next batch's first row with the target's row of this batch's last token. Drafts
    // are verified: only the acceptance moves. Host rows only, as the window
    bool skip_prompt(const common_batch & batch_in) override {
        if (dev_h || is_mem_shared || chain_heads || h_ext != nullptr || batch_in.size() <= 0 || (!batch_in.has_token() && !batch_in.has_embd())) {
            return process(batch_in);
        }
        const int32_t n_tokens = batch_in.size();
        std::vector<int32_t> i_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            const llama_seq_id s = batch_in.tokens[k].seq_id;
            if (s < 0 || s >= (llama_seq_id) n_seq || !batch_in.tokens[k].seq_ids_extra.empty()) {
                return process(batch_in);
            }
            i_end[s] = k;
        }
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * mem_dft = llama_get_memory(this->params.ctx_dft);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (i_end[s] < 0) {
                continue;
            }
            const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, nextn_row_offset + i_end[s]);
            if (h == nullptr) {
                return process(batch_in);
            }
            llama_memory_seq_rm(mem_dft, s, -1, -1);
            std::memcpy(pending_h[s].data(), h, row_bytes);
            if (s < (llama_seq_id) pending_dev.size()) {
                pending_dev[s] = 0;
            }
        }
        n_skipped_rows += n_tokens;
        return true;
    }
    int64_t n_skipped_rows = 0;

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
        // k19-host: with h_ext (LLAMA_MTP_PROMPT_BG) the rows come from the caller's pointer and stash, no target call at all
        const bool ext = h_ext != nullptr;
        const float * h_early = !ext && rows_early ? llama_get_embeddings_nextn_early(ctx_tgt) : nullptr;
        GGML_ASSERT(ext || !rows_early || h_early);
        auto tgt_row = [&](int32_t i) -> const float * {
            if (ext) {
                return i < h_ext_lo ? h_ext_stash.data() + (size_t) i * n_embd : h_ext + (size_t) i * n_embd;
            }
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
            const float * h_tgt = dev || ext ? nullptr : (h_early ? h_early : llama_get_embeddings_nextn(ctx_tgt)) + (size_t) nextn_row_offset * n_embd;
            if (t_tr0) {
                t_tr_sync = ggml_time_us();
            }

            // k18 (LLAMA_MTP_PROMPT_WINDOW=W, opt-in; 0 = every row): a batch of more than W rows of one sequence gives the drafter only
            // its last W rows; the draft's cells of that sequence are dropped first (it then attends to the last W prompt positions,
            // RoPE positions as in the target). Drafts are verified: only the acceptance moves. Host rows only (not the device sink,
            // shared KV or chained heads)
            static const int32_t pw_env = getenv("LLAMA_MTP_PROMPT_WINDOW") ? std::max(0, atoi(getenv("LLAMA_MTP_PROMPT_WINDOW"))) : 0;
            const int32_t pw = (!dev && !is_mem_shared && !chain_heads) ? pw_env : 0;
            std::vector<int32_t> k_first(n_seq, 0);
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (i_batch_beg[s] < 0) {
                    continue;
                }
                const int32_t n_rows_s = i_batch_end[s] - i_batch_beg[s] + 1;
                k_first[s] = pw > 0 && n_rows_s > pw ? i_batch_end[s] - pw + 1 : i_batch_beg[s];
            }

            for (int k = 0; k < n_tokens; ++k) {
                const auto & t = batch_in.tokens[k];

                const llama_seq_id seq_id = t.seq_id;
                if (k < k_first[seq_id]) {
                    continue; // outside the window
                }

                // vision tokens carry an embedding instead of an id
                const int32_t idx = t.id != LLAMA_TOKEN_NULL
                    ? batch.add(t.id, t.pos[0], seq_id, false)
                    : batch.add_embd(t.embd, t.pos.data(), seq_id, false);

                if (dev) {
                    continue; // row k of the sink
                }

                const float * h_row = k == i_batch_beg[seq_id]
                    ? pending_h[seq_id].data()
                    : ext ? tgt_row(nextn_row_offset + k - 1) : h_tgt + (size_t) (k - 1) * n_embd;

                batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd });
            }

            auto * mem_dft = llama_get_memory(ctx_dft);
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (i_batch_beg[s] >= 0 && k_first[s] > i_batch_beg[s]) {
                    llama_memory_seq_rm(mem_dft, s, -1, -1);
                }
            }

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
            if (ext && handoff_copy) {
                // k19-host: the copy below row by row (the stash breaks the rows' contiguity): the same n_keep rows and pending row
                const int32_t n_keep = verify_h_trim ? std::min<int32_t>(n_rows, std::max<int32_t>(256, n_max + 2)) : n_rows;
                verify_h_kept[seq_id] = n_keep;
                if (verify_h[seq_id].size() < (size_t) n_keep * n_embd) {
                    verify_h[seq_id].resize((size_t) n_keep * n_embd);
                }
                for (int32_t i = 0; i < n_keep; ++i) {
                    std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, tgt_row(nextn_row_offset + i_batch_beg[seq_id] + i), row_bytes);
                }
                std::memcpy(pending_h[seq_id].data(), h_end, row_bytes);
                continue;
            }
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
            // k19-sharednoise: so does a draft drawn with the target's noise (sample-and-match is what keeps it exact)
            if (!params.probabilistic || (dp.sn != nullptr && dp.sn->on)) {
                dp.result_q = nullptr;
            }

            if (cost_on) {
                // the time since the previous draft() call is one round (verify + sampling + this drafting) of the window drafted then
                auto & cs = cost_st[seq_id];
                const int64_t now = ggml_time_us();
                if (tpp_floor > 0) {
                    tpp_cost.set_context(dp.pos0);
                    if (cs.t_us > 0) {
                        cost_pol.n_rounds[std::clamp(cs.rows, 1, common_spec_mtp_cost::MAX_T)] += 1.0;
                        // a round that drafted a wider window and kept the floor carries a draft step the floor window does not have
                        if (cs.rows - 1 >= tpp_floor && cs.rows - 1 <= tpp_floor + 1 && cs.drafted == cs.rows - 1 && !cs.shared) {
                            tpp_cost.observe(cs.rows - 1 - tpp_floor, (now - cs.t_us)/1000.0);
                        }
                    }
                } else {
                    cost_pol.set_context(dp.pos0);
                    if (cs.t_us > 0) {
                        cost_pol.observe(cs.rows, (now - cs.t_us)/1000.0);
                    }
                }
                cs.t_us   = now;
                cs.rows   = 1;
                cs.chain  = {};
                cs.shared = dp.shared;
                cs.cal   = adraft_keep == 0 || sampled(dp);
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
                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                // k19-sharednoise: the draft takes the target's noise at its position (the probability fed to the window policies
                // is the picked token's under the mirrored chain; they calibrate it against what held)
                float p_draft = cur_p->data[0].p;
                const bool sn_on = dp.sn != nullptr && dp.sn->on;
                const llama_token id = sn_on ? common_shared_noise_draft_pick(*dp.sn, (int32_t) result.size(), result.data(), cur_p->data, cur_p->size, &p_draft) :
                                       dp.result_q ? id_sampled : cur_p->data[0].id;
                GGML_UNUSED(p_draft);

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                // k19-adraft: the confidence gate (the draft below its bar is not proposed)
                if (gated(dp) && !adraft_prev && !result.empty() && p_draft < (result.size() == 1 ? adraft_p2 : adraft_p)) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                result.push_back(id);

                if (FILE * lf = common_spec_acc_log()) {
                    // pos, picked id, its draft probability, the draft's top id / p, 2nd p
                    fprintf(lf, "D %d %d %.5f %d %.5f %.5f\n", (int) result.size() - 1, id, p_draft, cur_p->data[0].id, cur_p->data[0].p,
                            cur_p->size > 1 ? cur_p->data[1].p : 0.0f);
                }

                if (dp.result_q) {
                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                }

                if (tpp_floor > 0 && !dp.result_q) {
                    auto & cs = cost_st[seq_id];
                    cs.probs.push_back(p_draft);
                    // after the floor's drafts: one more where the wider window, its next draft as likely to hold as the last, pays
                    if ((int) result.size() == tpp_floor) {
                        bool more = !tpp_cost.need_floor() && (dp.n_max <= 0 || dp.n_max > tpp_floor);
                        if (more) {
                            double q = 1.0;
                            const double e_f = cost_pol.chain_e(cs.probs.data(), tpp_floor, q);
                            const double a   = cost_pol.draft_rate(cs.probs.back());
                            more = tpp_cost.wide_pays(e_f, e_f + q*a, q*a);
                        }
                        if (!more) {
                            drafting[seq_id] = false;
                            n_drafting--;
                            continue;
                        }
                    } else if ((int) result.size() == tpp_floor + 1 && !tpp_cost.deep()) {
                        // k19-depth: past the deep band a window ends at floor + 1 drafts, as with n_max floor + 1
                        drafting[seq_id] = false;
                        n_drafting--;
                        continue;
                    }
                } else if (priced(dp)) {
                    auto & cs = cost_st[seq_id];
                    cs.probs.push_back(p_draft);
                    const int cap = 1 + (dp.n_max > 0 ? std::min(params.n_max, dp.n_max) : params.n_max);
                    const bool more = cost_pol.keep_draft(cs.chain, (int) result.size() - 1, p_draft, cap);
                    if (!more && (int) result.size() >= adraft_floor) {
                        drafting[seq_id] = false;
                        n_drafting--;
                        continue;
                    }
                } else if (gated(dp) && adraft_prev && p_draft < (result.size() == 1 ? adraft_p2 : adraft_p)) {
                    // k19-adraft (LLAMA_SPEC_ADRAFT_PREV=1): the next draft is made only when this one clears the next one's bar
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                } else if (adraft_keep > 0 && !gated(dp) && (int) result.size() >= adraft_cap(seq_id, dp)) {
                    // k19-adraft: a greedy request's window, as with n_max = adraft_keep (modes 3 / 4: the stretch's width)
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
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

            if (tpp_floor > 0 && !dp.result_q) {
                // k19-tpp: the floor's drafts, and the one beyond it where the wider window pays with the drafted token's probability
                auto & cs = cost_st[seq_id];
                cs.drafted = (int) dp.result->size();
                if ((int) dp.result->size() > tpp_floor && (int) cs.probs.size() > tpp_floor) {
                    double q_f = 1.0, q_w = 1.0;
                    const double e_f = cost_pol.chain_e(cs.probs.data(), tpp_floor, q_f);
                    const double e_w = cost_pol.chain_e(cs.probs.data(), tpp_floor + 1, q_w);
                    if (!tpp_cost.wide_pays(e_f, e_w, q_w)) {
                        dp.result->resize(tpp_floor);
                    }
                }
            } else if (priced(dp)) {
                // the window keeps the drafts of the best expected tokens per ms (k19-adraft: at least adraft_floor)
                const size_t kept = (size_t) std::max(adraft_floor, cost_st[seq_id].chain.best_rows - 1);
                if (dp.result->size() > kept) {
                    dp.result->resize(kept);
                }
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            if (gated(dp) || (adraft_mode >= 3 && adraft_keep > 0 && !dp.result_q && sampled(dp))) {
                adraft_hist[std::min<size_t>(dp.result->size(), adraft_hist.size() - 1)]++;
            }
            if (adraft_mode == 3 && seq_id < (llama_seq_id) adraft_st.size()) {
                adraft_st[seq_id].drafted = sampled(dp) && !dp.result_q ? (int) dp.result->size() : 0;
            }

            if (cost_on) {
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

        if (adraft_mode == 3 && seq_id < (llama_seq_id) adraft_st.size() && adraft_st[seq_id].drafted > 0) {
            // k19-adraft: the share of this window's drafts that held, into the request's running share; a width switch past the dwell
            auto & a = adraft_st[seq_id];
            const double held = std::min(1.0, (double) n_accepted / a.drafted);
            a.ema = a.rounds == 0 ? held : a.ema + (held - a.ema)/adraft_ema_n;
            a.rounds++;
            a.since++;
            a.drafted = 0;
            if (a.rounds >= adraft_dwell && a.since >= adraft_dwell && (a.deep ? a.ema < adraft_off : a.ema >= adraft_on)) {
                a.deep  = !a.deep;
                a.since = 0;
                adraft_switches++;
            }
        }

        if (cost_on) {
            auto & cs = cost_st[seq_id];
            // k19-tpp: a kept draft past the floor, reached (the floor's drafts held): did it hold, against its predicted rate
            if (tpp_floor > 0 && (int) cs.probs.size() == tpp_floor + 1 && n_accepted >= tpp_floor) {
                tpp_cost.observe_marginal(cost_pol.draft_rate(cs.probs[tpp_floor]), n_accepted > tpp_floor);
            }
            if (!cs.probs.empty()) {
                if (cs.cal) {
                    cost_pol.observe_probs(cs.probs.data(), (int) cs.probs.size(), n_accepted);
                }
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
        int64_t n_win[common_spec_cost_policy::MAX_T + 1] = {};   // k19-lookup8: drafted rounds by window (rows), this request
        std::string win_text() const {
            std::string o;
            for (int t = 2; t <= common_spec_cost_policy::MAX_T; ++t) {
                if (n_win[t] > 0) {
                    o += " T" + std::to_string(t) + "=" + std::to_string(n_win[t]);
                }
            }
            return o;
        }
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
            // k19-lookup8: LLAMA_SPEC_SUFFIX_SIZES="2,3,4,8" prices every listed window size (rows), see common_spec_cost_policy::choose_sizes
            policy.sizes = common_spec_cost_policy::parse_sizes(getenv("LLAMA_SPEC_SUFFIX_SIZES"));
        }
        if (policy_on && policy.sizes != 0) {
            std::string ls;
            for (int t = 2; t <= common_spec_cost_policy::MAX_T; ++t) {
                if (policy.allowed(t)) {
                    ls += (ls.empty() ? "" : ",") + std::to_string(t);
                }
            }
            SPC_INF("ngram-suffix: k19-lookup8 window sizes %s (rows), each priced by its measured round time (LLAMA_SPEC_SUFFIX_SIZES)\n", ls.c_str());
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
            if (policy.sizes != 0) {
                SPC_INF("ngram-suffix seq %d: k19-lookup8 round ms by window:%s; drafts by window:%s\n", seq_id, policy.text().c_str(), si.win_text().c_str());
            }
        }
        si.n_lookups = si.n_matches = si.n_drafts = si.n_drafted = si.n_accepted = si.n_yield = 0;
        si.n_probe = 0;
        std::fill(si.n_win, si.n_win + common_spec_cost_policy::MAX_T + 1, 0);
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
                k = policy.sizes != 0 ? policy.choose_sizes(offered, q) :
                    policy.choose(k_full, std::min(k_full, si.n_short), q, si.n_short != params.n_max_short);
                if (k > 0) {
                    si.n_win[std::min(k + 1, common_spec_cost_policy::MAX_T)]++;
                }
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

FILE * common_spec_acc_log() {
    static FILE * f = [] () -> FILE * {
        const char * p = getenv("LLAMA_SPEC_ACC_LOG");
        if (p == nullptr || *p == 0) {
            return nullptr;
        }
        FILE * h = fopen(p, "a");
        if (h != nullptr) {
            setvbuf(h, nullptr, _IOLBF, 1 << 16);
        }
        return h;
    }();
    return f;
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

// k16-draft: the shape (and type) of tensor `name` in a GGUF (split-aware: a target's lm head is usually in a later shard): 1 found,
// 0 absent from every readable shard, -1 a shard could not be read
static int common_gguf_tensor_find(const std::string & path, const char * name, int64_t ne[GGML_MAX_DIMS], ggml_type * type = nullptr) {
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
            return -1;
        }
        ggml_context_ptr meta_ptr(meta);
        if (const ggml_tensor * t = ggml_get_tensor(meta, name)) {
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                ne[d] = t->ne[d];
            }
            if (type != nullptr) {
                *type = t->type;
            }
            return 1;
        }
    }
    return 0;
}

// k16-draft (idea from ik_llama.cpp by ikawrakow): a separate MTP draft model (-md with --spec-type draft-mtp) uses the target's
// token_embd and output when both files hold them with the same shapes (llama_model_loader::borrow_io checks again at load, against
// the loaded target). Decided here so the fit's measurements of the draft (no target loaded yet) count the lm head as the target's.
// LLAMA_MTP_BORROW_IO=0: the draft loads its own copies
// k19-pr374 (ported from LeoBorcherding's Qwen4Exp shared MTP heads, unslothai/llama.cpp, MIT): a shared drafter (mtp-*-shared-*.gguf)
// has neither tensor (a drafter missing only one keeps the old rule: no borrow); it borrows both, and `missing` gets the target's type of each one it lacks (the measurement prices them in it).
// LLAMA_MTP_BORROW_MISSING=0: a drafter without them is not borrowed for (its load fails on the missing tensor, as before)
static bool common_mtp_borrow_io_ok(const std::string & path_tgt, const std::string & path_dft,
        std::vector<std::pair<std::string, ggml_type>> * missing = nullptr) {
    const char * e = getenv("LLAMA_MTP_BORROW_IO");
    if ((e != nullptr && atoi(e) == 0) || path_tgt.empty() || path_dft.empty() || path_tgt == path_dft) {
        return false;
    }
    const char * em = getenv("LLAMA_MTP_BORROW_MISSING");
    const bool allow_missing = em == nullptr || atoi(em) != 0;
    std::vector<std::pair<std::string, ggml_type>> miss;
    for (const char * name : { "token_embd.weight", "output.weight" }) {
        int64_t a[GGML_MAX_DIMS] = {}, b[GGML_MAX_DIMS] = {};
        ggml_type ta = GGML_TYPE_COUNT;
        if (common_gguf_tensor_find(path_tgt, name, a, &ta) != 1) {
            return false;
        }
        const int fd = common_gguf_tensor_find(path_dft, name, b);
        if (fd == 0 && allow_missing) {
            miss.emplace_back(name, ta);
            continue;
        }
        if (fd != 1 || !std::equal(a, a + GGML_MAX_DIMS, b)) {
            return false;
        }
    }
    if (miss.size() == 1) {
        return false; // only a drafter with neither is a shared one: one with its own token_embd and no output ties its head to it
    }
    if (missing != nullptr) {
        *missing = std::move(miss);
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
        for (const char * rq_env : { "LLAMA_DRAFT_REQUANT", "LLAMA_DRAFT_REQUANT_OUTPUT" }) {
            const char * rq = getenv(rq_env);
            if (rq != nullptr && rq[0] != 0 && strcmp(rq, "off") != 0 && strcmp(rq, "0") != 0 && strcmp(rq, "none") != 0) {
                if (!result.kv_overrides.empty() && result.kv_overrides.back().key[0] == 0) {
                    result.kv_overrides.pop_back();
                }
                llama_model_kv_override o = {};
                snprintf(o.key, sizeof(o.key), "%s", strcmp(rq_env, "LLAMA_DRAFT_REQUANT") == 0 ? "general.requant_type" : "general.requant_output_type");
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
    std::vector<std::pair<std::string, ggml_type>> borrow_missing;
    result.mtp_borrow_io = spec_mtp && has_draft && common_mtp_borrow_io_ok(params.model.path, params_spec.mparams.path, &borrow_missing);
    // k19-pr374: a shared drafter's absent token_embd / output, in the target's types, for the memory measurement of the draft
    // (llama_model_loader::borrow_missing_type)
    for (const auto & [name, type] : borrow_missing) {
        if (!result.kv_overrides.empty() && result.kv_overrides.back().key[0] == 0) {
            result.kv_overrides.pop_back();
        }
        llama_model_kv_override o = {};
        snprintf(o.key, sizeof(o.key), "%s", name == "token_embd.weight" ? "general.mtp_borrow_token_embd_type" : "general.mtp_borrow_output_type");
        o.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        snprintf(o.val_str, sizeof(o.val_str), "%s", ggml_type_name(type));
        result.kv_overrides.push_back(o);
        result.kv_overrides.emplace_back();
        result.kv_overrides.back().key[0] = 0;
    }

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

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });

    // a block drafter reads the target features of a prompt in ubatches of its own (process() splits a target ubatch, each row is the
    // encoder and the K/V of each layer: no lm head) and decodes one noise block per sequence: the target's n_ubatch only grew its
    // compute buffer and its sliding-window cache (Qwen3.8-Flash-Next + DFlash2 at -ub 2048: 1980 -> 30 MiB of compute with
    // llama_context::sched_reserve, K/V 80 -> 50 MiB). Capped like a separate MTP drafter (LLAMA_DRAFT_UBATCH_MAX, default 512)
    if (has_block_draft && has_draft) {
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

int64_t common_speculative_rows_ext_floats(const common_speculative * spec) {
    if (spec == nullptr || spec->impls.empty()) {
        return 0;
    }
    int64_t n = 0;
    for (const auto & impl : spec->impls) {
        const int64_t f = impl->rows_ext_floats();
        if (f <= 0 || (n != 0 && f != n)) {
            return 0;
        }
        n = f;
    }
    return n;
}

bool common_speculative_rows_ext_set(common_speculative * spec, const float * h, int32_t n_rows, int32_t n_lo) {
    if (common_speculative_rows_ext_floats(spec) <= 0) {
        return false;
    }
    for (auto & impl : spec->impls) {
        if (!impl->rows_ext_set(h, n_rows, n_lo)) {
            common_speculative_rows_ext_clear(spec);
            return false;
        }
    }
    return true;
}

void common_speculative_rows_ext_clear(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->rows_ext_clear();
    }
}

bool common_speculative_process_skip(common_speculative * spec, const common_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->skip_prompt(batch);
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
