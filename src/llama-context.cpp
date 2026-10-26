#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-sysmem.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

//
// llama_context
//

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses,
        const llama_moe_cache * moe_cache) {
    // the MoE cache gives the matmul a copy of the expert weights, so the copy also needs the transform of its source
    llama_hadamard_rotations forward = rotations;
    for (const auto & [w, t] : rotations) {
        if (const ggml_tensor * cached = moe_cache ? moe_cache->get_experts(w) : nullptr) {
            forward.emplace(cached, t);
        }
    }

    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }

        const auto it = forward.find(node->src[0]);
        if (it == forward.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format(
                    "Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

// LLAMA_MTP_EXACT=1: the speculative verify batches (llama_set_verify) are computed batch invariant
static bool llama_exact_verify_env() {
    static const bool value = [] {
        const char * env = getenv("LLAMA_MTP_EXACT");
        return env != nullptr && atoi(env) != 0;
    }();
    return value;
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.verify                  = false;
    cparams.verify_exact            = false;
    cparams.batch_invariant         = false;
    cparams.exact_split             = 0;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload     = params.op_offload;
    cparams.kv_unified     = params.kv_unified;
    cparams.moe_cache_size = params.moe_cache_size;

    cparams.weight_stream         = params.weight_stream;
    cparams.spec_draft            = params.spec_draft;
    cparams.n_state_capture       = params.n_state_capture;
    cparams.weight_stream_io      = params.weight_stream_io;
    cparams.weight_stream_threads = params.weight_stream_threads;
    cparams.weight_stream_ram_mib = params.weight_stream_ram_mib;
    cparams.moe_host_cache_mib    = params.moe_host_cache_mib;

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    {
        // layer-major prompts (see decode_layer_major): LLAMA_MOE_LAYER_MAJOR=1 forces them with a MoE cache, =0 turns them off; unset
        // (auto) they run beside speculative decoding with a model (cparams.spec_draft) when the cache does not hold every host expert
        // and a batch holds at least two ubatches (checked once the cache exists)
        const char * env = getenv("LLAMA_MOE_LAYER_MAJOR");
        lm_auto    = env == nullptr || strcmp(env, "auto") == 0;
        // k15-small4: LLAMA_MOE_LAYER_MAJOR_AUTO_NODRAFT=1 (set by common where its rules keep the default n_ubatch with spilled experts,
        // see common_moe_cache_auto_ubatch): auto also without a draft
        const char * env_nd = getenv("LLAMA_MOE_LAYER_MAJOR_AUTO_NODRAFT");
        const bool   lm_nd  = env_nd != nullptr && atoi(env_nd) != 0;
        lm_enabled = (lm_auto ? (cparams.spec_draft || lm_nd) && cparams.n_batch >= 2*cparams.n_ubatch : atoi(env) != 0) && cparams.moe_cache_size > 0;
        if (const char * e = getenv("LLAMA_MOE_LAYER_MAJOR_UB_MAX")) {
            lm_ub_max = std::max(1, atoi(e));
        }
        // k9-prompt-pass: a batch holds at most ceil(n_batch / n_ubatch) ubatches, so no group is larger: the scheduler and the per-ubatch
        // graph results are sized for that (at -ub 2048 -b 8192: 4, not 16)
        if (cparams.n_ubatch > 0) {
            lm_ub_max = std::max(1, std::min<int>(lm_ub_max, (int) ((cparams.n_batch + cparams.n_ubatch - 1)/cparams.n_ubatch)));
        }
        lm_verbose = getenv("LLAMA_MOE_LAYER_MAJOR_LOG") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_LOG")) != 0;
        // k9-prompt-pass: a context whose prompt ubatches seed the cache on the device (the device-seed fallback, n_ubatch >= 1024, e.g.
        // Studio's -ub 2048) refused every group, so an 8k prompt copied each spilled expert once per ubatch (G4, Flash-Next UD-IQ4_XS
        // 32 GB: 147.7 GB in 4 passes). The group runs there too: its members do not seed on the device, a ubatch run alone after it
        // (the server's checkpoint batch) still does, and a group that ends the prompt seeds from the routing of its last but one member
        lm_devseed      = getenv("LLAMA_MOE_LAYER_MAJOR_DEVSEED") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_DEVSEED")) != 0;
        lm_devseed_seed = getenv("LLAMA_MOE_LAYER_MAJOR_DEVSEED_SEED") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_DEVSEED_SEED")) != 0;
    }

    {
        // smallest ubatch whose host weights the scheduler prefetches (see sched_prefetch_mode)
        const char * GGML_SCHED_PREFETCH           = getenv("GGML_SCHED_PREFETCH");
        const char * GGML_SCHED_PREFETCH_MIN_BATCH = getenv("GGML_SCHED_PREFETCH_MIN_BATCH");
        // LLAMA_MOE_CACHE_SEED=0 disables the seed of the MoE cache at the end of a prompt; by default (=2) a prefetched ubatch seeds
        // from the routing of an earlier ubatch, or else from its own ids read before its copies (see moe_cache_seed_begin)
        // =1 (also turns on LLAMA_MOE_CACHE_PREFILL): only a ubatch whose expert copies read its ids seeds, a prefetched ubatch does not
        // (faster prompts on a fast link, but the cache keeps less of the prompt's routing for the decode after it, and a slow link pays
        // for the extra uploads); =3 is the default (see llama_moe_cache::seed_mode)
        const int seed_mode = llama_moe_cache::seed_mode();
        if (cparams.moe_cache_size > 0 && getenv("LLAMA_MOE_CACHE_SEED") != nullptr && atoi(getenv("LLAMA_MOE_CACHE_SEED")) == 3) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_SEED=3 was removed, the default (2) is used\n", __func__);
        }
        moe_cache_seed_enabled = seed_mode != 0;
        moe_cache_seed_sync    = seed_mode == 2;
        // LLAMA_MOE_CACHE_SEED unset and a large n_ubatch (LLAMA_MOE_CACHE_SEED_AUTO, default 4096, 0 = off): run as =1 (prefill at load,
        // prompts do not seed from their own ids). A prompt of one or few large ubatches has no earlier routing to seed from, its own-ids
        // seed makes every expert upload of the ubatch wait for the routing of its layer, and its seed graph may not fit
        if (getenv("LLAMA_MOE_CACHE_SEED") == nullptr && cparams.moe_cache_size > 0) {
            const char * LLAMA_MOE_CACHE_SEED_AUTO = getenv("LLAMA_MOE_CACHE_SEED_AUTO");
            const int64_t seed_auto_ub = LLAMA_MOE_CACHE_SEED_AUTO == nullptr ? 4096 : atoll(LLAMA_MOE_CACHE_SEED_AUTO);
            moe_cache_seed_auto = seed_auto_ub > 0 && (int64_t) cparams.n_ubatch >= seed_auto_ub;
            if (moe_cache_seed_auto) {
                moe_cache_seed_sync = false;
                LLAMA_LOG_INFO("%s: MoE cache: n_ubatch = %u >= %" PRId64 ": prompts do not seed the cache from their own ids, it is prefilled at load "
                    "(LLAMA_MOE_CACHE_SEED_AUTO=0 or LLAMA_MOE_CACHE_SEED=2 to seed)\n", __func__, cparams.n_ubatch, seed_auto_ub);
            }
        }

        // smallest ubatch whose dense host weights (not MoE experts) are prefetched, below prefetch_min_batch
        // (e.g. dense FFN weights spilled by --fit, computed on the device from GGML_OP_OFFLOAD_MIN_BATCH = 32 tokens); >= prefetch_min_batch = off
        const char * GGML_SCHED_PREFETCH_DENSE_MIN = getenv("GGML_SCHED_PREFETCH_DENSE_MIN");
        if (GGML_SCHED_PREFETCH_DENSE_MIN) {
            prefetch_dense_min_batch = std::max(1, atoi(GGML_SCHED_PREFETCH_DENSE_MIN));
        }

        if (GGML_SCHED_PREFETCH && atoi(GGML_SCHED_PREFETCH) == 0) {
            prefetch_min_batch       = INT64_MAX;
            prefetch_dense_min_batch = INT64_MAX;
        } else if (GGML_SCHED_PREFETCH_MIN_BATCH) {
            prefetch_min_batch = std::max(1, atoi(GGML_SCHED_PREFETCH_MIN_BATCH));
        }

        // by default only if n_ubatch gets the staging arena anyway: the shorter ubatches of a prompt then reuse it, while with a smaller
        // n_ubatch the arena is reserved in place of weights (--fit keeps fewer of them on the device, which costs generation speed)
        if (!GGML_SCHED_PREFETCH_DENSE_MIN && (int64_t) cparams.n_ubatch < prefetch_min_batch) {
            prefetch_dense_min_batch = INT64_MAX;
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
        if (memory) {
            memory->wait_state_reads = [this]() { state_async_wait(); };
            // k16-draft: the replay commit of the delta-net states runs on this context's backends
            llama_memory_recurrent * mr = dynamic_cast<llama_memory_recurrent *>(memory.get());
            if (auto * mh = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
                mr = mh->get_mem_recr();
            } else if (auto * mhi = dynamic_cast<llama_memory_hybrid_iswa *>(memory.get())) {
                mr = mhi->get_mem_recr();
            }
            if (mr != nullptr && mr->replay && !model.hparams.no_alloc) {
                rs_replay_mem = mr;
                mr->commit_cb = [this](bool sync) { rs_replay_commit(sync); };
                LLAMA_LOG_INFO("%s: recurrent rollback by replay: one delta-net state row per sequence instead of %u "
                        "(LLAMA_RS_REPLAY=0: snapshot rows)\n", __func__, 1 + mr->n_rs_seq);
            }
        }
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // device memory a model or context created after this one takes (e.g. the draft of speculative decoding, see
        // llama_context_params::mem_reserve): kept free by what this context places in the free memory while it is created
        mem_reserve.assign(backend_ptrs.size(), 0);
        if (!model.hparams.no_alloc && params.mem_reserve != nullptr && params.mem_reserve_devs != nullptr) {
            for (size_t j = 0; j < params.n_mem_reserve; ++j) {
                for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                    if (params.mem_reserve[j] > 0 && ggml_backend_get_device(backend_ptrs[i]) == params.mem_reserve_devs[j] &&
                        !ggml_backend_buft_is_host(backend_buft[i])) {
                        mem_reserve[i] += params.mem_reserve[j];
                        LLAMA_LOG_INFO("%s: %s: %.2f MiB of device memory kept free for a later model or context (the draft of speculative "
                            "decoding, LLAMA_DRAFT_MEM_RESERVE=0 to disable)\n", __func__, ggml_backend_name(backend_ptrs[i]),
                            params.mem_reserve[j]/1024.0/1024.0);
                    }
                }
            }
        }

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // not supported by the MoE cache

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        // the host weights read from the mapping are uploaded from pinned memory filled with parallel file reads (only with a GPU to upload to)
        // (not in an MTP draft context: its NextN cache runs without a weight store or a RAM tier of its own)
        if (model.weight_extents && !model.hparams.no_alloc && cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
            const bool has_gpu = std::any_of(backend_ptrs.begin(), backend_ptrs.end(), [](ggml_backend_t b) {
                ggml_backend_dev_t dev = ggml_backend_get_device(b);
                return dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
                    ggml_backend_dev_host_buffer_type(dev) != nullptr;
            });
            const size_t bytes = model.weight_extents->bytes_streamable();
            const size_t avail = llama_weight_store::ram_available();
            if (has_gpu && llama_weight_store::enabled(cparams.weight_stream, bytes, avail)) {
                llama_weight_store_params wp;
                wp.io        = cparams.weight_stream_io;
                wp.ring_size = (size_t) cparams.weight_stream_ram_mib << 20;
                wp.n_threads = cparams.weight_stream_threads;
                wp.ram_avail = avail;
                if (const char * e = getenv("LLAMA_WEIGHT_STREAM_VERIFY")) {
                    wp.verify = atoi(e) != 0;
                }
                if (const char * e = getenv("LLAMA_WEIGHT_STREAM_WRAP")) {
                    wp.wrap = atoi(e) != 0;
                }
                if (const char * e = getenv("LLAMA_WEIGHT_STREAM_CHUNK_MIB")) {
                    wp.chunk = (size_t) std::max(1, atoi(e)) << 20;
                }
                wstore = std::make_unique<llama_weight_store>(model.weight_extents, wp);
                LLAMA_LOG_INFO("%s: weight stream on: %.1f MiB of mapped host weights, %.1f MiB of RAM available%s\n", __func__,
                        bytes/1048576.0, avail/1048576.0, wp.verify ? ", VERIFY" : "");
            }
        }

        // the main context caches the layers of its graph, an MTP context the NextN layers with its share of the same budget
        // (LLAMA_MOE_CACHE_MTP=0: the main context caches all layers, an MTP context none)
        const bool ctx_mtp = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP;
        const auto moe_cache_set = llama_moe_cache::select_layers(model, ctx_mtp);
        const size_t moe_cache_budget = cparams.moe_cache_size == 0 || (ctx_mtp && moe_cache_set != llama_moe_cache::layer_set::nextn) ? 0 :
            llama_moe_cache::budget(model, cparams.moe_cache_size, moe_cache_set);
        if (moe_cache_budget > 0 && moe_cache_set == llama_moe_cache::layer_set::nextn) {
            // the MTP cache is optional: the draft runs without it if its share is too small
            try {
                moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs, backend_buft, moe_cache_budget, moe_cache_set, wstore.get(), cparams.moe_host_cache_mib);
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("%s: no MoE cache for the MTP layers (%.2f MiB): %s\n", __func__, moe_cache_budget/1024.0/1024.0, e.what());
                moe_cache.reset();
            }
        } else if (moe_cache_budget > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs, backend_buft, moe_cache_budget, moe_cache_set, wstore.get(), cparams.moe_host_cache_mib);
        }
        if (lm_enabled && !moe_cache) {
            // nothing on the host is cached: no layer-major prompts, and the scheduler keeps its usual size
            lm_enabled = false;
        }
        if (lm_enabled && lm_auto && !(moe_cache->cached_frac() >= 0.0 && moe_cache->cached_frac() < 0.98)) {
            // the cache holds (nearly) every host expert: a prompt uploads (nearly) nothing, nothing to share
            lm_enabled = false;
        }
        if (moe_cache) {
            moe_cache->prompt_inplace_block(lm_enabled); // k15-lfm2: layer-major seeds read the staging copies
        }
        if (lm_enabled && !model.hparams.no_alloc) {
            LLAMA_LOG_INFO("%s: MoE cache: layer-major prompts%s, up to %d ubatches of %u tokens per graph (LLAMA_MOE_LAYER_MAJOR_UB_MAX; "
                "LLAMA_MOE_LAYER_MAJOR=0 disables)\n", __func__, lm_auto ? (cparams.spec_draft ? " beside speculative decoding" : " without a draft") : "", lm_ub_max, cparams.n_ubatch);
        }
        if (moe_cache) {
            n_threads_host = host_threads_init();
            if (n_threads_host > 0 && !model.hparams.no_alloc) {
                LLAMA_LOG_INFO("%s: MoE cache: ubatches whose missing experts the host may compute run on %d threads instead of %d "
                    "(LLAMA_MOE_HOST_THREADS, opt-in)\n", __func__, n_threads_host, (int) cparams.n_threads);
            }
            // the default hybrid path takes its fetch share from the decode threads, so it does not depend on timing
            moe_cache->hybrid_set_threads(n_threads_host > 0 ? n_threads_host : (int32_t) cparams.n_threads, model.hparams.no_alloc);
        }
        // LLAMA_MOE_CACHE_DEVSEED=1 (opt-in, default 0): with the default seeding (=2) a prefetched ubatch seeds the cache on the device from
        // its own routing (see llama_moe_cache::devseed_plan): its expert uploads never wait for the host to read the routing, and it needs
        // no stored routing of an earlier ubatch nor a larger seed graph; groups without device lookups are not seeded by prefetched ubatches
        if (moe_cache && llama_moe_cache::seed_mode() == 2 && prefetch_min_batch != INT64_MAX) {
            const char * env = getenv("LLAMA_MOE_CACHE_DEVSEED");
            // opt-in: measured faster prompts at 8-32 GB on fast links (L4 gemma +4..+30%, G4 Flash-Next IQ4_XS +7..+18%), but slower first
            // prompts on a throttled link (the #268 seed uploads only the routed experts of a single-ubatch prompt) and 2-3% slower decode
            // after a prompt on the G4 (a different cache mix than the host LRU seed)
            if (env != nullptr && atoi(env) != 0 && moe_cache->devseed_init(prefetch_min_batch)) {
                moe_cache_devseed_on = true;
                moe_cache_seed_sync  = false;
                moe_cache_seed_auto  = false;
                LLAMA_LOG_INFO("%s: MoE cache: prompt ubatches of %" PRId64 "+ tokens seed the cache on the device (LLAMA_MOE_CACHE_DEVSEED=1)\n",
                    __func__, prefetch_min_batch);
            }
        }
        if (moe_cache && moe_cache_seed_auto) {
            moe_cache->prompts_do_not_seed();
        }
        // k15-pp3: the capture snapshots fit counted (cap_planned_bytes) are allocated here, before the MoE cache's decisions read the free
        // memory, instead of at the first capture: with their bytes still free, the first reserve kept the staging copies outside the
        // cache and the first prompt kept its graph allocation out of the lent memory (G4 q4x 12 GB + MTP: lending off, then the graph
        // in its own compute buffer, 188 / 248 MiB free at exit instead of 408 with LLAMA_FIT_CAPTURE=0). cap_prepare takes these buffers
        if (moe_cache && !model.hparams.no_alloc) {
            const int n_cap = cap_count();
            ggml_backend_buffer_type_t cb = nullptr;
            const size_t cbytes = cap_planned_bytes(&cb);
            if (cbytes > 0 && cb != nullptr) {
                for (int c = 0; c < n_cap; ++c) {
                    cap[c].buf.reset(ggml_backend_buft_alloc_buffer(cb, cbytes / n_cap));
                }
                LLAMA_LOG_INFO("%s: %.2f MiB of device memory for %d state capture snapshots (%s; LLAMA_FIT_CAPTURE=0: at the first capture, "
                    "not counted by fit)\n", __func__, cbytes/1024.0/1024.0, n_cap, ggml_backend_buft_name(cb));
            }
        }
        if (moe_cache) {
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                moe_cache->set_mem_reserve(backend_ptrs[i], mem_reserve[i]);
            }
        }

        sched_reserve();

        // layer-major prompts: the host arena of the group inputs (mostly the KQ masks) and the graph results of the members. k9 made them
        // at load so that the first prompt does not pay for them: 2 x 256 MiB of pinned memory (main + MTP context) and ~290 MiB of zeroed
        // graph metadata (16 results per context) in every such process, also one that never reads a long prompt (k9 vs k8 host RSS +805 MiB
        // at idle on a B200, +884 MiB on gfx1151; plans/k9_prompt_pass.md). Now both are made by the first group, the arena sized by what
        // it needs (decode_layer_major grows it); LLAMA_MOE_LAYER_MAJOR_EAGER=1 allocates them here as before
        static const bool lm_eager = getenv("LLAMA_MOE_LAYER_MAJOR_EAGER") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_EAGER")) != 0;
        if (lm_enabled && moe_cache && lm_eager) {
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (auto * dev = model.dev_output()) {
                if (auto * hb = ggml_backend_dev_host_buffer_type(dev)) {
                    buft = hb;
                }
            }
            const size_t cap  = (size_t) (getenv("LLAMA_MOE_LAYER_MAJOR_ARENA_MIB") ? atoll(getenv("LLAMA_MOE_LAYER_MAJOR_ARENA_MIB")) : 256) << 20;
            const size_t want = std::min(cap, (size_t) lm_ub_max*((size_t) cparams.n_ctx*cparams.n_ubatch*sizeof(float) + (1u << 20)));
            if (want > 0) {
                lm_arena.reset(ggml_backend_buft_alloc_buffer(buft, want));
            }
            // the graph results of the group, created (and their metadata buffers touched) at load as well
            lm_res.resize(lm_ub_max);
            lm_inp_buf.resize(lm_ub_max);
            for (auto & r : lm_res) {
                r.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
            }
        }

        // the later model or context now takes its memory itself: a later reserve sees it in the free memory of the devices
        // (the lending decision of the MoE cache was made at the first graph allocation, in the reserve above)
        std::fill(mem_reserve.begin(), mem_reserve.end(), 0);
        if (moe_cache) {
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                moe_cache->set_mem_reserve(backend_ptrs[i], 0);
            }
        }
        if (sched && sched_pf_headroom_def != SIZE_MAX) {
            ggml_backend_sched_set_prefetch_headroom(sched.get(), sched_pf_headroom_def);
        }

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // advisory PLE prefetch
    {
        auto env_on = [](const char * name) {
            const char * v = getenv(name);
            return v && atoi(v) != 0;
        };

        ple.verify = env_on("LLAMA_PLE_PREFETCH_VERIFY");
        ple.stats  = getenv("LLAMA_PLE_PREFETCH_STATS") ? atoi(getenv("LLAMA_PLE_PREFETCH_STATS")) : 0;

        const enum llama_ple_prefetch_mode mode = llama_ple_prefetch_resolve(params.ple_prefetch);

        const ggml_tensor * t = model.per_layer_tok_embd;

        if (mode != LLAMA_PLE_PREFETCH_OFF && t && model.can_prefetch.count(t) && t->data && t->buffer &&
            ggml_backend_buffer_is_host(t->buffer) && ggml_is_matrix(t) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
            int    fd   = -1;
            size_t offs = 0;
            if (const auto it = model.lazy_files.find(t); it != model.lazy_files.end()) {
                fd   = it->second.fd;
                offs = it->second.offs;
            }
            const char * env_threads = getenv("LLAMA_PLE_PREFETCH_THREADS");
            const int n_threads = env_threads ? atoi(env_threads) : 0;

            ple_prefetcher = std::make_unique<llama_row_prefetcher>(mode, n_threads, t, fd, offs);

            LLAMA_LOG_INFO("%s: PLE prefetch of '%s' one ubatch ahead: mode = %s, threads = %d\n", __func__,
                    t->name, llama_ple_prefetch_mode_name(ple_prefetcher->get_mode()), ple_prefetcher->get_n_threads());
        }

        ple.enabled = t && (ple_prefetcher || ple.verify);
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

void llama_context::small_plan_free(uint32_t i) {
    if (gf_res_small_plan[i] != nullptr) {
        ggml_backend_sched_plan_free(gf_res_small_plan[i]);
        gf_res_small_plan[i] = nullptr;
    }
}

void llama_context::rs_replay_commit(bool sync) {
    llama_memory_recurrent * mem = rs_replay_mem;
    if (mem == nullptr || !mem->has_pending()) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    const size_t max_nodes = 16384;
    if (!rs_sched) {
        rs_sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, false));
    }
    // the same cells committing the same token counts (one slot drafting: the usual case) run the graph of the last commit again
    std::vector<uint32_t> key;
    for (uint32_t c = 0; c < mem->size; ++c) {
        if (mem->pend_n[c] != 0) {
            key.insert(key.end(), { c, mem->pend_n[c], mem->pend_rb[c] });
        }
    }
    static const bool reuse = getenv("LLAMA_RS_REPLAY_REUSE") == nullptr || atoi(getenv("LLAMA_RS_REPLAY_REUSE")) != 0;
    if (reuse && rs_gf != nullptr && key == rs_key) {
        for (uint32_t c = 0; c < mem->size; ++c) {
            mem->pend_n[c]  = 0;
            mem->pend_rb[c] = 0;
        }
        if (ggml_backend_sched_graph_compute_async(rs_sched.get(), rs_gf) != GGML_STATUS_SUCCESS) {
            GGML_ABORT("%s: the replay commit graph failed", __func__);
        }
        if (sync) {
            ggml_backend_sched_synchronize(rs_sched.get());
        }
        rs_reused++;
        mem->n_replay_commit++;
        rs_replay_us += ggml_time_us() - t0;
        return;
    }
    rs_gf = nullptr;
    rs_key = key;
    ggml_init_params ip = { ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true };
    rs_ctx.reset(ggml_init(ip));
    ggml_context * ctx = rs_ctx.get();
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
    const int64_t D = model.hparams.n_embd_s();
    for (uint32_t c = 0; c < mem->size; ++c) {
        const uint32_t n = mem->pend_n[c];
        if (n == 0) {
            continue;
        }
        const uint32_t rb   = mem->pend_rb[c];
        const uint32_t keep = n > rb ? n - rb : 0;
        for (size_t il = 0; il < mem->s_l.size(); ++il) {
            ggml_tensor * s  = mem->s_l[il];
            ggml_tensor * rk = mem->rk_l[il];
            if (s == nullptr || rk == nullptr) {
                continue;
            }
            if (keep > 0) {
                ggml_tensor * rv = mem->rv_l[il];
                ggml_tensor * rg = mem->rg_l[il];
                ggml_tensor * rbt = mem->rb_l[il];
                const int64_t H_k = mem->rk_h[il] > 0 ? mem->rk_h[il] : rk->ne[1];
                const int64_t S_v = rv->ne[0];
                const int64_t H_v = rv->ne[1];
                auto win = [&](ggml_tensor * t, int64_t ne0, int64_t ne1) {
                    return ggml_view_4d(ctx, t, ne0, ne1, keep, 1, t->nb[1], t->nb[2], t->nb[3], (size_t) c * t->nb[3]);
                };
                ggml_tensor * k  = win(rk, rk->ne[0], H_k);
                ggml_tensor * st = ggml_view_4d(ctx, s, S_v, S_v, H_v, 1, ggml_row_size(s->type, S_v), ggml_row_size(s->type, S_v*S_v),
                        ggml_row_size(s->type, D), (size_t) c * s->nb[1]);
                ggml_tensor * out = ggml_gated_delta_net(ctx, k, k, win(rv, S_v, H_v), win(rg, 1, H_v), win(rbt, 1, H_v), st, 1);
                ggml_tensor * ns  = ggml_view_1d(ctx, out, D, ggml_row_size(out->type, S_v*H_v*(int64_t) keep));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ns, ggml_view_1d(ctx, s, D, (size_t) c * s->nb[1])));
            }
            if (rb > 0) {
                // the conv histories keep their snapshot rows: group rb of this cell onto its live row
                for (ggml_tensor * t : { mem->r_l[il], il < mem->p_l.size() ? mem->p_l[il] : nullptr }) {
                    if (t == nullptr) {
                        continue;
                    }
                    ggml_tensor * src = ggml_view_1d(ctx, t, t->ne[0], ((size_t) rb * mem->size + c) * t->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, ggml_view_1d(ctx, t, t->ne[0], (size_t) c * t->nb[1])));
                }
            }
        }
        mem->pend_n[c]  = 0;
        mem->pend_rb[c] = 0;
    }
    if (ggml_graph_n_nodes(gf) > 0) {
        ggml_backend_sched_reset(rs_sched.get());
        if (!ggml_backend_sched_alloc_graph(rs_sched.get(), gf)) {
            GGML_ABORT("%s: cannot allocate the replay commit graph", __func__);
        }
        if (ggml_backend_sched_graph_compute_async(rs_sched.get(), gf) != GGML_STATUS_SUCCESS) {
            GGML_ABORT("%s: the replay commit graph failed", __func__);
        }
        rs_gf = gf;
        if (sync) {
            ggml_backend_sched_synchronize(rs_sched.get());
        }
    }
    mem->n_replay_commit++;
    rs_replay_us += ggml_time_us() - t0;
}

llama_context::~llama_context() {
    if (rs_replay_mem != nullptr && rs_replay_mem->n_replay_commit > 0) {
        LLAMA_LOG_INFO("%s: recurrent replay commits: %" PRIu64 " (%" PRIu64 " reused graphs), %.2f ms total, %.3f ms each (host)\n", __func__,
                rs_replay_mem->n_replay_commit, rs_reused, rs_replay_us/1e3, rs_replay_us/1e3/rs_replay_mem->n_replay_commit);
        if (rs_sched) {
            ggml_backend_sched_synchronize(rs_sched.get());
        }
        rs_replay_mem->commit_cb = nullptr;
    }
    if (n_plan_saves > 0) {
        size_t bytes = 0;
        int    n     = 0;
        for (auto * p : gf_res_small_plan) {
            bytes += ggml_backend_sched_plan_bytes(p);
            n     += p != nullptr;
        }
        LLAMA_LOG_INFO("%s: kept graph allocations (LLAMA_SCHED_SIZE_CACHE): %" PRIu64 " restored, %" PRIu64 " saved, %" PRIu64 " no longer valid, "
            "%d held, %.2f MiB of host memory, no device memory; kept graph swap-ins %" PRIu64 ", graph reuses %d\n", __func__, n_plan_restores,
            n_plan_saves, n_plan_invalid, n, bytes/1048576.0, n_small_swaps, (int) n_reused);
    }
    for (uint32_t i = 0; i < GF_SMALL_N; ++i) {
        small_plan_free(i);
    }

    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    if (threadpool_host != nullptr) {
        if (backend_cpu != nullptr) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
            auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
            if (set_threadpool_fn) {
                set_threadpool_fn(backend_cpu, nullptr);
            }
        }
        if (threadpool_host_free != nullptr) {
            threadpool_host_free(threadpool_host);
        }
        threadpool_host = nullptr;
    }

    if (vram_watch > 0) {
        vram_watch_sample(true);
        for (size_t i = 0; i < vram_watch_min.size(); ++i) {
            if (vram_watch_min[i] != SIZE_MAX) {
                LLAMA_LOG_INFO("%s: VRAM watch: %s least free %.1f MiB over %" PRId64 " graphs (LLAMA_VRAM_WATCH)\n", __func__,
                        ggml_backend_dev_name(model.devices[i].dev), vram_watch_min[i]/1048576.0, vram_watch_n);
            }
        }
    }

    if (mtp_dev_h_sink && mtp_dev_h_sink->tgt == this) {
        mtp_dev_h_sink->tgt = nullptr;
        mtp_dev_h_sink = nullptr;
    }
    if (mtp_dev_h_own) {
        if (mtp_dev_h_own->backend_dft) {
            ggml_backend_synchronize(mtp_dev_h_own->backend_dft);
        }
        if (mtp_dev_h_own->tgt) {
            mtp_dev_h_own->tgt->synchronize();
        }
        mtp_dev_h_own.reset();
    }

    for (auto & sa : state_async) {
        if (sa.event) {
            ggml_backend_event_free(sa.event);
            sa.event = nullptr;
        }
    }
    for (auto & c : cap) {
        if (c.ev) {
            ggml_backend_event_free(c.ev);
            c.ev = nullptr;
        }
    }
    if (cap_side_ev) {
        ggml_backend_event_free(cap_side_ev);
        cap_side_ev = nullptr;
    }
    if (cap_side) {
        ggml_backend_synchronize(cap_side);
        ggml_backend_free(cap_side);
        cap_side = nullptr;
    }

    if (ple_prefetcher || ple.verify || ple.stats) {
        const auto & st = ple_prefetcher ? &ple_prefetcher->st : nullptr;
        LLAMA_LOG_INFO("%s: PLE prefetch: mode = %s, predicted = %" PRIu64 ", unpredicted = %" PRIu64 ", rows = %" PRIu64
                ", skipped rows = %" PRIu64 ", predict = %.3f ms/ubatch\n", __func__,
                llama_ple_prefetch_mode_name(ple_prefetcher ? ple_prefetcher->get_mode() : LLAMA_PLE_PREFETCH_OFF),
                ple.n_predicted, ple.n_unpredicted, ple.n_rows, ple.n_skipped_rows,
                ple.n_predicted ? ple.t_predict_us/1e3/ple.n_predicted : 0.0);
        if (st) {
            LLAMA_LOG_INFO("%s: PLE prefetch: jobs = %" PRIu64 ", dropped = %" PRIu64 ", runs = %" PRIu64 ", %.1f MiB, "
                    "%.2f ms/job (submit to done), worker majflt = %" PRIu64 ", errors = %" PRIu64 "\n", __func__,
                    st->n_jobs.load(), st->n_dropped.load(), st->n_runs.load(), st->n_bytes.load()/1024.0/1024.0,
                    st->n_jobs ? st->t_busy_us.load()/1e3/st->n_jobs.load() : 0.0, st->n_majflt.load(), st->n_errors.load());
        }
        LLAMA_LOG_INFO("%s: PLE prefetch: set_inputs = %.2f ms/ubatch over %" PRIu64 ", decode = %.2f ms/call over %" PRIu64
                ", majflt = %" PRId64 ", minflt = %" PRId64 "\n", __func__,
                ple.n_set_inputs ? ple.t_set_inputs_us/1e3/ple.n_set_inputs : 0.0, ple.n_set_inputs,
                ple.n_decode ? ple.t_decode_us/1e3/ple.n_decode : 0.0, ple.n_decode, ple.majflt, ple.minflt);
        if (ple.verify) {
            LLAMA_LOG_INFO("%s: PLE prefetch verify: checked = %" PRIu64 " ubatches, mismatched rows = %" PRIu64 "\n", __func__,
                    ple.n_checked, ple.n_mismatch);
        }
    }

    // stop the workers before the model can go
    ple_prefetcher.reset();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);

    // the weight store is destroyed before the MoE cache whose RAM tier it reads through
    if (wstore) {
        wstore->set_tier(nullptr);
    }
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

void llama_context::sched_new(size_t max_nodes, bool parallel) {
    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, parallel, cparams.op_offload));

    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
    ggml_backend_sched_set_realloc_callback(sched.get(), sched_graph_realloc, this);
    ggml_backend_sched_set_prefetch_callback(sched.get(), sched_prefetch_mode, this);
    // the hybrid MoE cache computes the missing experts on the host while the device computes the cached ones
    ggml_backend_sched_set_overlap(sched.get(), moe_cache && moe_cache->hybrid_active());
    if (moe_cache) {
        ggml_backend_sched_set_lend_callback(sched.get(), sched_lend, this);
    }
    // spilled experts (the MoE cache holds less than the host experts): every prompt ubatch uploads the experts the cache misses in
    // every layer, and copies of 3 splits in flight hide them better than the default 2 when the staging is lent from the cache.
    // Start depth (GGML_SCHED_PREFETCH_DEPTH still wins; never for a cache of fewer than 4 tokens' experts in one bank, n_tokens_cached):
    // - LLAMA_MOE_PREFETCH_DEPTH_SPILL=N: N (0 = the scheduler default), no other check
    // - else 3 when no speculative decoding with a model runs beside this context (cparams.spec_draft: a -md draft or built-in MTP
    //   heads), the cache holds the experts of at least LLAMA_MOE_PREFETCH_DEPTH_SPILL_MIN_TOKENS (4) tokens over all layers, and the
    //   uncached host experts are at most LLAMA_MOE_PREFETCH_DEPTH_SPILL_MAX_RAM_FRAC (0.5) of the RAM; otherwise the default 2.
    //   LLAMA_MOE_PREFETCH_DEPTH_RULE=0 keeps the default 2 everywhere.
    // Why (G4 RTX PRO 6000, 3 reps, A/A <= 0.3%): no draft, staging lent: gpt-oss-20B Q2_K 8 GB 2k / 8k prompt 10864 -> 12487 /
    // 14409 -> 16276 tok/s, Qwen3.6-35B-A3B 8 GB 8k 9083 -> 10586, decode unchanged. With a drafter the depth-3 staging competes with
    // the draft: Flash-Next UD-IQ4_XS 64 GB (Studio argv) 2k prompt 3647 -> 2291, chat decode 163 -> 142 (the depth falls back after the
    // allocation and the cache is seeded empty instead of prefilled). A tiny cache (Flash-Next UD-IQ1_S 8 GB, the experts of 1.0 tokens)
    // pays the deeper staging with cache: n_ubatch 4096 -> 2048, prompt -1.8%, decode -1.5%. Uncached experts beyond the page cache read
    // more from the disk at depth 3 (L4 52 GB RAM, Flash-Next UD-IQ4_XS: 2k decode 13.2 -> 10.2)
    if (moe_cache && moe_cache->cached_frac() >= 0.0 && moe_cache->cached_frac() < 0.98 && moe_cache->n_tokens_cached() >= 4) {
        const char * env_depth = getenv("LLAMA_MOE_PREFETCH_DEPTH_SPILL");
        int depth_spill = 0;
        if (env_depth != nullptr) {
            depth_spill = atoi(env_depth);
        } else if (getenv("LLAMA_MOE_PREFETCH_DEPTH_RULE") == nullptr || atoi(getenv("LLAMA_MOE_PREFETCH_DEPTH_RULE")) != 0) {
            double min_tokens = 4.0;
            if (const char * e = getenv("LLAMA_MOE_PREFETCH_DEPTH_SPILL_MIN_TOKENS")) {
                min_tokens = std::max(0.0, atof(e));
            }
            double ram_frac = 0.5;
            if (const char * e = getenv("LLAMA_MOE_PREFETCH_DEPTH_SPILL_MAX_RAM_FRAC")) {
                ram_frac = std::max(0.0, atof(e));
            }
            // the experts of one token over all layers are n_expert_used / n_expert of the host experts
            const uint32_t n_used = model.hparams.n_expert_used_max();
            const double tokens = n_used > 0 ? moe_cache->cached_frac()*(double) model.hparams.n_expert/(double) n_used : 0.0;
            const uint64_t ram_total = llama_sysmem::total();
            const bool ram_ok = ram_total == 0 || moe_cache->host_uncached_bytes() <= ram_frac*(double) ram_total;
            const char * why = cparams.spec_draft ? "speculative decoding with a draft model or MTP heads" :
                tokens < min_tokens ? "the cache holds the experts of fewer tokens than LLAMA_MOE_PREFETCH_DEPTH_SPILL_MIN_TOKENS" :
                !ram_ok ? "the uncached host experts are more than LLAMA_MOE_PREFETCH_DEPTH_SPILL_MAX_RAM_FRAC of the RAM" : nullptr;
            depth_spill = why == nullptr ? 3 : 0;
            static bool logged = false; // once per process, not for a memory probe (no_alloc), whose log the fit silences
            if (!logged && !model.hparams.no_alloc) {
                logged = true;
                LLAMA_LOG_INFO("%s: MoE cache: host weight prefetch of the spilled experts starts at depth %s (cache = the experts of %.1f "
                    "tokens%s%s; LLAMA_MOE_PREFETCH_DEPTH_SPILL=N forces N, LLAMA_MOE_PREFETCH_DEPTH_RULE=0 keeps the default)\n", __func__,
                    why == nullptr ? "3" : "the default", tokens, why ? ": " : "", why ? why : "");
            }
        }
        if (depth_spill > 0) {
            ggml_backend_sched_set_prefetch_depth_init(sched.get(), depth_spill);
        }
    }
    // a staging arena without a lender (see ggml_backend_sched_prefetch_headroom_ok) also leaves the memory reserved for a later
    // model or context (one headroom for every device: the largest reserve)
    const size_t reserve_max = mem_reserve.empty() ? 0 : *std::max_element(mem_reserve.begin(), mem_reserve.end());
    if (reserve_max > 0) {
        sched_pf_headroom_def = ggml_backend_sched_get_prefetch_headroom(sched.get());
        if (sched_pf_headroom_def > 0) {
            ggml_backend_sched_set_prefetch_headroom(sched.get(), sched_pf_headroom_def + reserve_max);
        }
    }
    if (wstore) {
        ggml_backend_sched_set_prefetch_plan_callback(sched.get(), sched_stream_plan, this);
        ggml_backend_sched_set_prefetch_source_callback(sched.get(), sched_stream_source, this);
    }
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr; gf_res_epoch++;

    // a layer-major graph holds the graphs of up to lm_ub_max ubatches (LLAMA_MOE_LAYER_MAJOR_SCHED_PAD=1, debug: that size without them)
    static const bool lm_sched_pad = getenv("LLAMA_MOE_LAYER_MAJOR_SCHED_PAD") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_SCHED_PAD")) != 0;
    sched_new(lm_enabled || (lm_sched_pad && moe_cache) ? max_nodes*(size_t) (lm_ub_max + 1) : max_nodes, cparams.pipeline_parallel);
    // the usual graphs (decode, a ubatch alone) keep a table of their own size: the table sized for a merged graph cost the decode
    // 0.2-0.3% (G4, Qwen3.6-35B-A3B UD-Q3_K_M + MTP, LLAMA_MOE_LAYER_MAJOR_SCHED_PAD=1); decode_layer_major asks for the full one
    // (LLAMA_MOE_LAYER_MAJOR_SMALL_HASH=0: the full table for every graph)
    static const bool lm_small_hash = getenv("LLAMA_MOE_LAYER_MAJOR_SMALL_HASH") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_SMALL_HASH")) != 0;
    if ((lm_enabled || (lm_sched_pad && moe_cache)) && lm_small_hash) {
        ggml_backend_sched_set_hash_size_small(sched.get(), max_nodes);
    }

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched_new(max_nodes, false);
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        // a memory measurement (e.g. --fit) does not count the staging memory of the host weight prefetch: it only uses memory that is
        // left after the layers (the reserve drops the prefetch when the compute buffers do not fit with it or leave no headroom), so a fit
        // places the same layers as without the prefetch instead of moving one to the host for the staging copies
        // with the MoE cache the staging may live in lent cache slots and the cache sizing accounts for it: the measurement keeps it
        if (model.hparams.no_alloc && !moe_cache) {
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                backend_buf_exp_size[i] -= std::min(backend_buf_exp_size[i], ggml_backend_sched_get_staging_size(sched.get(), backend_ptrs[i]));
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;

        graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
    }

    // the staging memory of the pp graph (the prefetch arena), see the seed graph below
    std::vector<size_t> staging_pp(backend_ptrs.size());
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        staging_pp[i] = ggml_backend_sched_get_staging_size(sched.get(), backend_ptrs[i]);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;

        // the MoE cache lends memory to larger graphs (see llama_moe_cache::lend), the compute buffer must hold without it the small
        // batches whose experts are all read from the cache (larger ones copy experts, like the prompts, and may borrow)
        const uint32_t n_tokens_small = moe_cache ? std::min<uint32_t>(n_tokens, (uint32_t) moe_cache->n_tokens_cached()) : 0;
        if (moe_cache && moe_cache->lends_graph() && !model.hparams.no_alloc && n_tokens_small > n_seqs && n_tokens_small % n_seqs == 0) {
            if (!graph_reserve(n_tokens_small, n_seqs, std::min(n_tokens_small, cparams.n_outputs_max), mctx.get())) {
                throw std::runtime_error("failed to allocate compute small batch buffers");
            }
        }
    }

    // a prefetched ubatch that seeds the MoE cache from its own ids copies the experts of the cached layers once the ids are known, the
    // graph allocator holds these copies instead of the staging memory (see sched_copy_all_experts): reserve this graph too if the
    // devices have room for it, else such a ubatch does not seed (see moe_cache_seed_begin), before the pp graph that sizes the arena
    // a memory measurement (e.g. --fit) does not count it: the seed only uses memory that is left, it never moves weights off a device
    moe_cache_seed_fits = true;
    if (moe_cache && moe_cache_seed_sync && !model.hparams.no_alloc && n_tokens > llama_moe_cache::n_small_max && (int64_t) n_tokens >= prefetch_min_batch) {
        moe_cache_seed_fits = sched_reserve_seed(n_tokens, n_seqs, n_outputs_pp, mctx.get(), staging_pp);
        if (!moe_cache_seed_fits) {
            moe_cache_devseed_fallback();
            moe_cache->seed_unavailable();
        }
    }

    // the measure of the MoE cache auto n_ubatch (LLAMA_MOE_MEASURE_LENT=1, set by common_moe_cache_auto_ubatch) counts the compute buffer
    // the loaded context ends up with: the staging copies lent by the MoE cache (none of the arena), and the seed graph when it is larger
    // than the pp graph (a prompt ubatch that seeds the cache holds the expert copies of its cached layers in the graph allocation), which
    // a context with room for it reserves (see sched_reserve_seed). Measured with the staging and without the seed, the growth from
    // n_ubatch 512 to 2048 was 390 MiB where the device sees 254 (Qwen3.6 UD-Q4_K_XL) and 794 where it sees 548 (gemma-4 26B-A4B)
    // LLAMA_MOE_MEASURE_LENT=auto:<bytes>: lent only if the staging copies are at least <bytes>, the share of the cache from which the cache
    // lends them (LLAMA_MOE_CACHE_LEND_MIN_FRAC, see llama_moe_cache lend_decide); else the copies stay in the measure, as an arena next to it.
    // The copies are those the cache was asked to lend (a measurement always lends them, the arenas stay empty) or those of an arena
    const char * measure_env  = model.hparams.no_alloc ? getenv("LLAMA_MOE_MEASURE_LENT") : nullptr;
    const bool   measure_auto = measure_env != nullptr && strncmp(measure_env, "auto:", 5) == 0;
    int          measure_lent = measure_env != nullptr && !measure_auto ? atoi(measure_env) : 0;
    if (measure_env != nullptr && (measure_lent != 0 || measure_auto)) {
        const bool seeds = moe_cache && moe_cache_seed_sync && n_tokens > llama_moe_cache::n_small_max && (int64_t) n_tokens >= prefetch_min_batch;
        std::vector<size_t> sizes(backend_ptrs.size(), 0);
        std::vector<size_t> staging_seed(backend_ptrs.size(), 0);
        if (seeds) {
            moe_cache_seed = true;
            graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), /*split_only =*/ true, sizes.data());
            moe_cache_seed = false;
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                staging_seed[i] = std::min(sizes[i], ggml_backend_sched_get_staging_size(sched.get(), backend_ptrs[i]));
            }
        }
        if (measure_auto) {
            const size_t min_bytes = (size_t) strtoull(measure_env + 5, nullptr, 10);
            // the cache lends in a measurement (dry), which leaves no staging in the arena: the copies are what it was asked to lend
            size_t staging = moe_cache ? moe_cache->lend_need_max() : 0;
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                staging = std::max({ staging, staging_pp[i], staging_seed[i] });
            }
            measure_lent = staging > 0 && staging >= min_bytes ? 1 : 0;
            LLAMA_LOG_DEBUG("%s: measure: staging copies %.2f MiB, the cache lends from %.2f MiB: %s\n", __func__, staging/1024.0/1024.0,
                min_bytes/1024.0/1024.0, measure_lent ? "lent" : "counted");
        }
        if (measure_lent != 0) {
            std::vector<size_t> graph_sz(backend_ptrs.size());
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                graph_sz[i] = backend_buf_exp_size[i] - std::min(backend_buf_exp_size[i], staging_pp[i]);
                if (seeds) {
                    graph_sz[i] = std::max(graph_sz[i], sizes[i] - staging_seed[i]);
                }
                LLAMA_LOG_DEBUG("%s: %s: measure: compute %.2f MiB (pp with staging), %.2f MiB lent\n", __func__, ggml_backend_buft_name(backend_buft[i]),
                    backend_buf_exp_size[i]/1024.0/1024.0, graph_sz[i]/1024.0/1024.0);
                backend_buf_exp_size[i] = graph_sz[i];
            }
        }
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.hparams.no_alloc);
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

// the host seed of a prompt (its own ids read before its copies) does not fit in the compute buffer: such a context prefills the cache at
// load and its prompts do not seed it, so the decode after a prompt reads the experts of the prefill. With a cache of at most
// LLAMA_MOE_CACHE_DEVSEED_FALLBACK_MAX_FRAC (0.5) of the host experts, and of at least LLAMA_MOE_CACHE_DEVSEED_FALLBACK_MIN_TOKENS (4)
// tokens' experts over all layers, the large prompt ubatches (see min_ub below; never decode / verify) seed the cache on the device
// from their staged copies instead (as LLAMA_MOE_CACHE_DEVSEED=1, on top of the prefill): no larger graph, no wait for the routing, no
// lent slots. G4 (RTX PRO 6000) Flash-Next UD-IQ4_XS 32 GB, Studio argv (MTP drafter), 2k prompt + 256 decode: decode hit rate after
// the prompt 87.8% -> 90.3%, decode 95.7 -> 98.0 tok/s, prompt 2204 -> 2243; the host seed (lent at depth 3, or at depth 2) gets the same
// decode but costs the prompt 11-14% (its uploads wait for the routing of each layer, the lent slots are refilled after the prompt).
// A cache of most experts (Flash-Next 64 GB, 92%) keeps the prefill alone: there the decode reads 98.8% from it already.
// LLAMA_MOE_CACHE_DEVSEED_FALLBACK=0 keeps the prefill without seeding; an explicit LLAMA_MOE_CACHE_DEVSEED or LLAMA_MOE_CACHE_SEED wins
void llama_context::moe_cache_devseed_fallback() {
    if (!moe_cache || moe_cache_devseed_on || model.hparams.no_alloc || prefetch_min_batch == INT64_MAX ||
        getenv("LLAMA_MOE_CACHE_DEVSEED") != nullptr || getenv("LLAMA_MOE_CACHE_SEED") != nullptr) {
        return;
    }
    const char * env = getenv("LLAMA_MOE_CACHE_DEVSEED_FALLBACK");
    if (env != nullptr && atoi(env) == 0) {
        return;
    }
    double max_frac = 0.5;
    if (const char * e = getenv("LLAMA_MOE_CACHE_DEVSEED_FALLBACK_MAX_FRAC")) {
        max_frac = atof(e);
    }
    double min_tokens = 4.0;
    if (const char * e = getenv("LLAMA_MOE_CACHE_DEVSEED_FALLBACK_MIN_TOKENS")) {
        min_tokens = std::max(0.0, atof(e));
    }
    const double frac = moe_cache->cached_frac();
    // the experts of one token over all layers are n_expert_used / n_expert of the host experts (as for the prefetch depth of spilled experts)
    const uint32_t n_used = model.hparams.n_expert_used_max();
    const double tokens = n_used > 0 && frac > 0.0 ? frac*(double) model.hparams.n_expert/(double) n_used : 0.0;
    // only prompt ubatches of at least LLAMA_MOE_CACHE_DEVSEED_FALLBACK_MIN_UBATCH (1024) tokens seed: the routing of a chat-sized prompt
    // replaced the prefill for the next one (the 527-token warmup seed: chat decode -0.3% Flash-Next 32 GB, -1.2% Qwen3.6 8 GB MTP)
    int64_t min_ub = 1024;
    if (const char * e = getenv("LLAMA_MOE_CACHE_DEVSEED_FALLBACK_MIN_UBATCH")) {
        min_ub = std::max<int64_t>(1, atoll(e));
    }
    min_ub = std::max(min_ub, prefetch_min_batch);
    const bool on = frac >= 0.0 && frac <= max_frac && tokens >= min_tokens && (int64_t) cparams.n_ubatch >= min_ub;
    if (on && moe_cache->devseed_init(min_ub)) {
        moe_cache_devseed_on = true;
        moe_cache_seed_sync  = false;
    }
    static bool logged_off = false; // the message without the device seed once per process (every reserve of such a context repeats it)
    if (!moe_cache_devseed_on && logged_off) {
        return;
    }
    logged_off = logged_off || !moe_cache_devseed_on;
    LLAMA_LOG_INFO("%s: MoE cache: the host seed of a prompt does not fit: %s (cache = %.1f%% of the host experts, the experts of %.1f tokens; "
        "LLAMA_MOE_CACHE_DEVSEED_FALLBACK=0 keeps the prefill alone)\n", __func__,
        moe_cache_devseed_on ? "prompt ubatches seed the cache on the device, on top of the prefill" : "the cache is prefilled, prompts do not seed it",
        100.0*frac, tokens);
    if (moe_cache_devseed_on) {
        LLAMA_LOG_INFO("%s: MoE cache: device seed from prompt ubatches of %" PRId64 "+ tokens (LLAMA_MOE_CACHE_DEVSEED_FALLBACK_MIN_UBATCH)\n", __func__, min_ub);
    }
}

bool llama_context::sched_reserve_seed(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, const std::vector<size_t> & staging_pp) {
    // headroom for the memory the backends allocate while computing (e.g. the CUDA pool), which a larger compute buffer would take
    constexpr size_t margin = 512ull*1024*1024;

    // the measure may release a graph allocator buffer that is too small for this graph, the free memory then includes it
    std::vector<size_t> cur(backend_ptrs.size(), 0);
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        cur[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]);
    }

    std::vector<size_t> sizes(backend_ptrs.size(), 0);
    moe_cache_seed = true;
    graph_reserve(n_tokens, n_seqs, n_outputs, mctx, /*split_only =*/ true, sizes.data());
    moe_cache_seed = false;

    bool fits = true;
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        const size_t staging = std::min(sizes[i], ggml_backend_sched_get_staging_size(sched.get(), backend_ptrs[i]));
        const size_t graph   = sizes[i] - staging;
        const size_t cur_gph  = cur[i] - std::min(cur[i], staging_pp[i]);
        const size_t released = cur[i] - std::min(cur[i], ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]));
        ggml_backend_dev_t dev = ggml_backend_get_device(backend_ptrs[i]);
        if (graph <= cur_gph || dev == nullptr || ggml_backend_buft_is_host(backend_buft[i])) {
            continue;
        }
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        free -= std::min(free, released);
        free -= std::min(free, i < mem_reserve.size() ? mem_reserve[i] : (size_t) 0);
        // k11: the cache memory kept for layer-major graphs (common, LLAMA_MOE_LAYER_MAJOR_RESERVE_*) is not free for the seed
        // (LLAMA_MOE_LAYER_MAJOR_RESERVE_SEED=1: counted as free, as before)
        static const size_t lm_reserved = getenv("LLAMA_MOE_LAYER_MAJOR_RESERVED_BYTES") != nullptr &&
            (getenv("LLAMA_MOE_LAYER_MAJOR_RESERVE_SEED") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_RESERVE_SEED")) == 0) ?
            (size_t) atoll(getenv("LLAMA_MOE_LAYER_MAJOR_RESERVED_BYTES")) : 0;
        free -= std::min(free, lm_reserved);
        if (free < graph - cur_gph + margin) {
            LLAMA_LOG_WARN("%s: %s: the ubatch that seeds the MoE cache needs %.2f MiB more compute buffer, %.2f MiB free (with %.0f MiB of headroom): "
                "prompts do not seed the cache from their own ids (a smaller --moe-cache-mib keeps the seed)\n", __func__,
                ggml_backend_buft_name(backend_buft[i]), (graph - cur_gph)/1024.0/1024.0, free/1024.0/1024.0, margin/1024.0/1024.0);
            fits = false;
        }
    }
    if (!fits) {
        return false;
    }

    moe_cache_seed = true;
    ggml_cgraph * gf = graph_reserve(n_tokens, n_seqs, n_outputs, mctx);
    moe_cache_seed = false;
    if (!gf) {
        LLAMA_LOG_WARN("%s: the ubatch that seeds the MoE cache does not fit, prompts do not seed the cache from their own ids\n", __func__);
        return false;
    }
    return true;
}

static bool llama_sync_skip_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_SYNC_SKIP");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

void llama_context::synchronize_outputs() {
    if (!sync_pending && llama_sync_skip_enabled()) {
        return;
    }
    synchronize();
}

void llama_context::synchronize() {
    if (!sched) {
        state_async_wait();
        sync_pending = false;
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    state_async_wait();

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;

    sync_pending = false;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        // (k11: the kept small graphs stay: a swap-in allocates them in the scheduler again and checks them with can_reuse; a recurrent
        // rollback after a rejected draft comes here every time, LLAMA_GRAPH_SMALL_KEEP_UPDATE=0 drops them as before)
        static const bool keep_update = getenv("LLAMA_GRAPH_SMALL_KEEP_UPDATE") == nullptr || atoi(getenv("LLAMA_GRAPH_SMALL_KEEP_UPDATE")) != 0;
        gf_res_prev_active = nullptr;
        if (!keep_update) {
            gf_res_epoch++;
        }

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

int64_t llama_context::nextn_rows_wait(int64_t n) {
    // only rows of a single-sequence order (no output swaps to apply) and only rows the event covers
    if (n <= 0 || n > rows_ev_n || !rows_ev || !output_swaps.empty() || !embd_nextn.data) {
        return 0;
    }
    ggml_backend_event_synchronize(rows_ev.get());
    return rows_ev_n;
}

float * llama_context::nextn_rows_early() {
    return output_swaps.empty() ? embd_nextn.data : nullptr;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the toggle changes the graph shape (pooled: logits only; pooling NONE: the reserve graph outputs all tokens,
    // which crashed ggml_gallocr on the first /completion), so re-reserve on every toggle
    if (cparams.embeddings != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_verify(bool value) {
    cparams.verify = value;
}

void llama_context::set_verify_exact(bool value) {
    cparams.verify_exact = value;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

//
// LLAMA_MTP_DEV_H: device-resident MTP handoff (see llama_mtp_dev_h in llama-context.h)
//

llama_mtp_dev_h::~llama_mtp_dev_h() {
    LLAMA_LOG_INFO("%s: MTP handoff on the device: %" PRId64 " drafter passes read %" PRId64 " rows from the sink\n", __func__, n_batches, n_rows_dev);
    if (tgt && tgt->mtp_dev_h_sink == this) {
        tgt->mtp_dev_h_sink = nullptr;
    }
    if (ev_tgt) { ggml_backend_event_free(ev_tgt); }
    if (ev_dft) { ggml_backend_event_free(ev_dft); }
    if (buf)    { ggml_backend_buffer_free(buf); }
    if (meta)   { ggml_free(meta); }
}

// a device-to-device copy of rows [r_src, r_src + n) of src to rows [r_dst, r_dst + n) of dst on `backend`'s stream
static void mtp_dev_h_copy_rows(ggml_backend_t backend, ggml_tensor * src, int64_t r_src, ggml_tensor * dst, int64_t r_dst, int64_t n) {
    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_view_2d(ctx, src, src->ne[0], n, src->nb[1], r_src*src->nb[1]);
    ggml_tensor * b = ggml_view_2d(ctx, dst, dst->ne[0], n, dst->nb[1], r_dst*dst->nb[1]);
    a->buffer = nullptr; a->data = nullptr;
    b->buffer = nullptr; b->data = nullptr;
    GGML_ASSERT(ggml_backend_view_init(a) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_view_init(b) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_copy_async(backend, backend, a, b);
    ggml_free(ctx);
}

bool llama_context::mtp_dev_h_attach(llama_context * tgt, int32_t n_rows) {
    if (!tgt || tgt == this || n_rows <= 0 || mtp_dev_h_own) {
        return false;
    }
    switch (model.arch) {
        case LLM_ARCH_QWEN35MOE: case LLM_ARCH_QWEN35: case LLM_ARCH_QWEN3NEXT: case LLM_ARCH_QWEN4EXP: break;
        default: return false; // their MTP graphs read h through build_inp_h_state
    }
    // the sink lives on this context's first GPU backend
    ggml_backend_t bk = nullptr;
    for (auto & b : backends) {
        auto t = ggml_backend_dev_type(ggml_backend_get_device(b.get()));
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            bk = b.get();
            break;
        }
    }
    if (!bk) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(bk);
    auto st = std::make_unique<llama_mtp_dev_h>();
    st->tgt = tgt; st->dft = this; st->dev = dev; st->backend_dft = bk; st->n_rows = n_rows;
    if (const char * e = getenv("LLAMA_MTP_DEV_H_MIN")) {
        st->min_rows = std::max(1, atoi(e));
    }
    st->ev_tgt = ggml_backend_event_new(dev);
    st->ev_dft = ggml_backend_event_new(dev);
    if (!st->ev_tgt || !st->ev_dft) {
        LLAMA_LOG_INFO("%s: MTP handoff stays on the host: %s has no events\n", __func__, ggml_backend_dev_name(dev));
        return false;
    }
    const int64_t n_embd = model.hparams.n_embd_out();
    const size_t bytes = (size_t) n_embd * (n_rows + 1) * sizeof(float);
    size_t mfree = 0, mtotal = 0;
    ggml_backend_dev_memory(dev, &mfree, &mtotal);
    const size_t margin = (size_t) (getenv("LLAMA_MTP_DEV_H_MARGIN_MIB") ? std::max(0, atoi(getenv("LLAMA_MTP_DEV_H_MARGIN_MIB"))) : 512) << 20;
    if (mfree < bytes + margin) {
        LLAMA_LOG_INFO("%s: MTP handoff stays on the host: its %.1f MiB device buffer does not fit beside a %.0f MiB margin (%.1f MiB free, LLAMA_MTP_DEV_H_MARGIN_MIB)\n",
                __func__, bytes/1048576.0, margin/1048576.0, mfree/1048576.0);
        return false;
    }
    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    st->meta = ggml_init(ip);
    st->sink = ggml_new_tensor_2d(st->meta, GGML_TYPE_F32, n_embd, n_rows + 1);
    ggml_set_name(st->sink, "mtp_h_sink");
    st->buf = ggml_backend_alloc_ctx_tensors_from_buft(st->meta, ggml_backend_get_default_buffer_type(bk));
    if (!st->buf || ggml_backend_buffer_is_host(st->buf)) {
        return false;
    }
    ggml_backend_buffer_clear(st->buf, 0);
    // the scheduler runs the drafter's get_rows of the sink on the sink's device (as it does for weights)
    ggml_backend_buffer_set_usage(st->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (tgt->mtp_dev_h_sink) {
        return false;
    }
    tgt->mtp_dev_h_sink = st.get();
    LLAMA_LOG_INFO("%s: MTP handoff on the device: %.1f MiB on %s for batches of %d to %d rows (LLAMA_MTP_DEV_H=0 keeps the host copies)\n",
            __func__, bytes/1048576.0, ggml_backend_dev_name(dev), st->min_rows, n_rows);
    mtp_dev_h_own = std::move(st);
    return true;
}

int32_t llama_context::mtp_dev_h_last_rows() const {
    return mtp_dev_h_sink ? mtp_dev_h_sink->last_rows : 0;
}

bool llama_context::mtp_dev_h_begin() {
    auto * st = mtp_dev_h_own.get();
    if (!st || !st->tgt) {
        return false;
    }
    if (st->ev_tgt_rec) {
        ggml_backend_event_wait(st->backend_dft, st->ev_tgt);
    }
    mtp_dev_h_src = st->sink;
    return true;
}

void llama_context::mtp_dev_h_end(int32_t pending_row) {
    auto * st = mtp_dev_h_own.get();
    mtp_dev_h_src = nullptr;
    if (!st) {
        return;
    }
    // the next batch's first row: the last target row of this one (the target only writes rows >= 1)
    if (pending_row > 0) {
        mtp_dev_h_copy_rows(st->backend_dft, st->sink, pending_row, st->sink, 0, 1);
        st->n_batches++;
        st->n_rows_dev += pending_row;
    }
    ggml_backend_event_record(st->ev_dft, st->backend_dft);
    st->ev_dft_rec = true;
}

bool llama_context::mtp_dev_h_get_row(int32_t row, float * dst) {
    auto * st = mtp_dev_h_own.get();
    if (!st || row < 0 || row > st->n_rows) {
        return false;
    }
    // the draft stream waited the target's writes before its last use: its completion covers both
    ggml_backend_synchronize(st->backend_dft);
    ggml_backend_tensor_get(st->sink, dst, (size_t) row*st->sink->nb[1], st->sink->nb[1]);
    return true;
}

bool llama_context::mtp_dev_h_set_row(int32_t row, const float * src) {
    auto * st = mtp_dev_h_own.get();
    if (!st || row < 0 || row > st->n_rows) {
        return false;
    }
    ggml_backend_synchronize(st->backend_dft); // no draft graph still reads the row
    ggml_backend_tensor_set(st->sink, src, (size_t) row*st->sink->nb[1], st->sink->nb[1]);
    return true;
}

void llama_context::mtp_dev_h_decode_begin(const llama_batch_ext & batch_inp) {
    mtp_dev_h_this = false;
    mtp_dev_h_waited = false;
    mtp_dev_h_done = 0;
    mtp_dev_h_runs.clear();
    auto * st = mtp_dev_h_sink;
    if (!st) {
        return;
    }
    st->last_rows = 0;
    const int64_t n = (int64_t) batch_inp.tokens.size();
    if (!st->usable || !cparams.embeddings_nextn || cparams.embeddings_nextn_masked || cparams.pooling_type != LLAMA_POOLING_TYPE_NONE ||
            n < st->min_rows || n > st->n_rows || !embd_nextn.data) {
        return;
    }
    // one sequence: the drafter's rows are then the batch rows shifted by one (multi-sequence batches keep the host path)
    llama_seq_id s0 = -1;
    for (const auto & t : batch_inp.tokens) {
        if (t.seq_ids.size() != 1) {
            return;
        }
        const llama_seq_id s = *t.seq_ids.begin();
        if (s0 < 0) {
            s0 = s;
        } else if (s != s0) {
            return;
        }
    }
    mtp_dev_h_this = true;
}

bool llama_context::mtp_dev_h_store(ggml_backend_t backend_h, ggml_tensor * t_h_nextn, const llama_ubatch & ubatch, int64_t offset) {
    auto * st = mtp_dev_h_sink;
    const int64_t n_embd = model.hparams.n_embd_out();
    bool ok = st && backend_h && t_h_nextn && t_h_nextn->buffer && ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens &&
              ggml_backend_get_device(backend_h) == st->dev && !ggml_backend_buffer_is_host(t_h_nextn->buffer) &&
              t_h_nextn->type == GGML_TYPE_F32 && t_h_nextn->ne[0] == n_embd && t_h_nextn->ne[1] >= (int64_t) ubatch.n_tokens &&
              t_h_nextn->nb[0] == sizeof(float) && t_h_nextn->nb[1] == (size_t) n_embd*sizeof(float);
    if (!ok) {
        if (st) {
            st->usable = false;
            LLAMA_LOG_WARN("%s: MTP handoff back on the host: this graph's h_nextn cannot be written to the device sink\n", __func__);
            // the rows of this decode already on the device come back to the host buffer
            for (const auto & r : mtp_dev_h_runs) {
                ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * a = ggml_view_2d(ctx, st->sink, st->sink->ne[0], r[2], st->sink->nb[1], (1 + r[1])*st->sink->nb[1]);
                a->buffer = nullptr; a->data = nullptr;
                GGML_ASSERT(ggml_backend_view_init(a) == GGML_STATUS_SUCCESS);
                ggml_backend_tensor_get_async(backend_h, a, embd_nextn.data + r[0]*n_embd, 0, ggml_nbytes(a));
                ggml_free(ctx);
            }
            st->last_rows = 0;
        }
        mtp_dev_h_this = false;
        return false;
    }
    if (!mtp_dev_h_waited && st->ev_dft_rec) {
        // the drafter has read the rows this decode overwrites
        ggml_backend_event_wait(backend_h, st->ev_dft);
    }
    mtp_dev_h_waited = true;
    const auto & bi = ubatch.data->batch_idxs;
    for (size_t i = 0; i < bi.size(); ) {
        size_t j = i + 1;
        while (j < bi.size() && bi[j] == bi[j - 1] + 1) {
            ++j;
        }
        GGML_ASSERT(bi[i] >= 0 && bi[j - 1] < st->n_rows);
        mtp_dev_h_copy_rows(backend_h, t_h_nextn, (int64_t) i, st->sink, 1 + (int64_t) bi[i], (int64_t) (j - i));
        mtp_dev_h_runs.push_back({ offset + (int64_t) i, (int64_t) bi[i], (int64_t) (j - i) });
        i = j;
    }
    ggml_backend_event_record(st->ev_tgt, backend_h);
    st->ev_tgt_rec = true;
    mtp_dev_h_done += (int32_t) ubatch.n_tokens;
    st->last_rows = mtp_dev_h_done;
    return true;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
    //sched_need_reserve = true;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // hybrid MoE cache: the fetch share is measured with the threads of the small batches, before their first graph is built
    // (CPU misses: at the first ubatch of any size, with the decode threads, so that the first prompt can use the host path)
    if (moe_cache && moe_cache->hybrid_probe_pending() && (ubatch.n_tokens <= llama_moe_cache::n_small_max || moe_cache->cpu_miss_enabled()) &&
            backend_cpu != nullptr) {
        ggml_backend_sched_synchronize(sched.get());
        // the calibration times the host with the threads that compute the missing experts (k11 host threads when on)
        compute_n_tokens = -1;
        set_compute_threads(moe_cache->cpu_miss_enabled() ? false : ubatch.n_tokens > 1);
        compute_n_tokens = 0;
        moe_cache->hybrid_probe(backend_cpu);
        ggml_backend_sched_set_overlap(sched.get(), moe_cache->hybrid_active());
    }
    // fetch overlap auto: trials alternate single token ubatches with and without the overlap (the results are the same)
    if (moe_cache && moe_cache->overlap_trial_pending()) {
        if (moe_cache->overlap_tick_sync()) {
            ggml_backend_sched_synchronize(sched.get());
        }
        moe_cache->overlap_tick(ubatch.n_tokens);
    }
    // auto mode: the first small batches alternate between the plain cache and the hybrid path, then the faster one is kept
    if (moe_cache && moe_cache->hybrid_window_pending()) {
        if (moe_cache->hybrid_tick_sync()) {
            ggml_backend_sched_synchronize(sched.get());
        }
        moe_cache->hybrid_tick(ubatch.n_tokens);
        ggml_backend_sched_set_overlap(sched.get(), moe_cache->hybrid_active());
    }

    const int64_t t_tr0 = llama_mtp_handoff_trace() && ubatch.n_tokens >= 64 ? ggml_time_us() : 0;
    int64_t t_tr_graph = 0;
    int64_t t_tr_set   = 0;

    // k11-decode: LLAMA_UBATCH_TIMING=1 logs the host time of each small ubatch (graph build + alloc, inputs, compute launch)
    static const bool ub_timing = getenv("LLAMA_UBATCH_TIMING") != nullptr && atoi(getenv("LLAMA_UBATCH_TIMING")) != 0;
    const int64_t t_ub0 = ub_timing ? ggml_time_us() : 0;
    bool ub_reused = false;

    // k11-decode: a small output ubatch uses the kept graph of its size (see gf_res_small)
    // (only while n-gram drafting verifies exactly, see llama_set_verify_exact: without drafts the kept graphs bought nothing and cost
    // the next short prompt 7-8% on G4; LLAMA_GRAPH_SMALL_CACHE=2 keeps them always, =0 never)
    static const int  small_cache_mode = getenv("LLAMA_GRAPH_SMALL_CACHE") == nullptr ? 1 : atoi(getenv("LLAMA_GRAPH_SMALL_CACHE"));
    // k13-fn: =3 also with a MoE cache (verify windows of changing size each keep their graph, so their CUDA graphs stay captured)
    // k15-fn: also the MTP drafter's own context (no MoE cache): its single-token draft steps alternate with the catch-up batch of the
    // accepted rows (1 .. n_max + 1 rows, kept per size too), which rebuilt the shared graph every round, so the step's CUDA graph was
    // re-captured instead of replayed
    // (LLAMA_GRAPH_SMALL_CACHE_DRAFT=0: as before)
    static const bool small_cache_draft = getenv("LLAMA_GRAPH_SMALL_CACHE_DRAFT") == nullptr || atoi(getenv("LLAMA_GRAPH_SMALL_CACHE_DRAFT")) != 0;
    const bool small_cache_dft = small_cache_mode == 1 && small_cache_draft && cparams.embeddings_nextn_masked && moe_cache == nullptr;
    // k15-fn: with the saved allocations (LLAMA_SCHED_SIZE_CACHE=0: off) a target with a drafter keeps its decode and verify graphs
    // per size also with a MoE cache: a window of another size restores its own split, allocation and CUDA graph
    static const bool size_cache = getenv("LLAMA_SCHED_SIZE_CACHE") == nullptr || atoi(getenv("LLAMA_SCHED_SIZE_CACHE")) != 0;
    const bool small_cache_tgt = small_cache_mode == 1 && size_cache && cparams.spec_draft && (ubatch.n_tokens == 1 || cparams.verify);
    const bool small_cache_env = small_cache_mode == 2 || small_cache_mode == 3 || (small_cache_mode == 1 && cparams.verify_exact) || small_cache_dft ||
        small_cache_tgt;
    // (single token decodes and speculative verify windows only: a prompt's short tail keeps the shared graph, as before; G4 LFM2.5
    // 32 GB: with the tails kept too, chat / code prompts were 3 / 9% slower)
    // (the drafter's catch-up batch has no outputs: it only writes the KV of the accepted rows)
    const uint32_t n_small = small_cache_env && (moe_cache == nullptr || small_cache_mode == 3 || small_cache_tgt) && !graph_reuse_disable && (n_outputs > 0 || small_cache_dft) &&
        (ubatch.n_tokens == 1 || cparams.verify || small_cache_dft) && ubatch.n_tokens <= GF_SMALL_MAX ? ubatch.n_tokens : 0;

    llm_graph_result * res = nullptr;
    uint32_t i_small = 0; // the kept graph used: n_small*GF_SMALL_WAYS + way
    if (n_small > 0) {
        int way = -1;
        for (uint32_t w = 0; w < GF_SMALL_WAYS && way < 0; ++w) {
            const auto & r = gf_res_small[n_small*GF_SMALL_WAYS + w];
            if (r && r.get() == gf_res_prev_active) {
                way = (int) w;
            }
        }
        for (uint32_t w = 0; w < GF_SMALL_WAYS && way < 0; ++w) {
            const uint32_t i = n_small*GF_SMALL_WAYS + w;
            if (gf_res_small[i] && gf_res_small_built[i] && gf_res_small_epoch[i] == gf_res_epoch &&
                    gf_res_small[i]->can_reuse(graph_params(gf_res_small[i].get(), ubatch, mctx, gtype))) {
                way = (int) w;
            }
        }
        if (way < 0) {
            way = 0;
            for (uint32_t w = 1; w < GF_SMALL_WAYS; ++w) {
                if (gf_res_small_used[n_small*GF_SMALL_WAYS + w] < gf_res_small_used[n_small*GF_SMALL_WAYS + way]) {
                    way = (int) w;
                }
            }
        }
        i_small = n_small*GF_SMALL_WAYS + (uint32_t) way;
        auto & r = gf_res_small[i_small];
        if (!r) {
            r.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
            gf_res_small_built[i_small] = false;
        }
        gf_res_small_used[i_small] = ++n_small_uses;
        res = r.get();
    } else {
        res = get_gf_res_prev();
    }
    auto * gf  = res->get_gf();

    // the routing outputs of the previous graph were read before, if needed (see moe_cache_seed_begin)
    moe_cache_ids.clear();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // the copies of the experts are planned with the seed mode of the MoE cache (see sched_copy_all_experts)
    const int moe_cache_hybrid = moe_cache ? moe_cache->hybrid_mode() : 0;

    // k11-decode: a kept small graph built for these parameters goes back into the scheduler: only its allocation is redone (the same
    // graph allocates to the same addresses, so its CUDA graph is replayed as captured)
    auto small_swap_in = [&]() -> bool {
        if (n_small == 0 || gf == nullptr || !gf_res_small_built[i_small] || gf_res_small_epoch[i_small] != gf_res_epoch ||
                moe_cache_seed_graph != moe_cache_seed || moe_cache_hybrid_graph != moe_cache_hybrid || moe_cache_devseed_graph != moe_cache_devseed ||
                moe_inplace_graph != moe_inplace_ok ||
                !res->can_reuse(gparams)) {
            return false;
        }
        gf_res_prev_active = nullptr; // (k11: not an invalidation of the kept small graphs)
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        // k15-fn: the saved split and allocation of this graph: its tensors are as they were after that allocation (no other graph
        // writes them), the split context and tables go back in place
        if (size_cache && gf_res_small_plan[i_small] != nullptr) {
            if (ggml_backend_sched_plan_restore(sched.get(), gf_res_small_plan[i_small])) {
                prefetch_n_tokens  = ubatch.n_tokens;
                gf_res_prev_active = res;
                n_plan_restores++;
                return true;
            }
            small_plan_free(i_small);
            n_plan_invalid++;
        }
        for (const auto & st : gf_res_small_state[i_small]) {
            memcpy(st.t->src, st.src, sizeof(st.src));
            st.t->data   = st.data;
            st.t->buffer = st.buffer;
            st.t->extra  = st.extra;
            st.t->flags  = st.flags;
        }
        prefetch_n_tokens = ubatch.n_tokens;
        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            gf_res_small_built[i_small] = false;
            return false;
        }
        if (size_cache) {
            gf_res_small_plan[i_small] = ggml_backend_sched_plan_save(sched.get());
            n_plan_saves += gf_res_small_plan[i_small] != nullptr;
        }
        gf_res_prev_active = res;
        return true;
    };

    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams) && moe_cache_seed_graph == moe_cache_seed &&
            moe_inplace_graph == moe_inplace_ok &&
            moe_cache_hybrid_graph == moe_cache_hybrid && moe_cache_devseed_graph == moe_cache_devseed) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
        ub_reused = true;
    } else if (small_swap_in()) {
        n_small_swaps++;
        ub_reused = true;
    } else {
        gf_res_prev_active = nullptr; // (k11: not an invalidation of the kept small graphs)
        if (n_small > 0) {
            small_plan_free(i_small); // its tensors go now
        }
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        // verify batches computed batch invariant: every backend sees the flag on the nodes (it also keys the CUDA graphs)
        if (cparams.batch_invariant) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_graph_node(gf, i)->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
            }
        }

        prefetch_n_tokens = ubatch.n_tokens;
        if (n_small > 0) {
            auto & sv = gf_res_small_state[i_small];
            sv.clear();
            for (ggml_tensor * t = ggml_get_first_tensor(res->get_ctx()); t != nullptr; t = ggml_get_next_tensor(res->get_ctx(), t)) {
                gf_tensor_state st;
                st.t = t;
                memcpy(st.src, t->src, sizeof(st.src));
                st.data   = t->data;
                st.buffer = t->buffer;
                st.extra  = t->extra;
                st.flags  = t->flags;
                sv.push_back(st);
            }
        }
        if (moe_cache) {
            moe_cache->set_graph(gf);
        }
        bool alloc_ok = ggml_backend_sched_alloc_graph(sched.get(), gf);
        // a prompt ubatch with outputs whose graph the MoE cache would lend but cannot (its narrowed last layer reads the cache banks on
        // the host): built again with that layer at full width, so the cache lends its allocation (see llm_graph_context::crop_full_last)
        static const int full_last_env = getenv("LLAMA_MOE_LEND_FULL_LAST") != nullptr ? atoi(getenv("LLAMA_MOE_LEND_FULL_LAST")) : 1;
        if (!alloc_ok && full_last_env == 1 && !moe_full_last && moe_cache && n_outputs > 0 && (uint32_t) n_outputs < ubatch.n_tokens &&
                moe_cache->lends_graph_to(ubatch.n_tokens)) {
            LLAMA_LOG_WARN("%s: a lent prompt graph of %u tokens with %d outputs did not allocate: built again with the last layer at full width "
                "so the MoE cache lends its allocation (from now on; LLAMA_MOE_LEND_FULL_LAST=0 to disable)\n", __func__, ubatch.n_tokens, n_outputs);
            moe_full_last = true;
            res->reset();
            ggml_backend_sched_reset(sched.get());
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
            gf = model.build_graph(graph_params(res, ubatch, mctx, gtype));
            if (gf && cparams.batch_invariant) {
                for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                    ggml_graph_node(gf, i)->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                }
            }
            if (gf && moe_cache) {
                moe_cache->set_graph(gf);
            }
            alloc_ok = gf != nullptr && ggml_backend_sched_alloc_graph(sched.get(), gf);
        }
        if (!alloc_ok) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active     = res;
        if (n_small > 0) {
            gf_res_small_built[i_small] = true;
            gf_res_small_epoch[i_small] = gf_res_epoch;
            if (size_cache) {
                gf_res_small_plan[i_small] = ggml_backend_sched_plan_save(sched.get());
                n_plan_saves += gf_res_small_plan[i_small] != nullptr;
            }
        }
        moe_cache_seed_graph   = moe_cache_seed;
        moe_inplace_graph      = moe_inplace_ok;
        moe_cache_hybrid_graph = moe_cache_hybrid;
        // the seed plans are enabled in the op params of the graph, a reused graph keeps them (and may be replayed as captured)
        moe_cache_devseed_graph = moe_cache_devseed;
    }

    if (t_tr0) {
        t_tr_graph = ggml_time_us();
    }
    const int64_t t_ub1 = ub_timing ? ggml_time_us() : 0;

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        const int64_t t_set_us = ple.stats ? ggml_time_us() : 0;

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        if (ple.stats) {
            ple.t_set_inputs_us += ggml_time_us() - t_set_us;
            ple.n_set_inputs++;
        }

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }
    if (t_tr0) {
        t_tr_set = ggml_time_us();
    }

    if (moe_cache) {
        moe_cache->prepare(ubatch.n_tokens);
        if (moe_cache_devseed_on) {
            moe_cache->devseed_prepare(res->get_gf(), moe_cache_devseed);
        }
    }

    compute_n_tokens = ubatch.n_tokens;
    const int64_t t_ub2 = ub_timing ? ggml_time_us() : 0;
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    compute_n_tokens = 0;
    if (t_tr0) {
        const int64_t t_tr_end = ggml_time_us();
        LLAMA_LOG_INFO("mtp_trace ubatch ctx=%p n=%u state=%d graph=%.2f set_inputs=%.2f compute_launch=%.2f ms\n", (void *) this, ubatch.n_tokens,
                ubatch.embd_state != nullptr, (t_tr_graph - t_tr0)/1e3, (t_tr_set - t_tr_graph)/1e3, (t_tr_end - t_tr_set)/1e3);
    }
    if (ub_timing && ubatch.n_tokens <= 16) {
        const int64_t t_ub3 = ggml_time_us();
        LLAMA_LOG_INFO("%s: ubatch %u tokens %s: graph %.3f ms, inputs %.3f ms, compute launch %.3f ms\n", __func__, ubatch.n_tokens,
                       ub_reused ? "reused" : "built", (t_ub1 - t_ub0)/1000.0, (t_ub2 - t_ub1)/1000.0, (t_ub3 - t_ub2)/1000.0);
    }
    // CPU misses trial: the ubatch is timed to its end, without the work of other contexts
    if (moe_cache && status == GGML_STATUS_SUCCESS && moe_cache->hybrid_tick_timed()) {
        ggml_backend_sched_synchronize(sched.get());
        moe_cache->hybrid_tick_end();
    }
    if (moe_cache && moe_cache_devseed_on && moe_cache_devseed) {
        // a failed graph may have run only some of its seed plans and fills: the groups it seeds are cleared
        moe_cache->devseed_end(res->get_gf(), status == GGML_STATUS_SUCCESS);
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    // an asynchronous state save queued before this call has reached its destination when this call returns
    struct state_async_join {
        llama_context * ctx;
        ~state_async_join() { ctx->state_async_wait(); }
    } state_async_join_guard { this };

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

namespace {
// page faults and wall time of one decode() call, for the PLE prefetch stats
struct ple_decode_meter {
    llama_context::ple_state * ple;
    int64_t t0 = 0;
    int64_t maj0 = 0;
    int64_t min0 = 0;

    static void faults(int64_t & maj, int64_t & min) {
#if defined(__unix__) || defined(__APPLE__)
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            maj = ru.ru_majflt;
            min = ru.ru_minflt;
            return;
        }
#endif
        maj = min = 0;
    }

    int64_t  t_set0 = 0;
    uint64_t n_set0 = 0;
    int64_t  n_tokens = 0;

    explicit ple_decode_meter(llama_context::ple_state * ple, int64_t n_tokens) : ple(ple), n_tokens(n_tokens) {
        if (ple) {
            t0 = ggml_time_us();
            faults(maj0, min0);
            t_set0 = ple->t_set_inputs_us;
            n_set0 = ple->n_set_inputs;
        }
    }

    ~ple_decode_meter() {
        if (ple) {
            int64_t maj, min;
            faults(maj, min);
            const int64_t t = ggml_time_us() - t0;
            ple->t_decode_us += t;
            ple->n_decode++;
            ple->majflt += maj - maj0;
            ple->minflt += min - min0;
            if (ple->stats > 1) {
                LLAMA_LOG_INFO("ple_stats: decode n_tokens = %" PRId64 ", ubatches = %" PRIu64 ", set_inputs = %.2f ms, "
                        "majflt = %" PRId64 ", minflt = %" PRId64 ", wall = %.2f ms\n", n_tokens, ple->n_set_inputs - n_set0,
                        (ple->t_set_inputs_us - t_set0)/1e3, maj - maj0, min - min0, t/1e3);
            }
        }
    }
};
}

// k11 (LLAMA_STATE_ASYNC_LAZY=1, opt-in): an asynchronous state save queued before a decode() call completes at the next synchronize() /
// state call / llama_state_seq_async_wait() instead of when decode() returns. No gain measured (G4 Qwen3.6 32 GB chat / c400 within
// noise): the worker's copy into a fresh checkpoint buffer outlasts the 4-token decode it would overlap, so the sync for its logits waits anyway
static bool state_async_lazy() {
    static const bool on = getenv("LLAMA_STATE_ASYNC_LAZY") != nullptr && atoi(getenv("LLAMA_STATE_ASYNC_LAZY")) != 0;
    return on;
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    // an asynchronous state save queued before this call has reached its destination when this call returns (LLAMA_STATE_ASYNC_LAZY=0)
    struct state_async_join {
        llama_context * ctx;
        ~state_async_join() { if (!state_async_lazy()) { ctx->state_async_wait(); } }
    } state_async_join_guard { this };

    const ple_decode_meter ple_meter(ple.stats ? &ple : nullptr, (int64_t) batch_inp.tokens.size());

    struct handoff_trace {
        const llama_context * ctx;
        int64_t n;
        int64_t t0;
        int64_t t_init  = 0;
        int64_t t_split = 0;
        ~handoff_trace() {
            if (t0) {
                const int64_t t1 = ggml_time_us();
                LLAMA_LOG_INFO("mtp_trace decode ctx=%p n=%" PRId64 " init=%.2f split=%.2f total=%.2f ms\n", (const void *) ctx, n,
                        t_init ? (t_init - t0)/1e3 : 0.0, t_split && t_init ? (t_split - t_init)/1e3 : 0.0, (t1 - t0)/1e3);
            }
        }
    } handoff_tr { this, (int64_t) batch_inp.tokens.size(), llama_mtp_handoff_trace() && batch_inp.tokens.size() >= 64 ? ggml_time_us() : 0 };

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    // non-causal attention needs the whole sequence in one ubatch
    const bool embd_all     = output_all || !cparams.causal_attn;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    // k11 (LLAMA_LM_CKPT_MERGE): the breaks apply to this batch only
    struct ubatch_breaks_reset {
        llama_batch_allocr * b;
        ~ubatch_breaks_reset() { b->ubatch_breaks.clear(); }
    } ubatch_breaks_reset_guard { balloc.get() };
    std::vector<int32_t> ub_breaks = std::move(ubatch_breaks_next);
    ubatch_breaks_next.clear();
    // k15-pp: the captures asked for this decode (llama_set_state_capture); those of an earlier decode are dropped
    bool cap_active = false;
    for (auto & c : cap) {
        const bool act = c.pending;
        c.pending = false;
        if (act || c.done) {
            c.done   = false;
            c.ev_set = false;
            c.regions.clear();
        }
        if (!act) {
            c.idx = -1;
        }
        cap_active |= act;
    }
    cap_sel = -1;
    // k15-pp2: a snapshot is overwritten only after the side stream has read it for the save of an earlier decode
    if (cap_active && cap_side_pending) {
        if (ggml_backend_t be = cap_backend()) {
            ggml_backend_event_wait(be, cap_side_ev);
        }
        cap_side_pending = false;
    }
    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    mtp_dev_h_decode_begin(batch_inp);

    if (handoff_tr.t0) {
        handoff_tr.t_init = ggml_time_us();
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();
    balloc->ubatch_breaks = std::move(ub_breaks);
    static const bool tail_dbg_b = getenv("LLAMA_LM_TAIL_DEBUG") != nullptr;
    if (tail_dbg_b && n_tokens_all > 8) {
        const auto & v = batch_inp.tokens;
        std::string t;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i < 2 || i + 3 >= v.size()) {
                t += " " + std::to_string(v[i].id) + "@" + std::to_string(v[i].pos[0]) + (v[i].output ? "*" : "");
            }
        }
        LLAMA_LOG_INFO("%s: tail debug batch (%u tokens, %u outputs, ctx %p):%s\n", __func__, n_tokens_all, n_outputs_all, (void *) this, t.c_str());
    }
    if (tail_dbg_b && n_tokens_all <= 8) {
        std::string t;
        for (const auto & tk : batch_inp.tokens) {
            t += " " + std::to_string(tk.id) + "@" + std::to_string(tk.pos[0]) + (tk.output ? "*" : "");
        }
        LLAMA_LOG_INFO("%s: tail debug batch (%u tokens, %u outputs, ctx %p):%s\n", __func__, n_tokens_all, n_outputs_all, (void *) this, t.c_str());
    }

    // hybrid MoE cache with a fixed share: a new request (every sequence of the batch is empty) starts from an empty cache
    if (moe_cache && moe_cache->hybrid_fixed()) {
        bool fresh = true;
        for (const auto & tok : batch_inp.tokens) {
            for (auto seq_id : tok.seq_ids) {
                fresh = fresh && memory->seq_pos_max(seq_id) < 0;
            }
            if (!fresh) {
                break;
            }
        }
        if (fresh) {
            moe_cache->hybrid_reset();
        // the reset takes back the lent slots without a new allocation: a reused graph would keep its copies bound to them
        gf_res_prev_active = nullptr; gf_res_epoch++;
        }
    }

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    // k11-decode: a prompt of several ubatches after a small graph re-planned the graph allocator restores the worst-case pp plan once
    // (see restore_pp_plan); the server restores it when its slots go idle, so this is the fallback
    if (n_tokens_all > cparams.n_ubatch && !cparams.verify) {
        restore_pp_plan();
    }

    llama_memory_context_ptr mctx;

    // speculative verify batches with LLAMA_MTP_EXACT=1 are computed batch invariant (see llama_set_verify)
    const bool exact_verify = cparams.verify && (llama_exact_verify_env() || cparams.verify_exact) && n_tokens_all > 1 && cparams.causal_attn && !output_all;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, embd_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    if (handoff_tr.t0) {
        handoff_tr.t_split = ggml_time_us();
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // k15: a prompt batch (some tokens without an output) keeps its ubatches off the CPU-miss host path of a pinned cache (see
    // llama_moe_cache::set_prompt_batch); generation and verify batches (an output for every token) keep it
    struct moe_prompt_mark {
        llama_moe_cache * mc;
        moe_prompt_mark(llama_moe_cache * mc, bool prompt) : mc(mc) { if (mc) { mc->set_prompt_batch(prompt); } }
        ~moe_prompt_mark() { if (mc) { mc->set_prompt_batch(false); } }
    } moe_prompt_guard(moe_cache.get(), (int64_t) n_outputs_all < (int64_t) n_tokens_all);

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;
    rows_ev_n = 0; // k15-pp3: no early nextn rows from this decode yet

    for (bool more = true; more; ) {
        // k15-pp: the state before the ubatch that starts at the capture index (a layer-major group captures inside its graph)
        for (int s = 0; s < cap_max; ++s) {
            if (cap[s].idx >= 0 && !cap[s].done && n_tokens_prev == cap[s].idx) {
                cap_now(s, mctx->get_ubatch().pos[0] - 1);
            }
        }
        // layer-major prompts: the next ubatches as one graph; afterwards `more` says whether mctx is at a ubatch not processed yet
        if (lm_enabled) {
            int lm_ret = 0;
            const int n_lm = decode_layer_major(mctx.get(), n_outputs_all, n_tokens_all, n_outputs_prev, n_tokens_prev, more, lm_ret);
            if (n_lm < 0) {
                return lm_ret;
            }
            if (n_lm > 0) {
                continue;
            }
        }
        {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        // the PLE rows of the next ubatch (and of this one at the first) while this one is processed
        if (ple.enabled) {
            ple_prefetch_ahead(ubatch, mctx.get(), n_tokens_prev == 0);
        }
        // a ubatch that outputs only some of its tokens ends a prompt: the generation that follows prefetches nothing, so the weight
        // store does not read the next graph's copies ahead
        if (wstore) {
            wstore->set_wrap(!(n_outputs > 0 && n_outputs < ubatch.n_tokens));
        }

        // the MoE cache only serves batches of up to 32 tokens, the end of a prompt seeds it for the generation that follows
        if (moe_cache) {
            // CPU misses: the rule decides per batch size from the miss rate of the device lookups (see llama_moe_cache::cpu_miss_tick),
            // before anything of this ubatch depends on the decision
            if (moe_cache->cpu_miss_tick_sync(ubatch.n_tokens)) {
                ggml_backend_sched_synchronize(sched.get());
            }
            const bool was = moe_cache->hybrid_active();
            moe_cache->cpu_miss_tick(ubatch.n_tokens);
            if (moe_cache->hybrid_active() != was) {
                ggml_backend_sched_set_overlap(sched.get(), moe_cache->hybrid_active());
            }
            const bool large      = !moe_cache->small_batch(ubatch.n_tokens);
            const bool last_large = large && n_tokens_all - n_tokens_prev - ubatch.n_tokens <= llama_moe_cache::n_small_max;
            moe_cache_seed_begin(ubatch, last_large, n_outputs);
            if (large) {
                moe_cache->count_large_ubatch();
            }
        }

        cparams.batch_invariant = exact_verify && ubatch.n_tokens > 1;
        cparams.exact_split     = cparams.batch_invariant ? llm_graph_exact_split(ubatch) : 0;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        cparams.batch_invariant = false;
        cparams.exact_split     = 0;

        if (moe_cache) {
            moe_cache_seed_end(ubatch, res ? n_outputs : -1);
        }

        if (res && ple.verify) {
            ple_prefetch_verify(ubatch, mctx.get());
        }

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                if (!(mtp_dev_h_this && mtp_dev_h_store(backend_h, t_h_nextn, ubatch, offset))) {
                    ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                }
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
        }
        more = mctx->next();
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

void llama_context::lm_seed_resolve(lm_seed_pending & p) {
    auto & L = *p.L;
    const int64_t n_expert = L.w_first->ne[2];
    std::vector<int32_t> ids_data(ggml_nelements(L.ids));
    if (p.ev >= 0) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_event_synchronize(lm_async_ev[p.ev].get());
        lm_n_async_wait_us += ggml_time_us() - t0;
        memcpy(ids_data.data(), (const uint8_t *) ggml_backend_buffer_get_base(lm_async_buf.get()) + p.off, ggml_nbytes(L.ids));
    } else {
        ggml_backend_tensor_get(L.ids, ids_data.data(), 0, ggml_nbytes(L.ids));
    }
    std::vector<int32_t> counts(n_expert, 0);
    for (const int32_t id : ids_data) {
        GGML_ASSERT(id >= 0 && id < n_expert);
        counts[id]++;
    }
    moe_cache->set_seed_counts(L.w_first, counts);
    // the expert copies of the layer queued so far, in the order the synchronous seed issues them (the later ones seed when issued)
    for (const ggml_tensor * w : L.ws) {
        const auto st = lm_staged.find(w);
        if (st == lm_staged.end() || lm_seeded.count(w) > 0) {
            continue;
        }
        const std::vector<int32_t> * c = moe_cache->get_seed_counts(w);
        if (c != nullptr && !c->empty()) {
            lm_seeded.insert(w);
            moe_cache->seed(st->second.first, w, st->second.second, *c, std::vector<bool>(w->ne[2], true));
        }
    }
}

// a layer seeds (waiting for its routing, which the device computed by then or soon) once all of its expert copies are queued, or once the
// copies of a later split are queued after them (their seed must be queued before a copy that may reuse the staging slot: before the copies
// of the split after that one); pre: before the copies of a split, only the layers staged two or more splits back
void llama_context::lm_seed_resolve_due(bool pre) {
    for (size_t i = 0; i < lm_pending.size();) {
        auto & p = lm_pending[i];
        int64_t seq = -1;
        bool all = true;
        for (const ggml_tensor * w : p.L->ws) {
            const auto it = lm_staged_seq.find(w);
            if (it == lm_staged_seq.end()) {
                all = false;
            } else if (seq < 0 || it->second < seq) {
                seq = it->second;
            }
        }
        const bool due = seq >= 0 && (pre ? seq < lm_copy_seq - 1 : (all || seq < lm_copy_seq));
        if (!due) {
            i++;
            continue;
        }
        lm_seed_resolve(p);
        lm_pending.erase(lm_pending.begin() + i);
    }
}

void llama_context::lm_seed_resolve_all() {
    for (auto & p : lm_pending) {
        lm_seed_resolve(p);
    }
    lm_pending.clear();
    lm_async_used = 0;
    lm_async_nev  = 0;
}

bool llama_context::lm_eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);
    const auto it = lctx->lm_ask.find(t);
    const bool mine = it != lctx->lm_ask.end() && !it->second.done;
    if (ask) {
        return mine || (lctx->cparams.cb_eval && lctx->cparams.cb_eval(t, true, lctx->cparams.cb_eval_user_data));
    }
    if (mine && lctx->lm_seed_async) {
        // k15-pp: the nodes up to t are queued, not computed: read the routing behind them, the copy callback seeds the layer
        auto & L = it->second;
        L.done = true;
        const size_t nb = ggml_nbytes(L.ids);
        ggml_backend_t be = ggml_backend_sched_get_tensor_backend(lctx->sched.get(), L.ids);
        const size_t off = GGML_PAD(lctx->lm_async_used, 64);
        const int ev = lctx->lm_async_nev;
        if (be != nullptr && ggml_backend_get_device(be) == lctx->lm_async_dev && lctx->lm_async_buf &&
                off + nb <= ggml_backend_buffer_get_size(lctx->lm_async_buf.get()) &&
                ev < (int) lctx->lm_async_ev.size()) {
            ggml_backend_tensor_get_async(be, L.ids, (uint8_t *) ggml_backend_buffer_get_base(lctx->lm_async_buf.get()) + off, 0, nb);
            ggml_backend_event_record(lctx->lm_async_ev[ev].get(), be);
            lctx->lm_async_used = off + nb;
            lctx->lm_async_nev++;
            lctx->lm_pending.push_back({ &L, ev, off, be });
            lctx->lm_n_async++;
            lctx->lm_seed_resolve_due(false); // the layer's copies may all be queued already
            return true;
        }
        // no room (not expected): read it now
        ggml_backend_sched_synchronize(lctx->sched.get());
        lctx->lm_pending.push_back({ &L, -1, 0, nullptr });
        lctx->lm_seed_resolve(lctx->lm_pending.back());
        lctx->lm_pending.pop_back();
        return true;
    }
    if (mine) {
        // the scheduler synchronized the backend after t: the routing of the source ubatch for this layer is computed
        auto & L = it->second;
        L.done = true;
        const ggml_tensor * ids = L.ids;
        const int64_t n_expert = L.w_first->ne[2];
        std::vector<int32_t> ids_data(ggml_nelements(ids));
        ggml_backend_tensor_get(ids, ids_data.data(), 0, ggml_nbytes(ids));
        std::vector<int32_t> counts(n_expert, 0);
        for (const int32_t id : ids_data) {
            GGML_ASSERT(id >= 0 && id < n_expert);
            counts[id]++;
        }
        lctx->moe_cache->set_seed_counts(L.w_first, counts);
        // the expert copies of the layer issued so far: their seeds go on the copy queue behind them (the later ones seed when issued)
        for (const ggml_tensor * w : L.ws) {
            const auto st = lctx->lm_staged.find(w);
            if (st == lctx->lm_staged.end() || lctx->lm_seeded.count(w) > 0) {
                continue;
            }
            const std::vector<int32_t> * c = lctx->moe_cache->get_seed_counts(w);
            if (c != nullptr && !c->empty()) {
                lctx->lm_seeded.insert(w);
                lctx->moe_cache->seed(st->second.first, w, st->second.second, *c, std::vector<bool>(w->ne[2], true));
            }
        }
    }
    if (lctx->cparams.cb_eval && lctx->cparams.cb_eval(t, true, lctx->cparams.cb_eval_user_data)) {
        return lctx->cparams.cb_eval(t, false, lctx->cparams.cb_eval_user_data);
    }
    return true;
}

//
// layer-major prompts (LLAMA_MOE_LAYER_MAJOR=1)
//
// A prompt of K ubatches normally runs K graphs one after the other, and every graph copies every host expert it uses to the device: the
// experts that the MoE cache does not hold cross PCIe K times per prompt. Here the graphs of up to lm_ub_max consecutive ubatches are built
// as usual (each with its own inputs, KV view lengths and kernels) and merged into one graph, layer by layer: the nodes of each graph are cut
// before every first use of a host weight, and the merged graph runs piece s of every ubatch before piece s+1 of any. A host weight is then
// used only by the pieces that follow its cut, all in the split that copies it (ggml_backend_sched_set_reuse_weight_copies), so it is copied
// once for the group. Within a piece the ubatches run in order, so the KV rows and recurrent states one ubatch writes in a layer are there
// when the next ubatch of the group reads them in that layer, exactly as in the ubatch-by-ubatch order. Nothing of the math changes: the same
// ops on the same shapes in the same order per ubatch (only interleaved with the other ubatches), so the outputs are bit-identical.
// The prompt does not change the MoE cache (no seed, no lent slots): the decode after it finds the cache as it was.
// k13-small: the hybrid gate's fixed share (a cache of at most LLAMA_MOE_CACHE_HYBRID_MAX_FRAC of the host experts) computes the misses of
// small batches only, the prompt ubatches a layer-major group runs copy their experts as before. A cache below LLAMA_MOE_LAYER_MAJOR_HYBRID_FRAC
// (0.05) of the host experts runs its prompts layer-major too: G4 at 4 GB, Qwen3.6 Q3_K_M + MTP (0.7% cached) copied every host expert once
// per 512-token ubatch, 55 GB of copies for a 2k prompt (PolyStrata: 13.6 GB, one per prompt). LLAMA_MOE_LAYER_MAJOR_HYBRID_FRAC=0: k12
static bool lm_hybrid_frac_ok(double frac) {
    static const double max_frac = getenv("LLAMA_MOE_LAYER_MAJOR_HYBRID_FRAC") ? std::max(0.0, atof(getenv("LLAMA_MOE_LAYER_MAJOR_HYBRID_FRAC"))) : 0.05;
    return frac >= 0.0 && frac < max_frac;
}

// a cache below LLAMA_MOE_LM_RELEASE_FRAC (0.05) of the host experts (k13-small: see the release of the banks in decode_layer_major)
static bool lm_tiny_cache(double frac) {
    static const double max_frac = getenv("LLAMA_MOE_LM_RELEASE_FRAC") ? std::max(0.0, atof(getenv("LLAMA_MOE_LM_RELEASE_FRAC"))) : 0.05;
    return frac >= 0.0 && frac < max_frac;
}

int llama_context::decode_layer_major(llama_memory_context_i * mctx, int64_t n_outputs_all, int64_t n_tokens_all, int64_t & n_outputs_prev,
        int64_t & n_tokens_prev, bool & more, int & ret) {
    auto lm_hybrid_ok = [&]() { return lm_hybrid_frac_ok(moe_cache->cache_frac()); };
    const auto & hparams = model.hparams;
    const int64_t n_vocab = model.vocab.n_tokens();
    const int64_t lm_t_start = ggml_time_us(); // the CUDA graphs used from here on are the group's (evicted with its host memory)

    // ubatches that can run layer-major: prompt ubatches whose host experts are prefetched whole (the copy is the same for every ubatch)
    auto eligible = [&](const llama_ubatch & ub) {
        return ub.n_tokens > llama_moe_cache::n_small_max && (int64_t) ub.n_tokens >= prefetch_min_batch && ub.n_seqs_unq == 1;
    };
    // a merged graph of two ubatches was refused for its compute buffer growth: every graph would run alone, as usual
    if (lm_refused_min <= 2) {
        return 0;
    }
    if (!moe_cache || cparams.embeddings || !cparams.causal_attn || cparams.verify || ple.enabled ||
            std::any_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(), [](bool b) { return b; }) ||
            n_tokens_all - n_tokens_prev <= (int64_t) cparams.n_ubatch + llama_moe_cache::n_small_max || !eligible(mctx->get_ubatch()) ||
            moe_cache->lends_graph_to(mctx->get_ubatch().n_tokens) || (moe_cache->hybrid_fixed() && !lm_hybrid_ok()) ||
            (moe_cache_devseed_on && !lm_devseed)) {
        if (lm_verbose && moe_cache && n_tokens_all - n_tokens_prev > (int64_t) cparams.n_ubatch + llama_moe_cache::n_small_max) {
            const auto & ub = mctx->get_ubatch();
            LLAMA_LOG_INFO("%s: not layer-major: samplers %d embeddings %d causal %d verify %d ple %d ubatch %u (eligible %d, seqs %u, embd %d) "
                "lends %d hybrid_fixed %d devseed %d\n", __func__, (int) !sampling.samplers.empty(), (int) cparams.embeddings,
                (int) cparams.causal_attn, (int) cparams.verify, (int) ple.enabled, ub.n_tokens, (int) eligible(ub), ub.n_seqs_unq,
                (int) (ub.embd != nullptr), (int) moe_cache->lends_graph_to(ub.n_tokens), (int) moe_cache->hybrid_fixed(), (int) moe_cache_devseed_on);
        }
        return 0;
    }
    // a first ubatch with outputs would be a group of one
    if (n_outputs_all != n_tokens_all) {
        const auto & ub0 = mctx->get_ubatch();
        for (uint32_t i = 0; i < ub0.n_tokens; i++) {
            if (ub0.output[i]) {
                return 0;
            }
        }
    } else {
        return 0;
    }

    const int64_t t0_us = ggml_time_us();
    const auto gtype = ctx_type_to_graph_type(cparams.ctx_type);

    // the ubatches of the group: graphs built, inputs set in host memory of their own
    struct lm_ub {
        const llama_ubatch * ub;
        llm_graph_result   * res;
        ggml_cgraph        * gf;
        int32_t              n_outputs;
        size_t               n_leafs;
        int64_t              n_tokens_before; // tokens of the batch before this ubatch
    };
    std::vector<lm_ub> group;

    gf_res_prev_active = nullptr; gf_res_epoch++;
    moe_cache_ids.clear();
    ggml_backend_sched_reset(sched.get());
    ggml_backend_sched_use_full_hash(sched.get());
    ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

    // the prompt does not seed the cache and records no routing
    moe_cache_seed      = false;
    moe_cache_seed_prev = false;
    moe_cache_record    = false;
    moe_cache_devseed   = false;
    moe_inplace_ok      = false; // k15-lfm2: layer-major prompts keep their staging copies
    if (moe_cache) {
        moe_cache->prompt_ubatch(false);
    }

    // the arena is pinned only when it is made at load (LLAMA_MOE_LAYER_MAJOR_EAGER=1): made by a prompt, pinning it cost that prompt
    // 2.6% (2k) / 3.5% (8k) on G4 (Qwen3.6-35B-A3B UD-Q4_K_XL + MTP, 8 GB, fresh process per prompt), while the inputs it holds (mostly
    // KQ masks of a few MiB per ubatch) are copied synchronously either way; pageable memory is also not locked on a low-RAM machine.
    // LLAMA_MOE_LAYER_MAJOR_ARENA_PINNED=1 pins it anyway
    static const bool lm_pin = (getenv("LLAMA_MOE_LAYER_MAJOR_EAGER") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_EAGER")) != 0) ||
        (getenv("LLAMA_MOE_LAYER_MAJOR_ARENA_PINNED") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_ARENA_PINNED")) != 0);
    ggml_backend_buffer_type_t inp_buft = ggml_backend_cpu_buffer_type();
    if (auto * dev = model.dev_output(); dev != nullptr && lm_pin) {
        if (auto * hb = ggml_backend_dev_host_buffer_type(dev)) {
            inp_buft = hb;
        }
    }

    lm_arena_used = 0;
    lm_arena_old.clear();
    int64_t n_outputs_group = 0;
    int64_t n_tokens_group  = 0;
    int64_t t_build_us = 0, t_set_us = 0;
    more = true;
    // k15-pp2 (LLAMA_LM_TAIL=1): the last ubatch of the batch, too short to be eligible (e.g. the server's last 4 prompt tokens, now in
    // the same decode as the rest), joins the group as its last member: built as a large ubatch (llama_moe_cache::set_force_large) with
    // its expert products on the device, it reads the host experts the group already copies for each layer instead of looking them up,
    // copying the misses and computing some on the host in a pass of its own
    // opt-in for now (LLAMA_LM_TAIL=1): exact without speculative decoding, but beside an MTP drafter the first verify after the prompt
    // gives other logits (local B200, Qwen3.6 q4x 12 GB pinned: draft acceptance 33/58 -> 23/80 at 2k), cause not found yet
    static const bool lm_tail_env = getenv("LLAMA_LM_TAIL") != nullptr && atoi(getenv("LLAMA_LM_TAIL")) != 0;
    // the device assignments of the tail member, set again after every scheduler reset (a refused allocation, a split of the group)
    std::vector<std::pair<ggml_tensor *, ggml_backend_t>> lm_tail_pins;
    // not in a drafter's context (backend samplers: its outputs are the greedy drafts of the last member)
    const bool lm_tail_ctx = sampling.samplers.empty() && (getenv("LLAMA_LM_TAIL_ANY") != nullptr || cparams.embeddings_nextn || true);
    while ((int) group.size() < lm_ub_max) {
        const auto & ub = mctx->get_ubatch();
        const bool tail = lm_tail_env && lm_tail_ctx && !group.empty() && !eligible(ub) && ub.n_seqs_unq == 1 && ub.n_tokens > 0 &&
            (int64_t) ub.n_tokens < prefetch_min_batch && n_tokens_prev + n_tokens_group + (int64_t) ub.n_tokens == (int64_t) n_tokens_all;
        if (!group.empty() && !eligible(ub) && !tail) {
            break;
        }
        int32_t n_out = 0;
        if (n_outputs_all == n_tokens_all) {
            n_out = ub.n_tokens;
        } else {
            for (uint32_t i = 0; i < ub.n_tokens; i++) {
                n_out += (int32_t) (ub.output[i] != 0);
            }
        }
        // the outputs stay allocated until the merged graph ends: at most a quarter of a ubatch worth of logits per group
        if (!group.empty() && n_outputs_group + n_out > std::max<int64_t>(1, cparams.n_ubatch/4)) {
            break;
        }
        n_outputs = n_out;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
            ret = -3;
            return -1;
        }

        const size_t k = group.size();
        if (lm_res.size() <= k) {
            lm_res.resize(k + 1);
            lm_inp_buf.resize(k + 1);
        }
        if (!lm_res[k]) {
            lm_res[k].reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
        }
        auto * res = lm_res[k].get();
        res->reset();
        const int64_t tb0 = ggml_time_us();
        if (tail) {
            moe_cache->set_force_large(true);
        }
        ggml_cgraph * gf = model.build_graph(graph_params(res, ub, mctx, gtype));
        if (tail) {
            moe_cache->set_force_large(false);
            // its ops on host weights run on the device as the large members' do (the backends would leave a short batch on the CPU)
            ggml_backend_t be_dev = nullptr;
            for (const auto & b : backends) {
                if (b.get() != backend_cpu && ggml_backend_dev_type(ggml_backend_get_device(b.get())) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    be_dev = b.get();
                    break;
                }
            }
            for (int i = 0; gf != nullptr && be_dev != nullptr && i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                if (node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_NONE) {
                    continue;
                }
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    const ggml_tensor * t = node->src[j];
                    while (t != nullptr && t->view_src != nullptr) {
                        t = t->view_src;
                    }
                    if (t != nullptr && t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                            ggml_backend_buffer_is_host(t->buffer)) {
                        ggml_backend_sched_set_tensor_backend(sched.get(), node, be_dev);
                        lm_tail_pins.emplace_back(node, be_dev);
                        break;
                    }
                }
            }
            lm_n_tail++;
            if (getenv("LLAMA_LM_TAIL_DEBUG") != nullptr) {
                int n_pin = 0, n_mmid = 0, n_mmid_pin = 0;
                for (int i = 0; gf != nullptr && i < ggml_graph_n_nodes(gf); ++i) {
                    ggml_tensor * node = ggml_graph_node(gf, i);
                    ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                    n_pin += be == be_dev;
                    if (node->op == GGML_OP_MUL_MAT_ID) {
                        n_mmid++;
                        n_mmid_pin += be == be_dev;
                    }
                }
                LLAMA_LOG_INFO("%s: tail member of %u tokens: %d nodes pinned to %s, MUL_MAT_ID %d of %d\n", __func__, ub.n_tokens, n_pin,
                    be_dev ? ggml_backend_name(be_dev) : "none", n_mmid_pin, n_mmid);
            }
        }
        t_build_us += ggml_time_us() - tb0;
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to build the graph of ubatch %zu\n", __func__, k);
            ret = -3;
            return -1;
        }

        // the inputs: allocated here, in host memory, and set now (later ubatches of the group change the memory state they read)
        std::vector<ggml_tensor *> inps;
        size_t inp_size = 0;
        const size_t align = ggml_backend_buft_get_alignment(inp_buft);
        auto add_inp = [&](ggml_tensor * t) {
            if ((t->flags & GGML_TENSOR_FLAG_INPUT) && t->data == nullptr && t->view_src == nullptr && t->buffer == nullptr &&
                    std::find(inps.begin(), inps.end(), t) == inps.end()) {
                inps.push_back(t);
                inp_size += GGML_PAD(ggml_backend_buft_get_alloc_size(inp_buft, t), align);
            }
        };
        // the graph's leaves are the sources of its nodes that are not nodes themselves (no public accessor for the leaf list)
        size_t n_leafs = 0;
        {
            std::unordered_set<const ggml_tensor *> srcs;
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                add_inp(node);
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    ggml_tensor * src = node->src[j];
                    if (src != nullptr && src->op == GGML_OP_NONE && srcs.insert(src).second) {
                        add_inp(src);
                        n_leafs++;
                    }
                }
            }
        }
        inp_size += align;
        // one host arena for the inputs of the group (pinned where the device has a host buffer type), kept for the next prompts: the
        // scheduler copies user inputs synchronously, so the arena is free again when the previous group's compute call has returned.
        // A group that needs more starts a larger arena (the earlier members keep the old one until the group is computed)
        auto & ib = lm_inp_buf[k];
        ib.reset();
        if (!lm_arena || lm_arena_used + inp_size > ggml_backend_buffer_get_size(lm_arena.get())) {
            const size_t want = std::max<size_t>(inp_size*(size_t) (lm_ub_max - k), lm_arena ? ggml_backend_buffer_get_size(lm_arena.get())*3/2 : 0);
            if (lm_arena) {
                lm_arena_old.push_back(std::move(lm_arena));
            }
            lm_arena.reset(ggml_backend_buft_alloc_buffer(inp_buft, want));
            lm_arena_used = 0;
            if (!lm_arena) {
                LLAMA_LOG_ERROR("%s: failed to allocate %.2f MiB of inputs\n", __func__, want/1048576.0);
                ret = -2;
                return -1;
            }
            if (lm_verbose) {
                LLAMA_LOG_INFO("%s: input arena of %.2f MiB\n", __func__, want/1048576.0);
            }
        }
        {
            uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(lm_arena.get());
            size_t off = lm_arena_used;
            for (ggml_tensor * t : inps) {
                off = GGML_PAD(off, align);
                if (ggml_backend_tensor_alloc(lm_arena.get(), t, base + off) != GGML_STATUS_SUCCESS) {
                    LLAMA_LOG_ERROR("%s: failed to place an input\n", __func__);
                    ret = -2;
                    return -1;
                }
                off += ggml_backend_buft_get_alloc_size(inp_buft, t);
            }
            lm_arena_used = GGML_PAD(off, align);
        }
        const int64_t ts0 = ggml_time_us();
        res->set_inputs(&ub);
        t_set_us += ggml_time_us() - ts0;

        group.push_back({ &ub, res, gf, n_out, n_leafs, n_tokens_prev + n_tokens_group });
        n_tokens_group += ub.n_tokens;
        n_outputs_group += n_out;
        moe_cache->count_large_ubatch();

        if (!mctx->next()) {
            more = false;
            break;
        }
        // a ubatch with outputs narrows its last layer to them, and so few rows take the MoE cache lookup: its slot maps are host tensors
        // shared by every graph and filled for the ids of one, so such a ubatch ends the group
        if (n_out > 0) {
            break;
        }
    }

    // the pieces of each graph: cut before the first use of every host weight
    auto host_weight = [](const ggml_tensor * t) -> const ggml_tensor * {
        while (t->view_src != nullptr) {
            t = t->view_src;
        }
        return t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buffer_is_host(t->buffer) ? t : nullptr;
    };
    std::vector<std::vector<int>>               cuts(group.size()); // first node of each piece
    std::vector<std::vector<const ggml_tensor *>> cw(group.size());   // the weight that starts each piece (nullptr: the first piece)
    bool aligned = true;
    size_t n_total = 0;
    for (size_t k = 0; k < group.size(); ++k) {
        ggml_cgraph * gf = group[k].gf;
        std::unordered_set<const ggml_tensor *> seen;
        cuts[k].push_back(0);
        cw[k].push_back(nullptr);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            const ggml_tensor * w = nullptr;
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (node->src[j] != nullptr) {
                    const ggml_tensor * hw = host_weight(node->src[j]);
                    if (hw != nullptr && seen.insert(hw).second && w == nullptr) {
                        w = hw;
                    }
                }
            }
            if (w != nullptr) {
                if (i > 0) {
                    cuts[k].push_back(i);
                    cw[k].push_back(w);
                } else {
                    cw[k][0] = w;
                }
            }
        }
        n_total += (size_t) ggml_graph_n_nodes(gf) + group[k].n_leafs;
        if (getenv("LLAMA_LM_TAIL_DEBUG") != nullptr && (k + 1 == group.size() || k == 0) && cuts[k].size() > 4) {
            for (size_t c = 1; c < 5; ++c) {
                const int at = cuts[k][c];
                std::string sline;
                for (int i = std::max(0, at - 6); i < std::min(ggml_graph_n_nodes(gf), at + 3); ++i) {
                    const ggml_tensor * nd = ggml_graph_node(gf, i);
                    sline += (i == at ? " | " : " ") + std::string(ggml_op_desc(nd)) + ":" + nd->name;
                }
                LLAMA_LOG_INFO("%s: member %zu cut %zu at %d (%s):%s\n", __func__, k, c, at, cw[k][c] ? cw[k][c]->name : "-", sline.c_str());
            }
        }
    }
    // the pieces of all graphs in one order: the cuts of the first graph, a cut of a later graph at the same weight runs with it, a cut
    // the first graph does not have (e.g. the MoE cache lookup of a last layer narrowed to the outputs) right after the graph's previous one
    std::vector<const ggml_tensor *> order = cw[0];
    std::vector<std::vector<size_t>> slot(group.size());   // index of each piece in `order` (after all insertions)
    {
        std::vector<std::vector<int64_t>> ids(group.size());   // stable ids of the entries of order
        std::vector<int64_t> oid(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            oid[i] = (int64_t) i;
        }
        int64_t next_id = (int64_t) order.size();
        ids[0] = oid;
        for (size_t k = 1; k < group.size() && aligned; ++k) {
            size_t p = 0;
            for (size_t j = 0; j < cw[k].size(); ++j) {
                const ggml_tensor * w = cw[k][j];
                size_t q = p;
                while (q < order.size() && order[q] != w) {
                    q++;
                }
                if (j == 0 && q != 0) {
                    aligned = false; // the first pieces do not start alike
                    break;
                }
                if (q == order.size()) {
                    order.insert(order.begin() + (std::ptrdiff_t) p, w);
                    oid.insert(oid.begin() + (std::ptrdiff_t) p, next_id++);
                    q = p;
                }
                ids[k].push_back(oid[q]);
                p = q + 1;
            }
        }
        std::unordered_map<int64_t, size_t> pos;
        for (size_t i = 0; i < oid.size(); ++i) {
            pos[oid[i]] = i;
        }
        for (size_t k = 0; k < group.size() && aligned; ++k) {
            for (int64_t id : ids[k]) {
                slot[k].push_back(pos[id]);
            }
        }
    }
    std::vector<const ggml_tensor *> & cut_w = order;

    // host memory other than weights and the inputs set above (e.g. MoE cache slot maps, a host KV cache) read by more than one graph
    // would be copied once for all of them: such a group runs one graph after the other
    if (aligned && group.size() > 1) {
        std::unordered_map<const ggml_tensor *, size_t> owner;
        for (size_t k = 0; k < group.size() && aligned; ++k) {
            ggml_cgraph * gf = group[k].gf;
            for (int i = 0; i < ggml_graph_n_nodes(gf) && aligned; ++i) {
                const ggml_tensor * node = ggml_graph_node(gf, i);
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    const ggml_tensor * t = node->src[j];
                    while (t != nullptr && t->view_src != nullptr) {
                        t = t->view_src;
                    }
                    if (t == nullptr || t->op != GGML_OP_NONE || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer) ||
                            ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                            t->buffer == lm_arena.get() || std::any_of(lm_arena_old.begin(), lm_arena_old.end(),
                                [&](const ggml_backend_buffer_ptr & b) { return b.get() == t->buffer; })) {
                        continue;
                    }
                    auto it = owner.emplace(t, k).first;
                    if (it->second != k) {
                        LLAMA_LOG_WARN("%s: host tensor %s is read by ubatches %zu and %zu of the group\n", __func__, t->name, it->second, k);
                        aligned = false;
                        break;
                    }
                }
            }
        }
    }

    // the merged graph of the members [a, b) of the group: piece s of every member (in ubatch order) before piece s + 1
    ggml_context_ptr ctx_m;
    // k15-pp: the member that starts at the capture index, if a merged graph has to take the state before it
    int cap_k_group[cap_max];
    for (int c = 0; c < cap_max; ++c) {
        cap_k_group[c] = -1;
        if (cap[c].idx >= 0 && !cap[c].done) {
            for (size_t k = 1; k < group.size(); ++k) {
                if (group[k].n_tokens_before == cap[c].idx) {
                    cap_k_group[c] = (int) k;
                }
            }
        }
    }
    int  cap_k[cap_max];             // the member build_merged inserts the copies of capture c for
    bool cap_in_graph = false;       // the merged graph being computed takes a capture
    std::vector<bool> cap_seen[cap_max];
    for (int c = 0; c < cap_max; ++c) {
        cap_k[c] = -1;
    }
    int n_cap_nodes = 0;
    // the captures the members [a, b) take before one of them, run as separate graphs (between graphs, in stream order)
    auto cap_before = [&](size_t k) {
        for (int c = 0; c < cap_max; ++c) {
            if (cap_k_group[c] == (int) k && !cap[c].done) {
                cap_now(c, group[k].ub->pos[0] - 1);
            }
        }
    };
    auto build_merged = [&](size_t a, size_t b) -> ggml_cgraph * {
        size_t size = 64;
        for (size_t k = a; k < b; ++k) {
            size += (size_t) ggml_graph_n_nodes(group[k].gf) + group[k].n_leafs;
        }
        cap_in_graph = false;
        n_cap_nodes = 0;
        size_t n_cap_regions = 0;
        for (int c = 0; c < cap_max; ++c) {
            cap_k[c] = -1;
            if (cap_k_group[c] > (int) a && cap_k_group[c] < (int) b && !cap[c].done && cap_prepare(c)) {
                cap_k[c] = cap_k_group[c];
                cap_in_graph = true;
                cap_seen[c].assign(cap[c].regions.size(), false);
                size += 3*cap[c].regions.size();
                n_cap_regions += cap[c].regions.size();
            }
        }
        // k15-pp: the capture copies are tensors of this context too (a view of the state, the copy and its view of the snapshot)
        const size_t meta = ggml_graph_overhead_custom(size, false) + (1 + 4*n_cap_regions)*ggml_tensor_overhead();
        if (lm_meta.size() < meta) {
            lm_meta.resize(meta);
        }
        ggml_init_params ip = { lm_meta.size(), lm_meta.data(), true };
        ctx_m.reset(ggml_init(ip));
        ggml_cgraph * g = ggml_new_graph_custom(ctx_m.get(), size, false);
        const size_t n_pieces = order.size();
        std::vector<size_t> next(group.size(), 0); // the next piece of each graph
        for (size_t s = 0; s < n_pieces; ++s) {
            for (size_t k = a; k < b; ++k) {
                const size_t j = next[k];
                if (j >= slot[k].size() || slot[k][j] != s) {
                    continue;
                }
                ggml_cgraph * gf = group[k].gf;
                const int i0 = cuts[k][j];
                const int i1 = j + 1 < cuts[k].size() ? cuts[k][j + 1] : ggml_graph_n_nodes(gf);
                // order only: the compute flags stay as each graph set them (an input path that ggml_build_forward_select did not pick
                // is in the graph but not computed; expand would mark it computed, running e.g. the mixed-batch set_rows on unset indices)
                for (int i = i0; i < i1; ++i) {
                    ggml_tensor * node = ggml_graph_node(gf, i);
                    for (int c = 0; c < cap_max; ++c) {
                        if ((int) k != cap_k[c]) {
                            continue;
                        }
                        // k15-pp: the state rows as the members before this one left them, before its first node that reads or writes them
                        auto & rg = cap[c].regions;
                        for (int sj = 0; sj < GGML_MAX_SRC + 1; ++sj) {
                            const ggml_tensor * t = sj == 0 ? node : node->src[sj - 1];
                            while (t != nullptr && t->view_src != nullptr) {
                                t = t->view_src;
                            }
                            for (size_t r = 0; t != nullptr && r < rg.size(); ++r) {
                                if (rg[r].t == t && !cap_seen[c][r]) {
                                    cap_seen[c][r] = true;
                                    ggml_tensor * src = ggml_view_1d(ctx_m.get(), rg[r].t, rg[r].size/ggml_type_size(t->type), rg[r].off);
                                    const int n0 = ggml_graph_n_nodes(g);
                                    ggml_build_forward_expand(g, ggml_cpy(ctx_m.get(), src, rg[r].snap));
                                    n_cap_nodes += ggml_graph_n_nodes(g) - n0;
                                }
                            }
                        }
                    }
                    ggml_build_forward_order(g, node);
                }
                next[k]++;
            }
        }
        size_t n_nodes_sum = 0;
        for (size_t k = a; k < b; ++k) {
            GGML_ASSERT(next[k] == slot[k].size());
            n_nodes_sum += (size_t) ggml_graph_n_nodes(group[k].gf);
        }
        if (cap_in_graph) {
            n_nodes_sum += (size_t) n_cap_nodes;
            for (int c = 0; c < cap_max; ++c) {
                if (cap_k[c] >= 0 && std::count(cap_seen[c].begin(), cap_seen[c].end(), true) != (std::ptrdiff_t) cap_seen[c].size()) {
                    LLAMA_LOG_WARN("%s: capture: %zu of %zu state rows not used by ubatch %d of the group\n", __func__,
                        (size_t) std::count(cap_seen[c].begin(), cap_seen[c].end(), false), cap_seen[c].size(), cap_k[c]);
                }
            }
        }
        // every node of every graph is in the merged graph once: a node that two graphs share (none expected) would run only in the first
        if ((size_t) ggml_graph_n_nodes(g) != n_nodes_sum) {
            std::unordered_set<const ggml_tensor *> first;
            for (int i = 0; i < ggml_graph_n_nodes(group[a].gf); ++i) {
                first.insert(ggml_graph_node(group[a].gf, i));
            }
            int n_shared = 0;
            for (int i = 0; i < ggml_graph_n_nodes(group[a + 1].gf) && n_shared < 8; ++i) {
                const ggml_tensor * t = ggml_graph_node(group[a + 1].gf, i);
                if (first.count(t)) {
                    LLAMA_LOG_WARN("%s: node shared by the graphs: %s (%s)\n", __func__, t->name, ggml_op_name(t->op));
                    n_shared++;
                }
            }
            LLAMA_LOG_WARN("%s: merged graph has %d nodes, the graphs %zu\n", __func__, ggml_graph_n_nodes(g), n_nodes_sum);
        }
        return g;
    };
    ggml_cgraph * gm = aligned && group.size() > 1 ? build_merged(0, group.size()) : nullptr;

    // the seed of the MoE cache, as moe_cache_seed_begin plans it for the ubatch-by-ubatch order: the last large ubatch of the batch seeds,
    // from the routing of the ubatch before it if it has no outputs, else from its own (if its seed graph fits); every member of the group
    // is large and prefetched, so the seed copies from the shared expert copies (all experts staged)
    // k13-small: a cache below LLAMA_MOE_LM_RELEASE_FRAC (0.05) of the host experts frees its banks while the group runs (the compute
    // buffers grow in that memory), they are allocated again after it and refilled (see llama_moe_cache::release and the end of the group):
    // the slots of so small a cache are too few to lend the growth (G4 at 4 GB, Qwen3.6 Q3_K_M + MTP: 111 MiB in three bank tails of at most
    // 33 MiB, a 4-ubatch group grows by 68 MiB with 77 MiB free and 64 MiB of headroom, it was split in single ubatches: one pass over
    // every host expert per 512 tokens). The group does not seed the cache. LLAMA_MOE_LM_RELEASE=0 off
    static const bool lm_release_env = getenv("LLAMA_MOE_LM_RELEASE") == nullptr || atoi(getenv("LLAMA_MOE_LM_RELEASE")) != 0;
    bool lm_released = false;
    if (lm_release_env && group.size() >= 2 && lm_tiny_cache(moe_cache->cache_frac())) {
        ggml_backend_sched_synchronize(sched.get());
        lm_released = moe_cache->release();
    }
    lm_ask.clear();
    lm_staged.clear();
    lm_seeded.clear();
    moe_cache->clear_seed_counts();
    int seed_k = -1, src_k = -1;
    // k13-pp: LLAMA_MOE_LAYER_MAJOR_SEED=0 (opt-in) turns the seed off. The seed reads the source routing of every layer in an eval callback,
    // which synchronizes the device once per layer, so the next layer's expert copies are queued only after its router has run and the copy
    // engine idles through every layer's attention (G4, Qwen3.6-35B-A3B Q4 + MTP at 12 GB, 2k prompt: 42 gaps of ~2 ms in a 357 ms pass).
    // Off: G4 12 GB 2k +7.1%, 8 GB 2k +10.9% / 8k +5.7%; but L4 8 GB 8k -4.5% (the prompt after an unseeded one misses more experts on a
    // copy-bound link), so it stays opt-in until the seed reads the routing without a device sync (plans/k13_pp.md section 9)
    static const bool lm_seed_env = getenv("LLAMA_MOE_LAYER_MAJOR_SEED") == nullptr || atoi(getenv("LLAMA_MOE_LAYER_MAJOR_SEED")) != 0;
    // beside the device seed (seed_sync off: the host seed of a ubatch's own ids does not fit) only the seed from an earlier member's
    // routing, which needs no larger graph
    const bool lm_seed_dev = moe_cache_devseed_on && lm_devseed_seed;
    // k11 (LLAMA_MOE_LAYER_MAJOR_SEED_REPLAY=1, opt-in, default off: no model / tier has passed the KLD gate with it yet, plans/k11_prompt.md
    // 3n; it acts wherever the checkpoint merge is on, which other branches may turn on by default): a group merged across a ubatch break
    // (llama_set_ubatch_breaks) whose last member is the only ubatch after it, a prefetched large one without outputs that would end its own
    // batch: run by themselves, the members before the break seed as the batch they were (host seed from the routing of an earlier member),
    // then the last one seeds on the device from its own routing (moe_cache_devseed_on). The merged graph does the first seed as that batch
    // would, and the device seed of the last member runs after it (llama_moe_cache::devseed_replay), on the state the separate graph would
    // have seen, so the cache ends up as with the separate batches
    static const bool lm_replay_env = getenv("LLAMA_MOE_LAYER_MAJOR_SEED_REPLAY") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_SEED_REPLAY")) != 0;
    size_t rp_k = 0; // members before the break (0: no replay)
    // k14 merge: a group whose cache banks were released (k13-small) seeds nothing, so it replays nothing either
    if (!lm_released && lm_replay_env && moe_cache_devseed_on && !moe_cache_seed_sync && group.size() >= 3 && !balloc->ubatch_breaks.empty()) {
        const auto & last = group.back();
        const int64_t b = last.n_tokens_before;
        const auto & brk = balloc->ubatch_breaks;
        if (std::find(brk.begin(), brk.end(), (int32_t) b) != brk.end() && last.n_outputs == 0 &&
                (int64_t) last.ub->n_tokens >= prefetch_min_batch && n_tokens_all - b - (int64_t) last.ub->n_tokens <= llama_moe_cache::n_small_max &&
                b - group[0].n_tokens_before > (int64_t) cparams.n_ubatch + llama_moe_cache::n_small_max) {
            rp_k = group.size() - 1;
        }
    }
    if (!lm_released && lm_seed_env && moe_cache_seed_enabled && (moe_cache_seed_sync || lm_seed_dev) && !moe_cache_seed_auto) {
        // with the replay: the seed of the batch that ends at the break
        const int64_t n_seed_all = rp_k > 0 ? group[rp_k].n_tokens_before : n_tokens_all;
        for (size_t k = 0; k < (rp_k > 0 ? rp_k : group.size()); ++k) {
            const int64_t nt = group[k].ub->n_tokens;
            if (nt > llama_moe_cache::n_small_max && n_seed_all - group[k].n_tokens_before - nt <= llama_moe_cache::n_small_max) {
                seed_k = (int) k;
            }
        }
        if (seed_k >= 1 && group[seed_k].n_outputs == 0) {
            src_k = seed_k - 1;
        } else if (seed_k >= 1 && moe_cache_seed_fits && moe_cache_seed_sync) {
            src_k = seed_k;
        } else if (seed_k >= 1) {
            moe_cache->count_seed_skipped();
        }
    }
    if (src_k >= 0) {
        ggml_cgraph * gf = group[src_k].gf;
        std::unordered_map<const ggml_tensor *, int> index;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            index[ggml_graph_node(gf, i)] = i;
        }
        std::unordered_map<const ggml_tensor *, lm_seed_layer *> by_ids;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            const ggml_tensor * w = host_weight(node->src[0]);
            if (w == nullptr || moe_cache->get_experts(w) == nullptr) {
                continue;
            }
            ggml_tensor * ids = node->src[2];
            if (ggml_nelements(ids) == 0) {
                continue; // a layer narrowed to no rows copies and seeds nothing
            }
            while (ids->view_src != nullptr && index.count(ids) == 0) {
                ids = ids->view_src;
            }
            if (!ggml_is_contiguous(ids) || ids->type != GGML_TYPE_I32 || index.count(ids) == 0) {
                continue;
            }
            auto bi = by_ids.find(ids);
            if (bi == by_ids.end()) {
                // the last node of the piece of the source graph that computes the routing
                const int at = index[ids];
                const auto & c = cuts[src_k];
                const size_t j = (size_t) (std::upper_bound(c.begin(), c.end(), at) - c.begin()) - 1;
                const int last = (j + 1 < c.size() ? c[j + 1] : ggml_graph_n_nodes(gf)) - 1;
                lm_seed_layer L;
                L.ids     = ids;
                L.w_first = w;
                auto & ref = lm_ask[ggml_graph_node(gf, last)];
                if (!ref.ws.empty()) {
                    continue; // two routings end the same piece (not expected): the second does not seed
                }
                ref = L;
                bi = by_ids.emplace(ids, &ref).first;
            }
            if (std::find(bi->second->ws.begin(), bi->second->ws.end(), w) == bi->second->ws.end()) {
                bi->second->ws.push_back(w);
            }
        }
        lm_seed_active      = !lm_ask.empty();
        moe_cache_seed_prev = lm_seed_active;
        if (lm_seed_active) {
            ggml_backend_sched_set_eval_callback(sched.get(), lm_eval_cb, this);
            // k15-pp: LLAMA_MOE_LM_SEED_ASYNC=0 reads the routing in the callback after a device sync (k14). The sync held back the next
            // layer's expert copies until its router had run, so the copy engine idled through every layer's attention
            static const bool lm_async_env = getenv("LLAMA_MOE_LM_SEED_ASYNC") == nullptr || atoi(getenv("LLAMA_MOE_LM_SEED_ASYNC")) != 0;
            lm_seed_async = false;
            lm_pending.clear();
            lm_staged_seq.clear();
            lm_copy_graph = nullptr;
            lm_async_used = 0;
            lm_async_nev  = 0;
            // k15-pp2: a group with a tail member (LLAMA_LM_TAIL) seeds synchronously: with the asynchronous read the seed of some layers
            // copied experts from staging memory already reused (Qwen3.6 q4x + MTP 12 GB: the cache after the prompt held wrong experts, the
            // first MTP verify gave other logits); not found yet why the extra member breaks the "due" rule of lm_seed_resolve_due
            const bool lm_has_tail = !group.empty() && (int64_t) group.back().ub->n_tokens < prefetch_min_batch;
            if (lm_async_env && cparams.cb_eval == nullptr && !lm_has_tail) {
                size_t need = 0;
                ggml_backend_dev_t dev = nullptr;
                bool ok = true;
                for (const auto & a : lm_ask) {
                    need += GGML_PAD(ggml_nbytes(a.second.ids), 64);
                }
                // the device of the first GPU backend (the routing of an offloaded MoE layer is computed there)
                for (const auto & b : backends) {
                    ggml_backend_dev_t d = ggml_backend_get_device(b.get());
                    if (d != nullptr && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        dev = d;
                        break;
                    }
                }
                ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
                if (host_buft == nullptr) {
                    ok = false;
                }
                if (ok && (!lm_async_buf || ggml_backend_buffer_get_size(lm_async_buf.get()) < need)) {
                    lm_async_buf.reset(ggml_backend_buft_alloc_buffer(host_buft, need + (need >> 2)));
                    ok = lm_async_buf != nullptr;
                }
                while (ok && lm_async_ev.size() < lm_ask.size()) {
                    ggml_backend_event_t e = ggml_backend_event_new(dev);
                    if (e == nullptr) {
                        ok = false;
                        break;
                    }
                    lm_async_ev.emplace_back(e);
                }
                lm_seed_async = ok;
                lm_async_dev  = dev;
                if (ok) {
                    ggml_backend_sched_set_eval_callback_sync(sched.get(), false);
                }
            }
        }
    }

    // the routing that sched_copy_experts would record for the last ubatch of the group (moe_cache_record), taken before the scheduler
    // replaces the weight sources of the nodes with their copies
    std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> lm_record;
    {
        ggml_cgraph * gf = group.back().gf;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            const ggml_tensor * w = host_weight(node->src[0]);
            const ggml_tensor * ids = node->src[2];
            // not the empty routing of a layer narrowed to no outputs (it copies nothing, so nothing records it)
            if (w != nullptr && moe_cache->get_experts(w) != nullptr && (ids->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_nelements(ids) > 0 &&
                    std::none_of(lm_record.begin(), lm_record.end(), [&](const auto & r) { return r.second == ids; })) {
                lm_record.emplace_back(w, ids);
            }
        }
    }

    // the routing of the last member per layer, for the replayed device seed (the same outputs as lm_record, one per layer)
    std::vector<std::pair<int32_t, const ggml_tensor *>> rp_ids;
    if (rp_k > 0) {
        for (const auto & [w, ids] : lm_record) {
            int32_t il = -1;
            if (moe_cache->devseed_bank(w, il) != nullptr && il >= 0 && ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) &&
                    std::none_of(rp_ids.begin(), rp_ids.end(), [&](const auto & r) { return r.first == il; })) {
                rp_ids.emplace_back(il, ids);
            }
        }
        std::sort(rp_ids.begin(), rp_ids.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        if (rp_ids.empty()) {
            rp_k = 0;
        }
    }

    // the outputs of the members [n_extracted, n) (in order), right after the graph that computed them: the compute buffer of a later
    // graph of the group reuses their memory
    size_t n_extracted = 0;
    auto extract_upto = [&](size_t n) {
        for (; n_extracted < n; ++n_extracted) {
            const auto & g = group[n_extracted];
            const llama_ubatch & ubatch = *g.ub;
            auto * res = g.res;
            n_outputs = g.n_outputs;

            auto * t_logits  = res->get_logits();
            auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

            if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
                ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
                GGML_ASSERT(backend_res != nullptr);
                GGML_ASSERT(n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits.data + n_outputs_prev*n_vocab, 0, n_outputs*n_vocab*sizeof(float));
            }

            bool extract_all_idxs = false;
            {
                const bool masked    = cparams.embeddings_nextn_masked;
                const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
                const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

                if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                    ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                    GGML_ASSERT(backend_h != nullptr);
                    const uint32_t n_embd  = hparams.n_embd_out();
                    GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                    if (!(mtp_dev_h_this && mtp_dev_h_store(backend_h, t_h_nextn, ubatch, offset))) {
                        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data + offset*n_embd, 0, n_rows*n_embd*sizeof(float));
                    }
                    extract_all_idxs = !masked;
                }
            }
            if (extract_all_idxs) {
                GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
                GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
                const auto & batch_idxs = ubatch.data->batch_idxs;
                embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
            }

            // backend samplers (e.g. the greedy draft of an MTP context): only the last ubatch of a group has outputs
            if (!sampling.samplers.empty()) {
                const auto stride = n_vocab;
                copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
                copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
                copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
                copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
            }

            n_outputs_prev += n_outputs;
            n_tokens_prev  += ubatch.n_tokens;
        }
    };

    ggml_status status = GGML_STATUS_SUCCESS;
    const int64_t t1_us = ggml_time_us();
    int64_t n_tokens_max = 0;
    for (const auto & g : group) {
        n_tokens_max = std::max<int64_t>(n_tokens_max, g.ub->n_tokens);
    }
    if (wstore) {
        const auto & last = group.back();
        wstore->set_wrap(!(last.n_outputs > 0 && (uint32_t) last.n_outputs < last.ub->n_tokens));
    }

    // LLAMA_MOE_LAYER_MAJOR_CHECK=1 (debug): the KV row indices of every set_rows are inside the cache, checked on the host inputs
    static const bool lm_check = getenv("LLAMA_MOE_LAYER_MAJOR_CHECK") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_CHECK")) != 0;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> lm_sr; // (set_rows node, its index tensor before the scheduler replaces it)
    if (lm_check && gm != nullptr) {
        int n_bad = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gm); ++i) {
            ggml_tensor * node = ggml_graph_node(gm, i);
            if (node->op != GGML_OP_SET_ROWS || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            ggml_tensor * c = node->src[1];
            ggml_tensor * a = node->src[2];
            lm_sr.emplace_back(node, c);
            const ggml_tensor * root = c;
            while (root->view_src) root = root->view_src;
            if (c->type != GGML_TYPE_I64 || root->data == nullptr || !ggml_backend_buffer_is_host(root->buffer) || !ggml_is_contiguous(c)) {
                if (n_bad++ < 4) {
                    LLAMA_LOG_WARN("%s: check: set_rows %s indices %s not a host I64 input (data %p)\n", __func__, node->name, c->name, root->data);
                }
                continue;
            }
            const int64_t * d = (const int64_t *) c->data;
            for (int64_t j = 0; j < ggml_nelements(c); ++j) {
                if (d[j] < 0 || d[j] >= a->ne[1]) {
                    if (n_bad++ < 4) {
                        LLAMA_LOG_WARN("%s: check: set_rows %s index %" PRId64 " = %" PRId64 " outside [0, %" PRId64 ")\n", __func__, node->name, j, d[j], a->ne[1]);
                    }
                    break;
                }
            }
        }
        LLAMA_LOG_INFO("%s: check: %zu set_rows, %d bad\n", __func__, lm_sr.size(), n_bad);
    }

    auto run = [&](ggml_cgraph * g) {
        for (const auto & [node, be] : lm_tail_pins) {
            ggml_backend_sched_set_tensor_backend(sched.get(), node, be);
        }
        prefetch_n_tokens = n_tokens_max;
        moe_cache->set_graph(g);
        if (!lm_pending.empty()) {
            lm_seed_resolve_all();
        }
        lm_staged.clear(); // the expert copies a seed reads are those of this graph
        lm_staged_seq.clear();
        lm_copy_graph = nullptr;
        static const bool tail_dbg0 = getenv("LLAMA_LM_TAIL_DEBUG") != nullptr;
        if (tail_dbg0 && !group.empty() && (int64_t) group.back().ub->n_tokens < prefetch_min_batch) {
            int n_gpu = 0, n_all = 0;
            for (int i = 0; i < ggml_graph_n_nodes(group.back().gf); ++i) {
                ggml_tensor * node = ggml_graph_node(group.back().gf, i);
                if (node->op == GGML_OP_MUL_MAT_ID) {
                    ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                    n_all++;
                    n_gpu += be != nullptr && be != backend_cpu;
                }
            }
            LLAMA_LOG_INFO("%s: before alloc: tail MUL_MAT_ID on a GPU %d of %d, graph %d nodes\n", __func__, n_gpu, n_all, ggml_graph_n_nodes(g));
        }
        ggml_backend_sched_set_reuse_weight_copies(sched.get(), true);
        const bool ok = ggml_backend_sched_alloc_graph(sched.get(), g);
        ggml_backend_sched_set_reuse_weight_copies(sched.get(), false);
        static const bool tail_dbg = getenv("LLAMA_LM_TAIL_DEBUG") != nullptr;
        if (tail_dbg && ok && !group.empty() && (int64_t) group.back().ub->n_tokens < prefetch_min_batch) {
            std::map<std::string, int> by;
            for (int i = 0; i < ggml_graph_n_nodes(group.back().gf); ++i) {
                ggml_tensor * node = ggml_graph_node(group.back().gf, i);
                if (node->op == GGML_OP_MUL_MAT_ID) {
                    ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                    by[std::string(be ? ggml_backend_name(be) : "none") + " src0 " + (node->src[0]->buffer && ggml_backend_buffer_is_host(node->src[0]->buffer) ? "host" : "dev")]++;
                }
            }
            for (const auto & [k, v] : by) {
                LLAMA_LOG_INFO("%s: tail member MUL_MAT_ID on %s: %d\n", __func__, k.c_str(), v);
            }
        }
        if (!ok) {
            if (ggml_backend_sched_get_alloc_over(sched.get()) == 0) {
                LLAMA_LOG_ERROR("%s: failed to allocate the graph of %zu ubatches\n", __func__, group.size());
            }
            return GGML_STATUS_ALLOC_FAILED;
        }
        moe_cache->prepare(n_tokens_max);
        if (lm_check) {
            // the device copies of the indices: where they are, and whether two set_rows of different ubatches share one
            std::unordered_map<const void *, const ggml_tensor *> at;
            int n_same = 0;
            for (auto & [node, c] : lm_sr) {
                ggml_tensor * cc = node->src[1];
                auto it = at.emplace(cc->data, c);
                if (!it.second && it.first->second != c && n_same++ < 4) {
                    LLAMA_LOG_WARN("%s: check: indices %s and %s share device address %p (copy %s, buffer %s)\n", __func__,
                        it.first->second->name, c->name, cc->data, cc->name, cc->buffer ? ggml_backend_buffer_name(cc->buffer) : "-");
                }
            }
            LLAMA_LOG_INFO("%s: check: %zu index copies at %zu addresses, %d shared by different inputs\n", __func__, lm_sr.size(), at.size(), n_same);
        }
        const auto st = graph_compute(g, true);
        if (lm_seed_async && !lm_pending.empty()) {
            // the layers whose copies the graph queued last (their staging slots hold the copies until the next graph)
            if (st == GGML_STATUS_SUCCESS) {
                lm_seed_resolve_all();
            } else {
                lm_pending.clear();
            }
        }
        lm_async_used = 0;
        lm_async_nev  = 0;
        return st;
    };

    size_t buf_before = 0, buf_after = 0;
    for (auto & b : backend_ptrs) {
        buf_before += ggml_backend_sched_get_buffer_size(sched.get(), b);
    }
    // the compute buffers of a device may grow by at most LLAMA_MOE_LAYER_MAJOR_GROW_MIB (default 1024) for a merged graph, and must
    // leave the prefetch headroom free: a larger graph is split in two (and again), else its allocation would drop the prefetch for good
    // (gpt-oss-20b MXFP4, G4 8 GB tier: 16 ubatches of 512 grew the compute buffer 455 -> 1164 MiB)
    static const size_t lm_grow_max = (size_t) (getenv("LLAMA_MOE_LAYER_MAJOR_GROW_MIB") ? std::max(0, atoi(getenv("LLAMA_MOE_LAYER_MAJOR_GROW_MIB"))) : 1024) << 20;
    int n_sub = 0;
    size_t n_done = 0; // the members computed (in order): a fallback must not compute them again (recurrent states would advance twice)
    bool first_run = true;
    std::function<ggml_status(size_t, size_t)> run_range = [&](size_t a, size_t b) -> ggml_status {
        if (!first_run) {
            ggml_backend_sched_reset(sched.get());
            ggml_backend_sched_use_full_hash(sched.get());
        }
        first_run = false;
        cap_before(a);
        if (b - a == 1) {
            cap_in_graph = false;
            const ggml_status st = run(group[a].gf);
            if (st == GGML_STATUS_SUCCESS) {
                n_sub++;
                n_done = b;
                extract_upto(n_done);
            }
            return st;
        }
        if (b - a >= lm_refused_min) {
            // a graph of as many ubatches was refused before (the growth only rises with the context): split without measuring
            const size_t mid = a + (b - a)/2;
            const ggml_status st = run_range(a, mid);
            return st != GGML_STATUS_SUCCESS ? st : run_range(mid, b);
        }
        ggml_cgraph * g = (a == 0 && b == group.size()) ? gm : build_merged(a, b);
        // the scheduler replaces node sources with its copies while it splits a graph: kept to undo them if the merged graph does not
        // allocate and its members are split again
        std::vector<std::pair<ggml_tensor *, std::array<ggml_tensor *, GGML_MAX_SRC>>> srcs;
        srcs.reserve(ggml_graph_n_nodes(g));
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            ggml_tensor * node = ggml_graph_node(g, i);
            std::array<ggml_tensor *, GGML_MAX_SRC> arr;
            std::copy(node->src, node->src + GGML_MAX_SRC, arr.begin());
            srcs.emplace_back(node, arr);
        }
        lm_merged_active = true;
        ggml_backend_sched_set_alloc_limit(sched.get(), true, lm_grow_max);
        // k11 (LLAMA_MOE_LM_LEND=0 off): a merged graph whose compute buffers would grow past the free memory borrows its allocation from the
        // MoE cache (slots lent for the group and refilled after it, as its staging copies are) instead of being split; the merged graph
        // runs the same ubatches, only where its tensors live changes
        static const bool lm_lend = getenv("LLAMA_MOE_LM_LEND") == nullptr || atoi(getenv("LLAMA_MOE_LM_LEND")) != 0;
        // k13-small: the graph borrows only once its growth did not fit: the lend scope places the whole graph allocation in cache slots,
        // which were lent (and their experts refilled after the group) even where the compute buffers could grow in free memory. G4
        // gemma-4 26B-A4B Q3_K_M + drafter at 8 GB, 2k: 393 instead of 241 slots lent, a different cache after the prompt (another text than
        // k10) and 287 instead of 325 tok/s decode, prompt 4815 instead of 5014 (LLAMA_MOE_LM_LEND=0). LLAMA_MOE_LM_LEND_FIRST=1: lend at once (k12)
        static const bool lm_lend_first = getenv("LLAMA_MOE_LM_LEND_FIRST") != nullptr && atoi(getenv("LLAMA_MOE_LM_LEND_FIRST")) != 0;
        const bool lend_ok = lm_lend && moe_cache->lends_graph_any();
        // k13-small round 7: the growth-first rule only up to LLAMA_MOE_LM_LEND_FIRST_GROW_MIB (256) of growth (0 = any growth, the round 1
        // rule) and for groups starting before position LLAMA_MOE_LM_LEND_FIRST_POS (0 = any position); past either the group borrows at once
        // (k12). G4 run k13r8: q4x + MTP 8 GB 30k (its groups grow 346 MiB and more) 247.3 tok/s and k12's text with the cap, 225.3 without
        // (k12 247.7); gemma-4 8 GB 2k (120 MiB) keeps 326.5 (k12 288.2); POS=8192 does not fix 30k (230.2)
        static const size_t  lf_grow = (size_t) (getenv("LLAMA_MOE_LM_LEND_FIRST_GROW_MIB") ? std::max(0, atoi(getenv("LLAMA_MOE_LM_LEND_FIRST_GROW_MIB"))) : 256) << 20;
        static const int64_t lf_pos  = getenv("LLAMA_MOE_LM_LEND_FIRST_POS") ? std::max(0, atoi(getenv("LLAMA_MOE_LM_LEND_FIRST_POS"))) : 0;
        // a group refused by the cap is followed by groups at least as large and as far in the context: they borrow at once (the growth
        // only rises with both), saving the refused attempt (G4 q4x 8 GB 30k prompt: 6821 tok/s with the attempt, k12 6927)
        const int64_t g_pos = group[a].ub->pos[0];
        const bool lf_known = lf_grow > 0 && g_pos >= lm_lf_refused_pos && (int64_t) (b - a) >= lm_lf_refused_n;
        bool lend_g = lend_ok && (lm_lend_first || (lf_pos > 0 && g_pos >= lf_pos) || lf_known);
        if (lend_g) {
            moe_cache->set_graph_lend(true);
            ggml_backend_sched_set_alloc_lend_graph(sched.get(), true);
        }
        const bool lf_capped = lend_ok && !lend_g && lf_grow > 0 && lf_grow < lm_grow_max;
        if (lf_capped) {
            ggml_backend_sched_set_alloc_limit(sched.get(), true, lf_grow);
        }
        ggml_status st = run(g);
        if (lend_g) {
            moe_cache->set_graph_lend(false);
        }
        if (st == GGML_STATUS_ALLOC_FAILED && lf_known && !lm_lend_first && !(lf_pos > 0 && g_pos >= lf_pos)) {
            // the cache did not take it: the growth in free memory, as a group under the cap
            for (auto & [node, arr] : srcs) {
                std::copy(arr.begin(), arr.end(), node->src);
            }
            ggml_backend_sched_reset(sched.get());
            ggml_backend_sched_use_full_hash(sched.get());
            ggml_backend_sched_set_alloc_limit(sched.get(), true, lm_grow_max);
            st = run(g);
        }
        if (st == GGML_STATUS_ALLOC_FAILED && lend_ok && !lend_g && ggml_backend_sched_get_alloc_over(sched.get()) > 0) {
            // the growth did not fit in free memory: as the split below would, undo the scheduler's copies, then the same graph with the
            // allocation lent by the cache
            for (auto & [node, arr] : srcs) {
                std::copy(arr.begin(), arr.end(), node->src);
            }
            ggml_backend_sched_reset(sched.get());
            ggml_backend_sched_use_full_hash(sched.get());
            ggml_backend_sched_set_alloc_limit(sched.get(), true, lm_grow_max);
            lend_g = true;
            moe_cache->set_graph_lend(true);
            ggml_backend_sched_set_alloc_lend_graph(sched.get(), true);
            st = run(g);
            moe_cache->set_graph_lend(false);
            if (st != GGML_STATUS_ALLOC_FAILED && lf_capped) {
                lm_lf_refused_pos = std::min(lm_lf_refused_pos, g_pos);
                lm_lf_refused_n   = std::min(lm_lf_refused_n, (int64_t) (b - a));
            }
            if (st == GGML_STATUS_ALLOC_FAILED && lf_capped) {
                // the cache did not take it: the growth in free memory as without the cap
                for (auto & [node, arr] : srcs) {
                    std::copy(arr.begin(), arr.end(), node->src);
                }
                ggml_backend_sched_reset(sched.get());
                ggml_backend_sched_use_full_hash(sched.get());
                ggml_backend_sched_set_alloc_limit(sched.get(), true, lm_grow_max);
                st = run(g);
            }
        }
        ggml_backend_sched_set_alloc_limit(sched.get(), false, 0);
        lm_merged_active = false;
        if (st != GGML_STATUS_ALLOC_FAILED) {
            n_sub++;
            if (st == GGML_STATUS_SUCCESS) {
                n_done = b;
                extract_upto(n_done);
                for (int c = 0; cap_in_graph && c < cap_max; ++c) {
                    if (cap_k[c] >= 0) {
                        cap[c].done = true;
                        cap[c].pos  = group[cap_k[c]].ub->pos[0] - 1;
                        cap_record(c, cap_backend());
                    }
                }
            }
            cap_in_graph = false;
            return st;
        }
        cap_in_graph = false;
        const size_t over = ggml_backend_sched_get_alloc_over(sched.get());
        for (auto & [node, arr] : srcs) {
            std::copy(arr.begin(), arr.end(), node->src);
        }
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_use_full_hash(sched.get());
        // halves (the seed source may end up in an earlier graph than the ubatch it seeds: the seed reads the routing and the expert
        // copies of the source graph, so the cache ends up the same)
        const size_t mid = a + (b - a)/2;
        if (over == 0) {
            return GGML_STATUS_ALLOC_FAILED; // out of memory: the members not computed yet one by one, unseeded (below)
        }
        lm_n_split++;
        lm_refused_min = std::min(lm_refused_min, b - a);
        if (lm_verbose) {
            size_t free_min = SIZE_MAX;
            for (auto & bk : backend_ptrs) {
                ggml_backend_dev_t dev = ggml_backend_get_device(bk);
                if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                    size_t fr = 0, tot = 0;
                    ggml_backend_dev_memory(dev, &fr, &tot);
                    free_min = std::min(free_min, fr);
                }
            }
            LLAMA_LOG_INFO("decode_layer_major: the graph of ubatches [%zu, %zu) would grow the compute buffers by %.2f MiB (%.2f MiB free): split at %zu\n",
                a, b, over/1048576.0, free_min == SIZE_MAX ? -1.0 : free_min/1048576.0, mid);
        }
        st = run_range(a, mid);
        if (st != GGML_STATUS_SUCCESS) {
            return st;
        }
        return run_range(mid, b);
    };

    if (gm != nullptr) {
        status = run_range(0, group.size());
        lm_n_graphs += n_sub;
        lm_n_ubatches += (int64_t) group.size();
        if (status == GGML_STATUS_ALLOC_FAILED) {
            // the merged graphs need more compute memory than the device has: the members not computed yet one after the other, as usual
            LLAMA_LOG_WARN("%s: the graph of %zu ubatches did not allocate, ubatches %zu.. computed one after the other\n", __func__,
                group.size(), n_done);
            lm_n_fallback++;
            lm_seed_active = false;
            moe_cache_seed_prev = false;
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
            gm = nullptr;
            for (size_t k = n_done; k < group.size(); ++k) {
                ggml_backend_sched_reset(sched.get());
                ggml_backend_sched_use_full_hash(sched.get());
                cap_before(k);
                status = run(group[k].gf);
                if (status != GGML_STATUS_SUCCESS) {
                    break;
                }
                extract_upto(k + 1);
            }
        }
    } else {
        // one ubatch, or graphs that do not cut alike: one after the other, as usual
        if (group.size() > 1) {
            lm_n_fallback++;
            LLAMA_LOG_WARN("%s: the graphs of %zu ubatches do not cut alike, computed one after the other\n", __func__, group.size());
        }
        for (size_t k = 0; k < group.size(); ++k) {
            if (k > 0) {
                ggml_backend_sched_reset(sched.get());
                ggml_backend_sched_use_full_hash(sched.get());
            }
            cap_before(k);
            status = run(group[k].gf);
            if (status != GGML_STATUS_SUCCESS) {
                break;
            }
            extract_upto(k + 1);
        }
    }
    for (auto & b : backend_ptrs) {
        buf_after += ggml_backend_sched_get_buffer_size(sched.get(), b);
    }
    // the arenas a growing group outgrew: their inputs were copied when the graphs that read them returned
    lm_arena_old.clear();
    if (lm_verbose) {
        LLAMA_LOG_INFO("%s: compute buffers %.2f -> %.2f MiB\n", __func__, buf_before/1048576.0, buf_after/1048576.0);
        LLAMA_LOG_INFO("%s: %zu ubatches (%" PRId64 " tokens each at most), %zu pieces, %d nodes, %d splits: build %.2f ms (graphs %.2f, inputs %.2f), "
            "compute (queued) %.2f ms%s, %d graphs, last member %u tokens (%" PRId64 " tails so far)\n",
            __func__, group.size(), n_tokens_max, cut_w.size(), gm ? ggml_graph_n_nodes(gm) : -1, ggml_backend_sched_get_n_splits(sched.get()),
            (t1_us - t0_us)/1000.0, t_build_us/1000.0, t_set_us/1000.0, (ggml_time_us() - t1_us)/1000.0, gm ? "" : " (one by one)", n_sub,
            group.back().ub->n_tokens, lm_n_tail);
    }

    bool seeded = lm_seed_active && !lm_seeded.empty();
    if (seeded) {
        lm_n_seeded++;
    }
    // the device seed of the last member, as its own graph would have done it (see rp_k), from its routing where the graph left it
    bool replayed = false;
    if (rp_k > 0 && gm != nullptr && status == GGML_STATUS_SUCCESS) {
        ggml_backend_sched_synchronize(sched.get());
        std::vector<std::pair<int32_t, ggml_tensor *>> rec;
        for (const auto & [il, ids] : rp_ids) {
            rec.emplace_back(il, const_cast<ggml_tensor *>(ids));
        }
        const int64_t nt = group.back().ub->n_tokens;
        size_t rp_bytes = 0;
        double rp_ms    = 0.0;
        replayed = moe_cache->devseed_replay(rec, nt, rp_bytes, rp_ms);
        if (lm_verbose || !lm_replay_logged) {
            lm_replay_logged = true;
            LLAMA_LOG_INFO("%s: device seed of ubatch %zu (%" PRId64 " tokens, after the break at %" PRId64 ") replayed after the merged graph: %s, "
                "%zu layers, %.2f MiB uploaded again in %.2f ms (LLAMA_MOE_LAYER_MAJOR_SEED_REPLAY=1)\n", __func__, rp_k, nt,
                group[rp_k].n_tokens_before, replayed ? "done" : "not done", rec.size(), rp_bytes/1048576.0, rp_ms);
        }
    }
    if (lm_verbose && src_k >= 0) {
        size_t n_ws = 0;
        for (const auto & a : lm_ask) {
            n_ws += a.second.ws.size();
        }
        std::string miss;
        for (const auto & a : lm_ask) {
            for (const ggml_tensor * w : a.second.ws) {
                if (lm_seeded.count(w) == 0 && miss.size() < 200) {
                    miss += std::string(" ") + w->name + (lm_staged.count(w) ? "(staged)" : "(not staged)") + (a.second.done ? "" : "(no routing)");
                }
            }
        }
        LLAMA_LOG_INFO("%s: seed from the routing of ubatch %d of the group (seeding ubatch %d): %zu layers, %zu of %zu expert tensors seeded%s%s\n",
            __func__, src_k, seed_k, lm_ask.size(), lm_seeded.size(), n_ws, miss.empty() ? "" : "; not:", miss.c_str());
    }
    if (lm_verbose && lm_seed_async) {
        LLAMA_LOG_INFO("%s: async seed: %" PRId64 " routings read without a sync, %.2f ms waited for them in total\n", __func__, lm_n_async,
            lm_n_async_wait_us/1000.0);
    }
    lm_pending.clear();
    lm_staged_seq.clear();
    lm_seed_async       = false;
    lm_seed_active      = false;
    moe_cache_seed_prev = false;
    lm_ask.clear();
    lm_staged.clear();
    lm_seeded.clear();
    ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

    // the state moe_cache_seed_end leaves after the last ubatch of the group: its routing is kept for a later seed while the prompt goes on
    moe_cache_ids.clear();
    moe_cache_seeded = seeded || replayed;
    {
        const auto & last = group.back();
        if (status == GGML_STATUS_SUCCESS && last.n_outputs == 0 && last.ub->n_seqs_unq == 1 && moe_cache_seed_sync) {
            moe_cache_ids = lm_record;
            llama_pos pos_max = last.ub->pos[0];
            for (uint32_t i = 1; i < last.ub->n_tokens; ++i) {
                pos_max = std::max(pos_max, last.ub->pos[i]);
            }
            moe_cache_seq      = last.ub->seq_id_unq[0];
            moe_cache_pos_next = pos_max + 1;
        } else {
            moe_cache_seq      = -1;
            moe_cache_pos_next = -1;
            moe_cache_ids.clear();
            moe_cache->clear_seed_counts();
        }
    }

    static const bool lm_tail_sync = getenv("LLAMA_LM_TAIL_SYNC") != nullptr && atoi(getenv("LLAMA_LM_TAIL_SYNC")) != 0;
    if (lm_tail_sync && !group.empty() && (int64_t) group.back().ub->n_tokens < prefetch_min_batch) {
        synchronize();
        for (auto & b : backends) {
            ggml_backend_synchronize(b.get());
        }
    }
    if (lm_released) {
        // the slots come back with the experts they held before; LLAMA_MOE_LM_RELEASE_ROUTED=1: with those the last ubatch of the group
        // used most (its routing, read before the compute buffers are freed). G4 4 GB, Qwen3.6 Q3_K_M + MTP 2, 3 launches: decode 2k / 8k
        // 136.1 / 150.0 tok/s held before, 138.0 / 146.1 routed (release off 133.6 / 149.3); long-prompt KLD against the all-device run,
        // first token mean 0.00076 held before, 0.00134 routed (the -fa off band 0.00031, k12 0.00118)
        // k15-small4: on by default without a draft model (no-draft layer-major prompts, LLAMA_MOE_LAYER_MAJOR_NODRAFT): the slots of a fresh
        // cache held nothing before the group, so the held-before refill uploaded nothing and decode started from an empty cache (G4 4 GB,
        // Qwen3.6 Q4_K_XL: 238 experts admitted by the prompts against 1399 ubatch by ubatch, decode -0.7..-2.2%). It changes only which
        // experts the cache holds. LLAMA_MOE_LM_RELEASE_ROUTED=0 off, =1 on with a draft too
        static const char * lm_rr_env = getenv("LLAMA_MOE_LM_RELEASE_ROUTED");
        const bool lm_release_routed = lm_rr_env != nullptr ? atoi(lm_rr_env) != 0 : !cparams.spec_draft;
        if (lm_release_routed && status == GGML_STATUS_SUCCESS) {
            moe_cache->set_refill_routed_once(true);
            moe_cache_ids = lm_record;
            moe_cache_refill_from_ids();
        }
        // the routing of the group is in its compute buffers, which are freed: no later seed reads it
        moe_cache_ids.clear();
        moe_cache_seq      = -1;
        moe_cache_pos_next = -1;
        moe_cache->clear_seed_counts();
        // the compute buffers shrink back to what the next graph needs before the banks take their memory again
        ggml_backend_sched_free_device_buffers(sched.get());
        moe_cache->restore();
        lm_n_released++;
    } else if (buf_after > buf_before) {
        // k13-small: a group that grew the compute buffers leaves them at its size, and the decode after it computes in what is left: G4 at
        // 4 GB, Qwen3.6 Q3_K_M + MTP with LLAMA_MOE_LM_RELEASE=0, a fresh server's first request (2100 tokens): the group grew 502 -> 534 MiB
        // inside the 64 MiB headroom with 118 MiB left after it, the staging copies of the next graphs took the rest and the decode ran out
        // of device memory in the CUDA pool. With less than LLAMA_MOE_LM_SHRINK_FREE_MIB (192) free after the group the device compute
        // buffers are freed (the next graph allocates what it needs, as without the merge); the routing kept for the seed of the next ubatch
        // is copied to host memory first. At 512 the 8 GB cells (234-436 MiB left) shrank too, and the chat prompt after a long one lost
        // 8-12% to the reallocation (G4 run k13s6). 0 off
        static const size_t lm_shrink_free = (size_t) (getenv("LLAMA_MOE_LM_SHRINK_FREE_MIB") ? std::max(0, atoi(getenv("LLAMA_MOE_LM_SHRINK_FREE_MIB"))) : 192) << 20;
        size_t free_min = SIZE_MAX;
        for (auto & bk : backend_ptrs) {
            ggml_backend_dev_t dev = ggml_backend_get_device(bk);
            if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                size_t fr = 0, tot = 0;
                ggml_backend_dev_memory(dev, &fr, &tot);
                free_min = std::min(free_min, fr);
            }
        }
        if (free_min < lm_shrink_free) {
            ggml_backend_sched_synchronize(sched.get());
            lm_ids_host_keep();
            ggml_backend_sched_free_device_buffers(sched.get());
            if (lm_n_shrunk++ < 8) {
                LLAMA_LOG_INFO("%s: the group grew the compute buffers %.2f -> %.2f MiB with %.2f MiB free left: device compute buffers freed "
                    "(LLAMA_MOE_LM_SHRINK_FREE_MIB=0 keeps them)\n", __func__, buf_before/1048576.0, buf_after/1048576.0, free_min/1048576.0);
            }
        }
    }

    if (status != GGML_STATUS_SUCCESS) {
        llama_pos pos_min[LLAMA_MAX_SEQ];
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            pos_min[s] = std::numeric_limits<llama_pos>::max();
        }
        for (const auto & g : group) {
            for (uint32_t i = 0; i < g.ub->n_tokens; ++i) {
                const auto & seq_id = g.ub->seq_id[i][0];
                pos_min[seq_id] = std::min(pos_min[seq_id], g.ub->pos[i]);
            }
        }
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (pos_min[s] != std::numeric_limits<llama_pos>::max()) {
                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);
                memory->seq_rm(s, pos_min[s], -1);
            }
        }
        switch (status) {
            case GGML_STATUS_ABORTED:      ret =  2; break;
            case GGML_STATUS_ALLOC_FAILED: ret = -2; break;
            default:                       ret = -3; break;
        }
        return -1;
    }

    extract_upto(group.size()); // (none left: each graph extracts its members)

    // k15-pp3: the group's nextn rows are queued for the host; with ubatches still to come in this decode, an event after them lets an
    // MTP drafter read rows [0, n_tokens_prev) before the decode is done (llama_nextn_rows_wait)
    {
        static const bool rows_env = getenv("LLAMA_MTP_PROMPT_OVERLAP") == nullptr || atoi(getenv("LLAMA_MTP_PROMPT_OVERLAP")) != 0;
        auto * t_h = cparams.embeddings_nextn ? group.back().res->get_h_nextn() : nullptr;
        ggml_backend_t be_h = t_h ? ggml_backend_sched_get_tensor_backend(sched.get(), t_h) : nullptr;
        if (rows_env && more && be_h && embd_nextn.data && !cparams.embeddings_nextn_masked && !mtp_dev_h_this &&
                cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
            ggml_backend_dev_t dev = ggml_backend_get_device(be_h);
            if (rows_ev && rows_ev_dev != dev) {
                rows_ev.reset();
            }
            if (!rows_ev && dev) {
                rows_ev.reset(ggml_backend_event_new(dev));
                rows_ev_dev = dev;
            }
            if (rows_ev) {
                ggml_backend_event_record(rows_ev.get(), be_h);
                rows_ev_n = n_tokens_prev;
            }
        }
    }

    // the next graph is built anew
    gf_res_prev_active = nullptr; gf_res_epoch++;

    // k15-small4: the host memory of the group (its input arena, mostly the KQ masks of every member, and the members' graph results) is
    // freed after it instead of kept for the next prompt: kept, it stayed in the process after every long prompt (q3m 4 GB, 8k prompt:
    // RssAnon +235 MiB against ubatch by ubatch, T4 / G4 +160-176 MiB after the bench cells). The routing kept for the seed of the next
    // ubatch is copied to host memory first (as when the compute buffers shrink). LLAMA_MOE_LM_KEEP_HOST=1 (or LLAMA_MOE_LAYER_MAJOR_EAGER=1)
    // keeps them
    static const bool lm_keep_host = (getenv("LLAMA_MOE_LM_KEEP_HOST") != nullptr && atoi(getenv("LLAMA_MOE_LM_KEEP_HOST")) != 0) ||
                                     (getenv("LLAMA_MOE_LAYER_MAJOR_EAGER") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_EAGER")) != 0);
    if (!lm_keep_host) {
        ggml_backend_sched_synchronize(sched.get());
        lm_ids_host_keep();
        ggml_backend_sched_reset(sched.get());
        lm_arena.reset();
        lm_arena_old.clear();
        lm_arena_used = 0;
        lm_inp_buf.clear();
        lm_res.clear();
        // the backends' cached CUDA graphs of the group's splits (host-side node records of 16 ubatches, keyed by the freed nodes)
        for (auto & bk : backend_ptrs) {
            ggml_backend_dev_t dev = ggml_backend_get_device(bk);
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            auto fn = reg ? (size_t (*)(ggml_backend_t, int64_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_evict_since") : nullptr;
            if (fn != nullptr) {
                fn(bk, lm_t_start);
            }
        }
#if defined(__GLIBC__)
        // the graph results' metadata are large heap blocks that glibc keeps after free (its mmap threshold grows with the first such free)
        static const bool lm_trim = getenv("LLAMA_MOE_LM_TRIM") == nullptr || atoi(getenv("LLAMA_MOE_LM_TRIM")) != 0;
        if (lm_trim) {
            malloc_trim(0);
        }
#endif
    }

    return (int) group.size();
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = !cparams.embeddings || graph_has_logits;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    static const bool tail_dbg_h = getenv("LLAMA_LM_TAIL_DEBUG") != nullptr;
    if (tail_dbg_h && embd_nextn.has_data() && !embd_batch_idxs.empty() && embd_batch_idxs.size() >= 4) {
        synchronize();
        const size_t n = embd_batch_idxs.size();
        uint64_t h = 1469598103934665603ull;
        const uint8_t * p = (const uint8_t *) (embd_nextn.data + (n - 4)*n_embd_out);
        for (size_t i = 0; i < 4*n_embd_out*sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
        LLAMA_LOG_INFO("%s: tail debug: %zu rows, h_nextn of the last 4: %016llx\n", __func__, n, (unsigned long long) h);
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DEEPSEEK41 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    // k9-draft: recurrent per-token snapshots (n_rs_seq) add copies per recurrent layer and kept position; the 8 x n_tensors budget
    // of e.g. LFM2 ran out at n_rs_seq 8 (ngram-suffix drafts of 8 tokens: GGML_ASSERT(obj_new) in build_moe_ffn)
    if (cparams.n_rs_seq > 0) {
        res += 16u * cparams.n_rs_seq * model.hparams.n_layer();
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());
    moe_cache_ids.clear();

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr; gf_res_epoch++;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
    if (!hadamard_verified && gf && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
        llama_verify_hadamard_graph(gf, model.hdmd.rot, model.hdmd.inv, moe_cache.get());
        hadamard_verified = true;
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    prefetch_n_tokens = n_tokens;
    // a reserved graph sizes the memory with lending: the guard against lending to a graph that reads the cache through host
    // lookups applies when such a graph is allocated to run (see process_ubatch), which then takes the staging arena
    if (moe_cache) {
        moe_cache->set_graph(nullptr);
    }
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.hdmd        =*/ model.hdmd.rot.empty() ? nullptr : &model.hdmd,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.moe_full_last =*/ moe_full_last || getenv("LLAMA_MOE_LEND_FULL_LAST") == nullptr || atoi(getenv("LLAMA_MOE_LEND_FULL_LAST")) >= 2,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.h_dev_src   =*/ mtp_dev_h_src,
    };
}

// k11-decode: a small graph (a speculative verify window has its own node count) re-plans the graph allocator, so the ubatches of the
// next prompt would each re-plan it again as the KV view grows, every time with a synchronization of all backends (G4: prompts 2-7%
// slower with n-gram drafting on). This restores the worst-case pp plan of sched_reserve once. The allocator only plans addresses:
// outputs cannot change. LLAMA_PP_RERESERVE=0: off.
bool llama_context::restore_pp_plan() {
    // k13-fn: the slots are idle: the MoE cache's pending re-calibration of the CPU-miss host timings runs now, not inside a reply
    if (moe_cache && backend_cpu != nullptr && moe_cache->recal_pending()) {
        synchronize();
        compute_n_tokens = -1;
        set_compute_threads(false);
        compute_n_tokens = 0;
        moe_cache->recal(backend_cpu);
    }
    static const bool pp_rereserve = getenv("LLAMA_PP_RERESERVE") == nullptr || atoi(getenv("LLAMA_PP_RERESERVE")) != 0;
    if (!pp_rereserve || !sched_plan_small || !memory || moe_cache || model.hparams.no_alloc || sched_need_reserve) {
        return false;
    }
    sched_plan_small = false;
    synchronize();
    auto mctx_full = memory->init_full();
    if (!mctx_full) {
        return false;
    }
    const uint32_t n_seqs_r    = (model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_MINIMAX_01) ? 1 : cparams.n_seq_max;
    const uint32_t n_tokens_r  = std::min(cparams.n_ctx, cparams.n_ubatch);
    const uint32_t n_outputs_r = std::min(n_tokens_r, cparams.n_outputs_max);
    if (graph_reserve(n_tokens_r, n_seqs_r, n_outputs_r, mctx_full.get()) == nullptr) {
        LLAMA_LOG_WARN("%s: re-reserving the pp graph failed\n", __func__);
        return false;
    }
    if (++n_pp_rereserve == 1 || n_pp_rereserve % 100 == 0) {
        LLAMA_LOG_INFO("%s: a small graph re-planned the graph allocator: the pp graph is reserved again (%" PRId64 " times; LLAMA_PP_RERESERVE=0: off)\n",
                __func__, n_pp_rereserve);
    }
    return true;
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    if (lctx->prefetch_n_tokens <= 2*GF_SMALL_MAX) {
        lctx->sched_plan_small = true;
    }

    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), graph_size_changed, ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

void llama_context::ple_prefetch_ahead(const llama_ubatch & ubatch, const llama_memory_context_i * mctx, bool first) {
    const int64_t t_start_us = ggml_time_us();

    auto predict = [&](const llama_ubatch & ub, const llama_ubatch * pending, std::vector<int32_t> & out) {
        out.clear();
        const ggml_tensor * t = model.ple_rows(ub, pending, mctx, false, ple.scratch);
        if (!t) {
            ple.n_unpredicted++;
            return;
        }
        ple.n_predicted++;
        ple.n_rows += ple.scratch.size();
        for (const int32_t r : ple.scratch) {
            ple.n_skipped_rows += r < 0;
        }
        if (ple.verify) {
            out = ple.scratch;
        }
        if (ple_prefetcher) {
            ple_prefetcher->submit(std::move(ple.scratch));
        }
        ple.scratch.clear();
    };

    // the first ubatch: its predecessors are all in the cells already (this is the decode entry for generation)
    if (first) {
        predict(ubatch, nullptr, ple.pred_cur);
    } else {
        ple.pred_cur = std::move(ple.pred_next);
    }
    ple.pred_next.clear();

    // the next one, preceded by this one, which is not applied yet
    if (const llama_ubatch * next = mctx->peek_ubatch(1)) {
        predict(*next, &ubatch, ple.pred_next);
    }

    ple.t_predict_us += ggml_time_us() - t_start_us;
}

void llama_context::ple_prefetch_verify(const llama_ubatch & ubatch, const llama_memory_context_i * mctx) {
    if (ple.pred_cur.empty()) {
        return;
    }

    std::vector<int32_t> ref;
    if (!model.ple_rows(ubatch, nullptr, mctx, true, ref)) {
        return;
    }

    ple.n_checked++;

    if (ref.size() != ple.pred_cur.size()) {
        LLAMA_LOG_ERROR("%s: predicted %zu PLE rows, set_input gathers %zu\n", __func__, ple.pred_cur.size(), ref.size());
        ple.n_mismatch += std::max(ref.size(), ple.pred_cur.size());
    } else {
        for (size_t i = 0; i < ref.size(); ++i) {
            if (ple.pred_cur[i] >= 0 && ple.pred_cur[i] != ref[i]) {
                if (ple.n_mismatch < 8) {
                    LLAMA_LOG_ERROR("%s: PLE row %zu of the ubatch: predicted %d, set_input %d\n", __func__, i, ple.pred_cur[i], ref[i]);
                }
                ple.n_mismatch++;
            }
        }
    }

    ple.pred_cur.clear();
}

void llama_context::vram_watch_sample(bool force) {
    if (vram_watch < 0) {
        const char * env = getenv("LLAMA_VRAM_WATCH");
        vram_watch = env != nullptr && atoi(env) != 0 ? 1 : 0;
    }
    if (vram_watch == 0 || (!force && (vram_watch_n++ % 32) != 0)) {
        return;
    }
    if (vram_watch_min.size() != model.devices.size()) {
        vram_watch_min.assign(model.devices.size(), SIZE_MAX);
    }
    for (size_t i = 0; i < model.devices.size(); ++i) {
        ggml_backend_dev_t dev = model.devices[i].dev;
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        // the servers are usually killed, not freed: a new low (by 16 MiB or more) is logged when it is seen
        if (vram_watch_min[i] == SIZE_MAX || free + (16u << 20) <= vram_watch_min[i]) {
            LLAMA_LOG_INFO("%s: VRAM watch: %s free %.1f MiB after %" PRId64 " graphs\n", __func__, ggml_backend_dev_name(dev), free/1048576.0, vram_watch_n);
            vram_watch_min[i] = free;
        }
    }
}

// LLAMA_MOE_HOST_PIN=1 (opt-in, k11; the dedicated pool of ggml-org/llama.cpp#30971): the host-miss ubatches run on an own ggml
// threadpool, one worker per physical core (the first allowed SMT sibling, at most 32 and the cgroup quota), each pinned to its CPU
// (strict_cpu), polling LLAMA_MOE_HOST_POLL (default 50) before it sleeps; GGML_CPU_PIN_CALLER=1 also pins the calling thread to
// worker 0's CPU. Only builds without OpenMP use the pool's CPU masks (GGML_OPENMP=OFF); with OpenMP the team ignores them.
static const std::vector<int> & host_pin_cpus() {
    static const std::vector<int> cpus = [] {
        const char * e = getenv("LLAMA_MOE_HOST_PIN");
        return e != nullptr && atoi(e) != 0 ? llama_sysmem::physical_cpus(32) : std::vector<int>();
    }();
    return cpus;
}

int32_t llama_context::host_threads_init() const {
    const char * env = getenv("LLAMA_MOE_HOST_THREADS");
    const std::vector<int> & pin = host_pin_cpus();
    if (!pin.empty()) {
        // the pinned pool: its own threads even at the decode thread count (=N caps it)
        const int32_t n = env != nullptr && atoi(env) > 0 ? std::min<int32_t>(atoi(env), (int32_t) pin.size()) : (int32_t) pin.size();
        return n;
    }
    if (env != nullptr && strcmp(env, "auto") == 0) {
        // every usable logical CPU (affinity, cgroup quota), only when that is more than the decode threads
        const int32_t n = (int32_t) llama_sysmem::usable_cpus();
        return n > (int32_t) cparams.n_threads ? n : 0;
    }
    if (env != nullptr) {
        // =N: N threads (=0 off: the decode threads, as before k11)
        const int32_t n = atoi(env);
        return n > 0 && n != (int32_t) cparams.n_threads ? n : 0;
    }
    // opt-in: every usable logical CPU lost 8-9% decode on a G4 (48 vCPU, 24 -> 48 threads, Qwen3.6 UD-Q4_K_XL 8 GB + MTP, the same
    // host experts and text); =auto takes them anyway
    return 0;
}

bool llama_context::host_threads_for(int64_t n_tokens) const {
    // n_tokens < 0: the calibration of the host path
    return n_threads_host > 0 && moe_cache && n_tokens != 0 && (n_tokens < 0 || moe_cache->host_computes(n_tokens));
}

void llama_context::set_compute_threads(bool batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr && host_threads_for(compute_n_tokens)) {
        n_threads = n_threads_host;
        if (tp != nullptr) {
            // the attached pools have the threads of the caller: a pool of the host threads
            if (threadpool_host == nullptr) {
                auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
                auto * new_fn  = (decltype(ggml_threadpool_new)  *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
                auto * free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
                if (new_fn != nullptr && free_fn != nullptr) {
                    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads_host);
                    tpp.paused = true;
                    const std::vector<int> & pin = host_pin_cpus();
                    // one pinned pool for every context of the process (the MTP draft context computes between the target's graphs):
                    // two pools spinning on the same cores would compete, so it is shared and lives until the process exits
                    static ggml_threadpool_t pin_pool   = nullptr;
                    static int32_t           pin_pool_n = 0;
                    if (!pin.empty() && pin_pool != nullptr && pin_pool_n == n_threads_host) {
                        threadpool_host      = pin_pool;
                        threadpool_host_free = nullptr;
                    } else if (!pin.empty()) {
                        std::string list;
                        for (int32_t i = 0; i < n_threads_host && i < (int32_t) pin.size(); ++i) {
                            if (pin[i] < GGML_MAX_N_THREADS) {
                                tpp.cpumask[pin[i]] = true;
                                list += (list.empty() ? "" : ",") + std::to_string(pin[i]);
                            }
                        }
                        tpp.strict_cpu = true;
                        const char * poll = getenv("LLAMA_MOE_HOST_POLL");
                        tpp.poll = poll != nullptr ? (uint32_t) std::max(0, std::min(100, atoi(poll))) : 50;
                        LLAMA_LOG_INFO("%s: MoE host-miss pool: %d threads pinned to CPUs %s, poll %u (LLAMA_MOE_HOST_PIN)\n",
                            __func__, n_threads_host, list.c_str(), tpp.poll);
                        threadpool_host      = new_fn(&tpp);
                        threadpool_host_free = nullptr;
                        if (pin_pool == nullptr) {
                            pin_pool   = threadpool_host;
                            pin_pool_n = n_threads_host;
                        } else {
                            threadpool_host_free = free_fn;
                        }
                    } else {
                        threadpool_host      = new_fn(&tpp);
                        threadpool_host_free = free_fn;
                    }
                }
            }
            if (threadpool_host != nullptr) {
                tp = threadpool_host;
            } else {
                n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
            }
        }
    }

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    set_compute_threads(batched);

    if (moe_cache) {
        moe_cache->single_kick(compute_n_tokens);
    }

    copy_experts.reset();

    sync_pending = true;

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    if (moe_cache) {
        moe_cache->seed_end();
    }

    vram_watch_sample(false);

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

ggml_backend_sched_prefetch_mode llama_context::sched_prefetch_mode(const ggml_tensor * src, const ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // the slot maps of the MoE cache change between graph computations, they must go through the copy callback
    if (lctx->moe_cache && lctx->moe_cache->is_slot_map(src)) {
        return GGML_BACKEND_SCHED_PREFETCH_NONE;
    }
    // k15-lfm2: the prompt maps of in-place prompts: prefetched with the large ubatches (each staging slot of the copy queue has its own ring
    // of lent slots, see llama_moe_cache::prompt_copy), copied when the split starts otherwise
    if (lctx->moe_cache && lctx->moe_cache->prompt_layer(src) >= 0) {
        return lctx->prefetch_n_tokens >= lctx->prefetch_min_batch ? GGML_BACKEND_SCHED_PREFETCH_ASYNC : GGML_BACKEND_SCHED_PREFETCH_NONE;
    }

    // small ubatches are not prefetched, their copies reuse the staging arena of the prompt (if any) instead of growing the graph allocator next to it,
    // except the dense host weights of medium ubatches (the experts of MUL_MAT_ID keep the rules below, a medium ubatch never copies whole expert banks)
    if (lctx->prefetch_n_tokens < lctx->prefetch_min_batch) {
        if (lctx->prefetch_n_tokens < lctx->prefetch_dense_min_batch) {
            return GGML_BACKEND_SCHED_PREFETCH_SYNC;
        }
        if (lctx->moe_cache && lctx->moe_cache->get_experts(src) != nullptr) {
            return GGML_BACKEND_SCHED_PREFETCH_SYNC;
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if ((node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_ADD_ID) && (node->src[0] == dst || node->src[1] == dst || node->src[2] == dst)) {
                return GGML_BACKEND_SCHED_PREFETCH_SYNC;
            }
        }
        return GGML_BACKEND_SCHED_PREFETCH_ASYNC;
    }

    // host memory that is not a weight: only the rows of a host KV cache that a prompt's attention reads from its GPU copy
    // (kv-layer-prefetch, the scheduler checks that no earlier split writes them)
    const ggml_tensor * root = src;
    while (root->view_src != nullptr) {
        root = root->view_src;
    }
    if (root->buffer != nullptr && ggml_backend_buffer_get_usage(root->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return strncmp(root->name, "cache_", 6) == 0 ? GGML_BACKEND_SCHED_PREFETCH_ASYNC : GGML_BACKEND_SCHED_PREFETCH_NONE;
    }

    // experts: the copy callback can copy them ahead only if it does not need the ids, which are computed by an earlier split
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == dst && !lctx->sched_copy_all_experts(src, node->src[2])) {
            return GGML_BACKEND_SCHED_PREFETCH_SYNC;
        }
    }

    return GGML_BACKEND_SCHED_PREFETCH_ASYNC;
}

void llama_context::sched_stream_plan(const ggml_tensor * const * srcs, int n_srcs, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);
    lctx->wstore->plan(srcs, n_srcs);
}

bool llama_context::sched_stream_source(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, size_t offset, size_t size, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);
    return lctx->wstore->upload(backend, src, dst, offset, offset, size);
}

void llama_context::sched_lend(ggml_backend_t backend, ggml_backend_sched_lend_copy * copies, int n_copies, int depth, bool dry, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);
    // the copies of medium ubatches (see sched_prefetch_mode) use the staging arena of the large ones, the MoE cache lends only to large
    // ones; except the SYNC copies the scheduler offers (depth 1) on a device where the cache lends to the large ones: no arena exists
    // there, and the graph allocator cannot grow by the copies of a medium ubatch that copies its experts (e.g. the last, shorter ubatch
    // of a prompt); small ubatches that read every expert from the cache are left as they are
    if (n_copies > 0 && lctx->prefetch_n_tokens < lctx->prefetch_min_batch) {
        const bool medium = depth == 1 && lctx->moe_cache && lctx->moe_cache->lends_on(backend) &&
            lctx->prefetch_n_tokens > lctx->moe_cache->n_tokens_cached();
        if (!medium) {
            return;
        }
    }
    if (lctx->moe_cache) {
        lctx->moe_cache->lend(backend, copies, n_copies, depth, dry, lctx->prefetch_n_tokens);
    }
}

bool llama_context::sched_copy_all_experts(const ggml_tensor * src, const ggml_tensor * ids) const {
    // the ubatch that seeds the MoE cache counts the experts used by the layers it caches
    if (moe_cache_seed && moe_cache && moe_cache->get_experts(src) != nullptr) {
        return false;
    }
    // nearly every expert is used, copying all of them avoids waiting for the ids
    return ids->ne[1] >= prefetch_min_batch || ggml_nelements(ids) >= 8*src->ne[2];
}

// k15-pp: the seeds of the layer-major group whose routing the eval callback reads asynchronously are issued here, on the copy queue
// behind the layer's expert copies (as the synchronous seed issues them), see lm_seed_resolve_due
bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);
    if (!lctx->lm_seed_async || !lctx->lm_seed_active) {
        return sched_copy_experts_body(backend, src, dst, graph, user_data);
    }
    if (graph != lctx->lm_copy_graph) {
        lctx->lm_copy_graph = graph;
        lctx->lm_copy_seq++;
    }
    lctx->lm_seed_resolve_due(true);
    const bool ok = sched_copy_experts_body(backend, src, dst, graph, user_data);
    if (lctx->lm_staged.count(src) > 0 && lctx->lm_staged_seq.count(src) == 0) {
        lctx->lm_staged_seq[src] = lctx->lm_copy_seq;
    }
    lctx->lm_seed_resolve_due(false);
    return ok;
}

bool llama_context::sched_copy_experts_body(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // k15-lfm2: the prompt maps of in-place prompts (LLAMA_MOE_PROMPT_INPLACE=1)
    if (lctx->moe_cache) {
        const int32_t pil = lctx->moe_cache->prompt_layer(src);
        if (pil >= 0) {
            return lctx->moe_prompt_copy(backend, dst, graph, pil);
        }
    }

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the scheduler passes its copy queue backend only for prefetched copies
    const bool on_copy_queue = std::none_of(lctx->backends.begin(), lctx->backends.end(),
            [backend](const ggml_backend_ptr & b) { return b.get() == backend; });

    // other host weights: on the regular path the weight store reads them from the model file (prefetched copies go through the source callback)
    auto copy_whole = [&]() {
        return !on_copy_queue && lctx->wstore && lctx->wstore->covers(src) && lctx->wstore->upload(backend, src, dst, 0, 0, ggml_nbytes(src));
    };

    // the ids must be computed before the split starts. The product that reads the copy usually is the first node of the split, but a
    // backend may reorder the nodes of a split: the CUDA side-stream plan (ggml_cuda_side_mv_plan) moves a few-row product whose sources
    // come from an earlier split (e.g. the shared expert gate of Qwen3-Next style models) to its front. Taking only the first node then
    // uploaded the whole down_exps tensor of every layer, cached experts included (Qwen3.8-Flash-Next UD-IQ4_XS, G4 32 GB: 46.6 instead of
    // 36.9 GB per prompt ubatch, 2k prompt -15%). So the split is searched for the product, and its ids must not be computed in the split.
    // LLAMA_MOE_COPY_SCAN=0: only the first node (as before)
    const int n_nodes = ggml_graph_n_nodes(graph);
    if (n_nodes == 0) {
        return copy_whole();
    }
    static const bool scan = getenv("LLAMA_MOE_COPY_SCAN") == nullptr || atoi(getenv("LLAMA_MOE_COPY_SCAN")) != 0;
    const ggml_tensor * node = nullptr;
    for (int i = 0; i < (scan ? n_nodes : 1) && node == nullptr; i++) {
        const ggml_tensor * t = ggml_graph_node(graph, i);
        if (t->op == GGML_OP_MUL_MAT_ID && t->src[0] == dst) {
            node = t;
        }
    }
    if (node == nullptr) {
        return copy_whole();
    }
    if (node != ggml_graph_node(graph, 0)) {
        const ggml_tensor * ids_n = node->src[2];
        for (int i = 0; i < n_nodes; i++) {
            const ggml_tensor * t = ggml_graph_node(graph, i);
            if (t == ids_n || (ids_n->view_src != nullptr && t == ids_n->view_src)) {
                return copy_whole();
            }
        }
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    // the ubatch that seeds the MoE cache counts the experts used by the layers it caches
    const bool seed = lctx->moe_cache_seed && lctx->moe_cache && lctx->moe_cache->get_experts(src) != nullptr;

    // the scheduler passes its copy queue backend only for prefetched copies (nearly all experts used), which may run before the ids exist: copy all
    // on the regular path the ids are known: copy only the used experts
    // a graph of several ubatches (decode_layer_major) reads the copy for all of them, the product found routes only the first: copy all
    const bool all = on_copy_queue || lctx->lm_merged_active;

    if (ids != st.ids || (int64_t) st.used.size() != n_expert || (seed && st.counts.empty()) || all != st.all) {
        st.all = all;
        if (all) {
            st.used.assign(n_expert, true);
            st.counts.clear();
        } else {
            // the ids are computed by an earlier split, on the backend the scheduler assigned to them
            ggml_backend_t backend_ids = ggml_backend_sched_get_tensor_backend(lctx->sched.get(), const_cast<ggml_tensor *>(ids));
            if (backend_ids == nullptr) {
                backend_ids = backend;
            }
            st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
            ggml_backend_tensor_get_async(backend_ids, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
            ggml_backend_synchronize(backend_ids);

            st.used.assign(n_expert, false);
            st.counts.assign(n_expert, 0);
            for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                    const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                    GGML_ASSERT(id >= 0 && id < n_expert);
                    st.used[id] = true;
                    st.counts[id]++;
                }
            }
        }

        st.ids = ids;
    }

    // group consecutive experts and copy them together
    // batched (e.g. CUDA >= 12.8): every cached run is copied from the MoE cache in one call with the uploads
    // otherwise only long cached runs are copied from the cache (many small copies cost more than they save), before the uploads
    const bool batch = lctx->moe_cache && lctx->moe_cache->get_experts(src) != nullptr && ggml_backend_tensor_copy_regions_async(backend, nullptr, 0);

    auto plan = [&](bool batched) {
        st.uploads.clear();
        st.regions.clear();
        for (int64_t first = 0; first < n_expert; ) {
            if (!st.used[first]) {
                first++;
                continue;
            }
            int64_t last = first;
            while (last + 1 < n_expert && st.used[last + 1]) {
                last++;
            }

            // the experts in the MoE cache are copied from device memory, the others are uploaded
            int64_t next = first;
            for (int64_t e = first; e <= last && lctx->moe_cache; ) {
                const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last, batched ? &st.regions : nullptr);
                if (n <= 0) {
                    e += n == 0 ? 1 : -n;
                    continue;
                }
                if (next < e) {
                    st.uploads.push_back({ next*expert_size, (e - next)*expert_size });
                    if (!batched) {
                        lctx->moe_cache->count_upload(src, e - next, (e - next)*expert_size);
                    }
                }
                e   += n;
                next = e;
            }

            // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
            const size_t offset  = next*expert_size;
            const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
            const size_t size    = (last + 1 - next)*expert_size + padding;
            if (size > 0) {
                st.uploads.push_back({ offset, size });
                if (lctx->moe_cache && !batched) {
                    lctx->moe_cache->count_upload(src, last + 1 - next, size);
                }
            }

            first = last + 1;
        }
    };

    bool done = false;
    if (batch) {
        plan(true);
        for (const auto & [offset, size] : st.uploads) {
            st.regions.push_back({ src, offset, dst, offset, size });
        }
        // fails (nothing copied) e.g. when the host weights are not pinned
        if (ggml_backend_tensor_copy_regions_async(backend, st.regions.data(), st.regions.size())) {
            lctx->moe_cache->count_batch(src, st.regions);
            done = true;
        }
    }
    if (!done) {
        plan(false);
        // the weight store reads the uploads from the model file (prefetched ahead into its ring, or all of them at once)
        if (!lctx->wstore || !lctx->wstore->covers(src) || !lctx->wstore->upload_runs(backend, src, dst, st.uploads)) {
            for (const auto & [offset, size] : st.uploads) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);
            }
        }
    }

    // st.used: the experts staged in dst, the only ones a seed may copy into the cache
    if (seed && !st.counts.empty()) {
        lctx->moe_cache->seed(backend, src, dst, st.counts, st.used);
    } else if (lctx->moe_cache_seed_prev && lctx->moe_cache->get_experts(src) != nullptr) {
        // all experts copied (prefetched): use the routing of an earlier ubatch; else the ids of this ubatch were read and only its routed experts staged
        const std::vector<int32_t> * counts = st.all ? lctx->moe_cache->get_seed_counts(src) : &st.counts;
        if (counts != nullptr && !counts->empty() && (!lctx->lm_seed_active || lctx->lm_seeded.insert(src).second)) {
            lctx->moe_cache->seed(backend, src, dst, *counts, st.used);
        }
    }
    // layer-major graph: the routing that seeds this layer may come later, the seed then copies from this copy (see lm_eval_cb)
    if (lctx->lm_seed_active && lctx->moe_cache->get_experts(src) != nullptr) {
        static const bool tail_dbg_s = getenv("LLAMA_LM_TAIL_DEBUG") != nullptr;
        if (tail_dbg_s && lctx->lm_staged.count(src) > 0) {
            LLAMA_LOG_INFO("%s: tail debug: a second staged copy of %s (all %d, previous dst %p, this %p)\n", __func__, src->name, (int) st.all,
                (void *) lctx->lm_staged[src].second, (void *) dst);
        }
        lctx->lm_staged[src] = { backend, dst };
    }

    // the routing of a large ubatch is kept for a later seed, once per layer (the expert tensors of a layer share their ids)
    if (lctx->moe_cache_record && (ids->flags & GGML_TENSOR_FLAG_OUTPUT) && lctx->moe_cache->get_experts(src) != nullptr &&
            std::none_of(lctx->moe_cache_ids.begin(), lctx->moe_cache_ids.end(), [&](const auto & r) { return r.second == ids; })) {
        lctx->moe_cache_ids.emplace_back(src, ids);
    }

    return true;
}

void llama_context::moe_cache_seed_begin(const llama_ubatch & ubatch, bool last_large, int64_t n_outputs_ub) {
    moe_inplace_ok = true; // k15-lfm2: unless this ubatch seeds (below)
    moe_cache->prompt_ubatch(true);
    const bool large = !moe_cache->small_batch(ubatch.n_tokens);
    if (!moe_cache_seed_enabled) {
        return;
    }

    // the stored routing is used only while the prompt continues: same single sequence at the next position
    const bool cont = ubatch.n_seqs_unq == 1 && ubatch.seq_id_unq[0] == moe_cache_seq && ubatch.pos[0] == moe_cache_pos_next;
    if (!cont) {
        moe_cache->clear_seed_counts();
    }

    // a ubatch below the prefetch threshold copies only its routed experts, so it seeds from its own ids (a stored routing could name unstaged experts)
    const bool prefetch = ubatch.n_tokens >= prefetch_min_batch;

    // LLAMA_MOE_CACHE_SEED=2 (default): read the routing outputs of the previous large ubatch before this graph replaces them, unless this ubatch
    // records newer ones without seeding
    if (moe_cache_seed_sync && cont && !moe_cache_ids.empty() && n_outputs_ub == 0 && (!large || (last_large && prefetch))) {
        ggml_backend_sched_synchronize(sched.get());
        std::vector<int32_t> ids_data;
        std::vector<int32_t> counts;
        for (const auto & [w, ids] : moe_cache_ids) {
            GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) && ids->buffer != nullptr);
            ids_data.resize(ggml_nelements(ids));
            ggml_backend_tensor_get(ids, ids_data.data(), 0, ggml_nbytes(ids));
            counts.assign(w->ne[2], 0);
            for (const int32_t id : ids_data) {
                GGML_ASSERT(id >= 0 && id < w->ne[2]);
                counts[id]++;
            }
            moe_cache->set_seed_counts(w, counts);
            if (lm_verbose) {
                uint64_t h = 1469598103934665603ull;
                for (const int32_t c : counts) { h ^= (uint64_t) c; h *= 1099511628211ull; }
                LLAMA_LOG_INFO("%s: seed counts %s from %s: %016llx\n", __func__, w->name, ids->name, (unsigned long long) h);
            }
        }
    }
    // a ubatch with outputs after a large one that did not seed ends the prompt (e.g. the server runs the last tokens as a separate batch)
    if (cont && !moe_cache_ids.empty() && n_outputs_ub != 0 && !large && !moe_cache_seeded && moe_cache->refills_routed()) {
        moe_cache_refill_from_ids();
    }
    moe_cache_ids.clear();

    // the last large ubatch of the batch seeds the cache even without outputs: a caller may run the last prompt tokens as a separate small batch (e.g. server checkpoints)
    // a ubatch below the prefetch threshold reads its ids for its copies anyway, it seeds from them
    // a prefetched ubatch: =2 (default) seeds from the routing of an earlier ubatch if any, else reads its own ids first (this serializes
    // its expert uploads with the compute of every layer); =1 does not seed it
    // strata-gap: also when prompts do not seed (n_ubatch >= LLAMA_MOE_CACHE_SEED_AUTO) but the lent slots are refilled by routing
    moe_cache_record    = large && prefetch_min_batch != INT64_MAX && (moe_cache_seed_sync || moe_cache->refills_routed());
    moe_cache_seed      = false;
    moe_cache_seed_prev = false;
    moe_cache_devseed   = false;
    if (last_large) {
        if (!prefetch) {
            moe_cache_seed = true;
        } else if (moe_cache_devseed_on) {
            moe_cache_devseed = true;
        } else if (moe_cache_seed_sync) {
            if (n_outputs_ub == 0 && moe_cache->has_seed_counts()) {
                moe_cache_seed_prev = true;
            } else if (moe_cache_seed_fits) {
                moe_cache_seed = true;
            } else {
                // the graph of this seed was not reserved (see sched_reserve_seed), the ubatch runs as with LLAMA_MOE_CACHE_SEED=1
                moe_cache->count_seed_skipped();
            }
        }
    }
    // k15-lfm2: a ubatch that seeds the cache from its own routing keeps the staging copies (the seed reads them), the others may read in
    // place; one that seeds from the routing of an earlier ubatch plans its seeds before the graph (see prompt_seed_plan)
    moe_inplace_ok = !(moe_cache_seed || moe_cache_devseed || (moe_cache_seed_prev && lm_seed_active));
    moe_cache->prompt_ubatch(moe_inplace_ok, moe_cache_seed_prev);
    moe_inplace_ok = moe_cache->prompt_ubatch_on();
}

bool llama_context::moe_prompt_copy(ggml_backend_t backend, ggml_tensor * dst, ggml_cgraph * graph, int32_t il) {
    // the routing of the layer: the ids of the get_rows that reads the copy of the prompt map (a view of the selected experts output)
    const ggml_tensor * sel = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph) && sel == nullptr; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
            sel = node->src[1];
        }
    }
    while (sel != nullptr && sel->view_src != nullptr) {
        sel = sel->view_src;
    }
    const std::vector<ggml_tensor *> experts = moe_cache->prompt_experts(il);
    const bool sel_ok = sel != nullptr && (sel->flags & GGML_TENSOR_FLAG_OUTPUT) && sel->type == GGML_TYPE_I32 && ggml_is_contiguous(sel);
    if (moe_cache_record && sel_ok) {
        for (const ggml_tensor * w : experts) {
            if (std::none_of(moe_cache_ids.begin(), moe_cache_ids.end(), [&](const auto & r) { return r.first == w; })) {
                moe_cache_ids.emplace_back(w, sel);
            }
        }
    }
    // (a ubatch that seeds the cache is staged, see moe_cache_seed_begin)
    return moe_cache->prompt_copy(backend, il, dst, nullptr);
}

void llama_context::moe_cache_seed_end(const llama_ubatch & ubatch, int64_t n_outputs_ub) {
    moe_cache->prompt_flush();
    if (moe_cache->seed_check() && (moe_cache_seed || moe_cache_seed_prev)) {
        ggml_backend_sched_synchronize(sched.get());
        moe_cache->seed_verify();
    }

    moe_cache_seeded    = moe_cache_seed || moe_cache_seed_prev || moe_cache_devseed;
    moe_cache_devseed   = false;
    const bool recorded = moe_cache_record;
    moe_cache_seed      = false;
    moe_cache_seed_prev = false;
    moe_cache_record    = false;

    // the prompt ends at a ubatch with outputs (or a failed one), the next one may belong to another prompt
    if (n_outputs_ub != 0 || ubatch.n_seqs_unq != 1) {
        if (n_outputs_ub != 0 && recorded && !moe_cache_seeded && !moe_cache_ids.empty() && moe_cache->refills_routed()) {
            moe_cache_refill_from_ids();
        }
        moe_cache_seq      = -1;
        moe_cache_pos_next = -1;
        moe_cache_ids.clear();
        moe_cache->clear_seed_counts();
        return;
    }
    llama_pos pos_max = ubatch.pos[0];
    for (uint32_t i = 1; i < ubatch.n_tokens; ++i) {
        pos_max = std::max(pos_max, ubatch.pos[i]);
    }
    moe_cache_seq      = ubatch.seq_id_unq[0];
    moe_cache_pos_next = pos_max + 1;
}

void llama_context::lm_ids_host_keep() {
    if (moe_cache_ids.empty()) {
        return;
    }
    ggml_init_params params = { moe_cache_ids.size()*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> copies;
    for (const auto & [w, ids] : moe_cache_ids) {
        ggml_tensor * c = ggml_dup_tensor(ctx.get(), ids);
        ggml_set_name(c, ids->name);
        copies.emplace_back(w, c);
    }
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    if (!buf) {
        moe_cache_ids.clear();
        return;
    }
    std::vector<uint8_t> tmp;
    for (size_t i = 0; i < copies.size(); ++i) {
        const ggml_tensor * src = moe_cache_ids[i].second;
        tmp.resize(ggml_nbytes(src));
        ggml_backend_tensor_get(src, tmp.data(), 0, tmp.size());
        ggml_backend_tensor_set(copies[i].second, tmp.data(), 0, tmp.size());
        moe_cache_ids[i].second = copies[i].second;
    }
    lm_ids_ctx = std::move(ctx);
    lm_ids_buf = std::move(buf);
}

void llama_context::moe_cache_refill_from_ids() {
    ggml_backend_sched_synchronize(sched.get());
    std::vector<int32_t> ids_data;
    std::vector<int32_t> counts;
    for (const auto & [w, ids] : moe_cache_ids) {
        if (ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids) || ids->buffer == nullptr) {
            continue;
        }
        ids_data.resize(ggml_nelements(ids));
        ggml_backend_tensor_get(ids, ids_data.data(), 0, ggml_nbytes(ids));
        counts.assign(w->ne[2], 0);
        for (const int32_t id : ids_data) {
            if (id >= 0 && id < w->ne[2]) {
                counts[id]++;
            }
        }
        moe_cache->set_refill_counts(w, counts);
    }
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

// like llama_io_write_host, but the tensor reads are collected for llama_context::state_get_async()
class llama_io_write_host_async : public llama_io_write_i {
public:
    llama_io_write_host_async(uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

    std::vector<llama_state_write_info> winfos;

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    state_async_wait();

    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_async_wait();

    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    if (flags & LLAMA_STATE_SEQ_FLAGS_CAPTURED) {
        // k15-pp: the captured rows and position in place of the current ones for the whole write, then back (synchronous)
        int cs = cap_sel >= 0 && cap[cap_sel].done && cap[cap_sel].seq == seq_id && !cap[cap_sel].regions.empty() ? cap_sel : -1;
        if (cs < 0) {
            for (int c = 0; c < cap_max; ++c) {
                if (cap[c].done && cap[c].seq == seq_id && !cap[c].regions.empty() && (cs < 0 || cap[c].idx < cap[cs].idx)) {
                    cs = c;
                }
            }
        }
        if (cs < 0) {
            LLAMA_LOG_ERROR("%s: no captured state for sequence %d\n", __func__, seq_id);
            return 0;
        }
        // the snapshot tensors are read in place of the rows (asynchronously too: a later capture overwrites them in stream order)
        flags &= ~(LLAMA_STATE_SEQ_FLAGS_CAPTURED | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        auto * hyb = dynamic_cast<llama_memory_hybrid *>(memory.get());
        auto * rec = hyb ? hyb->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(memory.get());
        std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> map;
        for (const auto & r : cap[cs].regions) {
            map.emplace_back(r.t, r.snap);
        }
        rec->cap_map = &map;
        rec->cap_pos = cap[cs].pos;
        cap_save = cs;
        size_t n = 0;
        try {
            n = state_seq_get_data(seq_id, dst, size, flags);
        } catch (...) {
            rec->cap_map = nullptr;
            cap_save = -1;
            throw;
        }
        rec->cap_map = nullptr;
        cap_save = -1;
        return n;
    }
    if (flags & LLAMA_STATE_SEQ_FLAGS_ASYNC) {
        flags &= ~LLAMA_STATE_SEQ_FLAGS_ASYNC;

        if (!(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
            // the previous save may still be copying into dst, and must be complete even if this one fails
            state_async_wait();

            llama_io_write_host_async io(dst, size);
            try {
                io.write(&io_magic, sizeof(io_magic));
                io.write(&seq_id, sizeof(seq_id));

                const size_t n = state_seq_write_data(io, seq_id, flags);

                state_get_async(io.winfos);

                return n;
            } catch (const std::exception & err) {
                LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
                return 0;
            }
        }

        // the device copy below is not asynchronous
        synchronize();
    }

    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // the next batches continue the state read, not the batches before: see llama_moe_cache::hybrid_reset
    if (moe_cache) {
        moe_cache->hybrid_reset();
        // the reset takes back the lent slots without a new allocation: a reused graph would keep its copies bound to them
        gf_res_prev_active = nullptr; gf_res_epoch++;
    }

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

void llama_context::state_async_wait() {
    if (state_async_worker.joinable()) {
        state_async_worker.join();
    }
}

static bool state_async_disabled() {
    static const bool disabled = [] {
        const char * e = getenv("LLAMA_STATE_ASYNC");
        return e != nullptr && atoi(e) == 0;
    }();
    return disabled;
}

// the async-capable backends of state_get_async(), found on first use
void llama_context::state_async_init_backends() {
    if (!state_async_init) {
        state_async_init = true;

        for (auto & backend : backends) {
            ggml_backend_t b = backend.get();
            ggml_backend_dev_t dev = ggml_backend_get_device(b);
            if (b == backend_cpu || dev == nullptr) {
                continue;
            }

            // only backends known to support a cross-thread event wait while the host keeps queueing work.
            // A meta device (-sm tensor) has no registry: its rows take the synchronous path.
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (reg == nullptr) {
                continue;
            }
            const char * reg_name = ggml_backend_reg_name(reg);
            if (strcmp(reg_name, "CUDA") != 0 && strcmp(reg_name, "ROCm") != 0 && strcmp(reg_name, "MUSA") != 0) {
                continue;
            }

            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            if (!props.caps.async || !props.caps.events || ggml_backend_dev_host_buffer_type(dev) == nullptr) {
                continue;
            }

            ggml_backend_event_t event = ggml_backend_event_new(dev);
            if (event == nullptr) {
                continue;
            }

            state_async.push_back({ b, dev, event, nullptr });
        }
    }
}

// The device bytes a PARTIAL_ONLY save of one sequence can read at most, per buffer type: the share of one sequence in the
// buffers of the parts of the memory such a save writes. False for the memory types without a known bound (their staging
// buffers grow on demand).
static bool state_partial_bound(const llama_memory_i * mem, uint32_t n_swa, uint32_t n_ubatch, std::map<ggml_backend_buffer_type_t, size_t> & out) {
    auto add = [&out](const std::map<ggml_backend_buffer_type_t, size_t> & mb, double frac) {
        for (const auto & [buft, size] : mb) {
            out[buft] += (size_t) std::ceil(size * frac);
        }
    };
    // a sequence holds at most n_swa + n_ubatch cells of the SWA cache, in one of its streams
    auto add_swa = [&](const llama_kv_cache * kv) {
        const double cells = std::min<double>(kv->get_size(), (double) n_swa + n_ubatch + 256);
        add(kv->memory_breakdown(), cells / std::max<uint32_t>(1, kv->get_size()) / std::max<uint32_t>(1, kv->get_n_stream()));
    };
    // a recurrent sequence owns one cell
    auto add_recr = [&](const llama_memory_recurrent * mr) {
        add(mr->memory_breakdown(), 1.0 / std::max<uint32_t>(1, mr->size));
    };
    if (const auto * m = dynamic_cast<const llama_memory_hybrid_iswa *>(mem)) {
        add_swa(m->get_mem_attn()->get_swa());
        add_recr(m->get_mem_recr());
        return true;
    }
    if (const auto * m = dynamic_cast<const llama_memory_hybrid *>(mem)) {
        // also llama_memory_hybrid_idx: its indexer is written only with the attention cache
        add_recr(m->get_mem_recr());
        return true;
    }
    if (const auto * m = dynamic_cast<const llama_kv_cache_iswa *>(mem)) {
        add_swa(m->get_swa());
        return true;
    }
    if (const auto * m = dynamic_cast<const llama_kv_cache_dsa_iswa *>(mem)) {
        add_swa(m->get_swa());
        return true;
    }
    if (const auto * m = dynamic_cast<const llama_memory_recurrent *>(mem)) {
        add_recr(m);
        return true;
    }
    return false;
}

// Allocate the staging buffers of asynchronous PARTIAL_ONLY saves at their largest size now, while the context is idle.
// Grown on demand inside a save instead, the first large save paid for the pinned allocation (cudaMallocHost: page pinning
// plus an implicit synchronization) on the critical path of its prompt, with the device draining meanwhile.
// LLAMA_STATE_ASYNC_RESERVE=N caps each buffer at N MiB (default 1024; 0 = off: staging buffers grow on demand, as before).
size_t llama_context::state_async_reserve(llama_state_seq_flags flags) {
    static const size_t cap = [] {
        const char * e = getenv("LLAMA_STATE_ASYNC_RESERVE");
        return (size_t) (e != nullptr ? std::max(0, atoi(e)) : 1024) * 1024 * 1024;
    }();

    if (state_async_disabled() || cap == 0 || !memory || (flags & ~LLAMA_STATE_SEQ_FLAGS_ASYNC) != LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        return 0;
    }

    std::map<ggml_backend_buffer_type_t, size_t> bound;
    if (!state_partial_bound(memory.get(), model.hparams.n_swa, cparams.n_ubatch, bound)) {
        return 0;
    }

    state_async_wait();
    state_async_init_backends();

    // per backend, as in state_get_async(); a row no backend can read asynchronously sends every save down the
    // synchronous path, so nothing is reserved then
    std::vector<size_t> need(state_async.size(), 0);
    for (const auto & [buft, size] : bound) {
        if (size == 0) {
            continue;
        }
        if (ggml_backend_buft_is_host(buft)) {
            return 0;
        }
        size_t b = 0;
        for (; b < state_async.size(); ++b) {
            if (ggml_backend_buft_get_device(buft) == state_async[b].dev && ggml_backend_supports_buft(state_async[b].backend, buft)) {
                break;
            }
        }
        if (b == state_async.size()) {
            return 0;
        }
        need[b] += size;
    }

    size_t total = 0;
    for (size_t b = 0; b < state_async.size(); ++b) {
        auto & sa = state_async[b];
        if (need[b] == 0) {
            continue;
        }
        // room for the 64-byte alignment of each row in the staging buffer
        const size_t size = std::min(cap, GGML_PAD(need[b] + need[b]/32 + 1024*1024, 4096));
        if (sa.staging && ggml_backend_buffer_get_size(sa.staging.get()) >= size) {
            continue;
        }
        ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(sa.dev), size));
        if (!buf) {
            LLAMA_LOG_WARN("%s: failed to allocate a %.2f MiB staging buffer, it is allocated by the first save that needs it\n", __func__, size/1024.0/1024.0);
            continue;
        }
        sa.staging = std::move(buf);
        total += size;
        LLAMA_LOG_INFO("%s: %s: %.2f MiB pinned staging buffer for asynchronous state saves\n", __func__, ggml_backend_dev_name(sa.dev), size/1024.0/1024.0);
    }

    return total;
}

// Read the state rows of an asynchronous save into dst without synchronizing the context.
// Each row is read on the stream of the backend that owns it, so the read comes after the work already queued there
// (e.g. the previous ubatch) and before any later graph changes the row. The rows land in a pinned staging buffer per
// backend (reused across saves); a worker thread waits on an event per backend and copies them to dst, so the host is
// free to prepare the next ubatch meanwhile. Rows in host buffers, or on a device without async copies and events,
// take the synchronous path instead.
void llama_context::state_get_async(const std::vector<llama_state_write_info> & winfos) {
    // the previous save may still be reading the staging buffers
    state_async_wait();

    const bool disabled = state_async_disabled();

    state_async_init_backends();

    // owning backend of each row
    std::vector<int> owner(winfos.size(), -1);
    bool ok = !disabled && !state_async.empty();
    for (size_t i = 0; ok && i < winfos.size(); ++i) {
        const ggml_tensor * t = winfos[i].tensor;
        ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
        if (buf == nullptr || ggml_backend_buffer_is_host(buf)) {
            ok = false;
            break;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
        for (size_t b = 0; b < state_async.size(); ++b) {
            if (ggml_backend_buft_get_device(buft) == state_async[b].dev && ggml_backend_supports_buft(state_async[b].backend, buft)) {
                owner[i] = (int) b;
                break;
            }
        }
        ok = owner[i] >= 0;
    }

    // staging offsets
    std::vector<size_t> need(state_async.size(), 0);
    std::vector<size_t> offs(winfos.size(), 0);
    for (size_t i = 0; ok && i < winfos.size(); ++i) {
        offs[i] = need[owner[i]];
        need[owner[i]] += GGML_PAD(winfos[i].size, 64);
    }

    // staging buffers replaced by larger ones; freed by the worker thread below, after the copies.
    // Freeing pinned host memory (cudaFreeHost) synchronizes the device, so freeing the old buffer here, before the new
    // allocation, drained the queued ubatch and left the GPU idle for the whole cudaMallocHost of the new one.
    std::vector<ggml_backend_buffer_ptr> retired;

    for (size_t b = 0; ok && b < state_async.size(); ++b) {
        auto & sa = state_async[b];
        if (need[b] == 0 || (sa.staging && ggml_backend_buffer_get_size(sa.staging.get()) >= need[b])) {
            continue;
        }

        ggml_backend_buffer_ptr grown(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(sa.dev), need[b]));
        if (grown) {
            retired.push_back(std::move(sa.staging));
            sa.staging = std::move(grown);
        }
        if (!sa.staging || ggml_backend_buffer_get_size(sa.staging.get()) < need[b]) {
            LLAMA_LOG_WARN("%s: failed to allocate a %.2f MiB staging buffer, saving the state synchronously\n", __func__, need[b]/1024.0/1024.0);
            ok = false;
            break;
        }
        LLAMA_LOG_DEBUG("%s: %s staging buffer %.2f MiB\n", __func__, ggml_backend_dev_name(sa.dev), need[b]/1024.0/1024.0);
    }

    if (!ok) {
        synchronize();
        for (const auto & w : winfos) {
            ggml_backend_tensor_get(w.tensor, w.dst, w.offset, w.size);
        }
        return;
    }

    struct copy_job {
        uint8_t       * dst;
        const uint8_t * src;
        size_t          size;
    };
    std::vector<copy_job> jobs;
    jobs.reserve(winfos.size());

    // k15-pp2: a captured state (every row a snapshot of capture cap_save, on one device) is read on a side stream after the
    // event recorded right after its snapshot copies, so the reads overlap the ubatches queued since and the output wait of the
    // decode does not wait for them (LLAMA_STATE_CAPTURE_SIDE=0: on the compute stream, as k15-pp)
    static const bool side_env = getenv("LLAMA_STATE_CAPTURE_SIDE") == nullptr || atoi(getenv("LLAMA_STATE_CAPTURE_SIDE")) != 0;
    ggml_backend_t side = nullptr;
    if (side_env && cap_save >= 0 && cap[cap_save].ev_set && !winfos.empty()) {
        const int b0 = owner[0];
        bool one = true;
        for (size_t i = 0; i < winfos.size(); ++i) {
            one = one && owner[i] == b0;
        }
        if (one) {
            if (cap_side == nullptr) {
                cap_side    = ggml_backend_dev_init(state_async[b0].dev, nullptr);
                cap_side_ev = cap_side ? ggml_backend_event_new(state_async[b0].dev) : nullptr;
                if (cap_side == nullptr || cap_side_ev == nullptr) {
                    LLAMA_LOG_WARN("%s: no side stream for captured state reads, they run on the compute stream\n", __func__);
                }
            }
            if (cap_side != nullptr && cap_side_ev != nullptr && ggml_backend_get_device(cap_side) == state_async[b0].dev) {
                side = cap_side;
                ggml_backend_event_wait(side, cap[cap_save].ev);
            }
        }
    }

    for (size_t i = 0; i < winfos.size(); ++i) {
        auto & sa = state_async[owner[i]];
        uint8_t * stage = (uint8_t *) ggml_backend_buffer_get_base(sa.staging.get()) + offs[i];

        ggml_backend_tensor_get_async(side ? side : sa.backend, winfos[i].tensor, stage, winfos[i].offset, winfos[i].size);

        jobs.push_back({ winfos[i].dst, stage, winfos[i].size });
    }

    std::vector<ggml_backend_event_t> events;
    if (side) {
        ggml_backend_event_record(cap_side_ev, side);
        events.push_back(cap_side_ev);
        cap_side_pending = true;
    } else {
        for (size_t b = 0; b < state_async.size(); ++b) {
            if (need[b] > 0) {
                ggml_backend_event_record(state_async[b].event, state_async[b].backend);
                events.push_back(state_async[b].event);
            }
        }
    }

    if (events.empty()) {
        return;
    }

    sync_pending = true;

    state_async_worker = std::thread([jobs = std::move(jobs), events = std::move(events), retired = std::move(retired)]() mutable {
        for (auto * ev : events) {
            ggml_backend_event_synchronize(ev);
        }
        for (const auto & j : jobs) {
            memcpy(j.dst, j.src, j.size);
        }
        retired.clear();
    });
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // k15-pp debug (LLAMA_CAP_DEBUG=1): per layer hashes of the recurrent rows a partial state write reads
    static const bool cap_debug = getenv("LLAMA_CAP_DEBUG") != nullptr && atoi(getenv("LLAMA_CAP_DEBUG")) != 0;
    auto debug_rows = [&](const char * what) {
        if (!cap_debug || !(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) || dynamic_cast<llama_io_write_dummy *>(&io) != nullptr) {
            return;
        }
        auto * hyb = dynamic_cast<llama_memory_hybrid *>(memory.get());
        auto * rec = hyb ? hyb->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(memory.get());
        if (rec == nullptr) {
            return;
        }
        synchronize();
        for (uint32_t i = 0; i < rec->size; ++i) {
            if (!rec->cells[i].has_seq_id(seq_id)) {
                continue;
            }
            const uint32_t row = (uint32_t) (rec->cells[i].src >= 0 ? rec->cells[i].src : (int32_t) i);
            std::string line;
            int n = 0;
            for (uint32_t il = 0; il < rec->s_l.size() && n < 400; ++il) {
                for (ggml_tensor * t : { rec->r_l[il], rec->s_l[il] }) {
                    if (t == nullptr) {
                        continue;
                    }
                    const size_t rb = t == rec->r_l[il] ? ggml_row_size(t->type, model.hparams.n_embd_r()) : ggml_row_size(t->type, model.hparams.n_embd_s());
                    std::vector<uint8_t> b(rb);
                    ggml_backend_tensor_get(t, b.data(), (size_t) row*rb, rb);
                    uint32_t h = 2166136261u;
                    for (uint8_t c : b) { h ^= c; h *= 16777619u; }
                    char buf[24];
                    snprintf(buf, sizeof(buf), " %u%c:%08x", il, t == rec->r_l[il] ? 'r' : 's', h);
                    line += buf;
                    n++;
                }
            }
            LLAMA_LOG_INFO("%s: cap_debug %s seq %d cell %u row %u pos %d:%s\n", __func__, what, seq_id, i, row, rec->cells[i].pos, line.c_str());
        }
    };
    // k15-pp: a size query of the captured state: its rows have the sizes of the current ones (state_seq_get_data reads the snapshots)
    flags &= ~LLAMA_STATE_SEQ_FLAGS_CAPTURED;
    debug_rows("current");
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (moe_cache) {
        moe_cache->hybrid_reset();
        // the reset takes back the lent slots without a new allocation: a reused graph would keep its copies bound to them
        gf_res_prev_active = nullptr; gf_res_epoch++;
    }
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    // k15-pp3 (LLAMA_FIT_CAPTURE=0 off): the state capture snapshots, allocated at the first capture (cap_prepare), after fit placed
    // the model: planned in a context that only measures memory, the allocated ones otherwise
    {
        ggml_backend_buffer_type_t cb = nullptr;
        const size_t cbytes = cap_reserve_bytes(&cb);
        if (cb != nullptr && cbytes > 0) {
            ret[cb].context += cbytes;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr; gf_res_epoch++;
            res->reset();

            moe_cache_ids.clear();
            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            // like process_ubatch: the device lookups of small ubatches need the slot map state on the device
            if (moe_cache) {
                moe_cache->prepare(ubatch.n_tokens);
                if (moe_cache_devseed_on) {
                    moe_cache->devseed_prepare(gf, false);
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.moe_cache_size              =*/ 0,
        /*.ple_prefetch                =*/ LLAMA_PLE_PREFETCH_AUTO,
        /*.weight_stream               =*/ 1,
        /*.weight_stream_io            =*/ 0,
        /*.weight_stream_threads       =*/ 0,
        /*.weight_stream_ram_mib       =*/ 0,
        /*.moe_host_cache_mib          =*/ -1,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
        /*.mem_reserve_devs            =*/ nullptr,
        /*.mem_reserve                 =*/ nullptr,
        /*.n_mem_reserve               =*/ 0,
        /*.spec_draft                  =*/ false,
        /*.n_state_capture             =*/ -1,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4 || model->arch == LLM_ARCH_DEEPSEEK41) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const llama_moe_cache_too_small & err) {
        // a memory probe (no_alloc, e.g. the --fit search for a MoE cache size) tries sizes below the minimum and handles the failure
        if (model->hparams.no_alloc) {
            LLAMA_LOG_INFO("%s: memory probe: %s\n", __func__, err.what());
        } else {
            LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize_outputs();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_verify(llama_context * ctx, bool value) {
    ctx->set_verify(value);
}

void llama_context::set_ubatch_breaks(const int32_t * breaks, int32_t n) {
    ubatch_breaks_next.assign(breaks, breaks + std::max(0, n));
}

// k15-pp: llama_set_state_capture
bool llama_context::state_capture_supported() const {
    static const bool env = getenv("LLAMA_STATE_CAPTURE") == nullptr || atoi(getenv("LLAMA_STATE_CAPTURE")) != 0;
    if (!env || memory == nullptr || cap_count() == 0) {   // no snapshots (--fit, n_state_capture = 0): the server checkpoints as without capture
        return false;
    }
    auto * hyb = dynamic_cast<llama_memory_hybrid *>(memory.get());
    auto * rec = hyb ? hyb->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(memory.get());
    return rec != nullptr && !(hyb && model.hparams.swa_type != LLAMA_SWA_TYPE_NONE);
}

bool llama_context::set_state_capture(llama_seq_id seq_id, int32_t i_batch) {
    static const bool env = getenv("LLAMA_STATE_CAPTURE") == nullptr || atoi(getenv("LLAMA_STATE_CAPTURE")) != 0;
    if (!env || memory == nullptr || i_batch <= 0) {
        return false;
    }
    // only a recurrent part with a plain (non-SWA) KV cache beside it: the partial state is then the recurrent rows alone
    auto * hyb = dynamic_cast<llama_memory_hybrid *>(memory.get());
    auto * rec = hyb ? hyb->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(memory.get());
    if (rec == nullptr || (hyb && model.hparams.swa_type != LLAMA_SWA_TYPE_NONE)) {
        return false;
    }
    // k15-pp2: one slot per (sequence, index) for the next decode, up to cap_max (LLAMA_STATE_CAPTURE_MAX=1: one, as k15-pp)
    const int n_max = cap_count();
    int s = -1;
    for (int c = 0; c < n_max && s < 0; ++c) {
        if (cap[c].pending && cap[c].seq == seq_id && cap[c].idx == i_batch) {
            s = c;
        }
    }
    for (int c = 0; c < n_max && s < 0; ++c) {
        if (!cap[c].pending) {
            s = c;
        }
    }
    if (s < 0) {
        return false;
    }
    cap[s].seq     = seq_id;
    cap[s].idx     = i_batch;
    cap[s].pending = true;
    return true;
}

// the first capture of the sequence taken by the last decode (the one at the lowest index); it is also the one a CAPTURED read takes
bool llama_context::state_captured(llama_seq_id seq_id, llama_pos * pos_max) const {
    int best = -1;
    for (int c = 0; c < cap_max; ++c) {
        if (cap[c].done && cap[c].seq == seq_id && !cap[c].regions.empty() && (best < 0 || cap[c].idx < cap[best].idx)) {
            best = c;
        }
    }
    if (best < 0) {
        return false;
    }
    if (pos_max) {
        *pos_max = cap[best].pos;
    }
    return true;
}

// the capture of the sequence at batch index i_batch taken by the last decode; selects it for the next LLAMA_STATE_SEQ_FLAGS_CAPTURED read
bool llama_context::state_captured_at(llama_seq_id seq_id, int32_t i_batch, llama_pos * pos_max) {
    for (int c = 0; c < cap_max; ++c) {
        if (cap[c].done && cap[c].seq == seq_id && cap[c].idx == i_batch && !cap[c].regions.empty()) {
            cap_sel = c;
            if (pos_max) {
                *pos_max = cap[c].pos;
            }
            return true;
        }
    }
    return false;
}

ggml_backend_t llama_context::cap_backend() const {
    for (const auto & c : cap) {
        if (c.buf) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c.buf.get()));
            for (const auto & b : backends) {
                if (ggml_backend_get_device(b.get()) == dev) {
                    return b.get();
                }
            }
        }
    }
    return nullptr;
}

// k15-pp2: an event on the compute stream right after the snapshot copies of capture s, for the side-stream read of its save
void llama_context::cap_record(int s, ggml_backend_t be) {
    auto & C = cap[s];
    C.ev_set = false;
    if (be == nullptr) {
        return;
    }
    if (C.ev == nullptr) {
        C.ev = ggml_backend_event_new(ggml_backend_get_device(be));
    }
    if (C.ev != nullptr) {
        ggml_backend_event_record(C.ev, be);
        C.ev_set = true;
    }
}

size_t llama_context::cap_reserve_bytes(ggml_backend_buffer_type_t * buft_out) const {
    *buft_out = nullptr;
    if (!model.hparams.no_alloc) {
        size_t n = 0;
        for (const auto & c : cap) {
            if (c.buf) {
                *buft_out = ggml_backend_buffer_get_type(c.buf.get());
                n += ggml_backend_buffer_get_size(c.buf.get());
            }
        }
        return n;
    }
    return cap_planned_bytes(buft_out);
}

// the number of state capture snapshots: LLAMA_STATE_CAPTURE_MAX (1..cap_max) when set, else llama_context_params::n_state_capture
// when >= 0 (--fit lowers it where the snapshots would cost device layers; 0 = no capture), else cap_max
int llama_context::cap_count() const {
    static const int env_n = getenv("LLAMA_STATE_CAPTURE_MAX") ? std::max(1, std::min(cap_max, atoi(getenv("LLAMA_STATE_CAPTURE_MAX")))) : -1;
    if (cparams.n_state_capture == 0) {
        return 0;   // --fit dropped the capture (no room for its snapshots): also under LLAMA_STATE_CAPTURE_MAX
    }
    if (env_n > 0) {
        return env_n;
    }
    return cparams.n_state_capture >= 0 ? std::min<int>(cap_max, cparams.n_state_capture) : cap_max;
}

size_t llama_context::cap_planned_bytes(ggml_backend_buffer_type_t * buft_out) const {
    *buft_out = nullptr;
    // planned: captures happen where prompts run layer-major (a target with a draft whose experts spill to the host, see lm_enabled) on
    // a recurrent memory the server captures (llama_state_capture_supported, LLAMA_LM_CKPT_CAPTURE): one snapshot of a sequence's
    // recurrent rows per capture. The condition leaves out what fit varies between its measurements (the MoE cache size, n_ubatch), so
    // every measurement of one placement counts the same bytes: the n_ubatch rule compares two of them (common_fit_moe_cache)
    // LLAMA_FIT_CAPTURE: off by default for now (=1 counts them; pending a decision: +124.5 MiB headroom at q4x 12 GB for 1-8% default
    // decode). Off, the snapshots are allocated at the first capture, uncounted, as in k15-pp2
    static const bool fit_env = getenv("LLAMA_FIT_CAPTURE") != nullptr && atoi(getenv("LLAMA_FIT_CAPTURE")) != 0;
    static const bool srv_env = getenv("LLAMA_LM_CKPT_CAPTURE") == nullptr || atoi(getenv("LLAMA_LM_CKPT_CAPTURE")) != 0;
    static const bool lm_env  = getenv("LLAMA_MOE_LAYER_MAJOR") == nullptr || strcmp(getenv("LLAMA_MOE_LAYER_MAJOR"), "auto") == 0 ||
        atoi(getenv("LLAMA_MOE_LAYER_MAJOR")) != 0;
    const int  n_cap   = cap_count();
    if (!fit_env || !srv_env || !lm_env || !cparams.spec_draft || cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ||
            llama_moe_cache::host_bytes(model) == 0 || !state_capture_supported()) {
        return 0;
    }
    auto * hyb = dynamic_cast<llama_memory_hybrid *>(memory.get());
    auto * rec = hyb ? hyb->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(memory.get());
    if (rec == nullptr) {
        return 0;
    }
    const auto & hp = model.hparams;
    ggml_backend_buffer_type_t buft = nullptr;
    size_t total = 0;
    auto add = [&](const ggml_tensor * t, size_t row_bytes) {
        if (t == nullptr || row_bytes == 0 || t->buffer == nullptr) {
            return;
        }
        if (buft == nullptr) {
            buft = ggml_backend_buffer_get_type(t->buffer);
        }
        total += GGML_PAD(row_bytes, 256);
    };
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if (il < rec->r_l.size() && rec->r_l[il]) { add(rec->r_l[il], ggml_row_size(rec->r_l[il]->type, hp.n_embd_r())); }
        if (il < rec->p_l.size() && rec->p_l[il]) { add(rec->p_l[il], ggml_row_size(rec->p_l[il]->type, hp.ple_conv_state())); }
        if (il < rec->s_l.size() && rec->s_l[il]) { add(rec->s_l[il], ggml_row_size(rec->s_l[il]->type, hp.n_embd_s())); }
    }
    if (buft == nullptr || total == 0) {
        return 0;
    }
    *buft_out = buft;
    return (size_t) n_cap*(total + 4096);
}

bool llama_context::cap_prepare(int s) {
    auto & C = cap[s];
    if (!C.regions.empty()) {
        return true;
    }
    const llama_seq_id cap_seq = C.seq;
    auto * hyb = dynamic_cast<llama_memory_hybrid *>(memory.get());
    auto * rec = hyb ? hyb->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(memory.get());
    if (rec == nullptr || cap_seq < 0) {
        return false;
    }
    if (rec->n_rs_seq != 0 && ((size_t) cap_seq >= rec->rs_idx.size() || rec->rs_idx[cap_seq] != 0)) {
        return false;
    }
    int32_t cell = -1;
    for (uint32_t i = 0; i < rec->size; ++i) {
        if (rec->cells[i].has_seq_id(cap_seq)) {
            if (cell >= 0 || rec->cells[i].seq_id.size() != 1) {
                return false; // several cells, or a cell shared by sequences
            }
            cell = (int32_t) i;
        }
    }
    if (cell < 0) {
        return false;
    }
    const auto & hp = model.hparams;
    std::vector<cap_region> rg;
    ggml_backend_buffer_type_t buft = nullptr;
    size_t total = 0;
    const uint32_t row = (uint32_t) (rec->cells[cell].src >= 0 ? rec->cells[cell].src : cell);
    auto add = [&](ggml_tensor * t, size_t row_bytes) {
        if (t == nullptr || row_bytes == 0) {
            return true;
        }
        if (t->buffer == nullptr || (buft != nullptr && ggml_backend_buffer_get_type(t->buffer) != buft)) {
            return false;
        }
        buft = ggml_backend_buffer_get_type(t->buffer);
        rg.push_back({ t, (size_t) row*row_bytes, row_bytes, nullptr, nullptr });
        total += GGML_PAD(row_bytes, 256);
        return true;
    };
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if (!add(il < rec->r_l.size() ? rec->r_l[il] : nullptr, rec->r_l.size() > il && rec->r_l[il] ? ggml_row_size(rec->r_l[il]->type, hp.n_embd_r()) : 0) ||
            !add(il < rec->p_l.size() ? rec->p_l[il] : nullptr, rec->p_l.size() > il && rec->p_l[il] ? ggml_row_size(rec->p_l[il]->type, hp.ple_conv_state()) : 0) ||
            !add(il < rec->s_l.size() ? rec->s_l[il] : nullptr, rec->s_l.size() > il && rec->s_l[il] ? ggml_row_size(rec->s_l[il]->type, hp.n_embd_s()) : 0)) {
            return false;
        }
    }
    if (rg.empty()) {
        return false;
    }
    // the buffer may come from the context's creation (k15-pp3: allocated where fit counted it, see cap_planned_bytes)
    if (!C.buf || ggml_backend_buffer_get_size(C.buf.get()) < total + 4096 || ggml_backend_buffer_get_type(C.buf.get()) != buft) {
        C.buf.reset(ggml_backend_buft_alloc_buffer(buft, total + 4096));
        if (!C.buf) {
            return false;
        }
    }
    {
        ggml_free(C.ctx.release());
        ggml_init_params ip = { (2*rg.size() + 8)*ggml_tensor_overhead(), nullptr, true };
        C.ctx.reset(ggml_init(ip));
    }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(C.buf.get());
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t off = 0;
    for (auto & r : rg) {
        off = GGML_PAD(off, std::max<size_t>(align, 256));
        r.snap = ggml_new_tensor_1d(C.ctx.get(), r.t->type, (int64_t) (r.size/ggml_type_size(r.t->type)));
        if (ggml_backend_tensor_alloc(C.buf.get(), r.snap, base + off) != GGML_STATUS_SUCCESS) {
            return false;
        }
        r.view = ggml_view_1d(C.ctx.get(), r.t, (int64_t) (r.size/ggml_type_size(r.t->type)), r.off);
        off += r.size;
    }
    C.cell    = (uint32_t) cell;
    C.regions = std::move(rg);
    return true;
}

void llama_context::cap_now(int s, llama_pos pos) {
    if (!cap_prepare(s)) {
        return;
    }
    auto & C = cap[s];
    // device copies in stream order on the backend that holds the rows (after the ubatches queued so far, before the next one),
    // else through host memory after a sync
    ggml_backend_t be = nullptr;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(C.buf.get()));
    for (const auto & b : backends) {
        if (ggml_backend_get_device(b.get()) == dev) {
            be = b.get();
            break;
        }
    }
    if (be != nullptr) {
        for (const auto & r : C.regions) {
            ggml_backend_tensor_copy_async(be, be, r.view, r.snap);
        }
        cap_record(s, be);
    } else {
        ggml_backend_sched_synchronize(sched.get());
        for (const auto & r : C.regions) {
            std::vector<uint8_t> tmp(r.size);
            ggml_backend_tensor_get(r.t, tmp.data(), r.off, r.size);
            ggml_backend_tensor_set(r.snap, tmp.data(), 0, r.size);
        }
    }
    C.done = true;
    C.pos  = pos;
}


bool llama_context::layer_major_on() const {
    return lm_enabled && moe_cache != nullptr;
}

// a batch of n_before + n_after prompt tokens of one sequence, from the start of the batch, with a ubatch break after n_before: the first
// ubatch after the break runs in the layer-major group of the one before it (decode_layer_major: every member eligible, at most lm_ub_max
// per group). Otherwise the merge only drops the seed the separate batches made between them: a ubatch before the break that is too short
// for a group (Qwen3.6 q4x 12 GB, -ub 512: a 640-token prompt runs [124] [512] [4]) or a group that ends at the break
bool llama_context::layer_major_spans(int64_t n_before, int64_t n_after) const {
    if (!layer_major_on() || lm_refused_min <= 2 || n_before <= 0 || n_after <= 0) {
        return false;
    }
    const int64_t nu = cparams.n_ubatch;
    auto eligible = [&](int64_t n) { return n > llama_moe_cache::n_small_max && n >= prefetch_min_batch; };
    const int64_t r = n_before % nu;
    if ((r != 0 && !eligible(r)) || !eligible(std::min(nu, n_after))) {
        return false;
    }
    return ((n_before + nu - 1)/nu) % lm_ub_max != 0;
}

void llama_state_seq_async_wait(llama_context * ctx) {
    ctx->state_async_wait();
}

void llama_set_ubatch_breaks(llama_context * ctx, const int32_t * breaks, int32_t n) {
    ctx->set_ubatch_breaks(breaks, n);
}

bool llama_state_capture_supported(const llama_context * ctx) {
    return ctx->state_capture_supported();
}

bool llama_set_state_capture(llama_context * ctx, llama_seq_id seq_id, int32_t i_batch) {
    return ctx->set_state_capture(seq_id, i_batch);
}

bool llama_state_seq_captured_at(llama_context * ctx, llama_seq_id seq_id, int32_t i_batch, llama_pos * pos_max) {
    return ctx->state_captured_at(seq_id, i_batch, pos_max);
}

bool llama_state_seq_captured(const llama_context * ctx, llama_seq_id seq_id, llama_pos * pos_max) {
    return ctx->state_captured(seq_id, pos_max);
}

bool llama_layer_major_on(const llama_context * ctx) {
    return ctx->layer_major_on();
}

bool llama_layer_major_spans(const llama_context * ctx, int32_t n_before, int32_t n_after) {
    return ctx->layer_major_spans(n_before, n_after);
}

bool llama_exact_verify_enabled(void) {
    return llama_exact_verify_env();
}

bool llama_prepare_prompt_plan(llama_context * ctx) {
    return ctx->restore_pp_plan();
}

bool llama_context::moe_freeze_small() {
    return moe_cache != nullptr && moe_cache->freeze_small();
}

bool llama_moe_freeze_small(llama_context * ctx) {
    return ctx != nullptr && ctx->moe_freeze_small();
}

bool llama_context::moe_pin_small() {
    return moe_cache != nullptr && moe_cache->pin_small();
}

bool llama_moe_pin_small(llama_context * ctx) {
    return ctx != nullptr && ctx->moe_pin_small();
}

bool llama_context::moe_pin_small_set(bool on) {
    return moe_cache != nullptr && moe_cache->pin_small_set(on);
}

bool llama_moe_pin_small_set(llama_context * ctx, bool on) {
    return ctx != nullptr && ctx->moe_pin_small_set(on);
}

bool llama_context::moe_cpu_miss_drop_slow() {
    if (moe_cache == nullptr || !moe_cache->cpu_miss_drop_slow()) {
        return false;
    }
    ggml_backend_sched_set_overlap(sched.get(), moe_cache->hybrid_active());
    return true;
}

bool llama_moe_cpu_miss_drop_slow(llama_context * ctx) {
    return ctx != nullptr && ctx->moe_cpu_miss_drop_slow();
}

int32_t llama_moe_host_mode(const llama_context * ctx) {
    return ctx != nullptr && ctx->get_moe_cache() != nullptr ? ctx->get_moe_cache()->host_mode() : 0;
}

void llama_set_verify_exact(llama_context * ctx, bool value) {
    ctx->set_verify_exact(value);
}

bool llama_verify_exact_placement(const llama_context * ctx) {
    // a MoE cache computes misses on the host or reads them per batch, and host weights run CPU kernels whose batch path is not the
    // single token path: verify rows there do not give the bytes of single token decodes (G4, LFM2.5 8 GB: other text)
    if (ctx == nullptr) {
        return false;
    }
    // k13: a MoE cache whose host never computes an expert (no hybrid path, no CPU misses) runs every expert on the device from a slot,
    // with the kernels of the resident case: its host expert tensors are only the source of the uploads (LLAMA_SPEC_EXACT_CACHE=0: off)
    bool cache_ok = false;
    if (ctx->moe_cache_active()) {
        static const bool env_ok = [] {
            const char * e = getenv("LLAMA_SPEC_EXACT_CACHE");
            return e == nullptr || atoi(e) != 0;
        }();
        // frozen small batches (the host computes the misses of small batches, nothing admitted) only with LLAMA_SPEC_EXACT_FROZEN=1:
        // exact (k13, local B200, LFM2.5 5.5 GB cache: drafts = plain decode in chat / rep / 2k / 8k), opt-in because the frozen cache
        // does not follow the decode (G4 LFM2.5 8 GB: chat +13%, 2k +16%, 8k -22% against the default)
        static const bool frozen_ok = [] {
            const char * e = getenv("LLAMA_SPEC_EXACT_FROZEN");
            return e != nullptr && atoi(e) != 0;
        }();
        if (!env_ok || !(ctx->get_moe_cache()->host_never() || (frozen_ok && ctx->get_moe_cache()->host_split_frozen()))) {
            return false;
        }
        cache_ok = true;
    }
    for (const auto & [name, t] : ctx->get_model().tensors_by_name) {
        if (t != nullptr && t->buffer != nullptr && name.rfind("blk.", 0) == 0 && ggml_backend_buffer_is_host(t->buffer)) {
            if (cache_ok && name.find("_exps") != std::string::npos) {
                continue;
            }
            return false;
        }
    }
    return true;
}

bool llama_single_pass_equivalent(const llama_context * ctx) {
    if (ctx == nullptr || llama_moe_host_mode(ctx) != 0 || !llama_verify_exact_placement(ctx)) {
        return false;
    }
    const llama_model & model = ctx->get_model();
    if (model.devices.empty()) {
        return false;
    }
    for (const auto & d : model.devices) {
        const char * name = ggml_backend_dev_name(d.dev);
        if (name == nullptr || strncmp(name, "CUDA", 4) != 0) {   // the kernel paths were checked on CUDA (sm_100, sm_120) only
            return false;
        }
    }
    static const bool cublas_32f = getenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F") != nullptr;
    for (const auto & [name, t] : model.tensors_by_name) {
        if (t != nullptr && !cublas_32f && t->type == GGML_TYPE_F16 && t->ne[1] > 1 && name.rfind("blk.", 0) == 0 &&
            name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) {
            return false;
        }
    }
    return true;
}

int32_t llama_arch_rollback_state(const char * arch_name) {
    if (arch_name == nullptr) {
        return -1;
    }
    const llm_arch arch = llm_arch_from_string(arch_name);
    if (arch == LLM_ARCH_UNKNOWN) {
        return -1;
    }
    if (arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE) {
        return 1;   // a short-conv state of a few KiB per layer: per-token snapshots cost nothing
    }
    return llm_arch_is_recurrent(arch) || llm_arch_is_hybrid(arch) ? 2 : 0;
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

bool llama_mtp_dev_h_attach(llama_context * ctx_dft, llama_context * ctx_tgt, int32_t n_rows) {
    return ctx_dft && ctx_dft->mtp_dev_h_attach(ctx_tgt, n_rows);
}

int32_t llama_mtp_dev_h_last_rows(llama_context * ctx_tgt) {
    return ctx_tgt ? ctx_tgt->mtp_dev_h_last_rows() : 0;
}

bool llama_mtp_dev_h_begin(llama_context * ctx_dft) {
    return ctx_dft && ctx_dft->mtp_dev_h_begin();
}

void llama_mtp_dev_h_end(llama_context * ctx_dft, int32_t pending_row) {
    if (ctx_dft) {
        ctx_dft->mtp_dev_h_end(pending_row);
    }
}

bool llama_mtp_dev_h_get_row(llama_context * ctx_dft, int32_t row, float * dst) {
    return ctx_dft && ctx_dft->mtp_dev_h_get_row(row, dst);
}

bool llama_mtp_dev_h_set_row(llama_context * ctx_dft, int32_t row, const float * src) {
    return ctx_dft && ctx_dft->mtp_dev_h_set_row(row, src);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_nextn();
}

int64_t llama_nextn_rows_wait(llama_context * ctx, int64_t n_rows) {
    return ctx->nextn_rows_wait(n_rows);
}

float * llama_get_embeddings_nextn_early(llama_context * ctx) {
    return ctx->nextn_rows_early();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    if (data && mem->wait_state_reads) {
        mem->wait_state_reads();
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    // some memory types also clear the data of the removed sequence
    if (mem->wait_state_reads) {
        mem->wait_state_reads();
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    if (mem->wait_state_reads) {
        mem->wait_state_reads();
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_reserve_async(llama_context * ctx, llama_state_seq_flags flags) {
    return ctx->state_async_reserve(flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (!(flags & LLAMA_STATE_SEQ_FLAGS_ASYNC)) {
        ctx->synchronize();
    }

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    ctx->sync_pending = true; // the output copies queued after the last graph may still run
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    ctx->sync_pending = true; // the output copies queued after the last graph may still run
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    LLAMA_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    int32_t ret = -1;
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: ret = ctx->encode(*batch); break;
        case LLAMA_PROCESS_TYPE_DECODE: ret = ctx->decode(*batch); break;
    }
    ctx->sync_pending = true; // the output copies queued after the last graph may still run
    return ret;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
