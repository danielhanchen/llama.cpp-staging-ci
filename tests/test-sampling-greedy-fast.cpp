// the greedy fast path of common_sampler_sample (LLAMA_SAMPLE_GREEDY_FAST, LLAMA_SAMPLE_BIAS_SPARSE) must sample the token the
// sampler chain samples and leave the logits row as it found it. Every expected token below is the chain's (the test also passes with
// LLAMA_SAMPLE_GREEDY_FAST=0). Takes any model: the logits row of one decode is overwritten with fixed values.

#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static llama_model   * g_model = nullptr;
static llama_context * g_ctx   = nullptr;
static int32_t         g_n_vocab = 0;
static int             g_fail  = 0;

// the fast path leaves one candidate, the chain the whole row
static bool fast(common_sampler * s) {
    return common_sampler_get_candidates(s, false)->size == 1;
}

// one decode, then the row is set to -20 everywhere but the given tokens
static float * row(const std::vector<std::pair<llama_token, float>> & vals) {
    llama_memory_clear(llama_get_memory(g_ctx), true);
    llama_token t = 0;
    if (llama_decode(g_ctx, llama_batch_get_one(&t, 1)) != 0) {
        fprintf(stderr, "decode failed\n");
        exit(1);
    }
    float * lg = llama_get_logits_ith(g_ctx, -1);
    for (int32_t i = 0; i < g_n_vocab; ++i) {
        lg[i] = -20.0f;
    }
    for (const auto & v : vals) {
        lg[v.first] = v.second;
    }
    return lg;
}

static void check(const char * name, bool ok) {
    printf("%-60s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail = 1;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(argv[1], mparams);
    if (g_model == nullptr) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 64;
    cparams.n_batch = 64;
    g_ctx = llama_init_from_model(g_model, cparams);
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(g_model));
    if (g_ctx == nullptr || g_n_vocab < 100) {
        fprintf(stderr, "need a context and a vocab of at least 100 tokens\n");
        return 1;
    }

    const llama_token A = 50;
    const llama_token B = 70;

    // a biased token inside the penalties window: biased and penalized once (20 / 2 = 10 > 8), twice in a row, row intact
    {
        common_params_sampling sp;
        sp.temp           = 0.0f;
        sp.penalty_repeat = 2.0f;
        sp.penalty_last_n = 64;
        sp.logit_bias.push_back({A, 10.0f});
        common_sampler * s = common_sampler_init(g_model, sp);
        common_sampler_accept(s, A, false);
        row({{A, 10.0f}, {B, 8.0f}});
        const llama_token t1 = common_sampler_sample(s, g_ctx, -1);
        const float lg_a     = llama_get_logits_ith(g_ctx, -1)[A];
        const llama_token t2 = common_sampler_sample(s, g_ctx, -1);
        check("biased token in the penalties window", t1 == A && t2 == A && lg_a == 10.0f);
        // the default mode takes the fast path here (biases + penalties), so the cases above test it
        if (getenv("LLAMA_SAMPLE_GREEDY_FAST") == nullptr && getenv("LLAMA_SAMPLE_GREEDY_PENALTY") == nullptr) {
            check("fast path taken with a bias and penalties", fast(s));
        }
        common_sampler_free(s);
    }

    // an ignore_eos style -inf bias on the token with the highest logit, at every place of a 64-token penalties window
    {
        int n_banned = 0;
        int n_dirty  = 0;
        for (int tr = 0; tr < 64; ++tr) {
            common_params_sampling sp;
            sp.temp           = 0.0f;
            sp.penalty_repeat = 1.1f;
            sp.penalty_last_n = 64;
            const llama_token E = 1 + tr % 40;
            sp.logit_bias.push_back({E, -INFINITY});
            common_sampler * s = common_sampler_init(g_model, sp);
            unsigned x = 12345u + 977u*tr;
            for (int k = 0; k < 64; ++k) {
                x = x*1103515245u + 12345u;
                common_sampler_accept(s, k == tr ? E : (llama_token) (42 + (x >> 8) % 58), false);
            }
            row({{E, 9.0f}, {B, 4.5f}});
            n_banned += common_sampler_sample(s, g_ctx, -1) == E;
            n_dirty  += llama_get_logits_ith(g_ctx, -1)[E] != 9.0f;
            common_sampler_free(s);
        }
        printf("  banned token sampled %d / 64, row changed %d / 64\n", n_banned, n_dirty);
        check("-inf biased token in the penalties window", n_banned == 0 && n_dirty == 0);
    }

    // 20 suppressed tokens, all in the window, one of them with the highest logit
    {
        common_params_sampling sp;
        sp.temp           = 0.0f;
        sp.penalty_repeat = 1.1f;
        sp.penalty_last_n = 64;
        for (int k = 0; k < 20; ++k) {
            sp.logit_bias.push_back({80 + k, -INFINITY});
        }
        common_sampler * s = common_sampler_init(g_model, sp);
        for (int k = 0; k < 20; ++k) {
            common_sampler_accept(s, 80 + k, false);
        }
        row({{85, 9.0f}, {B, 4.5f}});
        check("suppressed tokens in the penalties window", common_sampler_sample(s, g_ctx, -1) == B);
        common_sampler_free(s);
    }

    // no penalties stage in the chain: the penalty parameters do nothing
    {
        common_params_sampling sp;
        sp.temp           = 0.0f;
        sp.penalty_repeat = 2.0f;
        sp.penalty_last_n = 64;
        sp.samplers       = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
        common_sampler * s = common_sampler_init(g_model, sp);
        common_sampler_accept(s, A, false);
        row({{A, 10.0f}, {B, 8.0f}});
        check("penalty parameters without a penalties stage", common_sampler_sample(s, g_ctx, -1) == A);
        if (getenv("LLAMA_SAMPLE_GREEDY_FAST") == nullptr) {
            check("fast path taken without a penalties stage", fast(s));
        }
        common_sampler_free(s);
    }

    // temperature before the penalties: the frequency penalty applies to logit / T (2*10 - 3 = 17 > 16)
    {
        common_params_sampling sp;
        sp.temp           = 0.5f;
        sp.top_k          = 1;
        sp.penalty_freq   = 3.0f;
        sp.penalty_last_n = 64;
        sp.samplers       = { COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K };
        common_sampler * s = common_sampler_init(g_model, sp);
        common_sampler_accept(s, A, false);
        row({{A, 10.0f}, {B, 8.0f}});
        check("temperature before the penalties stage", common_sampler_sample(s, g_ctx, -1) == A);
        common_sampler_free(s);
    }

    // two penalties stages apply twice: 10 - 2*1.5 = 7 < 8
    {
        common_params_sampling sp;
        sp.temp             = 0.0f;
        sp.penalty_present  = 1.5f;
        sp.penalty_last_n   = 64;
        sp.samplers         = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TEMPERATURE };
        common_sampler * s = common_sampler_init(g_model, sp);
        common_sampler_accept(s, A, false);
        row({{A, 10.0f}, {B, 8.0f}});
        check("two penalties stages", common_sampler_sample(s, g_ctx, -1) == B);
        common_sampler_free(s);
    }

    llama_free(g_ctx);
    llama_model_free(g_model);
    llama_backend_free();
    return g_fail;
}
