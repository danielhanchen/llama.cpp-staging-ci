// k19-sharednoise: the shared-noise final stage (LLAMA_SHARED_NOISE=1) must sample the target's distribution exactly, and a drafter
// with the target's logits and chain must propose the token the target draws. Takes any model (one decode, the logits row is then
// overwritten with fixed values). Run it with and without LLAMA_SHARED_NOISE=1: both must pass the goodness-of-fit checks against
// the probabilities the chain leaves (dist and shared noise sample the same distribution); the coupling checks run only with it on.
// Optional 2nd argument: draws per test (default 40000).

#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

static llama_model   * g_model   = nullptr;
static llama_context * g_ctx     = nullptr;
static int32_t         g_n_vocab = 0;
static int             g_fail    = 0;

static void check(const char * name, bool ok) {
    printf("%-72s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail = 1;
    }
}

// one decode, then the row is set to -40 everywhere but the given tokens (kept for every later read of row 0)
static std::vector<std::pair<llama_token, float>> g_vals;
static void set_row() {
    float * lg = llama_get_logits_ith(g_ctx, -1);
    for (int32_t i = 0; i < g_n_vocab; ++i) {
        lg[i] = -40.0f;
    }
    for (const auto & v : g_vals) {
        lg[v.first] = v.second;
    }
}

// chi-square upper tail, Wilson-Hilferty (dof >= 3)
static double chi2_p(double x, int k) {
    const double z = (std::cbrt(x / k) - (1.0 - 2.0 / (9.0 * k))) / std::sqrt(2.0 / (9.0 * k));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

// goodness of fit of counts against probabilities (bins with expected < 5 pooled)
static double gof(const std::map<llama_token, int> & cnt, const std::map<llama_token, double> & p, int n, int * dof_out) {
    double chi = 0.0, pool_e = 0.0, pool_o = 0.0;
    int dof = 0;
    for (const auto & [tok, q] : p) {
        const double e = q * n;
        const auto it = cnt.find(tok);
        const double o = it == cnt.end() ? 0.0 : it->second;
        if (e < 5.0) {
            pool_e += e;
            pool_o += o;
            continue;
        }
        chi += (o - e) * (o - e) / e;
        dof++;
    }
    // anything drawn outside p's support is an error by itself
    int outside = 0;
    for (const auto & [tok, c] : cnt) {
        if (p.find(tok) == p.end()) {
            outside += c;
        }
    }
    if (pool_e >= 5.0) {
        chi += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        dof++;
    }
    dof -= 1;
    *dof_out = dof;
    return outside > 0 ? 0.0 : chi2_p(chi, std::max(dof, 1));
}

struct cfg {
    const char * name;
    float temp;
    int32_t top_k;
    float top_p;
    float min_p;
    float presence;
    float repeat;
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_draws]\n", argv[0]);
        return 1;
    }
    const int N = argc > 2 ? atoi(argv[2]) : 40000;
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
    if (g_ctx == nullptr || g_n_vocab < 1000) {
        fprintf(stderr, "need a context and a vocab of at least 1000 tokens\n");
        return 1;
    }
    {
        llama_token t = 0;
        if (llama_decode(g_ctx, llama_batch_get_one(&t, 1)) != 0) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
    }

    const bool sn = common_shared_noise_enabled();
    printf("LLAMA_SHARED_NOISE: %s, %d draws per test, vocab %d\n", sn ? "on" : "off", N, g_n_vocab);

    // 48 tokens with a spread of logits (a few close at the top, a long tail)
    for (int j = 0; j < 48; ++j) {
        g_vals.emplace_back((llama_token) (100 + 7 * j), 4.0f - 0.18f * j - (j % 3 == 1 ? 0.35f : 0.0f));
    }
    // the "prompt": two of the top tokens are in the penalty window
    const std::vector<llama_token> prompt = { 5, 107, 6, 121, 7 };

    const cfg cfgs[] = {
        { "studio qwen thinking (T 0.6, top-p 0.95, top-k 20, presence 1.5)", 0.6f, 20, 0.95f, 0.0f, 1.5f, 1.0f },
        { "studio qwen non-thinking (T 0.7, top-p 0.8, top-k 20, presence 1.5)", 0.7f, 20, 0.80f, 0.0f, 1.5f, 1.0f },
        { "studio default (T 0.6, top-p 0.95, top-k 20, min-p 0.01)",             0.6f, 20, 0.95f, 0.01f, 0.0f, 1.0f },
        { "T 1.0, no truncation, repeat 1.1",                                     1.0f, 0,  1.0f,  0.0f, 0.0f, 1.1f },
    };

    for (const auto & c : cfgs) {
        common_params_sampling sp;
        sp.temp            = c.temp;
        sp.top_k           = c.top_k;
        sp.top_p           = c.top_p;
        sp.min_p           = c.min_p;
        sp.penalty_present = c.presence;
        sp.penalty_repeat  = c.repeat;
        sp.penalty_last_n  = 64;

        std::map<llama_token, int>    cnt;
        std::map<llama_token, double> p;
        int n_same = 0, n_coupled = 0, n_greedy_draft = 0, n_coupled_noisy = 0, n_noisy_greedy = 0;
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        common_shared_noise_draft snd;
        for (int t = 0; t < N; ++t) {
            sp.seed = 1000003u * (uint32_t) t + 17u;
            common_sampler * s = common_sampler_init(g_model, sp);
            for (const auto tok : prompt) {
                common_sampler_accept(s, tok, false);
            }
            set_row();
            const llama_token id = common_sampler_sample(s, g_ctx, 0);
            cnt[id]++;
            if (t == 0) {
                const auto * cur = common_sampler_get_candidates(s, true);
                for (size_t k = 0; k < cur->size; ++k) {
                    if (cur->data[k].p > 0.0f) {
                        p[cur->data[k].id] = cur->data[k].p;
                    }
                }
            }
            if (sn) {
                // the same position again (a grammar resample): the same token
                set_row();
                n_same += common_sampler_sample(s, g_ctx, 0) == id;
                // a drafter with the target's own logits proposes the target's token
                if (common_sampler_shared_noise_draft_init(s, snd)) {
                    std::vector<llama_token_data> cand;
                    for (const auto & v : g_vals) {
                        cand.push_back({ v.first, v.second, 0.0f });
                    }
                    n_coupled += common_shared_noise_draft_pick(snd, 0, nullptr, cand.data(), cand.size(), nullptr) == id;
                    // a drafter whose logits are off by N(0, 0.5): the coupled pick vs its argmax
                    llama_token gbest = cand[0].id;
                    float gv = -INFINITY;
                    for (auto & e : cand) {
                        e.logit += nd(rng);
                        if (e.logit > gv) {
                            gv = e.logit;
                            gbest = e.id;
                        }
                    }
                    n_coupled_noisy += common_shared_noise_draft_pick(snd, 0, nullptr, cand.data(), cand.size(), nullptr) == id;
                    n_noisy_greedy  += gbest == id;
                    n_greedy_draft  += g_vals[0].first == id; // the exact argmax (greedy draft with the target's logits)
                }
            }
            common_sampler_free(s);
        }
        int dof = 0;
        const double pv = gof(cnt, p, N, &dof);
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: gof p %.4f (dof %d, %zu candidates)", c.name, pv, dof, p.size());
        check(buf, pv > 1e-4);
        if (sn) {
            snprintf(buf, sizeof(buf), "  resample at the same position gives the same token (%d / %d)", n_same, N);
            check(buf, n_same == N);
            snprintf(buf, sizeof(buf), "  coupled draft with the target's logits = the target's token (%d / %d)", n_coupled, N);
            check(buf, n_coupled == N);
            printf("  agreement: exact-logit greedy draft %.3f; logits + N(0,0.5): coupled %.3f vs greedy %.3f\n",
                    (double) n_greedy_draft / N, (double) n_coupled_noisy / N, (double) n_noisy_greedy / N);
        }
    }

    // independence across positions: one sampler, no penalties, draws at n and n + 1 (same logits): the pair counts against p x p
    {
        common_params_sampling sp;
        sp.temp  = 0.8f;
        sp.top_k = 6;
        sp.top_p = 1.0f;
        sp.min_p = 0.0f;
        sp.penalty_last_n = 0;
        std::map<llama_token, int> pc;
        std::map<llama_token, double> pp;
        std::map<llama_token, double> p1;
        const int M = N / 2;
        for (int t = 0; t < M; ++t) {
            sp.seed = 7919u * (uint32_t) t + 3u;
            common_sampler * s = common_sampler_init(g_model, sp);
            set_row();
            const llama_token a = common_sampler_sample(s, g_ctx, 0);
            if (t == 0) {
                const auto * cur = common_sampler_get_candidates(s, true);
                for (size_t k = 0; k < cur->size; ++k) {
                    p1[cur->data[k].id] = cur->data[k].p;
                }
            }
            common_sampler_accept(s, a, true);
            set_row();
            const llama_token b = common_sampler_sample(s, g_ctx, 0);
            pc[a * 65536 + b]++;
            common_sampler_free(s);
        }
        for (const auto & [ta, qa] : p1) {
            for (const auto & [tb, qb] : p1) {
                pp[ta * 65536 + tb] = qa * qb;
            }
        }
        int dof = 0;
        const double pv = gof(pc, pp, M, &dof);
        char buf[256];
        snprintf(buf, sizeof(buf), "pairs (n, n+1) independent with equal marginals: gof p %.4f (dof %d)", pv, dof);
        check(buf, pv > 1e-4);
    }

    // a clone and a copy carry the key: the clone draws what the original draws
    if (sn) {
        common_params_sampling sp;
        sp.temp = 0.8f;
        sp.top_k = 20;
        sp.seed = 42;
        common_sampler * s = common_sampler_init(g_model, sp);
        int same = 0;
        for (int t = 0; t < 200; ++t) {
            common_sampler * c = common_sampler_clone(s);
            set_row();
            const llama_token a = common_sampler_sample(s, g_ctx, 0);
            set_row();
            const llama_token b = common_sampler_sample(c, g_ctx, 0);
            same += a == b;
            common_sampler_accept(s, a, true);
            common_sampler_copy(s, c);
            common_sampler_free(c);
        }
        check("clone draws the original's token at every position (200)", same == 200);
        check("seed reported", common_sampler_get_seed(s) == 42);
        common_sampler_free(s);
    }

    llama_free(g_ctx);
    llama_model_free(g_model);
    llama_backend_free();
    printf("%s\n", g_fail ? "FAILED" : "PASSED");
    return g_fail;
}
