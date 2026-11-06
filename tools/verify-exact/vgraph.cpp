// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// llama-vgraph (k19-vgraph): the verify passes of a speculative target as the context grows, without a drafter. After a prompt of
// L synthetic tokens, VG_PASSES verify batches of a width drawn from VG_WEIGHTS (or VG_FIXED) are decoded; each accepts a drawn number
// of its rows, the rest is removed from the memory (the rollback of a speculative step) and the next batch starts at the argmax of
// the last accepted row. The schedule depends only on VG_SEED, so two builds decode the same batches: the hash of every logits row
// must match (byte exact). Per length: host+device time per pass by width (median, after the first VG_SKIP passes), the passes over
// 2x their width's median (re-captures, warm-ups), and the CUDA graph captures of graphs with >= VG_BIG_NODES nodes (needs
// GGML_CUDA_GRAPH_CAPTURE_LOG=1).
//
// usage: llama-vgraph -m model.gguf -c N -ngl 999 -fa on --spec-type draft-mtp --spec-draft-n-max 3 [...]
// env:   VG_LENS=2048,8192   prompt lengths       VG_PASSES=300     verify passes per length
//        VG_WEIGHTS=1:5,2:15,3:60,4:20  width:weight     VG_FIXED=w   every pass w rows (all accepted but the last)
//        VG_SEED=1   VG_SKIP=20   VG_NEXTN=1 (the nextn embeddings output, as an MTP target)   VG_BIG_NODES=1000

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

static std::vector<std::pair<int, int>> env_pairs(const char * name, const char * def) {
    const char * v = getenv(name);
    std::string s = v ? v : def;
    std::vector<std::pair<int, int>> out;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) {
            q = s.size();
        }
        const std::string item = s.substr(p, q - p);
        const size_t c = item.find(':');
        if (c == std::string::npos) {
            out.push_back({ atoi(item.c_str()), 1 });
        } else {
            out.push_back({ atoi(item.substr(0, c).c_str()), atoi(item.substr(c + 1).c_str()) });
        }
        p = q + 1;
    }
    return out;
}

static std::atomic<int> g_big_captures{0};
static std::atomic<int> g_rebuilds{0}; // kept graphs not reused (LLAMA_GRAPH_SMALL_WHY=N)
static std::atomic<int> g_invalid{0};  // kept graphs invalidated (LLAMA_GRAPH_SMALL_WHY=N)
static int g_big_nodes = 1000;
static const bool g_log_all = getenv("VG_LOG_ALL") != nullptr; // k19-vwidth: every log line

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    const char * p = strstr(text, "CUDA graph capture ");
    if (p) {
        // "CUDA graph capture N: key K, NODES nodes, previous exec launched M times"
        const char * k = strstr(p, ", ");
        if (k && atoi(k + 2) >= g_big_nodes) {
            g_big_captures++;
        }
    }
    if (strstr(text, "not reused")) {
        g_rebuilds++;
    }
    if (strstr(text, "kept graphs invalidated")) {
        g_invalid++;
    }
    if (level >= GGML_LOG_LEVEL_WARN || p || strstr(text, "changed at node") || strstr(text, "n_kv buckets") ||
            strstr(text, "not reused") || strstr(text, "invalidated") || strstr(text, "k19-vwidth") || g_log_all) {
        fputs(text, stderr);
    }
}

static uint64_t rng_next(uint64_t & s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

static uint64_t fnv(uint64_t h, const void * data, size_t n) {
    const uint8_t * b = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 1099511628211ULL;
    }
    return h;
}

static void batch_add(llama_batch & b, llama_token id, llama_pos pos, bool logits) {
    b.token   [b.n_tokens] = id;
    b.pos     [b.n_tokens] = pos;
    b.n_seq_id[b.n_tokens] = 1;
    b.seq_id  [b.n_tokens][0] = 0;
    b.logits  [b.n_tokens] = logits;
    b.n_tokens++;
}

static llama_token argmax(const float * x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 2;
    }
    common_init();
    g_big_nodes = env_int("VG_BIG_NODES", 1000);
    llama_log_set(log_cb, nullptr);

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to load the model\n");
        return 2;
    }
    if (env_int("VG_NEXTN", 1) != 0) {
        llama_set_embeddings_nextn(ctx, true, false);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    // k19-vwidth: small vocabularies (the tiny test models) draw their tokens from the whole vocabulary
    const int      tok_lo  = n_vocab > 60000 ? 1000 : 0;
    const uint64_t tok_rnd = (uint64_t) std::max(1, std::min(50000, n_vocab - tok_lo));
    const int n_batch = llama_n_batch(ctx);

    std::vector<int> lens;
    for (const auto & p : env_pairs("VG_LENS", "2048,8192")) {
        lens.push_back(p.first);
    }
    const int n_pass = env_int("VG_PASSES", 300);
    const int fixed  = env_int("VG_FIXED", 0);
    const int skip   = env_int("VG_SKIP", 20);
    const auto wts   = env_pairs("VG_WEIGHTS", "1:5,2:15,3:60,4:20");
    int w_total = 0;
    for (const auto & p : wts) {
        w_total += p.second;
    }

    printf("vgraph: n_rs_seq %u, passes %d, %s\n", llama_n_rs_seq(ctx), n_pass, fixed > 0 ? "fixed width" : "weighted widths");

    uint64_t h_all = 1469598103934665603ULL;
    for (int L : lens) {
        uint64_t rs = 0x9E3779B97F4A7C15ULL ^ (uint64_t) (env_int("VG_SEED", 1)*1000003 + L);
        // the prompt: synthetic ids away from the special tokens
        std::vector<llama_token> prompt(L);
        for (int i = 0; i < L; ++i) {
            prompt[i] = (llama_token) (tok_lo + rng_next(rs) % (uint64_t) std::max(1, n_vocab > 60000 ? std::min(n_vocab, 150000) - 2000 : n_vocab));
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(std::max(n_batch, 16), 0, 1);
        const int64_t tp0 = ggml_time_us();
        for (int i0 = 0; i0 < L; i0 += n_batch) {
            batch.n_tokens = 0;
            const int n = std::min(L - i0, n_batch);
            for (int i = 0; i < n; ++i) {
                batch_add(batch, prompt[i0 + i], i0 + i, i0 + i == L - 1);
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "prompt decode failed at %d\n", i0);
                return 2;
            }
        }
        llama_token cur = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        const double t_prompt = (ggml_time_us() - tp0)/1e3;

        const int cap0 = g_big_captures.load();
        const int reb0 = g_rebuilds.load();
        std::map<int, std::vector<double>> t_by_w;
        std::vector<std::pair<int, double>> passes;
        uint64_t h_len = 1469598103934665603ULL;
        llama_pos P = L;
        int n_tok = 0;
        // VG_REQS=R: R requests per length, each after the first starts with a new turn of VG_TURN prompt tokens (as a chat's next
        // message); per request the large-graph captures, kept graph rebuilds and invalidations of its verify passes
        const int n_req  = std::max(1, env_int("VG_REQS", 1));
        const int n_turn = env_int("VG_TURN", 200);
        int rcap0 = g_big_captures.load(), rreb0 = g_rebuilds.load(), rinv0 = g_invalid.load(), req = 0;
        double t_turn = 0.0, t_req = 0.0;
        auto req_report = [&]() {
            printf("VGR L=%d req=%d turn_ms=%.1f passes_ms=%.1f big_captures=%d rebuilds=%d invalidations=%d\n", L, req, t_turn, t_req,
                g_big_captures.load() - rcap0, g_rebuilds.load() - rreb0, g_invalid.load() - rinv0);
        };
        for (int it = 0; it < n_pass; ++it) {
            if (n_req > 1 && it > 0 && it % (n_pass/n_req) == 0 && req + 1 < n_req) {
                req_report();
                req++;
                batch.n_tokens = 0;
                for (int i = 0; i < n_turn; ++i) {
                    batch_add(batch, (llama_token) (tok_lo + rng_next(rs) % tok_rnd), P + i, i == n_turn - 1);
                }
                const int64_t tt0 = ggml_time_us();
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "turn decode failed\n");
                    return 2;
                }
                cur = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
                t_turn = (ggml_time_us() - tt0)/1e3;
                P += n_turn;
                t_req = 0.0;
                rcap0 = g_big_captures.load(); rreb0 = g_rebuilds.load(); rinv0 = g_invalid.load();
            }
            int w = fixed;
            if (w <= 0) {
                int r = (int) (rng_next(rs) % (uint64_t) w_total);
                for (const auto & p : wts) {
                    if (r < p.second) {
                        w = p.first;
                        break;
                    }
                    r -= p.second;
                }
            }
            const int a = fixed > 0 ? w - 1 : (int) (rng_next(rs) % (uint64_t) w); // accepted drafts
            batch.n_tokens = 0;
            for (int i = 0; i < w; ++i) {
                const llama_token t = i == 0 ? cur : (llama_token) (tok_lo + rng_next(rs) % tok_rnd);
                batch_add(batch, t, P + i, true);
            }
            const int64_t t0 = ggml_time_us();
            llama_set_verify(ctx, w > 1);
            const int rc = llama_decode(ctx, batch);
            llama_set_verify(ctx, false);
            if (rc != 0) {
                fprintf(stderr, "verify decode failed at pass %d width %d\n", it, w);
                return 2;
            }
            const float * last = llama_get_logits_ith(ctx, w - 1);
            const double ms = (ggml_time_us() - t0)/1e3;
            GGML_UNUSED(last);
            for (int i = 0; i < w; ++i) {
                h_len = fnv(h_len, llama_get_logits_ith(ctx, i), (size_t) n_vocab*sizeof(float));
            }
            cur = argmax(llama_get_logits_ith(ctx, a), n_vocab);
            P += a + 1;
            n_tok += a + 1;
            if (a + 1 < w && !llama_memory_seq_rm(llama_get_memory(ctx), 0, P, -1)) {
                fprintf(stderr, "seq_rm failed at pass %d (p0 %d)\n", it, (int) P);
                return 2;
            }
            if (it >= skip) {
                t_by_w[w].push_back(ms);
            }
            passes.push_back({ w, ms });
            t_req += ms;
        }
        if (n_req > 1) {
            req_report();
        }
        const int caps = g_big_captures.load() - cap0;
        const int rebs = g_rebuilds.load() - reb0;

        std::map<int, double> med;
        for (auto & kv : t_by_w) {
            auto v = kv.second;
            std::sort(v.begin(), v.end());
            med[kv.first] = v[v.size()/2];
        }
        int n_slow = 0;
        double t_slow_extra = 0.0, t_total = 0.0;
        for (int it = skip; it < (int) passes.size(); ++it) {
            const auto & p = passes[it];
            t_total += p.second;
            if (med.count(p.first) && p.second > 2.0*med[p.first]) {
                n_slow++;
                t_slow_extra += p.second - med[p.first];
            }
        }
        printf("VG L=%d prompt_ms=%.0f n_pass=%d tokens=%d end_pos=%d big_captures=%d rebuilds=%d slow=%d slow_extra_ms=%.1f total_ms=%.1f hash=%016llx\n",
            L, t_prompt, n_pass, n_tok, (int) P, caps, rebs, n_slow, t_slow_extra, t_total, (unsigned long long) h_len);
        for (auto & kv : t_by_w) {
            auto v = kv.second;
            std::sort(v.begin(), v.end());
            printf("VGW L=%d w=%d n=%zu med_ms=%.3f p10=%.3f p90=%.3f max=%.3f\n", L, kv.first, v.size(), v[v.size()/2], v[v.size()/10],
                v[v.size()*9/10], v.back());
        }
        h_all = fnv(h_all, &h_len, sizeof(h_len));
        llama_batch_free(batch);
    }
    printf("VG_ALL hash=%016llx\n", (unsigned long long) h_all);
    return 0;
}
