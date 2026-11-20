// k19-l4b: ggml_cpu_moe_host_ffn (the host experts of a single-graph MoE mailbox batch) is bit-identical to the graph it replaces
// (mul_mat_id up / gate, swiglu_split, mul_mat_id down on the CPU backend) for every thread count, on the OpenMP region and on
// ggml_cpu_moe_host_ffn_team (the host team; the OpenMP region again with GGML_MOE_HOST_TEAM=0). Rows of id -1 must be 0. Any byte difference fails.
// "perf" as the first argument times both forms instead (LFM2.5-24B-A2B expert shapes, 1 token, 4 experts per batch).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef _WIN32
static int setenv(const char * name, const char * value, int /*overwrite*/) { return _putenv_s(name, value); }
#endif

struct ffn_case {
    ggml_context * ctx = nullptr;
    ggml_tensor  * up = nullptr, * gate = nullptr, * down = nullptr, * x = nullptr, * ids = nullptr;
    ~ffn_case() { if (ctx) { ggml_free(ctx); } }
};

static void quantize_into(ggml_tensor * t, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> w((size_t) ggml_nelements(t));
    for (float & v : w) {
        v = nd(rng);
    }
    ggml_quantize_init(t->type);
    ggml_quantize_chunk(t->type, w.data(), t->data, 0, t->ne[1]*t->ne[2], t->ne[0], nullptr);
}

static void make_case(ffn_case & c, ggml_type tu, ggml_type td, int64_t n_embd, int64_t n_ff, int n_exp, int n_used, int n_tokens, uint32_t seed,
        bool with_skip) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t mem = ggml_row_size(tu, n_embd)*n_ff*n_exp*2 + ggml_row_size(td, n_ff)*n_embd*n_exp + 64*1024*1024 +
        (size_t) n_tokens*n_used*(n_ff*3 + n_embd)*sizeof(float)*4;
    ggml_init_params ip = { mem, nullptr, false };
    c.ctx  = ggml_init(ip);
    c.up   = ggml_new_tensor_3d(c.ctx, tu, n_embd, n_ff, n_exp);
    c.gate = ggml_new_tensor_3d(c.ctx, tu, n_embd, n_ff, n_exp);
    c.down = ggml_new_tensor_3d(c.ctx, td, n_ff, n_embd, n_exp);
    c.x    = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    c.ids  = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, n_used, n_tokens);
    quantize_into(c.up, rng);
    quantize_into(c.gate, rng);
    quantize_into(c.down, rng);
    for (int64_t i = 0; i < ggml_nelements(c.x); ++i) {
        ((float *) c.x->data)[i] = nd(rng);
    }
    // distinct experts per token, from a small pool so that some experts serve several tokens (the multi-column kernels)
    const int pool_n = std::max(n_used, std::min(n_exp, n_used + 2));
    for (int t = 0; t < n_tokens; ++t) {
        std::vector<int> pool(n_exp);
        for (int e = 0; e < n_exp; ++e) {
            pool[e] = e;
        }
        std::shuffle(pool.begin(), pool.begin() + pool_n, rng);
        for (int u = 0; u < n_used; ++u) {
            int32_t e = pool[u];
            if (with_skip && (t + u) % 5 == 3) {
                e = -1; // a slot the device computes (the mailbox writes -1)
            }
            ((int32_t *) c.ids->data)[t*n_used + u] = e;
        }
    }
}

// the graph of the split path; rows of id -1 are zeroed after (mul_mat_id reads no row for them, the mailbox merge skips them)
static std::vector<float> run_graph(ffn_case & c, int n_threads) {
    const int64_t n_embd = c.up->ne[0];
    const int n_used = (int) c.ids->ne[0], n_tokens = (int) c.ids->ne[1];
    // ids of -1 are not valid for mul_mat_id: the graph computes the host slots with a valid stand-in id, and those rows are zeroed
    std::vector<int32_t> ids_save((int32_t *) c.ids->data, (int32_t *) c.ids->data + n_used*n_tokens);
    for (int j = 0; j < n_used*n_tokens; ++j) {
        if (ids_save[j] < 0) {
            ((int32_t *) c.ids->data)[j] = 0;
        }
    }
    ggml_tensor * up   = ggml_mul_mat_id(c.ctx, c.up,   c.x, c.ids);
    ggml_tensor * gate = ggml_mul_mat_id(c.ctx, c.gate, c.x, c.ids);
    ggml_tensor * act  = ggml_swiglu_split(c.ctx, gate, up);
    ggml_tensor * dn   = ggml_mul_mat_id(c.ctx, c.down, act, c.ids);
    ggml_cgraph * gf = ggml_new_graph(c.ctx);
    ggml_build_forward_expand(gf, dn);
    ggml_graph_compute_with_ctx(c.ctx, gf, n_threads);
    std::vector<float> out((size_t) n_embd*n_used*n_tokens);
    memcpy(out.data(), dn->data, out.size()*sizeof(float));
    for (int j = 0; j < n_used*n_tokens; ++j) {
        ((int32_t *) c.ids->data)[j] = ids_save[j];
        if (ids_save[j] < 0) {
            memset(out.data() + (size_t) j*n_embd, 0, n_embd*sizeof(float));
        }
    }
    return out;
}

static bool g_team = true; // ggml_cpu_moe_host_ffn_team, else ggml_cpu_moe_host_ffn (the OpenMP region)

// k19-overlap: GGML_MOE_HOST_LEAN bit 1 (default on) leaves the rows of id -1 as they were (run_fused fills them with 12345)
static bool lean_skip_rows() {
    const char * e = getenv("GGML_MOE_HOST_LEAN");
    return ((e != nullptr ? atoi(e) : 3) & 1) != 0;
}

static bool run_fused(ffn_case & c, int n_threads, std::vector<float> & out) {
    const int64_t n_embd = c.up->ne[0];
    const int n_used = (int) c.ids->ne[0], n_tokens = (int) c.ids->ne[1];
    const size_t ws = ggml_cpu_moe_host_ffn_wsize(c.up, c.gate, c.down, n_used, n_tokens, n_threads);
    std::vector<uint8_t> wbuf(ws);
    out.assign((size_t) n_embd*n_used*n_tokens, 12345.0f);
    auto fn = g_team ? ggml_cpu_moe_host_ffn_team : ggml_cpu_moe_host_ffn;
    // GGML_MOE_HOST_PF=1: a read-ahead of every expert runs on the team when the batch starts (the batch stops it first)
    if (g_team && getenv("GGML_MOE_HOST_PF") != nullptr && atoi(getenv("GGML_MOE_HOST_PF")) != 0) {
        std::vector<int32_t> all((size_t) c.up->ne[2]);
        for (size_t e = 0; e < all.size(); ++e) {
            all[e] = (int32_t) e;
        }
        ggml_cpu_moe_host_team_prefetch(c.up, c.gate, c.down, all.data(), (int) all.size(), n_threads);
    } else if (!g_team && getenv("GGML_MOE_HOST_PF") != nullptr && atoi(getenv("GGML_MOE_HOST_PF")) != 0) {
        // the OpenMP form: a read-ahead region that ends when its reads are done or after 200 us
        std::vector<int32_t> all((size_t) c.up->ne[2]);
        for (size_t e = 0; e < all.size(); ++e) {
            all[e] = (int32_t) e;
        }
        auto never = [](void *) { return false; };
        ggml_cpu_moe_host_prefetch_sync(c.up, c.gate, c.down, all.data(), (int) all.size(), n_threads, never, nullptr, 200);
    }
    return fn(c.up, c.gate, c.down, (const float *) c.x->data, (const int32_t *) c.ids->data, out.data(), n_used, n_tokens,
        n_threads, wbuf.data(), wbuf.size());
}

static int perf_main() {
    // LFM2.5-24B-A2B: n_embd 2048, n_ff 1536, 64 experts; gate / up Q4_K, down Q5_K (UD-Q4_K_XL); 4 used, the host gets 2-3 of them
    ffn_case c;
    make_case(c, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, 2048, 1536, 64, 4, 1, 7, false);
    for (int team = 0; team < 2; ++team) {
    g_team = team == 1;
    printf("%s\n", g_team ? (ggml_cpu_moe_host_team_on() ? "team" : "team entry, GGML_MOE_HOST_TEAM=0: OpenMP region") : "OpenMP region");
    for (int nt : { 1, 2, 4, 5, 6, 8, 11, 12, 16 }) {
        std::vector<float> out;
        for (int w = 0; w < 50; ++w) {
            run_fused(c, nt, out);
        }
        const int n = 400;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) {
            run_fused(c, nt, out);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/n;
        printf("threads %2d: %8.1f us per batch of 4 experts\n", nt, us);
    }
    }
    return 0;
}

static bool cpu_has_openmp() {
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    auto get_features = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (get_features == nullptr) {
        return true;
    }
    for (ggml_backend_feature * f = get_features(reg); f->name != nullptr; ++f) {
        if (strcmp(f->name, "OPENMP") == 0) {
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    if (getenv("GGML_CPU_TILED_MM") == nullptr) {
        setenv("GGML_CPU_TILED_MM", "0", 1); // the tiled kernels accumulate in another order (see test-mmid-multi)
    }
    ggml_cpu_init();
    if (argc > 1 && strcmp(argv[1], "perf") == 0) {
        return perf_main();
    }
    struct tc { ggml_type tu, td; int64_t n_embd, n_ff; int n_exp, n_used, n_tokens; bool skip; };
    const tc cases[] = {
        { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, 256, 512, 16, 4, 1, false },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, 256, 512, 16, 4, 1, true  },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 512, 256, 8,  4, 3, true  },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, 256, 768, 6,  4, 4, false },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 256, 8,  2, 2, false },
        { GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, 256, 512, 12, 8, 8, true  },
        { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, 128, 384, 4,  4, 8, false },
        { GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, 256, 512, 16, 8, 4, false },
        // k19-hostrow: Flash-Next's expert types (the AVX-512 IQ3_S kernel where the build has it), experts shared by 2-4 rows
        { GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, 512, 640, 6,  4, 6, false },
    };
    int fails = 0, runs = 0, unhandled = 0;
    printf("host team: %s\n", ggml_cpu_moe_host_team_on() ? "on" : "off (GGML_MOE_HOST_TEAM=0: OpenMP region)");
    // team 0 is the OpenMP-region path, which a build without OpenMP refuses by design: only the host team runs there
    const int team0 = cpu_has_openmp() ? 0 : 1;
    if (team0 != 0) {
        printf("CPU backend built without OpenMP: OpenMP-region runs skipped\n");
    }
    uint32_t seed = 1;
    for (const tc & k : cases) {
        ffn_case c;
        make_case(c, k.tu, k.td, k.n_embd, k.n_ff, k.n_exp, k.n_used, k.n_tokens, seed++, k.skip);
        const std::vector<float> ref = run_graph(c, 4);
        for (int team = team0; team < 2; ++team)
        // k19-hostadapt: then fewer threads than the team has, and back (the idle workers skip those batches)
        for (int nt : { 1, 2, 3, 5, 7, 11, 16, 6, 5, 4, 6, 5 }) {
            for (int rep = 0; rep < 3; ++rep) {
                std::vector<float> out;
                g_team = team == 1;
                ++runs;
                if (!run_fused(c, nt, out)) {
                    ++unhandled;
                    printf("  %s/%s tokens %d threads %d: not handled\n", ggml_type_name(k.tu), ggml_type_name(k.td), k.n_tokens, nt);
                    break;
                }
                if (lean_skip_rows()) {
                    // rows of id -1 must be untouched; then they take the reference's zeros for the comparison below
                    const int64_t n_embd = c.up->ne[0];
                    for (int j = 0; j < (int) ggml_nelements(c.ids); ++j) {
                        if (((const int32_t *) c.ids->data)[j] >= 0) {
                            continue;
                        }
                        for (int64_t i = 0; i < n_embd; ++i) {
                            if (out[(size_t) j*n_embd + i] != 12345.0f) {
                                printf("  FAIL row %d of id -1 written (GGML_MOE_HOST_LEAN bit 1)\n", j);
                                ++fails;
                                break;
                            }
                        }
                        memcpy(out.data() + (size_t) j*n_embd, ref.data() + (size_t) j*n_embd, n_embd*sizeof(float));
                    }
                }
                if (memcmp(out.data(), ref.data(), ref.size()*sizeof(float)) != 0) {
                    size_t bad = 0;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        bad += memcmp(&out[i], &ref[i], sizeof(float)) != 0;
                    }
                    printf("  FAIL %s/%s embd %" PRId64 " ff %" PRId64 " tokens %d used %d threads %d rep %d: %zu of %zu values differ\n",
                        ggml_type_name(k.tu), ggml_type_name(k.td), k.n_embd, k.n_ff, k.n_tokens, k.n_used, nt, rep, bad, ref.size());
                    ++fails;
                }
            }
        }
        printf("%s/%s embd %" PRId64 " ff %" PRId64 " experts %d used %d tokens %d%s: done\n", ggml_type_name(k.tu), ggml_type_name(k.td),
            k.n_embd, k.n_ff, k.n_exp, k.n_used, k.n_tokens, k.skip ? " (with -1 ids)" : "");
    }
    ggml_cpu_moe_host_team_stop();
    printf("%d runs, %d not handled, %d failed\n", runs, unhandled, fails);
    return fails == 0 && unhandled == 0 ? 0 : 1;
}
