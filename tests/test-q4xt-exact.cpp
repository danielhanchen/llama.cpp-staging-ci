// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// k19-q4xt: bit identity of the single token MUL_MAT_ID warp per row launch (mul_mat_vec_q_wpr_launch, GGML_CUDA_MMVQ_IDS_WPR) against
// the launch it replaces, byte for byte: K-quant / Q4_0 / Q5_0 / IQ4 experts whose rows one warp covers in one pass (Qwen3.6-35B-A3B
// UD-Q4_K_XL down: Q5_K K 512, 2048 rows, top-8; IQ4_XS K 512 / 1024), every warps x rows setting (GGML_CUDA_MMVQ_IDS_WPR_W 2 / 4,
// GGML_CUDA_MMVQ_IDS_WPR_ROWS 1 / 2 / 4), 1..4 tokens, plain and as a batch invariant verify batch (the exact verify path runs the
// single token launch per token), fused gate + up + SWIGLU and the product alone, rows that the rows per block do not divide (the
// launch then falls back) and shapes it never takes (K 2048, Q6_K / Q8_0: controls). Each mode runs in a child process (the switches
// are read once per process) that dumps every output; the parent compares the dumps byte for byte. The head child of a lever must
// log its engagement ("k19-q4xt:"), the base child none, so a silent fallback fails (only where the lever is enabled:
// gate_t, from the device compute capability; elsewhere neither child may log and the bytes are still compared); a negative control (GGML_CUDA_MMVQ_MOE_ONE=1:
// another K split of the single token MUL_MAT_ID) must differ. Q4XT_BASE_EXE=<a copy of this test in another build's bin dir>: that
// build's defaults vs this build's (cross-build identity). No CUDA device: SKIP, exit 0.
//
// --bench NAMES L REPS: microbench (q4xdn q4xgu q4xup q36dn q4xdn6 q4xdn4 shdn q4xdn8): L weight copies (cold in L2), REPS graph computes; prints
// median us per product, GB/s and a hash of the outputs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#if defined(_WIN32)
#define popen  _popen
#define pclose _pclose
#endif
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

static int n_logs = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "k19-q4xt:") != nullptr) {
        n_logs++;
        if (getenv("Q4XT_PRINT_LOGS")) {
            fputs(text, stderr);
        }
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

// Where the code under test enables a lever (read from mmvq.cu, the gates must follow it): GATE_ANY = everywhere the mode runs
// (the original expectation), GATE_SM75 = the default (unset) switch value, which engages only on sm_75 binaries on the TURING
// table (ggml_cuda_highest_compiled_arch(cc) == 750: a compute capability 7.5 device), GATE_NVIDIA = an explicit "every NVIDIA
// table" value, which engages on the TURING / GENERIC tables (not GB10, cc 12.1, whose table depends on whether the build has
// sm_121: engagement not checked there). A mode whose lever is not expected on this device must not log in either child (the
// default stays off there) and still compares bytes; one that is expected and does not log fails as before.
enum gate_t { GATE_ANY = 0, GATE_SM75, GATE_NVIDIA };

// the device compute capability (major*100 + minor*10), -1 if unknown (the backend lacks ggml_backend_dev_compute_capability)
static int device_cc(ggml_backend_dev_t dev) {
    typedef int (*cc_fn_t)(ggml_backend_dev_t);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    cc_fn_t fn = reg ? (cc_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
    return fn ? fn(dev) : -1;
}

// 1 = engagement expected, 0 = not expected, -1 = not checked; cc -1 (unknown) keeps the original expectation
static int gate_expect(gate_t g, int cc) {
    if (g == GATE_ANY || cc < 0) {
        return 1;
    }
    if (g == GATE_SM75) {
        return cc == 750 ? 1 : 0;
    }
    return cc == 1210 ? -1 : 1;
}

static uint64_t fnv(const void * d, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const uint8_t * p = (const uint8_t *) d;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

// N rows of K, each a copy of one of 64 pool rows quantized from N(0, 1) (imatrix all ones: IQ2 types need one)
static std::vector<uint8_t> make_rows(ggml_type type, int K, int64_t N, uint32_t seed) {
    const size_t rb    = ggml_row_size(type, K);
    const int    npool = 64;
    static std::map<std::pair<int, int>, std::vector<uint8_t>> pools;
    std::vector<uint8_t> & pool = pools[{ (int) type, K }];
    if (pool.empty()) {
        std::mt19937 gen(4321 + K + 7*(int) type);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> f((size_t) npool*K);
        for (auto & v : f) {
            v = nd(gen);
        }
        std::vector<float> imat(K, 1.0f);
        pool.resize((size_t) npool*rb);
        ggml_quantize_chunk(type, f.data(), pool.data(), 0, npool, K, imat.data());
    }
    std::mt19937 gen(seed);
    std::vector<uint8_t> m((size_t) N*rb);
    for (int64_t r = 0; r < N; ++r) {
        memcpy(&m[(size_t) r*rb], &pool[(size_t) (gen() % npool)*rb], rb);
    }
    return m;
}

struct iq_case {
    ggml_type type;
    int  K, N;
    int  E;        // experts (0 = dense)
    int  topk;
    int  T;        // tokens / columns
    bool flagged;  // batch invariant (verify batch)
    bool fused;    // gate + up + SWIGLU
};

struct iq_graph {
    ggml_context * wctx = nullptr, * ctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<ggml_tensor *> out, ids;

    bool build(ggml_backend_t be, const iq_case & c, int L, uint32_t seed) {
        const int nw = c.fused ? 2 : 1;
        const int E  = c.E > 0 ? c.E : 1;
        ggml_init_params wp = { (size_t) (nw*L + 4) * ggml_tensor_overhead(), nullptr, true };
        wctx = ggml_init(wp);
        std::vector<ggml_tensor *> a(L), g(L, nullptr);
        for (int l = 0; l < L; ++l) {
            a[l] = ggml_new_tensor_3d(wctx, c.type, c.K, c.N, E);
            if (c.fused) {
                g[l] = ggml_new_tensor_3d(wctx, c.type, c.K, c.N, E);
            }
        }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        if (!wbuf) {
            return false;
        }
        const std::vector<uint8_t> ha = make_rows(c.type, c.K, (int64_t) c.N*E, seed + 1);
        const std::vector<uint8_t> hg = c.fused ? make_rows(c.type, c.K, (int64_t) c.N*E, seed + 2) : std::vector<uint8_t>();
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_set(a[l], ha.data(), 0, ha.size());
            if (c.fused) {
                ggml_backend_tensor_set(g[l], hg.data(), 0, hg.size());
            }
        }
        ggml_init_params gp = { 32*(size_t) L*ggml_tensor_overhead() + ggml_graph_overhead_custom(16*L, false), nullptr, true };
        ctx = ggml_init(gp);
        // MUL_MAT_ID: the gate / up input is one row per token (broadcast over the slots), the down input one row per slot
        ggml_tensor * x = c.E > 0 ? (c.fused ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, 1, c.T) : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.topk, c.T))
                                  : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.T);
        ggml_set_input(x);
        gf = ggml_new_graph_custom(ctx, 16*L, false);
        out.resize(L);
        ids.assign(L, nullptr);
        for (int l = 0; l < L; ++l) {
            ggml_tensor * u, * gt = nullptr;
            if (c.E > 0) {
                ids[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.topk, c.T);
                ggml_set_input(ids[l]);
                u = ggml_mul_mat_id(ctx, a[l], x, ids[l]);
                if (c.fused) {
                    gt = ggml_mul_mat_id(ctx, g[l], x, ids[l]);
                }
            } else {
                u = ggml_mul_mat(ctx, a[l], x);
                if (c.fused) {
                    gt = ggml_mul_mat(ctx, g[l], x);
                }
            }
            ggml_tensor * o = c.fused ? ggml_swiglu_split(ctx, gt, u) : u;
            if (c.flagged) {
                u->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                o->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                if (gt) {
                    gt->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                }
            }
            out[l] = o;
            ggml_set_output(o);
            ggml_build_forward_expand(gf, o);
        }
        if (c.flagged) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_graph_node(gf, i)->flags |= GGML_TENSOR_FLAG_VERIFY;
            }
        }
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            return false;
        }
        std::mt19937 gen(seed + 3);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xh(ggml_nelements(x));
        for (auto & v : xh) {
            v = nd(gen);
        }
        ggml_backend_tensor_set(x, xh.data(), 0, xh.size()*sizeof(float));
        for (int l = 0; l < L; ++l) {
            if (!ids[l]) {
                continue;
            }
            std::vector<int32_t> ih((size_t) c.topk*c.T);
            for (int t = 0; t < c.T; ++t) {
                std::vector<int32_t> perm(E);
                for (int e = 0; e < E; ++e) {
                    perm[e] = e;
                }
                std::shuffle(perm.begin(), perm.end(), gen);
                for (int k = 0; k < c.topk; ++k) {
                    ih[(size_t) t*c.topk + k] = perm[k];
                }
            }
            ggml_backend_tensor_set(ids[l], ih.data(), 0, ih.size()*sizeof(int32_t));
        }
        return true;
    }

    void free_all() {
        if (ga) ggml_gallocr_free(ga);
        if (ctx) ggml_free(ctx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        ga = nullptr; ctx = nullptr; wbuf = nullptr; wctx = nullptr;
    }
};

static std::vector<iq_case> make_cases() {
    struct shp { int K, N, E, topk; bool fused; };
    const shp moe[] = {
        {  512, 2048, 16,  8, false },   // Qwen3.6-35B-A3B down (engaged for Q4_K / Q5_K / Q4_0 / Q5_0 / IQ4_XS / IQ4_NL)
        {  512,  512, 16,  8, true  },   // a fused gate + up at K 512
        { 1024,  256, 12,  8, true  },   // K 1024: IQ4_XS only (4 blocks per warp pass)
        {  256,   64, 12,  4, false },   // K of one super-block
        {  512,  132, 12,  6, true  },   // rows a multiple of 4, not of 8: falls back at 8 rows per block
        {  512,   66, 12,  6, false },   // rows not a multiple of 4: falls back
        { 2048,  512, 16,  8, true  },   // Qwen3.6 gate + up: never taken (control)
    };
    const ggml_type moe_types[] = { GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_IQ4_XS,
                                    GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0 };
    std::vector<iq_case> cs;
    for (const ggml_type t : moe_types) {
        for (const auto & s : moe) {
            if (s.K % ggml_blck_size(t) != 0) {
                continue;
            }
            for (int T = 1; T <= 4; ++T) {
                cs.push_back({ t, s.K, s.N, s.E, s.topk, T, false, s.fused });
                if (T > 1) {
                    cs.push_back({ t, s.K, s.N, s.E, s.topk, T, true,  s.fused });
                }
            }
        }
    }
    // dense single column (the shared expert down of Qwen3.6-35B-A3B: Q8_0 K 512, 2048 rows) and controls
    struct dshp { int K, N; bool fused; };
    const dshp dense[] = { { 512, 2048, false }, { 512, 512, true }, { 256, 132, false }, { 2048, 512, true } };
    const ggml_type dense_types[] = { GGML_TYPE_Q8_0, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K };
    for (const ggml_type t : dense_types) {
        for (const auto & s : dense) {
            for (int T = 1; T <= 3; ++T) {
                cs.push_back({ t, s.K, s.N, 0, 0, T, false, s.fused });
                if (T > 1) {
                    cs.push_back({ t, s.K, s.N, 0, 0, T, true, s.fused });
                }
            }
        }
    }
    return cs;
}

static int run_child(ggml_backend_t be, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return 1;
    }
    uint32_t seed = 100;
    std::vector<uint8_t> o;
    for (const auto & c : make_cases()) {
        iq_graph g;
        if (!g.build(be, c, 1, seed)) {
            fprintf(stderr, "build failed %s K %d N %d E %d T %d\n", ggml_type_name(c.type), c.K, c.N, c.E, c.T);
            return 1;
        }
        ggml_backend_graph_compute(be, g.gf);
        o.resize(ggml_nbytes(g.out[0]));
        ggml_backend_tensor_get(g.out[0], o.data(), 0, o.size());
        fwrite(o.data(), 1, o.size(), f);
        g.free_all();
        seed += 17;
    }
    fclose(f);
    printf("CHILD_LOGS %d\n", n_logs);
    return 0;
}

static int run_bench(ggml_backend_t be, int argc, char ** argv) {
    struct bshape { const char * name; ggml_type type; int K, N, E, topk; bool fused; };
    const bshape shapes[] = {
        { "q4xdn",  GGML_TYPE_Q5_K,   512, 2048, 256, 8, false },
        { "q4xgu",  GGML_TYPE_Q4_K,  2048,  512, 256, 8, true  },
        { "q4xup",  GGML_TYPE_Q4_K,  2048,  512, 256, 8, false },
        { "q36dn",  GGML_TYPE_IQ4_XS, 512, 2048, 256, 8, false },
        { "q4xdn6", GGML_TYPE_Q6_K,   512, 2048, 256, 8, false },
        { "q4xdn4", GGML_TYPE_Q4_K,   512, 2048, 256, 8, false },
        { "shdn",   GGML_TYPE_Q8_0,   512, 2048,   0, 0, false },
        { "q4xdn8", GGML_TYPE_Q8_0,   512, 2048, 256, 8, false },
    };
    const std::string name = argc > 2 ? argv[2] : "q4xdn";
    const int L    = argc > 3 ? atoi(argv[3]) : 4;
    const int REPS = argc > 4 ? atoi(argv[4]) : 40;
    for (const auto & s : shapes) {
        if (("," + name + ",").find("," + std::string(s.name) + ",") == std::string::npos) {
            continue;
        }
        const iq_case c = { s.type, s.K, s.N, s.E, s.topk, 1, false, s.fused };
        iq_graph g;
        if (!g.build(be, c, L, 1234)) {
            fprintf(stderr, "build failed\n");
            return 1;
        }
        ggml_backend_graph_compute(be, g.gf);
        uint64_t h = 0xcbf29ce484222325ull;
        std::vector<uint8_t> o;
        for (int l = 0; l < L; ++l) {
            o.resize(ggml_nbytes(g.out[l]));
            ggml_backend_tensor_get(g.out[l], o.data(), 0, o.size());
            h = fnv(o.data(), o.size(), h);
        }
        std::vector<double> ms;
        for (int r = 0; r < 5 + REPS; ++r) {
            ggml_backend_synchronize(be);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, g.gf);
            ggml_backend_synchronize(be);
            const auto t1 = std::chrono::steady_clock::now();
            if (r >= 5) {
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }
        std::sort(ms.begin(), ms.end());
        const double us = 1000.0*ms[ms.size()/2]/L;
        const double lo = 1000.0*ms[ms.size()/10]/L;
        const double hi = 1000.0*ms[ms.size()*9/10]/L;
        const double bytes = (double) ggml_row_size(s.type, s.K)*s.N*(s.E > 0 ? s.topk : 1)*(s.fused ? 2 : 1);
        printf("RES %s %s us %.2f p10 %.2f p90 %.2f GBps %.1f hash %016llx\n", s.name, ggml_type_name(s.type), us, lo, hi,
            bytes/(us*1e3), (unsigned long long) h);
        fflush(stdout);
        g.free_all();
    }
    return 0;
}

static int spawn_child(const std::string & exe, const std::string & env, const std::string & out, int * logs) {
    const std::string cmd = "env " + env + " \"" + exe + "\" --child \"" + out + "\"";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) {
        return -1;
    }
    char line[512];
    *logs = -1;
    while (fgets(line, sizeof(line), p)) {
        int n = 0;
        if (sscanf(line, "CHILD_LOGS %d", &n) == 1) {
            *logs = n;
        }
    }
    return pclose(p);
}

static std::vector<uint8_t> read_file(const std::string & path) {
    std::vector<uint8_t> d;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return d;
    }
    fseek(f, 0, SEEK_END);
    d.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(d.data(), 1, d.size(), f) != d.size()) {
        d.clear();
    }
    fclose(f);
    return d;
}

int main(int argc, char ** argv) {
#if defined(_WIN32)
    GGML_UNUSED(argc); GGML_UNUSED(argv);
    printf("SKIP: not on Windows\n");
    return 0;
#else
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && strstr(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "CUDA")) {
            dev = d;
            break;
        }
    }
    if (getenv("Q4XT_CPU")) {   // harness smoke only (the CUDA switches do nothing there)
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "--child") == 0) {
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        const int rc = run_child(be, argv[2]);
        ggml_backend_free(be);
        return rc;
    }
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0) {
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        printf("device %s\n", ggml_backend_dev_description(dev));
        const int rc = run_bench(be, argc, argv);
        ggml_backend_free(be);
        return rc;
    }

    printf("device %s, %zu cases\n", ggml_backend_dev_description(dev), make_cases().size());
    const int cc = device_cc(dev);
    printf("compute capability %d\n", cc);
    const char * tmp = getenv("TMPDIR");
    const std::string dir = tmp ? tmp : ".";
    const std::string tag = std::to_string((long) getpid());
    const std::string self = argv[0];
    const std::string off = "GGML_CUDA_MMVQ_IDS_WPR=0";
    // logs: 1 = the head child must log and the base child must not, 0 = neither logs, -1 = not checked; diff: the dumps must differ
    struct mode { const char * name; std::string env_base, env_head; int logs; bool diff; std::string exe_base = ""; gate_t gate = GATE_ANY; };
    std::vector<mode> modes = {
        { "neg_moe_one",  off, off + " GGML_CUDA_MMVQ_MOE_ONE=1", 0, true },
        { "off_twice",    off, off, 0, false },
        { "off_twice_exact", off + " GGML_CUDA_VERIFY_REASSOC=0", off + " GGML_CUDA_VERIFY_REASSOC=0", 0, false },
        { "dflt_twice",   "", "", -1, false },
        { "dflt_twice_exact", "GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_VERIFY_REASSOC=0", -1, false },
        { "default",      off, "", 1, false, "", GATE_SM75 },
        { "default_exact", off + " GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_VERIFY_REASSOC=0", 1, false, "", GATE_SM75 },
        { "default_exact_pt", off + " GGML_CUDA_VERIFY_REASSOC=0 GGML_CUDA_MMVQ_EXACT_IDS_MERGE=0", "GGML_CUDA_VERIFY_REASSOC=0 GGML_CUDA_MMVQ_EXACT_IDS_MERGE=0", 1, false, "", GATE_SM75 },
        { "any_table",    off, "GGML_CUDA_MMVQ_IDS_WPR=2", 1, false, "", GATE_NVIDIA },
        { "w2r1",         off, "GGML_CUDA_MMVQ_IDS_WPR_W=2 GGML_CUDA_MMVQ_IDS_WPR_ROWS=1", 1, false, "", GATE_SM75 },
        { "w2r2",         off, "GGML_CUDA_MMVQ_IDS_WPR_W=2 GGML_CUDA_MMVQ_IDS_WPR_ROWS=2", 1, false, "", GATE_SM75 },
        { "w2r4",         off, "GGML_CUDA_MMVQ_IDS_WPR_W=2 GGML_CUDA_MMVQ_IDS_WPR_ROWS=4", 1, false, "", GATE_SM75 },
        { "w4r1",         off, "GGML_CUDA_MMVQ_IDS_WPR_W=4 GGML_CUDA_MMVQ_IDS_WPR_ROWS=1", 1, false, "", GATE_SM75 },
        { "w4r4",         off, "GGML_CUDA_MMVQ_IDS_WPR_W=4 GGML_CUDA_MMVQ_IDS_WPR_ROWS=4", 1, false, "", GATE_SM75 },
        { "w4r4_exact",   off + " GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_VERIFY_REASSOC=0 GGML_CUDA_MMVQ_IDS_WPR_W=4 GGML_CUDA_MMVQ_IDS_WPR_ROWS=4", 1, false, "", GATE_SM75 },
        { "turing_rows4", off + " GGML_CUDA_MMVQ_TURING_IDS_ROWS=4", "GGML_CUDA_MMVQ_TURING_IDS_ROWS=4", 1, false, "", GATE_SM75 },
    };
    if (const char * be = getenv("Q4XT_BASE_EXE")) {
        // a copy of this test in the other build's bin dir (the CUDA backend is loaded from the executable's directory, so
        // LD_LIBRARY_PATH alone does not switch it): that build's defaults (no k19-q4xt code: never logs) vs this build's
        const std::string b = be;
        modes.push_back({ "xbin_default", "", "", -1, false, b });
        modes.push_back({ "xbin_off",     "", off, 0, false, b });
        modes.push_back({ "xbin_exact",   "GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_VERIFY_REASSOC=0", -1, false, b });
        modes.push_back({ "xbin_off_exact", "GGML_CUDA_VERIFY_REASSOC=0", off + " GGML_CUDA_VERIFY_REASSOC=0", 0, false, b });
    }
    int n_fail = 0;
    for (const auto & md : modes) {
        const std::string fb = dir + "/q4xt_" + tag + "_" + md.name + "_b.bin";
        const std::string fh = dir + "/q4xt_" + tag + "_" + md.name + "_h.bin";
        int lb = -1, lh = -1;
        const int rb = spawn_child(md.exe_base.empty() ? self : md.exe_base, md.env_base, fb, &lb);
        const int rh = spawn_child(self, md.env_head, fh, &lh);
        const std::vector<uint8_t> db = read_file(fb), dh = read_file(fh);
        remove(fb.c_str());
        remove(fh.c_str());
        size_t ndiff = 0;
        if (db.size() == dh.size()) {
            for (size_t i = 0; i + 4 <= db.size(); i += 4) {
                ndiff += memcmp(&db[i], &dh[i], 4) != 0;
            }
        }
        if (ndiff > 0 && !md.diff && db.size() == dh.size()) {
            size_t off_b = 0;
            int nrep = 0;
            for (const auto & c : make_cases()) {
                const size_t nb = (size_t) c.N * (c.E > 0 ? (size_t) c.topk : 1) * c.T * sizeof(float);
                size_t nd = 0;
                for (size_t i = off_b; i + 4 <= off_b + nb && i + 4 <= db.size(); i += 4) {
                    nd += memcmp(&db[i], &dh[i], 4) != 0;
                }
                if (nd > 0 && nrep++ < 12) {
                    printf("  DIFF %s %s K %d N %d E %d topk %d T %d flagged %d fused %d: %zu floats\n", md.name, ggml_type_name(c.type),
                        c.K, c.N, c.E, c.topk, c.T, (int) c.flagged, (int) c.fused, nd);
                }
                off_b += nb;
            }
        }
        const bool same_ok = md.diff ? ndiff > 0 : ndiff == 0;
        int logs = md.logs;
        if (logs == 1) {
            const int ex = gate_expect(md.gate, cc);
            if (ex <= 0) {
                printf("  NOTE %s: engagement %s on cc %d (bytes still compared)\n", md.name, ex == 0 ? "not expected" : "not checked", cc);
                logs = ex;   // 0: neither child may log (the lever stays off here), -1: not checked
            }
        }
        const bool logs_ok = logs < 0 || (logs == 0 ? lb == 0 && lh == 0 : lb == 0 && lh > 0);
        const bool ok = rb == 0 && rh == 0 && !db.empty() && db.size() == dh.size() && same_ok && logs_ok;
        if (!ok) {
            printf("  WHY %s: rc %d / %d, sizes %zu / %zu, same_ok %d, logs_ok %d\n", md.name, rb, rh, db.size(), dh.size(), (int) same_ok, (int) logs_ok);
        }
        printf("%s %-15s base rc %d logs %d hash %016llx | head rc %d logs %d hash %016llx | %zu bytes, %zu floats differ\n",
            ok ? "OK  " : "FAIL", md.name, rb, lb, (unsigned long long) fnv(db.data(), db.size()), rh, lh,
            (unsigned long long) fnv(dh.data(), dh.size()), db.size(), ndiff);
        fflush(stdout);
        // the exact verify modes count too: their run-to-run variation (66-row MUL_MAT_ID) was the single column row guard, fixed in k19-exactdet
        n_fail += !ok;
    }
    printf("%s: %d of %zu modes failed\n", n_fail ? "FAIL" : "PASS", n_fail, modes.size());
    return n_fail ? 1 : 0;
#endif
}
