// k19-iqsmem: bit identity of the IQ shared-memory table launches against the launches they replace, byte for byte:
//   - GGML_CUDA_MMVQ_GRID_IDS: IQ3_S / IQ3_XXS single token MUL_MAT_ID (MoE experts) with 2 / 4 / 8 rows per block (0 = one row);
//   - GGML_CUDA_MMVQ_IQ2_TAB: IQ2_XS / IQ2_S single column mat-vec (dense and MUL_MAT_ID) with the codebook in shared memory
//     (GGML_CUDA_MMVQ_IQ2_TAB_ROWS 2 / 4 / 8 rows per dense block);
// at the expert shapes of Qwen3.6-35B-A3B (K 2048, 512 rows, top-8) and Qwen3.8-Flash-Next (K 2560, 640 rows, top-10), the dense
// shapes of Qwen3.8-27B (K 5120 / 6144 / 17408) and odd ones (rows not a multiple of 4 / 8, K of one super-block), 1..4 tokens,
// plain and as a batch invariant verify batch, with and without a fused gate. Each mode runs in a child process (the switches are
// read once per process) that dumps every output; the parent compares the dumps byte for byte. The head child of a lever must log
// its engagement ("k19-iqsmem:"), the base child none, so a silent fallback fails; a negative control (GGML_CUDA_MMVQ_MOE_ONE=1:
// another K split of the single token MUL_MAT_ID) must differ. IQS_BASE_LIBS=<dir of another build's libggml*>: that build's
// defaults vs this build's (cross-build identity). No CUDA device: SKIP, exit 0.
//
// --bench NAME L REPS: microbench of one shape (q36gu q36gun fn4gu fn4gu2s fn4gu2xs d2sgu d2sdn d2xsgu d3sgu): L weight copies (cold
// in L2), REPS graph computes; prints median us per product and a hash of the outputs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
    if (strstr(text, "k19-iqsmem:") != nullptr) {
        n_logs++;
        if (getenv("IQS_PRINT_LOGS")) {
            fputs(text, stderr);
        }
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
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
        { 2048,  512, 24,  8, true  },   // Qwen3.6-35B-A3B gate + up (fused)
        { 2048,  512, 24,  8, false },   // gate or up alone (MoE cache tiers)
        { 2560,  640, 24, 10, true  },   // Qwen3.8-Flash-Next gate + up
        {  512, 2048, 16,  8, false },   // a down shape
        {  256,   36, 12,  4, false },   // K of one super-block, rows a multiple of 4 but not 8
        { 1024,   66, 12,  6, true  },   // rows not a multiple of 4
    };
    const shp dense[] = {
        {  5120, 384, 0, 0, false },   // Qwen3.8-27B gate / up / qkv
        { 17408, 160, 0, 0, false },   // ffn_down
        {  6144, 130, 0, 0, false },   // rows not a multiple of 4
        {  5120, 256, 0, 0, true  },   // gate + up + SWIGLU
        {   256,  37, 0, 0, false },   // odd rows
    };
    const ggml_type moe_types[]   = { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_XS };
    const ggml_type dense_types[] = { GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_XXS };
    std::vector<iq_case> cs;
    for (const ggml_type t : moe_types) {
        for (const auto & s : moe) {
            for (int T = 1; T <= 4; ++T) {
                cs.push_back({ t, s.K, s.N, s.E, s.topk, T, false, s.fused });
                cs.push_back({ t, s.K, s.N, s.E, s.topk, T, true,  s.fused });
            }
        }
    }
    for (const ggml_type t : dense_types) {
        for (const auto & s : dense) {
            for (int T = 1; T <= 3; ++T) {
                cs.push_back({ t, s.K, s.N, 0, 0, T, false, s.fused });
                cs.push_back({ t, s.K, s.N, 0, 0, T, true,  s.fused });
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
        { "q36gu",    GGML_TYPE_IQ3_XXS, 2048,   512, 256,  8, true  },
        { "q36gun",   GGML_TYPE_IQ3_XXS, 2048,   512, 256,  8, false },
        { "fn4gu",    GGML_TYPE_IQ3_XXS, 2560,   640, 128, 10, true  },
        { "fn4gu2s",  GGML_TYPE_IQ2_S,   2560,   640, 128, 10, true  },
        { "fn4gu2xs", GGML_TYPE_IQ2_XS,  2560,   640, 128, 10, true  },
        { "d2sgu",    GGML_TYPE_IQ2_S,   5120, 17408,   0,  0, false },
        { "d2sdn",    GGML_TYPE_IQ2_S,  17408,  5120,   0,  0, false },
        { "d2xsgu",   GGML_TYPE_IQ2_XS,  5120, 17408,   0,  0, false },
        { "d3sgu",    GGML_TYPE_IQ3_S,   5120, 17408,   0,  0, false },
    };
    const std::string name = argc > 2 ? argv[2] : "q36gu";
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
    if (getenv("IQS_CPU")) {   // harness smoke only (the CUDA switches do nothing there)
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
    const char * tmp = getenv("TMPDIR");
    const std::string dir = tmp ? tmp : ".";
    const std::string tag = std::to_string((long) getpid());
    const std::string self = argv[0];
    const std::string off = "GGML_CUDA_MMVQ_GRID_IDS=0 GGML_CUDA_MMVQ_IQ2_TAB=0";
    // logs: 1 = the head child must log and the base child must not, 0 = neither logs, -1 = not checked; diff: the dumps must differ
    struct mode { const char * name; std::string env_base, env_head; int logs; bool diff; };
    std::vector<mode> modes = {
        { "neg_moe_one",  off, off + " GGML_CUDA_MMVQ_MOE_ONE=1", 0, true },
        { "off_twice",    off, off, 0, false },
        { "grid_ids",     off, "GGML_CUDA_MMVQ_IQ2_TAB=0", 1, false },
        { "grid_ids4",    off, "GGML_CUDA_MMVQ_IQ2_TAB=0 GGML_CUDA_MMVQ_GRID_IDS=4", 1, false },
        { "grid_ids8",    off, "GGML_CUDA_MMVQ_IQ2_TAB=0 GGML_CUDA_MMVQ_GRID_IDS=8", 1, false },
        { "grid_ids_gsm0", off + " GGML_CUDA_MMVQ_GRID_SMEM=0", "GGML_CUDA_MMVQ_IQ2_TAB=0 GGML_CUDA_MMVQ_GRID_SMEM=0", 1, false },
        { "iq2tab",       off, "GGML_CUDA_MMVQ_GRID_IDS=0", 1, false },
        { "iq2tab_r4",    off, "GGML_CUDA_MMVQ_GRID_IDS=0 GGML_CUDA_MMVQ_IQ2_TAB_ROWS=4", 1, false },
        { "iq2tab_r8",    off, "GGML_CUDA_MMVQ_GRID_IDS=0 GGML_CUDA_MMVQ_IQ2_TAB_ROWS=8", 1, false },
        { "iq2tab_ids4",  off, "GGML_CUDA_MMVQ_GRID_IDS=0 GGML_CUDA_MMVQ_TURING_IDS_ROWS=4", 1, false },
        { "default",      off, "", 1, false },
        { "default_exact", off + " GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_VERIFY_REASSOC=0", 1, false },
        { "default_iqtab0", off + " GGML_CUDA_MMVQ_IQ_TAB=0", "GGML_CUDA_MMVQ_IQ_TAB=0", 1, false },
    };
    if (const char * bl = getenv("IQS_BASE_LIBS")) {
        // the other build's defaults (no k19-iqsmem code: never logs) vs this build's defaults and kill switches
        const std::string b = std::string("LD_LIBRARY_PATH=") + bl;
        modes.push_back({ "xbin_default", b, "", -1, false });
        modes.push_back({ "xbin_off",     b, off, 0, false });
        modes.push_back({ "xbin_gsm0",    b + " GGML_CUDA_MMVQ_GRID_SMEM=0", "GGML_CUDA_MMVQ_GRID_SMEM=0", -1, false });
    }
    int n_fail = 0;
    for (const auto & md : modes) {
        const std::string fb = dir + "/iqsmem_" + tag + "_" + md.name + "_b.bin";
        const std::string fh = dir + "/iqsmem_" + tag + "_" + md.name + "_h.bin";
        int lb = -1, lh = -1;
        const int rb = spawn_child(self, md.env_base, fb, &lb);
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
        const bool same_ok = md.diff ? ndiff > 0 : ndiff == 0;
        const bool logs_ok = md.logs < 0 || (md.logs == 0 ? lb == 0 && lh == 0 : lb == 0 && lh > 0);
        const bool ok = rb == 0 && rh == 0 && !db.empty() && db.size() == dh.size() && same_ok && logs_ok;
        printf("%s %-15s base rc %d logs %d hash %016llx | head rc %d logs %d hash %016llx | %zu bytes, %zu floats differ\n",
            ok ? "OK  " : "FAIL", md.name, rb, lb, (unsigned long long) fnv(db.data(), db.size()), rh, lh,
            (unsigned long long) fnv(dh.data(), dh.size()), db.size(), ndiff);
        fflush(stdout);
        n_fail += !ok;
    }
    printf("%s: %d of %zu modes failed\n", n_fail ? "FAIL" : "PASS", n_fail, modes.size());
    return n_fail ? 1 : 0;
#endif
}
