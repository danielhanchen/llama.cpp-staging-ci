// k19-mmvqcols: the rows-per-block lever of the dense 2..8 column mul_mat_vec_q (IQ3_S / IQ2_XXS / Q3_K, default on per arch,
// GGML_CUDA_MMVQ_COLS_CFG = AUTO | 0 | R4 | R8) against the launch it replaces (GGML_CUDA_MMVQ_COLS_CFG=0), byte for byte, at the shapes of
// Qwen3.8-27B (K 5120 / 6144 / 17408) and odd ones (rows not a multiple of 4, K of one super-block, broadcast channels), 1..8 columns,
// plain and as a verify batch (batch invariant + verify flags, so also the exact verify path under GGML_CUDA_VERIFY_REASSOC=0), with and
// without a fused gate. Each mode runs in a child process (the switches are read once per process) that dumps every output; the parent
// compares the dumps byte for byte. GGML_CUDA_MMVQ_COLS_LOG=1 makes the backend log every launch the lever takes: the R4 / R8 children
// must log launches (a silent fallback fails the test). No CUDA device: SKIP, exit 0.
//
// --bench L REPS [TYPES] [SHAPES]: per type and Qwen3.8-27B shape (gu 5120x17408, dn 17408x5120, qkv 5120x10240, o 6144x5120), 1..8
// columns: L weight copies (cold in L2), REPS graph computes of L products; prints the median us per product and a hash of the outputs.
// MMVQCOLS_BENCH_FLAG=1: the products are verify batches (batch invariant + verify flags), as a speculative verify pass builds them.

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
    if (strstr(text, "k19-mmvqcols:") != nullptr) {
        n_logs++;
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

// FNV-1a 64 of a byte range (identity of a dump, the comparison itself is byte for byte)
static uint64_t fnv(const void * d, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const uint8_t * p = (const uint8_t *) d;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

static const ggml_type all_types[] = {
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K,
};

// N rows of K, every row a copy of one of npool rows quantized from N(0, 1) values (quantizing large matrices to IQ2 is slow; the pool
// of a (type, K) is quantized once per process); the importance matrix is all ones (IQ2_XXS / IQ2_XS need one)
static std::vector<uint8_t> make_rows(ggml_type type, int K, int64_t N, uint32_t seed) {
    const size_t rb    = ggml_row_size(type, K);
    const int    npool = 64;
    static std::map<std::pair<int, int>, std::vector<uint8_t>> pools;
    std::vector<uint8_t> & pool = pools[{ (int) type, K }];
    if (pool.empty()) {
        std::mt19937 gen(4321 + K);
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

struct mm_case {
    ggml_type type;
    int  K;
    int  N;
    int  C;        // channels (weights and input broadcast 1:1)
    int  T;        // columns
    bool flagged;  // batch invariant + verify (a speculative verify batch)
    bool fused;    // gate and up of one input + SWIGLU (the fused gate path)
};

// one graph: L products (each its own weight copy), the same input
struct mm_graph {
    ggml_context * wctx = nullptr, * ctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<ggml_tensor *> out;

    bool build(ggml_backend_t be, const mm_case & c, int L, uint32_t seed) {
        const int nw = c.fused ? 2 : 1;
        ggml_init_params wp = { (size_t) (nw*L + 4) * ggml_tensor_overhead(), nullptr, true };
        wctx = ggml_init(wp);
        std::vector<ggml_tensor *> a(L), g(L, nullptr);
        for (int l = 0; l < L; ++l) {
            a[l] = ggml_new_tensor_3d(wctx, c.type, c.K, c.N, c.C);
            if (c.fused) {
                g[l] = ggml_new_tensor_3d(wctx, c.type, c.K, c.N, c.C);
            }
        }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        if (!wbuf) {
            return false;
        }
        const std::vector<uint8_t> ha = make_rows(c.type, c.K, (int64_t) c.N*c.C, seed + 1);
        const std::vector<uint8_t> hg = c.fused ? make_rows(c.type, c.K, (int64_t) c.N*c.C, seed + 2) : std::vector<uint8_t>();
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_set(a[l], ha.data(), 0, ha.size());
            if (c.fused) {
                ggml_backend_tensor_set(g[l], hg.data(), 0, hg.size());
            }
        }
        ggml_init_params gp = { 16*(size_t) L*ggml_tensor_overhead() + ggml_graph_overhead_custom(8*L, false), nullptr, true };
        ctx = ggml_init(gp);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.T, c.C);
        ggml_set_input(x);
        gf = ggml_new_graph_custom(ctx, 8*L, false);
        out.resize(L);
        for (int l = 0; l < L; ++l) {
            ggml_tensor * u = ggml_mul_mat(ctx, a[l], x);
            ggml_tensor * o = u;
            if (c.fused) {
                ggml_tensor * gt = ggml_mul_mat(ctx, g[l], x);
                o = ggml_swiglu_split(ctx, gt, u);
                if (c.flagged) {
                    gt->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                }
            }
            if (c.flagged) {
                u->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                o->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
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
        std::vector<float> xh(ggml_nelements(x));
        std::mt19937 gen(seed + 3);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & v : xh) {
            v = nd(gen);
        }
        ggml_backend_tensor_set(x, xh.data(), 0, xh.size()*sizeof(float));
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

static std::vector<mm_case> make_cases() {
    struct shp { int K, N, C; bool fused; };
    const shp shapes[] = {
        {  5120, 384, 1, false },   // Qwen3.8-27B gate / up / qkv (K 5120)
        { 17408, 160, 1, false },   // ffn_down (K 17408)
        {  6144, 130, 1, false },   // attention / GDN output (K 6144), rows not a multiple of 4
        {  5120, 256, 1, true  },   // gate + up + SWIGLU
        {   256,  37, 1, false },   // K of one super-block, odd rows
        {  1536,  66, 3, false },   // broadcast channels
    };
    std::vector<mm_case> cs;
    for (const ggml_type t : all_types) {
        for (const auto & s : shapes) {
            for (int T = 1; T <= 8; ++T) {
                cs.push_back({ t, s.K, s.N, s.C, T, false, s.fused });
                cs.push_back({ t, s.K, s.N, s.C, T, true,  s.fused });
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
        mm_graph mg;
        if (!mg.build(be, c, 1, seed)) {
            fprintf(stderr, "build failed %s K %d N %d T %d\n", ggml_type_name(c.type), c.K, c.N, c.T);
            return 1;
        }
        ggml_backend_graph_compute(be, mg.gf);
        o.resize(ggml_nbytes(mg.out[0]));
        ggml_backend_tensor_get(mg.out[0], o.data(), 0, o.size());
        fwrite(o.data(), 1, o.size(), f);
        mg.free_all();
        seed += 17;
    }
    fclose(f);
    printf("CHILD_LOGS %d\n", n_logs);
    return 0;
}

static int run_bench(ggml_backend_t be, int argc, char ** argv) {
    const int L    = argc > 2 ? atoi(argv[2]) : 6;
    const int REPS = argc > 3 ? atoi(argv[3]) : 40;
    const std::string types  = argc > 4 ? argv[4] : "all";
    const std::string shapes = argc > 5 ? argv[5] : "gu,dn,qkv,o";
    struct shp { const char * name; int K, N; };
    const shp all_shapes[] = { { "gu", 5120, 17408 }, { "dn", 17408, 5120 }, { "qkv", 5120, 10240 }, { "o", 6144, 5120 } };
    printf("bench L %d REPS %d COLS_CFG=%s KQ_ROWS=%s FLAG=%s\n", L, REPS, getenv("GGML_CUDA_MMVQ_COLS_CFG") ? getenv("GGML_CUDA_MMVQ_COLS_CFG") : "-",
        getenv("GGML_CUDA_MMVQ_KQ_ROWS") ? getenv("GGML_CUDA_MMVQ_KQ_ROWS") : "-", getenv("MMVQCOLS_BENCH_FLAG") ? getenv("MMVQCOLS_BENCH_FLAG") : "-");
    for (const ggml_type t : all_types) {
        if (types != "all" && (("," + types + ",").find("," + std::string(ggml_type_name(t)) + ",") == std::string::npos)) {
            continue;
        }
        for (const auto & s : all_shapes) {
            if (("," + shapes + ",").find("," + std::string(s.name) + ",") == std::string::npos) {
                continue;
            }
            for (int T = 1; T <= 8; ++T) {
                mm_graph mg;
                static const bool bflag = getenv("MMVQCOLS_BENCH_FLAG") != nullptr && atoi(getenv("MMVQCOLS_BENCH_FLAG")) != 0;
                const mm_case c = { t, s.K, s.N, 1, T, bflag, false };
                if (!mg.build(be, c, L, 1234)) {
                    fprintf(stderr, "build failed\n");
                    return 1;
                }
                ggml_backend_graph_compute(be, mg.gf);
                uint64_t h = 0xcbf29ce484222325ull;
                std::vector<uint8_t> o;
                for (int l = 0; l < L; ++l) {
                    o.resize(ggml_nbytes(mg.out[l]));
                    ggml_backend_tensor_get(mg.out[l], o.data(), 0, o.size());
                    h = fnv(o.data(), o.size(), h);
                }
                std::vector<double> ms;
                for (int r = 0; r < 5 + REPS; ++r) {
                    ggml_backend_synchronize(be);
                    const auto t0 = std::chrono::steady_clock::now();
                    ggml_backend_graph_compute(be, mg.gf);
                    ggml_backend_synchronize(be);
                    const auto t1 = std::chrono::steady_clock::now();
                    if (r >= 5) {
                        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    }
                }
                std::sort(ms.begin(), ms.end());
                const double us  = 1000.0*ms[ms.size()/2]/L;
                const double lo  = 1000.0*ms[ms.size()/10]/L;
                const double hi  = 1000.0*ms[ms.size()*9/10]/L;
                const double gbs = (double) ggml_row_size(t, s.K)*s.N/(us*1e3);
                printf("RES %s %s T %d us %.2f p10 %.2f p90 %.2f GBps %.1f hash %016llx\n", ggml_type_name(t), s.name, T, us, lo, hi, gbs,
                    (unsigned long long) h);
                fflush(stdout);
                mg.free_all();
            }
        }
    }
    return 0;
}

static int spawn_child(const std::string & exe, const std::string & env, const std::string & out, int * logs) {
    const std::string cmd = "env " + env + " GGML_CUDA_MMVQ_COLS_LOG=1 \"" + exe + "\" --child \"" + out + "\"";
#if defined(_WIN32)
    FILE * p = _popen(cmd.c_str(), "r");
#else
    FILE * p = popen(cmd.c_str(), "r");
#endif
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
#if defined(_WIN32)
    return _pclose(p);
#else
    return pclose(p);
#endif
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
    // the child processes are spawned through `env`
    GGML_UNUSED(argc); GGML_UNUSED(argv);
    printf("SKIP: not on Windows\n");
    return 0;
#endif
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
    if (getenv("MMVQCOLS_CPU")) {   // harness smoke only (the CUDA switches do nothing there)
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
    // base side: the default launch (never logs); head side: the lever (must log)
    struct mode { const char * name; std::string exe_base, env_base, env_head; int logs; };
    std::vector<mode> modes = {
        { "r4",          self, "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0", "GGML_CUDA_MMVQ_COLS_CFG=R4", 1 },
        { "r8",          self, "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0", "GGML_CUDA_MMVQ_COLS_CFG=R8", 1 },
        { "r4_reassoc0", self, "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_MMVQ_COLS_CFG=R4 GGML_CUDA_VERIFY_REASSOC=0", 1 },
        { "r8_reassoc0", self, "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_MMVQ_COLS_CFG=R8 GGML_CUDA_VERIFY_REASSOC=0", 1 },
        // k19-q38next: the K-quant / IQ4_XS rows (GGML_CUDA_MMVQ_KQ_ROWS=R4) against the same launch without them, plain, verify and exact
        // (the IQ lever off on both sides, so only the K-quant launches log)
        { "kq_r4",       self, "GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_MMVQ_KQ_ROWS=0", "GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_MMVQ_KQ_ROWS=R4", 1 },
        { "kq_r4_reassoc0", self, "GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_VERIFY_REASSOC=0",
                         "GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_MMVQ_KQ_ROWS=R4 GGML_CUDA_VERIFY_REASSOC=0", 1 },
        { "kq_r4_mmq0",  self, "GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_VERIFY_MMQ=0",
                         "GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_MMVQ_KQ_ROWS=R4 GGML_CUDA_VERIFY_MMQ=0", 1 },
        { "r4_mmq0",     self, "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_VERIFY_MMQ=0", "GGML_CUDA_MMVQ_COLS_CFG=R4 GGML_CUDA_VERIFY_MMQ=0", 1 },
        // the shipped default (unset = AUTO; off on archs without a measurement, so bytes only)
        { "default",     self, "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0", "MMVQCOLS_DEFAULT=1", -1 },
        { "default_exact", self, "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_VERIFY_REASSOC=0", -1 },
        // k19-q38t: IQ3_S / IQ3_XXS grid tables in shared memory (default) against the global tables (GGML_CUDA_MMVQ_GRID_SMEM=0),
        // with the rows lever off, at its default and on the exact verify path (logs: not counted, -2)
        { "grid",        self, "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0 GGML_CUDA_MMVQ_GRID_SMEM=0", "GGML_CUDA_MMVQ_KQ_ROWS=0 GGML_CUDA_MMVQ_COLS_CFG=0", -2 },
        { "grid_cols",   self, "GGML_CUDA_MMVQ_GRID_SMEM=0", "MMVQCOLS_DEFAULT=1", -2 },
        { "grid_exact",  self, "GGML_CUDA_MMVQ_GRID_SMEM=0 GGML_CUDA_VERIFY_REASSOC=0", "GGML_CUDA_VERIFY_REASSOC=0", -2 },
        { "grid_mmq0",   self, "GGML_CUDA_MMVQ_GRID_SMEM=0 GGML_CUDA_VERIFY_MMQ=0", "GGML_CUDA_VERIFY_MMQ=0", -2 },
    };
    int n_fail = 0;
    for (const auto & md : modes) {
        const std::string fb = dir + "/mmvqcols_" + tag + "_" + md.name + "_b.bin";
        const std::string fh = dir + "/mmvqcols_" + tag + "_" + md.name + "_h.bin";
        int lb = -1, lh = -1;
        const int rb = spawn_child(md.exe_base, md.env_base, fb, &lb);
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
        const bool head_off = false;
        const bool ok = rb == 0 && rh == 0 && !db.empty() && db.size() == dh.size() && ndiff == 0 && (md.logs == -2 || lb == 0) &&
                        (md.logs < 0 || (head_off ? lh == 0 : lh > 0));
        printf("%s %-14s base rc %d logs %d hash %016llx | head rc %d logs %d hash %016llx | %zu bytes, %zu floats differ\n",
            ok ? "OK  " : "FAIL", md.name, rb, lb, (unsigned long long) fnv(db.data(), db.size()), rh, lh,
            (unsigned long long) fnv(dh.data(), dh.size()), db.size(), ndiff);
        n_fail += !ok;
    }
    printf("%s: %d of %zu modes failed\n", n_fail ? "FAIL" : "PASS", n_fail, modes.size());
    return n_fail ? 1 : 0;
}
