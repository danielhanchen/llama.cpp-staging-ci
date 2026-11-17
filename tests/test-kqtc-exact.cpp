// k19-kqtc: the few-row int8 MMQ ported from upstream (GGML_CUDA_MMQ_FEW: unset = AUTO per architecture, 0 = off, 1 = upstream's
// per-type table, N = from N columns) on dense Q4_K / Q5_K / Q6_K / IQ4_XS / Q8_0 products of 2..8 columns, against the mul_mat_vec_q /
// MMQ launch it replaces (GGML_CUDA_MMQ_FEW=0) and an fp64 CPU reference (dequantized weights x f32 activations). It quantizes the
// activations itself (x * (127 / amax)), so it cannot be bit-identical; the gates are:
//  - accuracy: NMSE(lever, ref) <= 1.10 * NMSE(replaced launch, ref) + 1e-12 per case (the q8_1 activation rounding, ~1e-5, dominates;
//    a wrong scale / nibble / layout gives >= 1e-4), the replaced launch's NMSE being the smaller of the case's and its verify-flag
//    twin's (same data: where the device table sends the unflagged product to MMQ, the flagged one keeps mmvq); a negative control
//    (outputs x (1 + 2^-7)) must fail it;
//  - exact properties: 1-column products and products it does not take keep the bytes of the replaced launch; under the bit-exact
//    verify (batch invariant + GGML_CUDA_VERIFY_REASSOC=0) every flagged product keeps them and the lever does not engage;
//  - engagement (GGML_CUDA_MMQ_FEW_LOG=1 logs each launch): every type engages on Ampere and newer; the AUTO mode only on the devices
//    of its table (cc 8.0 / 8.9), never elsewhere; never on Turing. Column invariance across widths is reported (informational).
// Each mode runs in a child process (the switches are read once per process). No CUDA device: SKIP, exit 0.
//
// --bench L REPS [TYPES] [SHAPES]: Qwen3.8-27B shapes (gu = gate + up 5120 -> 2 x 17408 on one activation, dn 17408 -> 5120,
// qkv 5120 -> 10240, o 6144 -> 5120), 1..8 columns, L weight copies (cold in L2), REPS graph computes; median us per product (gu: per
// pair) and a hash of the outputs. KQTC_BENCH_FLAG=1: verify batches (batch invariant + verify flags).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

static int n_logs = 0, n_logs_multi = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "k19-kqtc:") != nullptr) {
        if (strstr(text, " columns ") != nullptr) { // per launch line (GGML_CUDA_MMQ_FEW_LOG=1)
            n_logs++;
            if (strstr(text, " 2 matrices") != nullptr) {
                n_logs_multi++;
            }
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

static const ggml_type all_types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0 };

// N rows of K quantized from N(0, 1) values (a pool of 64 distinct rows per (type, K), repeated with a seeded pick)
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

struct shp {
    int  K, N;
    bool pair; // two products of one activation (the multi-matrix launch)
    int  N2 = 0; // rows of the second matrix of a pair (0: N), e.g. Q (6144 rows) and K (1024) of one activation
};

// rows of matrix i of a shape, and the output floats of a T-column case (the matrices back to back)
static int shp_rows(const shp & s, int i) { return i == 0 || s.N2 == 0 ? s.N : s.N2; }
static size_t shp_out(const shp & s, int T) { return (size_t) T*(shp_rows(s, 0) + (s.pair ? shp_rows(s, 1) : 0)); }

static const shp test_shapes[] = {
    { 5120, 64, false }, { 17408, 40, false }, { 6144, 37, false }, { 256, 19, false }, { 5120, 48, true }, { 1280, 21, true }, { 5120, 48, true, 16 }, { 6144, 21, true, 64 },
    { 320, 33, false }, { 1056, 16, false }, // K not a multiple of 256 (Q8_0 only)
};

struct mm_case {
    ggml_type type;
    int       shape;
    int       T;
    bool      flagged;
};

static std::vector<mm_case> make_cases() {
    std::vector<mm_case> cs;
    for (const ggml_type t : all_types) {
        for (int si = 0; si < (int) (sizeof(test_shapes)/sizeof(test_shapes[0])); ++si) {
            if (test_shapes[si].K % ggml_blck_size(t) != 0) {
                continue;
            }
            for (int T = 1; T <= 8; ++T) {
                cs.push_back({ t, si, T, false });
                cs.push_back({ t, si, T, true });
            }
        }
    }
    return cs;
}

// host data of a (type, shape): weights (1 or 2 matrices) and 8 activation columns (a T-column case takes the first T)
struct host_data {
    std::vector<uint8_t> w[2];
    std::vector<float>   x;
};

static const host_data & get_host(ggml_type type, int si) {
    static std::map<std::pair<int, int>, host_data> cache;
    host_data & h = cache[{ (int) type, si }];
    if (h.x.empty()) {
        const shp & s = test_shapes[si];
        const uint32_t seed = 1000u*(uint32_t) type + 17u*(uint32_t) si;
        h.w[0] = make_rows(type, s.K, s.N, seed + 1);
        if (s.pair) {
            h.w[1] = make_rows(type, s.K, shp_rows(s, 1), seed + 2);
        }
        h.x.resize((size_t) 8*s.K);
        std::mt19937 gen(seed + 3);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & v : h.x) {
            v = nd(gen);
        }
    }
    return h;
}

// one graph: L groups of (1 or 2) products over one activation each (bench: own activation per group, so groups do not merge)
struct mm_graph {
    ggml_context * wctx = nullptr, * ctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<ggml_tensor *> out;

    bool build(ggml_backend_t be, ggml_type type, int K, int N, bool pair, int T, bool flagged, int L,
               const std::vector<uint8_t> * w, const float * xh, int N2 = 0) {
        const int nw = pair ? 2 : 1;
        ggml_init_params wp = { (size_t) (nw*L + 4) * ggml_tensor_overhead(), nullptr, true };
        wctx = ggml_init(wp);
        std::vector<ggml_tensor *> a((size_t) nw*L);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = ggml_new_tensor_2d(wctx, type, K, i % nw == 1 && N2 > 0 ? N2 : N);
        }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        if (!wbuf) {
            return false;
        }
        for (int l = 0; l < L; ++l) {
            for (int i = 0; i < nw; ++i) {
                ggml_backend_tensor_set(a[(size_t) nw*l + i], w[i].data(), 0, w[i].size());
            }
        }
        ggml_init_params gp = { 16*(size_t) L*ggml_tensor_overhead() + ggml_graph_overhead_custom(8*L, false), nullptr, true };
        ctx = ggml_init(gp);
        gf = ggml_new_graph_custom(ctx, 8*L, false);
        std::vector<ggml_tensor *> xs(L);
        for (int l = 0; l < L; ++l) {
            xs[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
            ggml_set_input(xs[l]);
            for (int i = 0; i < nw; ++i) {
                ggml_tensor * o = ggml_mul_mat(ctx, a[(size_t) nw*l + i], xs[l]);
                if (flagged) {
                    o->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                }
                ggml_set_output(o);
                out.push_back(o);
                ggml_build_forward_expand(gf, o);
            }
        }
        if (flagged) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_graph_node(gf, i)->flags |= GGML_TENSOR_FLAG_VERIFY;
            }
        }
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            return false;
        }
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_set(xs[l], xh, 0, (size_t) K*T*sizeof(float));
        }
        return true;
    }

    void free_all() {
        if (ga) ggml_gallocr_free(ga);
        if (ctx) ggml_free(ctx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        ga = nullptr; ctx = nullptr; wbuf = nullptr; wctx = nullptr;
        out.clear();
    }
};

// child: every case's outputs (both products of a pair) back to back
static int run_child(ggml_backend_t be, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return 1;
    }
    std::vector<uint8_t> o;
    std::vector<int32_t> eng;
    for (const auto & c : make_cases()) {
        const shp & s = test_shapes[c.shape];
        const host_data & h = get_host(c.type, c.shape);
        const int logs0 = n_logs;
        mm_graph mg;
        if (!mg.build(be, c.type, s.K, s.N, s.pair, c.T, c.flagged, 1, h.w, h.x.data(), s.N2)) {
            fprintf(stderr, "build failed %s K %d N %d T %d\n", ggml_type_name(c.type), s.K, s.N, c.T);
            return 1;
        }
        ggml_backend_graph_compute(be, mg.gf);
        for (ggml_tensor * t : mg.out) {
            o.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, o.data(), 0, o.size());
            fwrite(o.data(), 1, o.size(), f);
        }
        eng.push_back(n_logs - logs0);
        mg.free_all();
    }
    fwrite(eng.data(), sizeof(int32_t), eng.size(), f); // trailer: lever launches per case
    fclose(f);
    printf("CHILD_LOGS %d %d\n", n_logs, n_logs_multi);
    return 0;
}

static int run_bench(ggml_backend_t be, int argc, char ** argv) {
    const int L    = argc > 2 ? atoi(argv[2]) : 6;
    const int REPS = argc > 3 ? atoi(argv[3]) : 30;
    const std::string types  = argc > 4 ? argv[4] : "all";
    const std::string shapes = argc > 5 ? argv[5] : "gu,dn,qkv,o";
    struct bshp { const char * name; int K, N; bool pair; };
    const bshp all_shapes[] = { { "gu", 5120, 17408, true }, { "dn", 17408, 5120, false }, { "qkv", 5120, 10240, false }, { "o", 6144, 5120, false } };
    static const bool bflag = getenv("KQTC_BENCH_FLAG") != nullptr && atoi(getenv("KQTC_BENCH_FLAG")) != 0;
    printf("bench L %d REPS %d FLAG %d MMQ_FEW=%s KQ_ROWS=%s VERIFY_MMQ=%s\n", L, REPS, (int) bflag,
        getenv("GGML_CUDA_MMQ_FEW") ? getenv("GGML_CUDA_MMQ_FEW") : "-",
        getenv("GGML_CUDA_MMVQ_KQ_ROWS") ? getenv("GGML_CUDA_MMVQ_KQ_ROWS") : "-", getenv("GGML_CUDA_VERIFY_MMQ") ? getenv("GGML_CUDA_VERIFY_MMQ") : "-");
    for (const ggml_type t : all_types) {
        if (types != "all" && (("," + types + ",").find("," + std::string(ggml_type_name(t)) + ",") == std::string::npos)) {
            continue;
        }
        for (const auto & s : all_shapes) {
            if (("," + shapes + ",").find("," + std::string(s.name) + ",") == std::string::npos) {
                continue;
            }
            std::vector<uint8_t> w[2];
            w[0] = make_rows(t, s.K, s.N, 77);
            if (s.pair) {
                w[1] = make_rows(t, s.K, s.N, 78);
            }
            std::vector<float> xh((size_t) 8*s.K);
            std::mt19937 gen(99);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto & v : xh) {
                v = nd(gen);
            }
            for (int T = 1; T <= 8; ++T) {
                mm_graph mg;
                if (!mg.build(be, t, s.K, s.N, s.pair, T, bflag, L, w, xh.data())) {
                    fprintf(stderr, "build failed\n");
                    return 1;
                }
                ggml_backend_graph_compute(be, mg.gf);
                uint64_t h = 0xcbf29ce484222325ull;
                std::vector<uint8_t> o;
                for (ggml_tensor * ot : mg.out) {
                    o.resize(ggml_nbytes(ot));
                    ggml_backend_tensor_get(ot, o.data(), 0, o.size());
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
                const double gbs = (double) ggml_row_size(t, s.K)*s.N*(s.pair ? 2 : 1)/(us*1e3);
                printf("RES %s %s T %d us %.2f p10 %.2f p90 %.2f GBps %.1f hash %016llx\n", ggml_type_name(t), s.name, T, us, lo, hi, gbs,
                    (unsigned long long) h);
                fflush(stdout);
                mg.free_all();
            }
        }
    }
    return 0;
}

static int spawn_child(const std::string & exe, const std::string & env, const std::string & out, int * logs, int * logs_multi) {
    const std::string cmd = "env " + env + " GGML_CUDA_MMQ_FEW_LOG=1 \"" + exe + "\" --child \"" + out + "\"";
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
    *logs_multi = -1;
    while (fgets(line, sizeof(line), p)) {
        int n = 0, nm = 0;
        if (sscanf(line, "CHILD_LOGS %d %d", &n, &nm) == 2) {
            *logs = n;
            *logs_multi = nm;
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

// fp64 reference of one case: out[i][c*N + r] = sum_k W_i[r][k] * x[c][k]
static std::vector<double> reference(ggml_type type, int si, int T) {
    const shp & s = test_shapes[si];
    const host_data & h = get_host(type, si);
    const auto * tr = ggml_get_type_traits(type);
    const size_t rb = ggml_row_size(type, s.K);
    const int nw = s.pair ? 2 : 1;
    std::vector<double> out(shp_out(s, T));
    std::vector<float> wr(s.K);
    size_t base = 0;
    for (int i = 0; i < nw; ++i) {
        const int N = shp_rows(s, i);
        for (int r = 0; r < N; ++r) {
            tr->to_float(h.w[i].data() + (size_t) r*rb, wr.data(), s.K);
            for (int c = 0; c < T; ++c) {
                double acc = 0.0;
                const float * xc = h.x.data() + (size_t) c*s.K;
                for (int k = 0; k < s.K; ++k) {
                    acc += (double) wr[k]*xc[k];
                }
                out[base + (size_t) c*N + r] = acc;
            }
        }
        base += (size_t) T*N;
    }
    return out;
}

static double nmse(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += ((double) a[i] - b[i])*((double) a[i] - b[i]);
        den += (double) b[i]*b[i];
    }
    return den > 0.0 ? num/den : num;
}

static double nmse_ref(const float * a, const double * r, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += (a[i] - r[i])*(a[i] - r[i]);
        den += r[i]*r[i];
    }
    return den > 0.0 ? num/den : num;
}

static int device_cc(ggml_backend_dev_t dev) {
    typedef int (*cc_fn_t)(ggml_backend_dev_t);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    cc_fn_t fn = reg ? (cc_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
    return fn ? fn(dev) : -1;
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

    const int cc = device_cc(dev);
    const std::vector<mm_case> cases = make_cases();
    printf("device %s (cc %d), %zu cases\n", ggml_backend_dev_description(dev), cc, cases.size());
    if (cc >= 0 && cc < 750) {
        printf("SKIP: no int8 MMA below cc 7.5 (the lever does not engage)\n");
        return 0;
    }
    const char * tmp = getenv("TMPDIR");
    const std::string dir  = tmp ? tmp : ".";
    const std::string tag  = std::to_string((long) getpid());
    const std::string self = argv[0];

    struct mode { const char * name; const char * env; };
    const mode modes[] = {
        { "base",       "GGML_CUDA_MMQ_FEW=0" },
        { "base_exact", "GGML_CUDA_MMQ_FEW=0 GGML_CUDA_VERIFY_REASSOC=0" },
        { "few",        "GGML_CUDA_MMQ_FEW=2" },
        { "few_up",     "GGML_CUDA_MMQ_FEW=1" },
        { "few_exact",  "GGML_CUDA_MMQ_FEW=2 GGML_CUDA_VERIFY_REASSOC=0" },
        { "few_auto",   "GGML_CUDA_MMQ_FEW_LOG=1" },   // the shipped default: GGML_CUDA_MMQ_FEW unset = the per-architecture table
    };
    const int nm = (int) (sizeof(modes)/sizeof(modes[0]));
    std::vector<std::vector<uint8_t>> dumps(nm);
    std::vector<int> logs(nm), logs_m(nm);
    int n_fail = 0;
    for (int i = 0; i < nm; ++i) {
        const std::string f = dir + "/kqtc_" + tag + "_" + modes[i].name + ".bin";
        const int rc = spawn_child(self, modes[i].env, f, &logs[i], &logs_m[i]);
        dumps[i] = read_file(f);
        remove(f.c_str());
        printf("mode %-10s rc %d logs %d (multi %d) bytes %zu hash %016llx\n", modes[i].name, rc, logs[i], logs_m[i], dumps[i].size(),
            (unsigned long long) fnv(dumps[i].data(), dumps[i].size()));
        if (rc != 0 || dumps[i].empty() || dumps[i].size() != dumps[0].size()) {
            printf("FAIL mode %s: child failed or dump size differs\n", modes[i].name);
            n_fail++;
        }
    }
    if (n_fail) {
        printf("FAIL: %d children failed\n", n_fail);
        return 1;
    }
    const auto M = [&](const char * name) {
        for (int i = 0; i < nm; ++i) {
            if (strcmp(modes[i].name, name) == 0) {
                return i;
            }
        }
        return -1;
    };
    const int ib = M("base"), it = M("few"), ibe = M("base_exact"), ite = -1, ibm = -1, itm = -1;
    const int ife = M("few_exact");
    const auto is_lever = [&](int i) { return strncmp(modes[i].name, "few", 3) == 0; };
    const auto is_few   = [&](int i) { return strncmp(modes[i].name, "few", 3) == 0; };
    const bool few_dev  = cc < 0 || cc >= 800; // the few-row MMQ needs the Ampere MMA
    const bool auto_dev = cc == 800 || cc == 890; // devices of its AUTO table (k19-int3fix: cc 12.0 off)

    // engagement: base never logs, every lever mode logs (single and multi-matrix launches)
    for (int i = 0; i < nm; ++i) {
        const bool lever = is_lever(i);
        const bool dev_i = strcmp(modes[i].name, "few_auto") == 0 ? auto_dev : few_dev;
        const bool ok = !lever ? logs[i] == 0 : is_few(i) ? (dev_i ? logs[i] > 0 : logs[i] == 0) : logs[i] > 0 && logs_m[i] > 0;
        if (!ok) {
            printf("FAIL engagement %s: %d launches (%d multi)\n", modes[i].name, logs[i], logs_m[i]);
            n_fail++;
        }
    }

    // per case. The trailer of a dump holds the lever launches of every case: a product the lever does not take (1 column, the
    // bit-exact verify, or routed elsewhere by the device tables, e.g. MMQ) must keep the bytes of its base mode
    const size_t ncase = cases.size();
    const size_t body  = dumps[0].size() - ncase*sizeof(int32_t);
    // the byte offset of every case in a dump, and its twin: the same type / shape / columns with the other verify flag. Both take the
    // same weights and activations (get_host), so the twin's base output is a second launch of the product the lever replaces: on devices
    // whose mmvq table sends a K-quant of 5..8 columns to MMQ (cc 12.0: Q4_K from 6, Q5_K from 7, Q6_K at 8) the unflagged base is MMQ
    // and the flagged one keeps mmvq (k19-vrow3), so the accuracy gate is anchored to the better of the two
    std::vector<size_t> case_off(cases.size());
    std::vector<int>    twin(cases.size(), -1);
    {
        size_t o = 0;
        for (size_t ci = 0; ci < cases.size(); ++ci) {
            case_off[ci] = o;
            o += shp_out(test_shapes[cases[ci].shape], cases[ci].T)*sizeof(float);
            for (size_t cj = 0; cj < cases.size(); ++cj) {
                if (cases[cj].type == cases[ci].type && cases[cj].shape == cases[ci].shape && cases[cj].T == cases[ci].T &&
                        cases[cj].flagged != cases[ci].flagged) {
                    twin[ci] = (int) cj;
                }
            }
        }
    }
    size_t off = 0;
    double worst_tc = 0.0, worst_ratio = 0.0;
    int n_tc_cases = 0, n_neg_fail = 0, n_neg = 0, n_few_cases = 0;
    double worst_few = 0.0, worst_few_ratio = 0.0;
    std::map<int, int> engaged_by_type;
    struct pending { int mode, type, shape, T; bool flagged; size_t off; };
    std::vector<pending> pend;
    for (size_t ci = 0; ci < ncase; ++ci) {
        const mm_case & c = cases[ci];
        const shp & s = test_shapes[c.shape];
        const size_t n = shp_out(s, c.T);
        const std::vector<double> ref = reference(c.type, c.shape, c.T);
        for (int i = 0; i < nm; ++i) {
            if (!is_lever(i)) {
                continue;
            }
            const int32_t eng = ((const int32_t *) (dumps[i].data() + body))[ci];
            const float * fi  = (const float *) (dumps[i].data() + off);
            const int    ib_i = i == ite || i == ife ? ibe : i == itm ? ibm : ib;
            const float * fbi = (const float *) (dumps[ib_i].data() + off);
            const bool allowed = c.T >= 2 && !((i == ite || i == ife) && c.flagged);
            if (eng > 0 && !allowed) {
                printf("FAIL %s %s K %d N %d T %d%s: the lever engaged where it must not\n", modes[i].name, ggml_type_name(c.type), s.K, s.N,
                    c.T, c.flagged ? " flagged" : "");
                n_fail++;
                continue;
            }
            if (eng == 0) {
                if (memcmp(fi, fbi, n*sizeof(float)) != 0) {
                    printf("FAIL %s %s K %d N %d T %d%s: not engaged but bytes differ from %s\n", modes[i].name, ggml_type_name(c.type), s.K,
                        s.N, c.T, c.flagged ? " flagged" : "", modes[ib_i].name);
                    n_fail++;
                }
                continue;
            }
            if (i == it) {
                engaged_by_type[(int) c.type]++;
            }
            const double d_base = nmse(fi, fbi, n);
            const double e_tc   = nmse_ref(fi, ref.data(), n);
            const double e_bi   = nmse_ref(fbi, ref.data(), n);
            // few-row gate anchor: the closer of this case's base launch and its twin's (see case_off / twin above)
            const double e_tw   = twin[ci] >= 0 ? nmse_ref((const float *) (dumps[ib_i].data() + case_off[twin[ci]]), ref.data(), n) : e_bi;
            const double e_anc  = std::min(e_bi, e_tw);
            // few-row MMQ: its own src1 quantization, and the product it replaces may be MMQ (the device table) with yet another one, so
            // only the distance to the fp64 reference is gated (G4: within 1.065x of mmvq at 2..3 columns, 4x closer than MMQ at 6..8)
            const bool ok = is_few(i) ? e_tc <= 1.10*e_anc + 1e-12 : d_base <= 1e-10 && e_tc <= 1.02*e_bi + 1e-12;
            if (is_few(i)) {
                worst_few       = std::max(worst_few, d_base);
                worst_few_ratio = std::max(worst_few_ratio, e_anc > 0 ? e_tc/e_anc : 0.0);
                n_few_cases++;
            } else {
                worst_tc    = std::max(worst_tc, d_base);
                worst_ratio = std::max(worst_ratio, e_bi > 0 ? e_tc/e_bi : 0.0);
                n_tc_cases++;
            }
            if (!ok) {
                printf("FAIL %s %s K %d N %d T %d%s: nmse vs base %.3g, vs ref %.3g (base %.3g, twin base %.3g)\n", modes[i].name,
                    ggml_type_name(c.type), s.K, s.N, c.T, c.flagged ? " flagged" : "", d_base, e_tc, e_bi, e_tw);
                n_fail++;
            }
            // negative control (mode few only): the outputs x (1 + 2^-7) must fail the accuracy gate
            if (i == it) {
                std::vector<float> pert(fi, fi + n);
                for (auto & v : pert) {
                    v *= 1.0f + 1.0f/128.0f;
                }
                n_neg++;
                const double e_pert = nmse_ref(pert.data(), ref.data(), n);
                if (e_pert <= 1.10*e_anc + 1e-12) {
                    n_neg_fail++;
                }
                // KQTC_DIAG=1: the per-case errors vs the fp64 reference where the two base launches differ by more than 10%
                if (getenv("KQTC_DIAG") && (e_bi > 1.10*e_tw || e_tw > 1.10*e_bi || e_pert <= 1.10*e_anc + 1e-12)) {
                    printf("DIAG %s K %d N %d T %d%s: few %.3g base %.3g twin base %.3g perturbed %.3g\n", ggml_type_name(c.type), s.K, s.N,
                        c.T, c.flagged ? " flagged" : "", e_tc, e_bi, e_tw, e_pert);
                }
            }
            pend.push_back({ i, (int) c.type, c.shape, c.T, c.flagged, off });
        }
        off += n*sizeof(float);
    }
    for (const ggml_type t : all_types) {
        if (few_dev && engaged_by_type[(int) t] == 0) {
            printf("FAIL %s: the lever never engaged\n", ggml_type_name(t));
            n_fail++;
        }
    }

    // the other base modes against base: GGML_CUDA_VERIFY_REASSOC=0 / VERIFY_MMQ=0 only change flagged products (no check), so nothing here

    // column invariance: a lever mode's T-column outputs (2 <= T <= 8) == the first T columns of its 8-column outputs, byte for byte
    int n_inv = 0, n_inv_fail = 0, n_inv_few = 0, n_inv_few_diff = 0;
    for (const auto & p : pend) {
        if (p.T == 8) {
            continue;
        }
        const pending * p8 = nullptr;
        for (const auto & q : pend) {
            if (q.mode == p.mode && q.type == p.type && q.shape == p.shape && q.flagged == p.flagged && q.T == 8) {
                p8 = &q;
            }
        }
        if (!p8) {
            continue;
        }
        const shp & s = test_shapes[p.shape];
        const int nw = s.pair ? 2 : 1;
        const float * a = (const float *) (dumps[p.mode].data() + p.off);
        const float * b = (const float *) (dumps[p.mode].data() + p8->off);
        bool same = true;
        for (int i = 0; i < nw && same; ++i) {
            const size_t oa = i == 0 ? 0 : (size_t) p.T*shp_rows(s, 0);
            const size_t ob = i == 0 ? 0 : (size_t) 8*shp_rows(s, 0);
            same = memcmp(a + oa, b + ob, (size_t) p.T*shp_rows(s, i)*sizeof(float)) == 0;
        }
        if (is_few(p.mode)) {
            n_inv_few++;
            n_inv_few_diff += !same;
            continue;
        }
        n_inv++;
        if (!same) {
            n_inv_fail++;
            if (n_inv_fail <= 10) {
                printf("FAIL invariance %s %s K %d N %d T %d%s: columns differ from the 8-column launch\n", modes[p.mode].name,
                    ggml_type_name((ggml_type) p.type), s.K, s.N, p.T, p.flagged ? " flagged" : "");
            }
        }
    }
    n_fail += n_inv_fail;

    // (no lever on this device, e.g. Turing: nothing to perturb, SKIP)
    printf("%s negative control: %d of %d perturbed cases passed the gate (must be 0)\n", !few_dev && n_neg == 0 ? "SKIP" :
        n_neg_fail == 0 && n_neg > 0 ? "OK  " : "FAIL",
        n_neg_fail, n_neg);
    n_fail += n_neg_fail != 0 || (few_dev && n_neg == 0);
    printf("(own-kernel cases %d, worst %.3g / %.6f; invariance %d checked, %d failed: none since the own kernel was dropped)\n",
        n_tc_cases, worst_tc, worst_ratio, n_inv, n_inv_fail);
    printf("few-row MMQ cases %d: worst nmse vs mmvq %.3g, worst nmse(few, ref) / nmse(mmvq, ref) %.6f; column invariance %d of %d differ "
        "(informational)\n", n_few_cases, worst_few, worst_few_ratio, n_inv_few_diff, n_inv_few);
    printf("%s: %d failures\n", n_fail ? "FAIL" : "PASS", n_fail);
    return n_fail ? 1 : 0;
#endif
}
