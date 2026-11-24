// the int8 tensor-core multi-token MUL_MAT_ID (ggml-cuda mmvq-moetc.cu, GGML_CUDA_MOE_TC) against
//   - an fp64 reference from the dequantized weights and the f32 activations (NMSE per case; the kernel's error may be at most
//     MOE_TC_TOL (1.05) x the mul_mat_vec_q path's own error + 1e-9: both read the same q8_1 activations, only the f32 order differs),
//   - the mul_mat_vec_q MoE kernels (kill switch GGML_CUDA_MOE_TC=0): NMSE between the two <= 4 x the mmvq error + 1e-6,
//   - itself: a token's bytes do not depend on the other tokens of the launch (the same token in a 2-token launch with another partner),
//   - the switch: mode 1 (verify windows) engages on a verify-tagged graph and leaves an untagged one byte-identical to the kill switch,
//   - a negative control (the kernel's output with one weight row perturbed must fail the reference bound).
// Each mode runs in a child process (this binary with --child MODE FILE; the switches are read once per process) that dumps every output;
// GGML_CUDA_MOE_TC_LOG=1 makes the backend log every launch ("int8 tensor-core MoE:"), counted per child. Types: Q4_K / Q5_K / Q6_K / IQ4_XS
// (256-value units), Q8_0 / IQ4_NL / Q5_0 (blocks of 32, K tails), fused gate + up + SWIGLU and down products, 2..8 tokens, shared
// experts across tokens (P), skipped slots, rows not a multiple of 16. No CUDA device / no Ampere-class device: SKIP, exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

static int n_tc_logs = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "int8 tensor-core MoE:") != nullptr && strstr(text, " tokens ") != nullptr && strstr(text, " slots ") != nullptr) {
        n_tc_logs++;
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct tc_shape {
    const char * name;
    ggml_type    type;
    int          K;
    int          N;
    int          E;
    int          topk;
    bool         fused;
};

static const tc_shape shapes[] = {
    { "q4k_gu",   GGML_TYPE_Q4_K,   2048,  512, 24, 8, true  },
    { "q5k_dn",   GGML_TYPE_Q5_K,    512, 2048, 24, 8, false },
    { "q6k_dn",   GGML_TYPE_Q6_K,    512, 2048, 24, 8, false },
    { "q4k_dn",   GGML_TYPE_Q4_K,    768,  200, 12, 4, false },   // 3 units, rows not a multiple of 16
    { "q5k_gu",   GGML_TYPE_Q5_K,   2048,  512, 16, 8, true  },
    { "q6k_gu",   GGML_TYPE_Q6_K,   1024,  272, 12, 6, true  },
    { "iq4xs_gu", GGML_TYPE_IQ4_XS, 2560,  640, 24, 10, true },   // 10 units: warps of 3, 3, 2, 2 units
    { "iq4xs_dn", GGML_TYPE_IQ4_XS,  512, 1000, 12, 4, false },
    { "iq4nl_dn", GGML_TYPE_IQ4_NL,  640, 2560, 24, 10, false },  // 20 blocks of 32: a tail unit of 4
    { "q5_0_dn",  GGML_TYPE_Q5_0,    640, 2560, 24, 10, false },
    { "q8_0_dn",  GGML_TYPE_Q8_0,    640, 2560, 24, 10, false },
    { "q5_0_k32", GGML_TYPE_Q5_0,     32,  100, 12, 4, false },   // one block of 32
    { "q8_0_gu",  GGML_TYPE_Q8_0,   2080,  304, 12, 6, true  },   // 65 blocks: tail unit of 1
    { "iq4nl_gu", GGML_TYPE_IQ4_NL, 2560,  640, 24, 10, true },
    { "iq3xxs_gu", GGML_TYPE_IQ3_XXS, 2560, 640, 24, 10, true },
    { "iq2s_gu",  GGML_TYPE_IQ2_S,  2560,  640, 24, 10, true },
    { "iq3xxs_dn", GGML_TYPE_IQ3_XXS, 512, 304, 12, 4, false },
};

struct tc_case {
    const tc_shape * s;
    int   T;
    float P;
    float p_skip;
};

static std::vector<tc_case> make_cases() {
    std::vector<tc_case> cs;
    for (const auto & s : shapes) {
        for (int T = 2; T <= 8; ++T) {
            cs.push_back({ &s, T, 0.55f, 0.0f });
        }
        cs.push_back({ &s, 8, 0.0f, 0.0f });
        cs.push_back({ &s, 8, 0.9f, 0.0f });
        cs.push_back({ &s, 5, 0.55f, 0.2f });
    }
    return cs;
}

// E experts of N rows, every row a copy of one of npool rows quantized from N(0, 1) values
static std::vector<uint8_t> make_experts(ggml_type type, int K, int N, int E, uint32_t seed) {
    const size_t rb    = ggml_row_size(type, K);
    const int    npool = 96;
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f((size_t) npool*K);
    for (auto & v : f) {
        v = nd(gen);
    }
    std::vector<uint8_t> pool((size_t) npool*rb);
    ggml_quantize_chunk(type, f.data(), pool.data(), 0, npool, K, nullptr);
    std::vector<uint8_t> m((size_t) E*N*rb);
    for (size_t r = 0; r < (size_t) E*N; ++r) {
        memcpy(&m[r*rb], &pool[(size_t) (gen() % npool)*rb], rb);
    }
    return m;
}

// T tokens x topk slots: a slot takes an expert of an earlier token with probability P (distinct per token); p_skip: -1
static std::vector<int32_t> make_routing(std::mt19937 & gen, int T, int topk, int E, float P, float p_skip) {
    std::vector<int32_t> r((size_t) topk*T);
    std::vector<int32_t> prev;
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> mine;
        for (int s = 0; s < topk; ++s) {
            std::vector<int32_t> cand;
            for (int32_t p : prev) {
                if (std::find(mine.begin(), mine.end(), p) == mine.end()) {
                    cand.push_back(p);
                }
            }
            int32_t e;
            if (!cand.empty() && std::uniform_real_distribution<float>(0, 1)(gen) < P) {
                e = cand[gen() % cand.size()];
            } else {
                do { e = gen() % E; } while (std::find(mine.begin(), mine.end(), e) != mine.end());
            }
            mine.push_back(e);
            r[(size_t) t*topk + s] = e;
        }
        for (int32_t e : mine) {
            if (std::find(prev.begin(), prev.end(), e) == prev.end()) {
                prev.push_back(e);
            }
        }
    }
    if (p_skip > 0.0f) {
        for (auto & e : r) {
            if (std::uniform_real_distribution<float>(0, 1)(gen) < p_skip) {
                e = -1;
            }
        }
    }
    return r;
}

// the data of one case: weights, activations, routing (the same in every child: fixed seeds)
struct tc_data {
    std::vector<uint8_t> a, g;
    std::vector<float>   x;
    std::vector<int32_t> ids;
};

static tc_data make_data(const tc_case & c, uint32_t seed) {
    tc_data d;
    const tc_shape & s = *c.s;
    d.a = make_experts(s.type, s.K, s.N, s.E, seed + 1);
    if (s.fused) {
        d.g = make_experts(s.type, s.K, s.N, s.E, seed + 2);
    }
    const int nx = s.fused ? c.T : c.T*s.topk;
    d.x.resize((size_t) nx*s.K);
    std::mt19937 gen(seed + 3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : d.x) {
        v = nd(gen);
    }
    std::mt19937 gr(seed + 9);
    d.ids = make_routing(gr, c.T, s.topk, s.E, c.P, c.p_skip);
    return d;
}

// one MUL_MAT_ID (+ SWIGLU) graph on the backend; returns the output floats [N, topk, T]
static bool run_graph(ggml_backend_t be, const tc_shape & s, int T, const tc_data & d, bool vtag, std::vector<float> & out) {
    ggml_init_params wp = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(wp);
    ggml_tensor * a = ggml_new_tensor_3d(ctx, s.type, s.K, s.N, s.E);
    ggml_tensor * g = s.fused ? ggml_new_tensor_3d(ctx, s.type, s.K, s.N, s.E) : nullptr;
    ggml_tensor * x = s.fused ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, 1, T) : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, s.topk, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.topk, T);
    ggml_set_input(x);
    ggml_set_input(ids);
    ggml_tensor * o;
    if (s.fused) {
        ggml_tensor * gt = ggml_mul_mat_id(ctx, g, x, ids);
        ggml_tensor * ut = ggml_mul_mat_id(ctx, a, x, ids);
        o = ggml_swiglu_split(ctx, gt, ut);
    } else {
        o = ggml_mul_mat_id(ctx, a, x, ids);
    }
    ggml_set_output(o);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    if (vtag) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_graph_node(gf, i)->flags |= GGML_TENSOR_FLAG_VERIFY;
        }
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!wbuf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(a, d.a.data(), 0, d.a.size());
    if (g) {
        ggml_backend_tensor_set(g, d.g.data(), 0, d.g.size());
    }
    ggml_backend_tensor_set(x, d.x.data(), 0, d.x.size()*sizeof(float));
    ggml_backend_tensor_set(ids, d.ids.data(), 0, d.ids.size()*sizeof(int32_t));
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    out.resize(ggml_nelements(o));
    ggml_backend_tensor_get(o, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    return ok;
}

// fp64 reference: dequantized weights x f32 activations (skipped slots: 0)
static std::vector<double> reference(const tc_shape & s, int T, const tc_data & d) {
    const ggml_type_traits * tt = ggml_get_type_traits(s.type);
    const size_t rb = ggml_row_size(s.type, s.K);
    std::vector<float> wa(s.K), wg(s.K);
    std::vector<double> out((size_t) s.N*s.topk*T, 0.0);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < s.topk; ++k) {
            const int e = d.ids[(size_t) t*s.topk + k];
            if (e < 0) {
                continue;
            }
            const float * xv = s.fused ? &d.x[(size_t) t*s.K] : &d.x[((size_t) t*s.topk + k)*s.K];
            for (int r = 0; r < s.N; ++r) {
                tt->to_float(&d.a[((size_t) e*s.N + r)*rb], wa.data(), s.K);
                double up = 0.0, gate = 0.0;
                for (int i = 0; i < s.K; ++i) {
                    up += (double) wa[i]*xv[i];
                }
                if (s.fused) {
                    tt->to_float(&d.g[((size_t) e*s.N + r)*rb], wg.data(), s.K);
                    for (int i = 0; i < s.K; ++i) {
                        gate += (double) wg[i]*xv[i];
                    }
                    up *= gate/(1.0 + std::exp(-gate));
                }
                out[((size_t) t*s.topk + k)*s.N + r] = up;
            }
        }
    }
    return out;
}

static double nmse(const std::vector<float> & a, const std::vector<double> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (a[i] - ref[i])*(a[i] - ref[i]);
        den += ref[i]*ref[i];
    }
    return den > 0.0 ? num/den : num;
}

// child: every case (plus the pair launches of the invariance check) -> FILE
static int run_child(ggml_backend_t be, const char * path, bool vtag) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return 1;
    }
    uint32_t seed = 1000;
    for (const auto & c : make_cases()) {
        const tc_data d = make_data(c, seed);
        std::vector<float> o;
        if (!run_graph(be, *c.s, c.T, d, vtag, o)) {
            fprintf(stderr, "compute failed %s\n", c.s->name);
            return 1;
        }
        fwrite(o.data(), sizeof(float), o.size(), f);
        // invariance: tokens (T-1, 0) as a 2-token launch, swapped (other partners and column positions)
        if (c.T >= 3) {
            const tc_shape & s = *c.s;
            tc_data d2;
            d2.a = d.a;
            d2.g = d.g;
            const int nxr = s.fused ? 1 : s.topk;
            const int tk[2] = { c.T - 1, 0 };
            for (int j = 0; j < 2; ++j) {
                d2.x.insert(d2.x.end(), d.x.begin() + (size_t) tk[j]*nxr*s.K, d.x.begin() + (size_t) (tk[j] + 1)*nxr*s.K);
                d2.ids.insert(d2.ids.end(), d.ids.begin() + (size_t) tk[j]*s.topk, d.ids.begin() + (size_t) (tk[j] + 1)*s.topk);
            }
            std::vector<float> o2;
            if (!run_graph(be, s, 2, d2, vtag, o2)) {
                return 1;
            }
            fwrite(o2.data(), sizeof(float), o2.size(), f);
        }
        seed += 31;
    }
    fclose(f);
    printf("CHILD_LOGS %d\n", n_tc_logs);
    return 0;
}

static int spawn_child(const std::string & exe, const std::string & env, const std::string & mode, const std::string & out, int * n_logs) {
    const std::string cmd = "env " + env + " GGML_CUDA_MOE_TC_MIN=2 GGML_CUDA_MOE_TC_TYPES=all GGML_CUDA_MOE_TC_LOG=1 \"" + exe + "\" --child " + mode + " \"" + out + "\"";
#if defined(_WIN32)
    FILE * p = _popen(cmd.c_str(), "r");
#else
    FILE * p = popen(cmd.c_str(), "r");
#endif
    if (!p) {
        return -1;
    }
    char line[512];
    *n_logs = -1;
    while (fgets(line, sizeof(line), p)) {
        int n = 0;
        if (sscanf(line, "CHILD_LOGS %d", &n) == 1) {
            *n_logs = n;
        }
    }
#if defined(_WIN32)
    return _pclose(p);
#else
    return pclose(p);
#endif
}

static std::vector<float> read_floats(const std::string & path) {
    std::vector<float> d;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return d;
    }
    fseek(f, 0, SEEK_END);
    d.resize(ftell(f)/sizeof(float));
    fseek(f, 0, SEEK_SET);
    if (fread(d.data(), sizeof(float), d.size(), f) != d.size()) {
        d.clear();
    }
    fclose(f);
    return d;
}

int main(int argc, char ** argv) {
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
    if (argc >= 4 && strcmp(argv[1], "--child") == 0) {
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        const int rc = run_child(be, argv[3], strcmp(argv[2], "v") == 0);
        ggml_backend_free(be);
        return rc;
    }
    printf("device %s\n", ggml_backend_dev_description(dev));
    const char * tmp = getenv("TMPDIR");
    const std::string dir = tmp ? tmp : ".";
    const std::string tag = dir + "/moe_tc_" + std::to_string((long) getpid());
    const std::string self = argv[0];
    // GGML_CUDA_MOE_TC_ANYARCH=1 (correctness runs on sm_90+ hosts) is passed through by the environment
    struct child { const char * name; const char * env; const char * mode; int logs; };   // logs: 1 must, 0 must not
    const child ch[] = {
        { "off",   "GGML_CUDA_MOE_TC=0", "u", 0 },
        { "tc2",   "GGML_CUDA_MOE_TC=2", "u", 1 },
        { "tc1v",  "GGML_CUDA_MOE_TC=1", "v", 1 },
        { "tc1u",  "GGML_CUDA_MOE_TC=1", "u", 0 },
        { "offv",  "GGML_CUDA_MOE_TC=0", "v", 0 },
    };
    std::vector<std::vector<float>> res;
    int n_fail = 0;
    for (const auto & c : ch) {
        const std::string fp = tag + "_" + c.name + ".bin";
        int nl = -1;
        const int rc = spawn_child(self, c.env, c.mode, fp, &nl);
        res.push_back(read_floats(fp));
        remove(fp.c_str());
        const bool ok = rc == 0 && !res.back().empty() && (c.logs ? nl > 0 : nl == 0);
        printf("%s child %-5s rc %d launches logged %d, %zu floats\n", ok ? "OK  " : "FAIL", c.name, rc, nl, res.back().size());
        if (!ok && rc == 0 && c.logs && nl == 0) {
            printf("SKIP: the kernel does not engage on this device (Ampere / Ada only; GGML_CUDA_MOE_TC_ANYARCH=1 elsewhere)\n");
            return 0;
        }
        n_fail += !ok;
    }
    if (n_fail) {
        printf("FAIL: %d children\n", n_fail);
        return 1;
    }
    const auto & off = res[0], & tc2 = res[1], & tc1v = res[2], & tc1u = res[3], & offv = res[4];
    const bool sizes = tc2.size() == off.size() && tc1v.size() == off.size() && tc1u.size() == off.size() && offv.size() == off.size();
    printf("%s mode 1 untagged == kill switch byte for byte; tagged kill switch == untagged kill switch\n",
        sizes && memcmp(tc1u.data(), off.data(), off.size()*4) == 0 && memcmp(offv.data(), off.data(), off.size()*4) == 0 ? "OK  " : "FAIL");
    n_fail += !(sizes && memcmp(tc1u.data(), off.data(), off.size()*4) == 0 && memcmp(offv.data(), off.data(), off.size()*4) == 0);
    printf("%s mode 1 tagged == mode 2 byte for byte\n", sizes && memcmp(tc1v.data(), tc2.data(), off.size()*4) == 0 ? "OK  " : "FAIL");
    n_fail += !(sizes && memcmp(tc1v.data(), tc2.data(), off.size()*4) == 0);
    if (!sizes) {
        return 1;
    }
    const double tol = getenv("MOE_TC_TOL") ? atof(getenv("MOE_TC_TOL")) : 1.05;
    size_t pos = 0;
    uint32_t seed = 1000;
    int n_case = 0, n_inv = 0;
    double worst_ratio = 0.0, worst_cross = 0.0;
    bool neg_ok = false;
    for (const auto & c : make_cases()) {
        const tc_shape & s = *c.s;
        const tc_data d = make_data(c, seed);
        const std::vector<double> ref = reference(s, c.T, d);
        const size_t n = ref.size();
        const std::vector<float> vo(off.begin() + pos, off.begin() + pos + n), vt(tc2.begin() + pos, tc2.begin() + pos + n);
        const double e_off = nmse(vo, ref), e_tc = nmse(vt, ref);
        std::vector<double> rofff(vo.begin(), vo.end());
        const double e_x = nmse(vt, rofff);
        bool ok = std::isfinite(e_tc) && e_tc <= tol*e_off + 1e-9 && e_x <= 4.0*e_off + 1e-6;
        // IQ3_XXS / IQ2_S take mul_mat_vec_q's integer scale rounding (GGML_CUDA_MOE_TC_IROUND, default on): every integer term is
        // the same, only the f32 summation order differs, so the two paths agree to f32 rounding (the exact product: ~e_off apart)
        const bool iround = getenv("GGML_CUDA_MOE_TC_IROUND") == nullptr || atoi(getenv("GGML_CUDA_MOE_TC_IROUND")) != 0;
        if (iround && (s.type == GGML_TYPE_IQ3_XXS || s.type == GGML_TYPE_IQ2_S) && !(e_x <= 1e-10)) {
            printf("FAIL %s T=%d: tensor-core vs mul_mat_vec_q NMSE %.3e > 1e-10 with the integer scale rounding\n", s.name, c.T, e_x);
            ok = false;
        }
        // skipped slots are zeros
        for (size_t i = 0; i < n; ++i) {
            if (d.ids[i / s.N] < 0 && vt[i] != 0.0f) {
                ok = false;
            }
        }
        worst_ratio = std::max(worst_ratio, e_off > 0 ? e_tc/e_off : 0.0);
        worst_cross = std::max(worst_cross, e_x);
        // negative control (first case of each shape): a perturbed copy of the kernel's output must fail the bound
        if (c.T == 2 && c.P > 0.5f && c.p_skip == 0.0f) {
            std::vector<float> vn = vt;
            for (int r = 0; r < s.N; ++r) {
                vn[r] *= 1.0f + 1.0f/64.0f;
            }
            const bool neg = nmse(vn, ref) > tol*e_off + 1e-9;
            neg_ok = neg_ok || neg;
            if (!neg) {
                printf("WARN negative control within the bound for %s\n", s.name);
            }
        }
        pos += n;
        bool inv = true;
        if (c.T >= 3) {
            // the pair launch: token T-1 first, token 0 second
            const size_t per = (size_t) s.N*s.topk;
            const size_t base2 = pos;
            for (int j = 0; j < 2; ++j) {
                const int tk = j == 0 ? c.T - 1 : 0;
                inv = inv && memcmp(&tc2[base2 + j*per], &tc2[pos - n + tk*per], per*4) == 0;
                inv = inv && memcmp(&tc1v[base2 + j*per], &tc1v[pos - n + tk*per], per*4) == 0;
            }
            pos += 2*per;
            n_inv += inv;
        }
        if (!ok || !inv) {
            printf("FAIL %-9s T %d P %.2f skip %.2f: nmse mmvq %.3e tc %.3e (x%.3f) tc vs mmvq %.3e invariance %s\n", s.name, c.T, c.P,
                c.p_skip, e_off, e_tc, e_off > 0 ? e_tc/e_off : 0.0, e_x, inv ? "ok" : "BROKEN");
            n_fail++;
        } else if (c.T == 8 && c.P == 0.0f) {
            printf("OK   %-9s T %d: nmse mmvq %.3e tc %.3e (x%.4f) tc vs mmvq %.3e\n", s.name, c.T, e_off, e_tc, e_off > 0 ? e_tc/e_off : 0.0, e_x);
        }
        n_case++;
        seed += 31;
    }
    printf("%s negative control\n", neg_ok ? "OK  " : "FAIL");
    n_fail += !neg_ok;
    printf("%s: %d cases (%d invariance pairs), worst nmse ratio tc / mmvq %.4f, worst tc vs mmvq %.3e, %d failed\n",
        n_fail ? "FAIL" : "PASS", n_case, n_inv, worst_ratio, worst_cross, n_fail);
    return n_fail ? 1 : 0;
}
