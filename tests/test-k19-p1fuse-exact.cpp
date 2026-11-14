// Bit identity of the k19-p1fuse CUDA changes, at the real decode / verify / prompt shapes of Qwen3.5 / 3.6 (qwen35, qwen35moe):
//   8.1a: the gated_delta_net fusion predicate (GGML_CUDA_GDN_FUSE_PRED=0: shape-only) lets a 2..8-token replay window (and any launch
//         that takes the AR kernel) read its state in place from the cache rows and apply its gates in the kernel;
//   6.2a: the gated output norm RMS_NORM -> MUL -> SILU(z) -> MUL in one kernel (GGML_CUDA_GATED_NORM_SILU=0: sigmoid only), with the
//         q8_1 copy of its Q8_0 out projection (decode) or the MMQ copy (prompt);
//   2.3:  mul_mat_vec_f with one barrier for all its columns (GGML_CUDA_MMVF_ONEBAR=0: a barrier pair per column), F32 / F16 / BF16
//         weights at 1..8 columns, also the multi-matrix launch (alpha + beta of a delta-net layer).
// Three children of this binary run every case and dump the outputs: "head" (defaults), "kill" (the three switches off = the code
// before k19-p1fuse) and "unfused" (kill + the gated norm, the in-place state and the gate fusions off). The parent compares the dumps
// byte for byte. GGML_CUDA_P1FUSE_STATS=1 makes each child print its launch counts at exit (stderr): the head child must take every
// new path (counts > 0), the kill and unfused children none, so a silent fallback fails. No CUDA device: SKIP, exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

enum case_kind { K_GDN, K_GN, K_MMVF };

struct p1_case {
    case_kind kind;
    // GDN: v heads, n_t, replay (1) / plain (0), w_par;  GN: v heads, n_t, silu (1) / sigmoid (0), -;  MMVF: K, n_t, type (0 F32, 1 F16,
    // 2 BF16), multi (two products over one activation)
    int64_t a, b, c, d;
    std::string str() const {
        char buf[160];
        if (kind == K_GDN) {
            snprintf(buf, sizeof(buf), "8.1a gdn %s heads=%lld n_t=%lld w_par=%lld", c ? "replay" : "plain ", (long long) a, (long long) b,
                     (long long) d);
        } else if (kind == K_GN) {
            snprintf(buf, sizeof(buf), "6.2a gated norm %s heads=%lld n_t=%lld", c ? "silu   " : "sigmoid", (long long) a, (long long) b);
        } else {
            static const char * tn[3] = { "f32", "f16", "bf16" };
            snprintf(buf, sizeof(buf), "2.3 mmvf %s K=%lld n_t=%lld%s", tn[c], (long long) a, (long long) b, d ? " multi" : "");
        }
        return buf;
    }
};

static std::vector<p1_case> cases() {
    std::vector<p1_case> s;
    // qwen35moe GDN: 16 key heads x 128, 32 value heads (Qwen3.6-35B-A3B); 48 (Qwen3.6-27B dense). Replay windows of 1..8 tokens
    // (MTP / DFlash verify), plain launches below and at the 64-token mma limit
    for (int64_t h : { 32, 48 }) {
        for (int64_t nt : { 1, 2, 3, 5, 8 }) {
            s.push_back({ K_GDN, h, nt, 1, nt % 2 });
        }
        for (int64_t nt : { 2, 8, 33, 64, 100 }) {
            s.push_back({ K_GDN, h, nt, 0, 0 });
        }
    }
    for (int64_t h : { 32, 48 }) {
        for (int64_t nt : { 1, 2, 4, 8, 33, 512 }) {
            s.push_back({ K_GN, h, nt, 1, 0 });
        }
        s.push_back({ K_GN, h, 4, 0, 0 });   // sigmoid: unchanged
        s.push_back({ K_GN, h, 512, 0, 0 });
    }
    for (int64_t ty : { 0, 1, 2 }) {
        for (int64_t nt = 1; nt <= 8; ++nt) {
            s.push_back({ K_MMVF, 2048, nt, ty, 0 });
        }
        s.push_back({ K_MMVF, 512, 3, ty, 0 });
        s.push_back({ K_MMVF, 1000, 5, ty, 0 });
    }
    for (int64_t nt = 1; nt <= 8; ++nt) {
        s.push_back({ K_MMVF, 2048, nt, 0, 1 });
    }
    return s;
}

struct built {
    ggml_context *            ctx = nullptr;
    ggml_cgraph *             gf  = nullptr;
    std::vector<ggml_tensor *> outs;
};

static built build_case(const p1_case & t) {
    built r;
    ggml_init_params ip = { 256 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    r.ctx = ggml_init(ip);
    ggml_context * ctx = r.ctx;
    r.gf = ggml_new_graph(ctx);
    if (t.kind == K_GDN) {
        // build_rs (state gather by the rollback ids) + the gates (beta = sigmoid(b), g = softplus(a + dt_bias) * A) + the delta net
        const int64_t S = 128, Hk = 16, H = t.a, n_t = t.b, n_win = 8, n_rows = 4;
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, n_t, 1);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, n_t, 1);
        ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_t, 1);
        ggml_set_name(v, "v");
        ggml_tensor * braw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_t);
        ggml_set_name(braw, "braw");
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_t);
        ggml_set_name(a, "a");
        ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_set_name(dt, "dt");
        ggml_tensor * A = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_set_name(A, "gscale");
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * S * H, n_rows);
        ggml_set_name(cache, "cache");
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(ids, "rsid");
        ggml_tensor * beta = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, braw), 1, H, n_t, 1);
        ggml_tensor * g    = ggml_reshape_4d(ctx, ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, a, dt)), A), 1, H, n_t, 1);
        ggml_tensor * st   = ggml_reshape_4d(ctx, ggml_get_rows(ctx, cache, ids), S, S, H, 1);
        ggml_tensor * out;
        if (t.c) {
            ggml_tensor * wk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, 2 * n_win, 1);
            ggml_set_name(wk, "wk");
            ggml_tensor * wv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 2 * n_win, 1);
            ggml_set_name(wv, "wv");
            ggml_tensor * wgb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, H, 2 * n_win, 1);
            ggml_set_name(wgb, "wgb");
            ggml_tensor * wp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            ggml_set_name(wp, t.d ? "wp1" : "wp0");
            out = ggml_gated_delta_net_replay(ctx, q, k, v, g, beta, st, wk, wv, wgb, wp);
            ggml_build_forward_expand(r.gf, out);
            r.outs = { out, wk, wv, wgb };
        } else {
            out = ggml_gated_delta_net(ctx, q, k, v, g, beta, st, 1);
            ggml_build_forward_expand(r.gf, out);
            r.outs = { out };
        }
    } else if (t.kind == K_GN) {
        // qwen35 / qwen35moe build_norm_gated (gated delta net output) into ssm_out (Q8_0)
        const int64_t hd = 128, heads = t.a, n_t = t.b;
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, n_t);
        ggml_set_name(x, "x");
        ggml_tensor * z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, n_t);
        ggml_set_name(z, "z");
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_set_name(w, "w");
        ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hd * heads, 2048);
        ggml_set_name(wo, "wo");
        ggml_tensor * normed = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
        ggml_tensor * gate   = t.c ? ggml_silu(ctx, z) : ggml_sigmoid(ctx, z);
        ggml_tensor * gated  = ggml_mul(ctx, normed, gate);
        ggml_tensor * out    = ggml_mul_mat(ctx, wo, ggml_reshape_2d(ctx, gated, hd * heads, n_t));
        ggml_build_forward_expand(r.gf, out);
        r.outs = { gated, out };
    } else {
        static const ggml_type types[3] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 };
        const int64_t K = t.a, n_t = t.b, M = 48;
        ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_t);
        ggml_set_name(y, "y");
        ggml_tensor * w0 = ggml_new_tensor_2d(ctx, types[t.c], K, M);
        ggml_set_name(w0, "w0");
        ggml_tensor * o0 = ggml_mul_mat(ctx, w0, y);
        ggml_build_forward_expand(r.gf, o0);
        r.outs = { o0 };
        if (t.d) {
            ggml_tensor * w1 = ggml_new_tensor_2d(ctx, types[t.c], K, M / 2);
            ggml_set_name(w1, "w1");
            ggml_tensor * o1 = ggml_mul_mat(ctx, w1, y);
            ggml_build_forward_expand(r.gf, o1);
            r.outs.push_back(o1);
        }
    }
    return r;
}

static void init_inputs(ggml_context * ctx, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (ggml_tensor * x = ggml_get_first_tensor(ctx); x != nullptr; x = ggml_get_next_tensor(ctx, x)) {
        if (x->buffer == nullptr || x->view_src != nullptr || x->op != GGML_OP_NONE) {
            continue;
        }
        const int64_t n = ggml_nelements(x);
        if (strcmp(x->name, "rsid") == 0) {
            const int32_t id = 2; // not row 0: the in-place read must follow the id
            ggml_backend_tensor_set(x, &id, 0, sizeof(id));
        } else if (strncmp(x->name, "wp", 2) == 0) {
            const int32_t wp[2] = { 3, x->name[2] == '1' ? 1 : 0 }; // commit 3 stored tokens of window half w_par
            ggml_backend_tensor_set(x, wp, 0, sizeof(wp));
        } else if (ggml_is_quantized(x->type)) {
            const int64_t n_per_row = x->ne[0];
            std::vector<float> f(n);
            for (auto & v : f) {
                v = u(rng);
            }
            std::vector<uint8_t> q(ggml_nbytes(x));
            ggml_quantize_chunk(x->type, f.data(), q.data(), 0, n / n_per_row, n_per_row, nullptr);
            ggml_backend_tensor_set(x, q.data(), 0, q.size());
        } else if (x->type == GGML_TYPE_F32) {
            std::vector<float> d(n);
            const bool decay = strcmp(x->name, "gscale") == 0; // A = -exp(A_log) < 0: a decaying state
            const float s = (strcmp(x->name, "q") == 0 || strcmp(x->name, "k") == 0 || strcmp(x->name, "wk") == 0) ? 0.088f : 1.0f;
            for (auto & v : d) {
                v = decay ? -0.1f - 0.9f * std::fabs(u(rng)) : s * u(rng);
            }
            ggml_backend_tensor_set(x, d.data(), 0, ggml_nbytes(x));
        } else if (x->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> d(n);
            for (auto & v : d) {
                v = ggml_fp32_to_fp16(u(rng));
            }
            ggml_backend_tensor_set(x, d.data(), 0, ggml_nbytes(x));
        } else if (x->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> d(n);
            for (auto & v : d) {
                v = ggml_fp32_to_bf16(u(rng));
            }
            ggml_backend_tensor_set(x, d.data(), 0, ggml_nbytes(x));
        }
    }
}

static std::vector<uint8_t> run_case(ggml_backend_t backend, const p1_case & t, int seed) {
    built b = build_case(t);
    std::vector<uint8_t> res;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    if (buf != nullptr) {
        init_inputs(b.ctx, seed);
        if (ggml_backend_graph_compute(backend, b.gf) == GGML_STATUS_SUCCESS) {
            for (ggml_tensor * o : b.outs) {
                const size_t off = res.size();
                res.resize(off + ggml_nbytes(o));
                ggml_backend_tensor_get(o, res.data() + off, 0, ggml_nbytes(o));
            }
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(b.ctx);
    return res;
}

static ggml_backend_dev_t gpu_dev() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "CUDA") == 0) {
            return d;
        }
    }
    return nullptr;
}

static int run_child(const char * path) {
    ggml_backend_dev_t dev = gpu_dev();
    ggml_backend_t backend = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (backend == nullptr) {
        return 3;
    }
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        return 4;
    }
    const std::vector<p1_case> sh = cases();
    for (size_t i = 0; i < sh.size(); ++i) {
        const std::vector<uint8_t> o = run_case(backend, sh[i], 4321 + (int) i);
        const uint64_t n = o.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(o.data(), 1, o.size(), f);
    }
    fclose(f);
    ggml_backend_free(backend);
    return 0;
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static bool read_dump(const std::string & path, std::vector<std::vector<uint8_t>> & outs, size_t n_cases) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    for (size_t i = 0; i < n_cases; ++i) {
        uint64_t n = 0;
        if (fread(&n, sizeof(n), 1, f) != 1) {
            fclose(f);
            return false;
        }
        outs.emplace_back(n);
        if (n > 0 && fread(outs.back().data(), 1, n, f) != n) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

// the "k19-p1fuse stats:" line of a child's stderr: gdn_fused_new gdn_fused silu sigmoid onebar twobar
static bool read_stats(const std::string & path, long long st[6]) {
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    char line[1024];
    bool ok = false;
    while (fgets(line, sizeof(line), f) != nullptr) {
        const char * p = strstr(line, "k19-p1fuse stats:");
        if (p != nullptr) {
            ok = sscanf(p, "k19-p1fuse stats: gdn_fused_new=%lld gdn_fused=%lld gated_norm_silu=%lld gated_norm_sigmoid=%lld mmvf_onebar=%lld "
                           "mmvf_twobar=%lld", &st[0], &st[1], &st[2], &st[3], &st[4], &st[5]) == 6;
        } else if (strstr(line, "error") != nullptr || strstr(line, "ERROR") != nullptr) {
            fputs(line, stderr);
        }
    }
    fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[3]);
    }
    if (gpu_dev() == nullptr) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    const std::vector<p1_case> sh = cases();
    static const char * names[3] = { "head", "kill", "unfused" };
    std::vector<std::vector<uint8_t>> outs[3];
    long long st[3][6] = {};
    const std::string dir = argc > 1 ? argv[1] : ".";
    for (int mode = 0; mode < 3; ++mode) {
        const std::string path = dir + "/k19_p1fuse_exact_" + std::to_string(mode) + ".bin";
        const std::string errp = path + ".err";
        set_env("GGML_CUDA_P1FUSE_STATS", "1");
        set_env("GGML_CUDA_GDN_FUSE_PRED", mode ? "0" : "1");
        set_env("GGML_CUDA_GATED_NORM_SILU", mode ? "0" : "1");
        set_env("GGML_CUDA_MMVF_ONEBAR", mode ? "0" : "1");
        set_env("GGML_CUDA_DISABLE_GATED_NORM_Q8", mode == 2 ? "1" : "0");
        set_env("GGML_CUDA_GDN_STATE_IN_PLACE", mode == 2 ? "0" : "1");
        set_env("GGML_CUDA_GDN_GATES", mode == 2 ? "0" : "1");
        std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\" 2> \"" + errp + "\"";
#ifdef _WIN32
        cmd = "\"" + cmd + "\"";
#endif
        const int rc = std::system(cmd.c_str());
        if (rc != 0 || !read_dump(path, outs[mode], sh.size()) || !read_stats(errp, st[mode])) {
            fprintf(stderr, "FAIL: child %s rc %d (dump or stats missing, see %s)\n", names[mode], rc, errp.c_str());
            return 1;
        }
        std::remove(path.c_str());
        std::remove(errp.c_str());
        printf("child %-7s: gdn fused new %lld (all %lld), gated norm silu %lld / sigmoid %lld, mmvf one-barrier %lld / per-column %lld\n",
               names[mode], st[mode][0], st[mode][1], st[mode][2], st[mode][3], st[mode][4], st[mode][5]);
    }
    int n_fail = 0;
    for (size_t i = 0; i < sh.size(); ++i) {
        bool ok = !outs[0][i].empty();
        for (int m = 1; m < 3; ++m) {
            ok = ok && outs[m][i].size() == outs[0][i].size() && memcmp(outs[m][i].data(), outs[0][i].data(), outs[0][i].size()) == 0;
        }
        printf("  head == kill == unfused: %-48s bytes=%zu: %s\n", sh[i].str().c_str(), outs[0][i].size(), ok ? "memcmp == 0" : "DIFFER");
        n_fail += !ok;
    }
    // engagement: the head child takes every new path, the others none of them
    const bool eng = st[0][0] > 0 && st[0][2] > 0 && st[0][4] > 0 && st[0][5] == 0 && st[1][0] == 0 && st[1][2] == 0 && st[1][4] == 0 &&
        st[1][5] > 0 && st[2][0] == 0 && st[2][1] == 0 && st[2][2] == 0 && st[2][3] == 0 && st[2][4] == 0;
    if (!eng) {
        printf("FAIL: engagement (head must take the new GDN fusions, the silu gated norm and the one-barrier mmvf; kill / unfused none)\n");
        return 1;
    }
    if (n_fail) {
        printf("FAIL: %d / %zu cases differ\n", n_fail, sh.size());
        return 1;
    }
    printf("OK: %zu / %zu cases bit-identical in head, kill and unfused\n", sh.size(), sh.size());
    return 0;
}
