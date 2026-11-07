// k19-dsv4: the fused CUDA top-k MoE routing for 384 experts (DeepSeek-V4.1: sqrt-softplus gating, exp_probs_b bias for the selection,
// 6 used, weights normalised and scaled by 1.5) vs the unfused graph (GGML_CUDA_TOPK_MOE_384=0) and vs the CPU backend.
// The graph is test-backend-ops' test_topk_moe. Each CUDA mode runs in a child process (this binary with --child MODE FILE; the CUDA
// switches are read once per process) that dumps ids + weights per case; GGML_CUDA_TOPK_MOE_LOG=1 counts the fused launches: the fused
// child must log one per case, the unfused child none. Reports: selected-expert sets equal per row, max |weight delta| matched by expert
// id, and whether the outputs are byte-identical. No CUDA / ROCm device: SKIP, exit 0.

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

static int n_fused_logs = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "topk moe fused launch:") != nullptr) {
        n_fused_logs++;
        return;
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct tcase {
    int n_tokens;
    bool with_norm;
    float scale_w;
    std::string str() const {
        char b[128];
        snprintf(b, sizeof(b), "experts 384 used 6 tokens %3d norm %d scale %.1f", n_tokens, (int) with_norm, scale_w);
        return b;
    }
};

static std::vector<tcase> cases() {
    std::vector<tcase> v;
    for (int nt : {1, 2, 8, 22, 64}) {
        for (bool wn : {true, false}) {
            v.push_back({nt, wn, 1.5f});
        }
    }
    return v;
}

static const int N_EXP = 384, N_USED = 6;

// outputs: ids (I32, n_used x n_tokens) then weights (F32, n_used x n_tokens)
static std::vector<uint8_t> run_case(ggml_backend_t backend, const tcase & c, int seed) {
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EXP, c.n_tokens);
    ggml_tensor * bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_EXP);
    ggml_tensor * probs  = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    ggml_tensor * sel    = ggml_add(ctx, probs, bias);
    ggml_tensor * ids    = ggml_argsort_top_k(ctx, sel, N_USED);
    ggml_tensor * w      = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, N_EXP, c.n_tokens), ids);
    if (c.with_norm) {
        w = ggml_reshape_2d(ctx, w, N_USED, c.n_tokens);
        ggml_tensor * s = ggml_sum_rows(ctx, w);
        s = ggml_clamp(ctx, s, 6.103515625e-5, INFINITY);
        w = ggml_div(ctx, w, s);
        w = ggml_reshape_3d(ctx, w, 1, N_USED, c.n_tokens);
    }
    if (c.scale_w) {
        w = ggml_scale(ctx, w, c.scale_w);
    }
    // no ggml_set_output: an output flag on a node inside the pattern blocks the fusion (ggml_can_fuse); every tensor has its own
    // buffer (ggml_backend_alloc_ctx_tensors), so ids and weights are read after the compute as they are
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ids);
    ggml_build_forward_expand(gf, w);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> l((size_t) N_EXP * c.n_tokens), b(N_EXP);
    for (auto & x : l) x = 2.0f * nd(rng);
    for (auto & x : b) x = 0.3f * nd(rng);
    ggml_backend_tensor_set(logits, l.data(), 0, ggml_nbytes(logits));
    ggml_backend_tensor_set(bias, b.data(), 0, ggml_nbytes(bias));
    ggml_backend_graph_compute(backend, gf);

    std::vector<uint8_t> out(ggml_nbytes(ids) + ggml_nbytes(w));
    // ids is a view (argsort_top_k): read it as n_used rows of its stride
    std::vector<int32_t> idv((size_t) N_USED * c.n_tokens);
    for (int t = 0; t < c.n_tokens; ++t) {
        ggml_backend_tensor_get(ids, idv.data() + (size_t) t * N_USED, t * ids->nb[1], N_USED * sizeof(int32_t));
    }
    memcpy(out.data(), idv.data(), idv.size() * sizeof(int32_t));
    out.resize(idv.size() * sizeof(int32_t) + (size_t) N_USED * c.n_tokens * sizeof(float));
    ggml_backend_tensor_get(w, out.data() + idv.size() * sizeof(int32_t), 0, (size_t) N_USED * c.n_tokens * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static ggml_backend_dev_t gpu_dev() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        if (strcmp(reg, "CUDA") == 0 || strcmp(reg, "ROCm") == 0) {
            return d;
        }
    }
    return nullptr;
}

static int run_child(const char * path) {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t dev = gpu_dev();
    if (dev == nullptr) return 3;
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) return 3;
    FILE * f = fopen(path, "wb");
    if (f == nullptr) return 4;
    const auto cs = cases();
    for (size_t i = 0; i < cs.size(); ++i) {
        const std::vector<uint8_t> o = run_case(backend, cs[i], 1000 + (int) i);
        const uint64_t n = o.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(o.data(), 1, o.size(), f);
    }
    fwrite(&n_fused_logs, sizeof(n_fused_logs), 1, f);
    fclose(f);
    ggml_backend_free(backend);
    return 0;
}

static bool read_dump(const std::string & path, std::vector<std::vector<uint8_t>> & outs, int & logs, size_t n) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    for (size_t i = 0; i < n; ++i) {
        uint64_t m = 0;
        if (fread(&m, sizeof(m), 1, f) != 1) { fclose(f); return false; }
        outs.emplace_back(m);
        if (m > 0 && fread(outs.back().data(), 1, m, f) != m) { fclose(f); return false; }
    }
    const bool ok = fread(&logs, sizeof(int), 1, f) == 1;
    fclose(f);
    return ok;
}

// rows with equal expert sets, and the max |weight delta| matched by expert id over the rows that agree
static void cmp(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, int nt, int & rows_eq, double & maxd) {
    const int32_t * ia = (const int32_t *) a.data();
    const int32_t * ib = (const int32_t *) b.data();
    const float * wa = (const float *) (a.data() + (size_t) N_USED * nt * sizeof(int32_t));
    const float * wb = (const float *) (b.data() + (size_t) N_USED * nt * sizeof(int32_t));
    rows_eq = 0;
    maxd = 0.0;
    for (int t = 0; t < nt; ++t) {
        int eq = 0;
        double md = 0.0;
        for (int j = 0; j < N_USED; ++j) {
            for (int k = 0; k < N_USED; ++k) {
                if (ia[t * N_USED + j] == ib[t * N_USED + k]) {
                    eq++;
                    md = std::max(md, (double) std::fabs(wa[t * N_USED + j] - wb[t * N_USED + k]));
                }
            }
        }
        if (eq == N_USED) {
            rows_eq++;
            maxd = std::max(maxd, md);
        }
    }
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[3]);
    }
    ggml_log_set(log_cb, nullptr);
    if (gpu_dev() == nullptr) {
        printf("SKIP: no CUDA / ROCm device\n");
        return 0;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    const auto cs = cases();
    std::vector<std::vector<uint8_t>> outs[2];
    int logs[2] = { -1, -1 };
    for (int mode = 0; mode < 2; ++mode) {
        const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/topk384_exact_" + std::to_string(mode) + ".bin";
        set_env("GGML_CUDA_TOPK_MOE_384", mode ? "1" : "0");
        set_env("GGML_CUDA_TOPK_MOE_LOG", "1");
        std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\"";
        const int rc = std::system(cmd.c_str());
        if (rc != 0 || !read_dump(path, outs[mode], logs[mode], cs.size())) {
            printf("FAIL: child mode %d rc %d\n", mode, rc);
            return 1;
        }
        std::remove(path.c_str());
    }
    int n_bad = 0, n_bytes_eq = 0;
    double max_all = 0.0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const int nt = cs[i].n_tokens;
        const std::vector<uint8_t> ref = run_case(cpu, cs[i], 1000 + (int) i);
        int r_fu = 0, r_fc = 0, r_uc = 0;
        double d_fu = 0, d_fc = 0, d_uc = 0;
        cmp(outs[1][i], outs[0][i], nt, r_fu, d_fu);
        cmp(outs[1][i], ref, nt, r_fc, d_fc);
        cmp(outs[0][i], ref, nt, r_uc, d_uc);
        const bool bytes = outs[0][i] == outs[1][i];
        n_bytes_eq += bytes;
        max_all = std::max(max_all, d_fu);
        const bool ok = r_fu == nt && r_fc == nt;
        n_bad += !ok;
        printf("  %s: fused vs unfused rows %d/%d max|dw| %.3g%s; fused vs CPU %d/%d %.3g; unfused vs CPU %d/%d %.3g%s\n",
               cs[i].str().c_str(), r_fu, nt, d_fu, bytes ? " (byte-identical)" : "", r_fc, nt, d_fc, r_uc, nt, d_uc, ok ? "" : "  <- FAIL");
    }
    printf("fused launches logged: unfused child %d, fused child %d (expected 0 / %zu)\n", logs[0], logs[1], cs.size());
    ggml_backend_free(cpu);
    if (logs[0] != 0 || logs[1] != (int) cs.size()) {
        printf("FAIL: the fusion did not run once per case in the fused child (or ran in the unfused child)\n");
        return 1;
    }
    if (n_bad) {
        printf("FAIL: %d / %zu cases select different experts\n", n_bad, cs.size());
        return 1;
    }
    printf("OK: %zu / %zu cases select the same experts (fused / unfused / CPU), max |weight delta| fused vs unfused %.3g, %d / %zu byte-identical\n",
           cs.size(), cs.size(), max_all, n_bytes_eq, cs.size());
    return 0;
}
