// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// the tensor-core flash attention reading a q8_0 K / V cache in place (default: the raw rows pipelined where the f16 kernel
// pipelines, else synchronous loads; and GGML_CUDA_FATTN_Q8_MODE=1: synchronous loads everywhere) against the staged path that converts
// the whole K and V to f16 first (GGML_CUDA_FATTN_Q8_INPLACE=0), byte for byte on the GPU backend. Each mode runs in a child process (the
// switch is read once per process) that dumps every output, the median time of the timed graph runs per case and whether the in-place
// kernel logged. K / V live in the KV cache layout (a q8_0 [D*HKV, cells] tensor, viewed per head and permuted as llama-graph does),
// filled from normal draws with per cell scales from 1e-6 to 1e3 (subnormal and large f16 block scales), the mask causal over the last
// rows with -inf elsewhere, and a fraction of fully masked rows. Head sizes 128 / 256, GQA 1 / 2 / 4 / 6 / 8 (Qwen3.8-27B: 24 / 4,
// Qwen3.6-35B-A3B: 16 / 2), 2 .. 2048 query rows (verify windows and prompt ubatches), 256 .. 32768 cells, some not a multiple of 256.
// Gate: all three modes byte-identical in every case, the in-place kernel logged in modes 0 / 1 only. Timings: in place vs staged (the staged
// time includes the K / V conversion), printed per case. No CUDA device: SKIP, exit 0.

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
#include <random>
#include <string>
#include <vector>

static bool g_inplace_logged = false;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "q8_0-KV flash attention: tensor-core flash attention reads") != nullptr) {
        g_inplace_logged = true;
        fputs(text, stdout);
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct tcase {
    int  D;
    int  H;      // query heads
    int  HKV;    // K / V heads
    int  T;      // query rows
    int  n_kv;   // cells
    bool timed;  // a prompt-sized case: timed with more runs
    int  nsel;   // > 0: a sparse selection of nsel visible cells per row (ggml_flash_attn_ext_set_n_kv_max: the MMA gather)
    std::string str() const {
        char b[128];
        snprintf(b, sizeof(b), "D %3d H %2d / %d T %4d cells %5d%s", D, H, HKV, T, n_kv, nsel ? (" sel " + std::to_string(nsel)).c_str() : "");
        return b;
    }
};

static std::vector<tcase> cases() {
    std::vector<tcase> v;
    for (int D : { 128, 256 }) {
        for (int g : { 1, 2, 4, 6, 8 }) {
            const int HKV = g == 6 ? 4 : 2;
            for (int T : { 2, 3, 4, 8, 17, 64, 512 }) {
                for (int n_kv : { 256, 1000, 4096 }) {
                    v.push_back({ D, g*HKV, HKV, T, n_kv, false, 0 });
                }
            }
        }
    }
    // the priority models' prompt ubatches and verify windows (Qwen3.8-27B 24 / 4, Qwen3.6-35B-A3B 16 / 2, head size 256)
    for (int n_kv : { 2048, 8192, 32768 }) {
        v.push_back({ 256, 24, 4, 2048, n_kv, true, 0 });
        v.push_back({ 256, 16, 2, 2048, n_kv, true, 0 });
        v.push_back({ 256, 24, 4, 3, n_kv, true, 0 });
        v.push_back({ 256, 16, 2, 3, n_kv, true, 0 });
    }
    // the sparse gather (head size 256, GQA 8: the (1, 8) / (8, 8) instances)
    for (int T : { 2, 3, 4, 8, 64 }) {
        v.push_back({ 256, 16, 2, T, 8192, false, 1027 });
        v.push_back({ 256, 16, 2, T, 16384, false, 2051 });
    }
    return v;
}

struct case_data {
    std::vector<float>   q;    // [D, H, T]
    std::vector<uint8_t> k, v; // q8_0 [D*HKV, n_kv]
    std::vector<ggml_fp16_t> m; // [n_kv, T]
};

static case_data make_data(const tcase & c, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    case_data d;
    d.q.resize((size_t) c.D*c.H*c.T);
    for (auto & x : d.q) {
        x = nd(rng);
    }
    const int64_t row = (int64_t) c.D*c.HKV;
    std::vector<float> f((size_t) row*c.n_kv);
    for (int which = 0; which < 2; ++which) {
        for (int64_t cell = 0; cell < c.n_kv; ++cell) {
            // per (cell, head) scale: most O(1), some tiny (subnormal f16 scales), some large
            for (int h = 0; h < c.HKV; ++h) {
                const float r = u(rng);
                const float s = r < 0.05f ? 1e-6f : (r < 0.10f ? 1e2f : (r < 0.15f ? 3e-3f : 1.0f));
                for (int i = 0; i < c.D; ++i) {
                    f[(size_t) cell*row + (size_t) h*c.D + i] = s*nd(rng);
                }
            }
        }
        std::vector<uint8_t> & dst = which == 0 ? d.k : d.v;
        dst.resize(ggml_row_size(GGML_TYPE_Q8_0, row)*c.n_kv);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), dst.data(), 0, c.n_kv, row, nullptr);
    }
    // causal over the last T cells; ~1/16 of the rows of the larger cases see only a few cells
    d.m.assign((size_t) c.n_kv*c.T, ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < c.T; ++t) {
        int64_t n_vis = std::max<int64_t>(1, c.n_kv - (c.T - 1 - t));
        if (c.nsel > 0) {
            // the last 3 visible cells and a random choice of the others, nsel in all
            int n_set = 0;
            for (int64_t j = n_vis - 3; j < n_vis; ++j, ++n_set) {
                d.m[(size_t) t*c.n_kv + j] = ggml_fp32_to_fp16(0.0f);
            }
            while (n_set < c.nsel) {
                const int64_t j = rng() % (n_vis - 3);
                if (ggml_fp16_to_fp32(d.m[(size_t) t*c.n_kv + j]) != 0.0f) {
                    d.m[(size_t) t*c.n_kv + j] = ggml_fp32_to_fp16(0.0f);
                    n_set++;
                }
            }
            continue;
        }
        if (c.T >= 64 && (rng() % 16) == 0) {
            n_vis = 1 + rng() % 7;
        }
        for (int64_t j = 0; j < n_vis; ++j) {
            d.m[(size_t) t*c.n_kv + j] = ggml_fp32_to_fp16(0.0f);
        }
    }
    return d;
}

static ggml_backend_dev_t gpu_dev() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "CUDA") == 0) {
            return d;
        }
    }
    return nullptr;
}

static std::vector<uint8_t> run_case(ggml_backend_t backend, const tcase & c, const case_data & d, double & us) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t row = (int64_t) c.D*c.HKV;
    ggml_tensor * q_s  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.D, c.H, c.T);
    ggml_tensor * k_s  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, row, c.n_kv);
    ggml_tensor * v_s  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, row, c.n_kv);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv, c.T);
    // the KV cache views of llama-graph: [D, HKV, cells] then permuted to [D, cells, HKV]
    ggml_tensor * q = ggml_permute(ctx, q_s, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(ctx, ggml_view_3d(ctx, k_s, c.D, c.HKV, c.n_kv, ggml_row_size(GGML_TYPE_Q8_0, c.D), k_s->nb[1], 0), 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx, ggml_view_3d(ctx, v_s, c.D, c.HKV, c.n_kv, ggml_row_size(GGML_TYPE_Q8_0, c.D), v_s->nb[1], 0), 0, 2, 1, 3);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf((float) c.D), 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    if (c.nsel > 0) {
        ggml_flash_attn_ext_set_n_kv_max(out, c.nsel);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(q_s, d.q.data(), 0, ggml_nbytes(q_s));
    ggml_backend_tensor_set(k_s, d.k.data(), 0, ggml_nbytes(k_s));
    ggml_backend_tensor_set(v_s, d.v.data(), 0, ggml_nbytes(v_s));
    ggml_backend_tensor_set(mask, d.m.data(), 0, ggml_nbytes(mask));
    // a poisoned output first: every byte the run leaves must be written by it
    std::vector<uint8_t> poison(ggml_nbytes(out), 0x7f);
    ggml_backend_tensor_set(out, poison.data(), 0, poison.size());
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed: %s\n", c.str().c_str());
        exit(5);
    }
    std::vector<uint8_t> o(ggml_nbytes(out));
    ggml_backend_tensor_get(out, o.data(), 0, o.size());
    std::vector<double> t;
    const int n_runs = c.timed ? 10 : 3;
    for (int it = 0; it < n_runs; ++it) {
        ggml_backend_synchronize(backend);
        const auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        t.push_back(std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    us = t[t.size()/2];
    // the repeat runs must give the same bytes as the first
    std::vector<uint8_t> o2(o.size());
    ggml_backend_tensor_get(out, o2.data(), 0, o2.size());
    if (o2 != o) {
        fprintf(stderr, "repeat run differs: %s\n", c.str().c_str());
        exit(6);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return o;
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
        double us = 0.0;
        const std::vector<uint8_t> o = run_case(backend, cs[i], make_data(cs[i], 11 + (int) i), us);
        const uint64_t n = o.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(o.data(), 1, o.size(), f);
        fwrite(&us, sizeof(us), 1, f);
    }
    const int32_t eng = g_inplace_logged ? 1 : 0;
    fwrite(&eng, sizeof(eng), 1, f);
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

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[3]);
    }
    ggml_log_set(log_cb, nullptr);
    if (gpu_dev() == nullptr) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    const auto cs = cases();
    const char * names[3] = { "in-place", "in-place sync", "staged" };
    std::vector<std::vector<uint8_t>> outs[3];
    std::vector<double> times[3];
    int32_t engaged[3] = { 0, 0, 0 };
    for (int mode = 0; mode < 3; ++mode) {
        const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/fattn_q8_exact_" + std::to_string(mode) + ".bin";
        set_env("GGML_CUDA_FATTN_Q8_INPLACE", mode == 2 ? "0" : "1");
        set_env("GGML_CUDA_FATTN_Q8_INPLACE_MAX_ROWS", "1000000"); // every case in place (the default keeps prompt batches staged)
        set_env("GGML_CUDA_FATTN_Q8_MODE", mode == 1 ? "1" : "0");
        set_env("GGML_CUDA_FATTN_I8", "0"); // the staged reference is the f16 kernel (the INT8 Q*K^T route of q8_0 K prompts is not byte-equal)
        fflush(stdout);
        std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\"";
#if defined(_WIN32)
        cmd = "\"" + cmd + "\""; // cmd.exe /c strips the outer pair of quotes of a line with more than two
#endif
        const int rc = std::system(cmd.c_str());
        FILE * f = rc == 0 ? fopen(path.c_str(), "rb") : nullptr;
        if (f == nullptr) {
            printf("FAIL: child %s rc %d\n", names[mode], rc);
            return 1;
        }
        for (size_t i = 0; i < cs.size(); ++i) {
            uint64_t n = 0;
            double us = 0.0;
            bool ok = fread(&n, sizeof(n), 1, f) == 1;
            outs[mode].emplace_back(ok ? n : 0);
            ok = ok && fread(outs[mode].back().data(), 1, n, f) == n && fread(&us, sizeof(us), 1, f) == 1;
            if (!ok) {
                printf("FAIL: short dump from child %s\n", names[mode]);
                return 1;
            }
            times[mode].push_back(us);
        }
        if (fread(&engaged[mode], sizeof(int32_t), 1, f) != 1) {
            printf("FAIL: short dump from child %s\n", names[mode]);
            return 1;
        }
        fclose(f);
        std::remove(path.c_str());
    }
    int n_bad = 0, n_nonfinite = 0;
    double t_sum[3] = { 0.0, 0.0, 0.0 };
    for (size_t i = 0; i < cs.size(); ++i) {
        const bool eq = outs[0][i] == outs[2][i] && outs[1][i] == outs[2][i];
        n_bad += !eq;
        for (int m = 0; m < 3; ++m) {
            t_sum[m] += cs[i].timed ? times[m][i] : 0.0;
        }
        const float * o = (const float *) outs[0][i].data();
        int nf = 0;
        for (size_t j = 0; j < outs[0][i].size()/sizeof(float); ++j) {
            nf += !std::isfinite(o[j]);
        }
        n_nonfinite += nf > 0;
        printf("  %s: %s | us in-place %9.1f sync %9.1f staged %9.1f (%.3fx / %.3fx)%s%s\n", cs[i].str().c_str(),
            eq ? "equal" : (outs[0][i] == outs[2][i] ? "DIFF(sync)" : "DIFF"), times[0][i], times[1][i], times[2][i],
            times[0][i] / std::max(times[2][i], 1e-9), times[1][i] / std::max(times[2][i], 1e-9), nf ? " NONFINITE" : "", eq ? "" : "  <- FAIL");
    }
    printf("timed cases, sum of medians: in-place %.1f us, sync %.1f us, staged %.1f us\n", t_sum[0], t_sum[1], t_sum[2]);
    printf("in-place kernel logged: in-place %d, sync %d, staged %d\n", engaged[0], engaged[1], engaged[2]);
    if (n_bad || n_nonfinite || engaged[0] != 1 || engaged[1] != 1 || engaged[2] != 0) {
        printf("FAIL: %d / %zu cases differ, %d with non-finite output, engagement %d / %d / %d (want 1 / 1 / 0)\n", n_bad, cs.size(),
            n_nonfinite, engaged[0], engaged[1], engaged[2]);
        return 1;
    }
    printf("OK: %zu / %zu cases byte-identical in place (pipelined and synchronous) vs staged (q8_0 K / V)\n", cs.size(), cs.size());
    return 0;
}
