// k19-dsv4long: the CUDA lightning indexer (ggml_lightning_indexer, DeepSeek-V4.1 index scores) with several key tiles per block
// (GGML_CUDA_LID_TILES: auto / 2 / 4) vs one tile per block (GGML_CUDA_LID_TILES=1, the previous kernel), byte for byte, plus the
// median time of 30 graph runs per case. Heads 32 / 64, 128 dims, 1 / 2 / 4 query rows, 1000 .. 122880 keys, F16 and Q8_0 keys, a
// -inf masked suffix. Each mode runs in a child process (the switch is read once per process). Gate: all modes byte-identical.
// No CUDA device: SKIP, exit 0.

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

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct tcase {
    int n_head;
    int n_rows;
    int n_kv;
    ggml_type tk;
    std::string str() const {
        char b[128];
        snprintf(b, sizeof(b), "heads %2d rows %d keys %6d %s", n_head, n_rows, n_kv, ggml_type_name(tk));
        return b;
    }
};

static std::vector<tcase> cases() {
    std::vector<tcase> v;
    for (int nh : { 32, 64 }) {
        for (int nr : { 1, 2, 4 }) {
            for (int nkv : { 1000, 20000, 65573, 122880 }) {
                for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
                    v.push_back({ nh, nr, nkv, t });
                }
            }
        }
    }
    return v;
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

static std::vector<float> run_case(ggml_backend_t backend, const tcase & c, int seed, double & us) {
    const int D = 128;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, c.n_head, c.n_rows, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, c.tk, D, 1, c.n_kv, 1);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.n_head, c.n_rows, 1, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_rows, 1, 1);
    ggml_tensor * s = ggml_lightning_indexer(ctx, q, k, w, m);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, s);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qv((size_t) D * c.n_head * c.n_rows), kf((size_t) D * c.n_kv), wv((size_t) c.n_head * c.n_rows);
    for (auto & x : qv) x = nd(rng);
    for (auto & x : kf) x = nd(rng);
    for (auto & x : wv) x = 0.1f * nd(rng);
    std::vector<uint8_t> kb(ggml_nbytes(k));
    if (c.tk == GGML_TYPE_F16) {
        for (size_t i = 0; i < kf.size(); ++i) {
            ((ggml_fp16_t *) kb.data())[i] = ggml_fp32_to_fp16(kf[i]);
        }
    } else {
        ggml_quantize_chunk(c.tk, kf.data(), kb.data(), 0, c.n_kv, D, nullptr);
    }
    std::vector<ggml_fp16_t> mv((size_t) c.n_kv * c.n_rows);
    for (int r = 0; r < c.n_rows; ++r) {
        const int n_vis = c.n_kv - (int) (rng() % (c.n_kv / 4 + 1));
        for (int i = 0; i < c.n_kv; ++i) {
            mv[(size_t) r * c.n_kv + i] = ggml_fp32_to_fp16(i < n_vis ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kb.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(m, mv.data(), 0, ggml_nbytes(m));
    ggml_backend_graph_compute(backend, gf);
    std::vector<double> t;
    for (int it = 0; it < 30; ++it) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        t.push_back(std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    us = t[t.size() / 2];
    std::vector<float> out((size_t) c.n_kv * c.n_rows);
    ggml_backend_tensor_get(s, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
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
        const std::vector<float> o = run_case(backend, cs[i], 11 + (int) i, us);
        const uint64_t n = o.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(o.data(), sizeof(float), o.size(), f);
        fwrite(&us, sizeof(us), 1, f);
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
    const char * modes[4] = { "1", "2", "4", "" }; // "" = auto
    std::vector<std::vector<float>> outs[4];
    std::vector<double> times[4];
    for (int mode = 0; mode < 4; ++mode) {
        const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/lid_tiles_exact_" + std::to_string(mode) + ".bin";
        if (modes[mode][0]) {
            set_env("GGML_CUDA_LID_TILES", modes[mode]);
        } else {
#ifdef _WIN32
            _putenv_s("GGML_CUDA_LID_TILES", "");
#else
            unsetenv("GGML_CUDA_LID_TILES");
#endif
        }
        std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\"";
        const int rc = std::system(cmd.c_str());
        FILE * f = rc == 0 ? fopen(path.c_str(), "rb") : nullptr;
        if (f == nullptr) {
            printf("FAIL: child mode %d rc %d\n", mode, rc);
            return 1;
        }
        for (size_t i = 0; i < cs.size(); ++i) {
            uint64_t n = 0;
            double us = 0.0;
            bool ok = fread(&n, sizeof(n), 1, f) == 1;
            outs[mode].emplace_back(ok ? n : 0);
            ok = ok && fread(outs[mode].back().data(), sizeof(float), n, f) == n && fread(&us, sizeof(us), 1, f) == 1;
            if (!ok) {
                printf("FAIL: short dump from child mode %d\n", mode);
                return 1;
            }
            times[mode].push_back(us);
        }
        fclose(f);
        std::remove(path.c_str());
    }
    int n_bad = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        bool ok = true;
        for (int m = 1; m < 4; ++m) {
            ok = ok && memcmp(outs[m][i].data(), outs[0][i].data(), outs[0][i].size() * sizeof(float)) == 0;
        }
        n_bad += !ok;
        printf("  %s: %s | us tiles 1 %8.1f  2 %8.1f  4 %8.1f  auto %8.1f%s\n", cs[i].str().c_str(), ok ? "byte-identical" : "DIFFER",
               times[0][i], times[1][i], times[2][i], times[3][i], ok ? "" : "  <- FAIL");
    }
    if (n_bad) {
        printf("FAIL: %d / %zu cases differ between tile counts\n", n_bad, cs.size());
        return 1;
    }
    printf("OK: %zu / %zu cases byte-identical (tiles 1 / 2 / 4 / auto)\n", cs.size(), cs.size());
    return 0;
}
