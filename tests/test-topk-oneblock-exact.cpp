// the CUDA deterministic top-k (ggml_top_k) with the 8-columns-per-thread one-block radix select
// (GGML_CUDA_TOPK_ONEBLOCK_V2, default on) vs the previous one-block kernel (GGML_CUDA_TOPK_ONEBLOCK_V2=0, up to 65536 columns, then
// the multi-block path) vs the multi-block path only (GGML_CUDA_TOPK_ONEBLOCK=0), and vs a host reference of the documented order
// (keys above the k-th largest in column order, then the first kept ties in column order; -0 == +0). Each mode runs in a child
// process (the switches are read once per process) that dumps the ids and the median time of 30 graph runs per case. Rows of 1 / 2 /
// 40 (one T4 wave), 1500 .. 1048576 columns (DeepSeek-V4.1 decode rows: compressed rows at long context), k 512 / 2048, values from
// a normal draw, a coarse grid (many ties), a masked suffix (-inf, as the indexer's kq mask) and signed zeros. Gate: every mode
// byte-identical to the reference. No CUDA / ROCm device: SKIP, exit 0.

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
    int nrows;
    int ncols;
    int k;
    int dist; // 0 normal, 1 coarse grid (ties), 2 normal with a -inf masked suffix, 3 signed zeros + grid
    std::string str() const {
        char b[128];
        snprintf(b, sizeof(b), "rows %2d cols %7d k %4d dist %d", nrows, ncols, k, dist);
        return b;
    }
};

static std::vector<tcase> cases() {
    std::vector<tcase> v;
    for (int nc : { 4097, 15360, 30000, 60000, 65536, 65537, 122880, 262144, 524288, 1048576 }) {
        for (int d = 0; d < 4; ++d) {
            v.push_back({ 1, nc, 512, d });
        }
        v.push_back({ 1, nc, 2048, 0 });
        if (nc <= 131072) {
            v.push_back({ 2, nc, 512, 1 });
            v.push_back({ 40, nc, 512, 2 });
        }
    }
    return v;
}

static std::vector<float> make_data(const tcase & c, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) c.nrows * c.ncols);
    for (int r = 0; r < c.nrows; ++r) {
        const int n_vis = c.dist == 2 ? c.ncols - (int) (rng() % (c.ncols / 3 + 1)) : c.ncols;
        for (int i = 0; i < c.ncols; ++i) {
            float v = nd(rng);
            if (c.dist == 1 || c.dist == 3) {
                v = std::round(v * 4.0f) / 4.0f;
            }
            if (c.dist == 3 && v == 0.0f && (rng() & 1)) {
                v = -0.0f;
            }
            if (c.dist == 2 && i >= n_vis) {
                v = -INFINITY;
            }
            x[(size_t) r * c.ncols + i] = v;
        }
    }
    return x;
}

static uint32_t ordered(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    if (v == 0.0f) {
        bits = 0;
    }
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

static std::vector<int32_t> reference(const tcase & c, const std::vector<float> & x) {
    std::vector<int32_t> out;
    std::vector<uint32_t> keys(c.ncols);
    for (int r = 0; r < c.nrows; ++r) {
        for (int i = 0; i < c.ncols; ++i) {
            keys[i] = ordered(x[(size_t) r * c.ncols + i]);
        }
        std::vector<uint32_t> s = keys;
        std::nth_element(s.begin(), s.begin() + (c.k - 1), s.end(), std::greater<uint32_t>());
        const uint32_t thr = s[c.k - 1];
        int n_greater = 0;
        for (uint32_t kk : keys) {
            n_greater += kk > thr;
        }
        int n_eq = c.k - n_greater;
        for (int i = 0; i < c.ncols; ++i) {
            if (keys[i] > thr) {
                out.push_back(i);
            }
        }
        for (int i = 0; i < c.ncols && n_eq > 0; ++i) {
            if (keys[i] == thr) {
                out.push_back(i);
                n_eq--;
            }
        }
    }
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

// ids (k x nrows I32) and the median us of 30 timed graph runs
static std::vector<int32_t> run_case(ggml_backend_t backend, const tcase & c, const std::vector<float> & x, double & us) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ncols, c.nrows);
    ggml_tensor * ids = ggml_top_k(ctx, a, c.k);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ids);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(a, x.data(), 0, ggml_nbytes(a));
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
    std::vector<int32_t> out((size_t) c.k * c.nrows);
    for (int r = 0; r < c.nrows; ++r) {
        ggml_backend_tensor_get(ids, out.data() + (size_t) r * c.k, r * ids->nb[1], c.k * sizeof(int32_t));
    }
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
        const std::vector<int32_t> o = run_case(backend, cs[i], make_data(cs[i], 7 + (int) i), us);
        const uint64_t n = o.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(o.data(), sizeof(int32_t), o.size(), f);
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
        printf("SKIP: no CUDA / ROCm device\n");
        return 0;
    }
    const auto cs = cases();
    const char * names[3] = { "v2", "v1", "multi" };
    std::vector<std::vector<int32_t>> outs[3];
    std::vector<double> times[3];
    for (int mode = 0; mode < 3; ++mode) {
        const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/topk_oneblock_exact_" + std::to_string(mode) + ".bin";
        set_env("GGML_CUDA_TOPK_DETERMINISTIC", "1");
        set_env("GGML_CUDA_TOPK_ONEBLOCK_V2", mode == 0 ? "1" : "0");
        set_env("GGML_CUDA_TOPK_ONEBLOCK", mode == 2 ? "0" : "1");
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
            ok = ok && fread(outs[mode].back().data(), sizeof(int32_t), n, f) == n && fread(&us, sizeof(us), 1, f) == 1;
            if (!ok) {
                printf("FAIL: short dump from child %s\n", names[mode]);
                return 1;
            }
            times[mode].push_back(us);
        }
        fclose(f);
        std::remove(path.c_str());
    }
    int n_bad = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const std::vector<int32_t> ref = reference(cs[i], make_data(cs[i], 7 + (int) i));
        bool ok = true;
        std::string tag;
        for (int m = 0; m < 3; ++m) {
            const bool eq = outs[m][i] == ref;
            ok = ok && eq;
            tag += std::string(" ") + names[m] + (eq ? " =ref" : " DIFF");
        }
        n_bad += !ok;
        printf("  %s:%s | us v2 %8.1f v1 %8.1f multi %8.1f%s\n", cs[i].str().c_str(), tag.c_str(), times[0][i], times[1][i], times[2][i],
               ok ? "" : "  <- FAIL");
    }
    if (n_bad) {
        printf("FAIL: %d / %zu cases differ from the reference order\n", n_bad, cs.size());
        return 1;
    }
    printf("OK: %zu / %zu cases byte-identical to the reference in all three modes (v2 / v1 / multi-block)\n", cs.size(), cs.size());
    return 0;
}
