// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// k19-fattnq4: two-source flash attention (ggml_flash_attn_ext_set_src2: a body K / V with its mask plus a ring K / V with its own mask)
// on the CUDA MMA kernel, byte for byte:
//   (a) the two-source node in place (body q4_0 / q4_0 or q8_0 / q8_0 read in place, ring q8_0 / q8_0 in place; default load mode and
//       GGML_CUDA_FATTN_Q8_MODE=1 synchronous loads) == the same node staged (GGML_CUDA_FATTN_Q4_INPLACE=0, GGML_CUDA_FATTN_Q8_INPLACE=0:
//       both sources converted to f16);
//   (b) the staged two-source node == one plain flash attention node over the concatenated cells (K / V dequantized on the host to f16,
//       body cells then ring cells, the two masks concatenated): the kernel loops over the tiles of both sources with one online
//       softmax, so it computes exactly what the one-source kernel computes over the concatenation.
// Each mode runs in a child process (the switches are read once per process). K / V in the KV cache layout ([D*HKV, cells], viewed per
// head and permuted), per cell scales from 1e-6 to 1e2; masks: the body sees a causal prefix minus the cells in the ring window, the
// ring sees its window slots (some empty, some masked), so the two are disjoint as k19-kvwin builds them. Head sizes 64 / 128 / 256 and
// the MLA 576 / 512 (V = view of K, staged), GQA 1 / 2 / 6 / 8, 1 .. 512 query rows, body 256 .. 8192 cells, ring 256 / 1792.
// Gate: every case equal in (a) and (b), the two-source kernel logged in every mode. No CUDA device: SKIP, exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-fa-src2.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static bool g_src2_logged = false;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "k19-fattnq4: two-source flash attention") != nullptr) {
        g_src2_logged = true;
        fputs(text, stdout);
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct tcase {
    ggml_type tb;   // body K / V type
    ggml_type tr;   // ring K / V type
    int D, DV, H, HKV, T, n1, n2;
    bool mla;       // V = view of K (D 576, DV 512)
    std::string str() const {
        char b[160];
        snprintf(b, sizeof(b), "body %s ring %s D %3d/%3d H %2d / %d T %4d body %5d ring %4d%s", ggml_type_name(tb), ggml_type_name(tr), D, DV,
            H, HKV, T, n1, n2, mla ? " mla" : "");
        return b;
    }
};

static std::vector<tcase> cases() {
    std::vector<tcase> v;
    const ggml_type Q4 = GGML_TYPE_Q4_0, Q8 = GGML_TYPE_Q8_0, F16 = GGML_TYPE_F16;
    for (const auto & tp : { std::make_pair(Q4, Q8), std::make_pair(Q8, Q8), std::make_pair(F16, F16), std::make_pair(F16, Q8) }) {
        for (int D : { 64, 128, 256 }) {
            for (int g : { 1, 2, 6, 8 }) {
                const int HKV = g == 6 ? 4 : 2;
                for (int T : { 1, 2, 3, 8, 16, 64 }) {
                    for (int n1 : { 256, 4096 }) {
                        v.push_back({ tp.first, tp.second, D, D, g*HKV, HKV, T, n1, n1 == 256 ? 256 : 1792, false });
                    }
                }
            }
        }
    }
    // the priority shapes (head size 256, Qwen3.8-27B 24 / 4, Qwen3.6-35B-A3B 16 / 2): verify windows and a prompt ubatch
    for (int n1 : { 8192, 32768 }) {
        for (int T : { 1, 3, 8, 512, 2048 }) {
            v.push_back({ Q4, Q8, 256, 256, 24, 4, T, n1, n1 == 32768 && T == 2048 ? 3328 : 1792, false });
            v.push_back({ Q4, Q8, 256, 256, 16, 2, T, n1, 1792, false });
        }
    }
    // MLA (K-only cache, V = view of K): staged
    for (int T : { 1, 3, 16 }) {
        v.push_back({ F16, F16, 576, 512, 16, 1, T, 1024, 256, true });
        v.push_back({ Q8, Q8, 576, 512, 16, 1, T, 1024, 256, true });
    }
    return v;
}

struct case_data {
    std::vector<float>   q;           // [D, H, T]
    std::vector<uint8_t> k1, v1;      // body [D*HKV, n1] (V: [DV*HKV, n1])
    std::vector<uint8_t> k2, v2;      // ring [D*HKV, n2]
    std::vector<ggml_fp16_t> m1, m2;  // [n1, T], [n2, T]
    std::vector<ggml_fp16_t> kc, vc;  // concatenated f16 [D*HKV, n1 + n2], [DV*HKV, n1 + n2]
    std::vector<ggml_fp16_t> mc;      // [n1 + n2, T]
};

static void fill(std::mt19937 & rng, std::vector<float> & f, int64_t row, int64_t n, int HKV, int D) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    f.resize((size_t) row*n);
    for (int64_t c = 0; c < n; ++c) {
        for (int h = 0; h < HKV; ++h) {
            const float r = u(rng);
            const float s = r < 0.05f ? 1e-6f : (r < 0.10f ? 1e2f : (r < 0.15f ? 3e-3f : 1.0f));
            for (int i = 0; i < D; ++i) {
                f[(size_t) c*row + (size_t) h*D + i] = s*nd(rng);
            }
        }
    }
}

// quantize f (rows of `row` values) to type t, and its f16 values as the GPU staging conversion produces them (dequantized, rounded once)
static void quant(ggml_type t, const std::vector<float> & f, int64_t row, int64_t n, std::vector<uint8_t> & q, std::vector<ggml_fp16_t> & h16,
        size_t h_off) {
    q.resize(ggml_row_size(t, row)*n);
    if (t == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) q.data(), row*n);
    } else {
        ggml_quantize_chunk(t, f.data(), q.data(), 0, n, row, nullptr);
    }
    std::vector<float> dq((size_t) row*n);
    ggml_get_type_traits(t)->to_float(q.data(), dq.data(), row*n);
    ggml_fp32_to_fp16_row(dq.data(), h16.data() + h_off, row*n);
}

static case_data make_data(const tcase & c, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    case_data d;
    d.q.resize((size_t) c.D*c.H*c.T);
    for (auto & x : d.q) {
        x = nd(rng);
    }
    const int64_t rowk = (int64_t) c.D*c.HKV, rowv = (int64_t) c.DV*c.HKV, nc = c.n1 + c.n2;
    d.kc.resize((size_t) rowk*nc);
    d.vc.resize((size_t) rowv*nc);
    std::vector<float> f;
    fill(rng, f, rowk, c.n1, c.HKV, c.D);
    quant(c.tb, f, rowk, c.n1, d.k1, d.kc, 0);
    fill(rng, f, rowk, c.n2, c.HKV, c.D);
    quant(c.tr, f, rowk, c.n2, d.k2, d.kc, (size_t) rowk*c.n1);
    if (!c.mla) {
        fill(rng, f, rowv, c.n1, c.HKV, c.DV);
        quant(c.tb, f, rowv, c.n1, d.v1, d.vc, 0);
        fill(rng, f, rowv, c.n2, c.HKV, c.DV);
        quant(c.tr, f, rowv, c.n2, d.v2, d.vc, (size_t) rowv*c.n1);
    }
    // positions: body cell j holds position j; the ring holds the last W positions of the body's visible range (slot s -> position
    // n1 - W + s, or rotated, see rot), W = min(n2 - 3, 1024) (the slots after W empty, -inf for everyone), plus every 97th slot masked (a dead slot)
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY), zero = ggml_fp32_to_fp16(0.0f);
    d.m1.assign((size_t) c.n1*c.T, ninf);
    d.m2.assign((size_t) c.n2*c.T, ninf);
    const int W = std::min(c.n2 - 3, 1024); // slots W .. n2 - 1 empty (-inf for every row), like a ring with spare capacity
    // half of the cases rotate the slots (slot s -> position n1 - W + (s + rot) % W): a ring that wrapped, so the visible slots of a
    // query tile are two runs with masked tiles between them (the ring-tile skip walks the runs)
    const int rot = (c.n1 + c.T) % 2 ? W/3 + 5 : 0;
    for (int t = 0; t < c.T; ++t) {
        const int p_max = c.n1 - 1 - (c.T - 1 - t); // causal: row t sees positions <= p_max
        for (int j = 0; j <= p_max && j < c.n1; ++j) {
            const bool in_ring = j > p_max - W && j >= c.n1 - W && (((j - (c.n1 - W) - rot) % W + W) % W) % 97 != 1;
            if (!in_ring) {
                d.m1[(size_t) t*c.n1 + j] = (rng() % 7 == 0) ? ggml_fp32_to_fp16(-0.5f) : zero; // some ALiBi-like finite offsets
            }
        }
        for (int s = 0; s < W; ++s) {
            const int pos = c.n1 - W + (s + rot) % W;
            if (pos >= 0 && pos <= p_max && pos > p_max - W && s % 97 != 1) {
                d.m2[(size_t) t*c.n2 + s] = zero;
            }
        }
    }
    d.mc.resize((size_t) nc*c.T);
    for (int t = 0; t < c.T; ++t) {
        std::copy(d.m1.begin() + (size_t) t*c.n1, d.m1.begin() + (size_t) (t + 1)*c.n1, d.mc.begin() + (size_t) t*nc);
        std::copy(d.m2.begin() + (size_t) t*c.n2, d.m2.begin() + (size_t) (t + 1)*c.n2, d.mc.begin() + (size_t) t*nc + c.n1);
    }
    return d;
}

static ggml_backend_dev_t gpu_dev() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "CUDA") == 0) {
            return d;
        }
    }
    return nullptr;
}

static uint64_t fnv(const std::vector<uint8_t> & o) {
    uint64_t h = 1469598103934665603ULL;
    for (const uint8_t b : o) {
        h = (h ^ b) * 1099511628211ULL;
    }
    return h;
}

// ref: one plain node over the concatenated f16 cells; else the two-source node
static std::vector<uint8_t> run_case(ggml_backend_t backend, const tcase & c, const case_data & d, bool ref, double * us = nullptr) {
    ggml_init_params ip = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t rowk = (int64_t) c.D*c.HKV, rowv = (int64_t) c.DV*c.HKV, nc = c.n1 + c.n2;
    ggml_tensor * q_s = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.D, c.H, c.T);
    ggml_tensor * q   = ggml_permute(ctx, q_s, 0, 2, 1, 3);
    const auto kview = [&](ggml_tensor * s, int Dh) {
        return ggml_permute(ctx, ggml_view_3d(ctx, s, Dh, c.HKV, s->ne[1], ggml_row_size(s->type, Dh), s->nb[1], 0), 0, 2, 1, 3);
    };
    // MLA: V = the first DV values of each K row (a view of K, as llama-graph builds it)
    const auto vview_mla = [&](ggml_tensor * s) {
        return ggml_permute(ctx, ggml_view_3d(ctx, s, c.DV, c.HKV, s->ne[1], ggml_row_size(s->type, c.D), s->nb[1], 0), 0, 2, 1, 3);
    };
    ggml_tensor * out = nullptr;
    std::vector<std::pair<ggml_tensor *, const void *>> uploads = { { q_s, d.q.data() } };
    if (ref) {
        ggml_tensor * k_s = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rowk, nc);
        ggml_tensor * v_s = c.mla ? nullptr : ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rowv, nc);
        ggml_tensor * m   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nc, c.T);
        ggml_tensor * k = kview(k_s, c.D);
        ggml_tensor * v = c.mla ? vview_mla(k_s) : kview(v_s, c.DV);
        out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) c.D), 0.0f, 0.0f);
        uploads.push_back({ k_s, d.kc.data() });
        if (v_s) uploads.push_back({ v_s, d.vc.data() });
        uploads.push_back({ m, d.mc.data() });
    } else {
        ggml_tensor * k1_s = ggml_new_tensor_2d(ctx, c.tb, rowk, c.n1);
        ggml_tensor * v1_s = c.mla ? nullptr : ggml_new_tensor_2d(ctx, c.tb, rowv, c.n1);
        ggml_tensor * k2_s = ggml_new_tensor_2d(ctx, c.tr, rowk, c.n2);
        ggml_tensor * v2_s = c.mla ? nullptr : ggml_new_tensor_2d(ctx, c.tr, rowv, c.n2);
        ggml_tensor * m1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n1, c.T);
        ggml_tensor * m2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n2, c.T);
        ggml_tensor * k1 = kview(k1_s, c.D);
        ggml_tensor * v1 = c.mla ? vview_mla(k1_s) : kview(v1_s, c.DV);
        ggml_tensor * k2 = kview(k2_s, c.D);
        ggml_tensor * v2 = c.mla ? vview_mla(k2_s) : kview(v2_s, c.DV);
        out = ggml_flash_attn_ext(ctx, q, k1, v1, m1, 1.0f/sqrtf((float) c.D), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_src2(out, k2, v2, m2);
        uploads.push_back({ k1_s, d.k1.data() });
        if (v1_s) uploads.push_back({ v1_s, d.v1.data() });
        uploads.push_back({ k2_s, d.k2.data() });
        if (v2_s) uploads.push_back({ v2_s, d.v2.data() });
        uploads.push_back({ m1, d.m1.data() });
        uploads.push_back({ m2, d.m2.data() });
    }
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto & u : uploads) {
        ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
    }
    std::vector<uint8_t> poison(ggml_nbytes(out), 0x7f);
    ggml_backend_tensor_set(out, poison.data(), 0, poison.size());
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed: %s\n", c.str().c_str());
        exit(5);
    }
    std::vector<uint8_t> o(ggml_nbytes(out));
    ggml_backend_tensor_get(out, o.data(), 0, o.size());
    if (us) {
        std::vector<double> t;
        for (int it = 0; it < 7; ++it) {
            ggml_backend_synchronize(backend);
            const auto t0 = std::chrono::high_resolution_clock::now();
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            t.push_back(std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        *us = t[t.size()/2];
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return o;
}

static bool supported(ggml_backend_dev_t dev, const tcase & c) {
    // ask the device (a node without data)
    ggml_init_params ip = { 32*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.T, c.H, 1);
    ggml_tensor * k1 = ggml_new_tensor_4d(ctx, c.tb, c.D, c.n1, c.HKV, 1);
    ggml_tensor * v1 = c.mla ? ggml_view_4d(ctx, k1, c.DV, c.n1, c.HKV, 1, k1->nb[1], k1->nb[2], k1->nb[3], 0) : ggml_new_tensor_4d(ctx, c.tb, c.DV, c.n1, c.HKV, 1);
    ggml_tensor * k2 = ggml_new_tensor_4d(ctx, c.tr, c.D, c.n2, c.HKV, 1);
    ggml_tensor * v2 = c.mla ? ggml_view_4d(ctx, k2, c.DV, c.n2, c.HKV, 1, k2->nb[1], k2->nb[2], k2->nb[3], 0) : ggml_new_tensor_4d(ctx, c.tr, c.DV, c.n2, c.HKV, 1);
    ggml_tensor * m1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n1, c.T);
    ggml_tensor * m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n2, c.T);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k1, v1, m1, 1.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_src2(out, k2, v2, m2);
    const bool ok = ggml_backend_dev_supports_op(dev, out);
    ggml_free(ctx);
    return ok;
}

static int run_child(const char * path) {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t dev = gpu_dev();
    if (dev == nullptr) return 3;
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) return 3;
    FILE * f = fopen(path, "wb");
    if (f == nullptr) return 4;
    const bool want_ref = getenv("FATTN_SRC2_REF") != nullptr;
    const auto cs = cases();
    int n_unsup = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const case_data d = make_data(cs[i], 23 + (int) i);
        uint64_t rec[3] = { 0, 0, 0 };
        double us = 0.0;
        if (supported(dev, cs[i])) {
            rec[0] = fnv(run_case(backend, cs[i], d, false, cs[i].n1 >= 8192 ? &us : nullptr));
            // (one row: the plain node may take the vector kernel, so (b) is checked from 2 rows)
            rec[1] = want_ref && cs[i].T >= 2 ? fnv(run_case(backend, cs[i], d, true)) : 0;
            rec[2] = 1;
        } else {
            n_unsup++;
        }
        fwrite(rec, sizeof(rec), 1, f);
        fwrite(&us, sizeof(us), 1, f);
    }
    const int32_t eng = g_src2_logged ? 1 : 0;
    fwrite(&eng, sizeof(eng), 1, f);
    fclose(f);
    ggml_backend_free(backend);
    if (n_unsup) {
        printf("child: %d cases unsupported by the device\n", n_unsup);
    }
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
    std::vector<std::array<uint64_t, 3>> outs[3];
    std::vector<double> times[3]; // median us of the two-source node (body >= 8192 cells only)
    int32_t engaged[3] = { 0, 0, 0 };
    for (int mode = 0; mode < 3; ++mode) {
        const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/fattn_src2_exact_" + std::to_string(mode) + ".bin";
        set_env("GGML_CUDA_FATTN_Q8_INPLACE", mode == 2 ? "0" : "1");
        set_env("GGML_CUDA_FATTN_Q4_INPLACE", mode == 2 ? "0" : "1");
        set_env("GGML_CUDA_FATTN_Q8_MODE", mode == 1 ? "1" : "0");
        // the reference is a plain f16 node: keep it on the f16 kernel (the INT8 Q*K^T route of f16 prompt nodes is not byte-equal)
        set_env("GGML_CUDA_FATTN_I8", "0");
        set_env("GGML_CUDA_FATTN_I8_F16", "0");
        // the ring-tile skip (fully masked ring tiles of a query tile) on at every row count in modes 0 / 2, off in mode 1: the three
        // must agree, and mode 2 with the plain node over the concatenation (which never skips)
        set_env("GGML_CUDA_FATTN_SRC2_SKIP", mode == 1 ? "0" : "1");
        set_env("GGML_CUDA_FATTN_SRC2_SKIP_MIN_ROWS", "1");
        if (mode == 2) {
            set_env("FATTN_SRC2_REF", "1");
        }
        fflush(stdout);
        const std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\"";
        const int rc = std::system(cmd.c_str());
        FILE * f = rc == 0 ? fopen(path.c_str(), "rb") : nullptr;
        if (f == nullptr) {
            printf("FAIL: child %s rc %d\n", names[mode], rc);
            return 1;
        }
        for (size_t i = 0; i < cs.size(); ++i) {
            std::array<uint64_t, 3> rec = {};
            double us = 0.0;
            if (fread(rec.data(), sizeof(uint64_t), 3, f) != 3 || fread(&us, sizeof(us), 1, f) != 1) {
                printf("FAIL: short dump from child %s\n", names[mode]);
                return 1;
            }
            outs[mode].push_back(rec);
            times[mode].push_back(us);
        }
        if (fread(&engaged[mode], sizeof(int32_t), 1, f) != 1) {
            printf("FAIL: short dump from child %s\n", names[mode]);
            return 1;
        }
        fclose(f);
        std::remove(path.c_str());
    }
    int n_bad = 0, n_run = 0, n_unsup = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        if (!outs[2][i][2]) {
            n_unsup++;
            printf("  %s: unsupported (not run)\n", cs[i].str().c_str());
            continue;
        }
        n_run++;
        const bool eq_a = outs[0][i][0] == outs[2][i][0] && outs[1][i][0] == outs[2][i][0];
        const bool eq_b = cs[i].T < 2 || outs[2][i][0] == outs[2][i][1];
        n_bad += !(eq_a && eq_b);
        char tb[160] = "";
        if (times[2][i] > 0.0) {
            snprintf(tb, sizeof(tb), " | us in place+skip %.1f sync noskip %.1f staged+skip %.1f (%.3fx)", times[0][i], times[1][i], times[2][i], times[0][i]/times[2][i]);
        }
        printf("  %s: in place == staged %s, staged == concatenated %s%s%s\n", cs[i].str().c_str(), eq_a ? "yes" : "NO",
            cs[i].T < 2 ? "(1 row: n/a)" : (eq_b ? "yes" : "NO"), tb,
            eq_a && eq_b ? "" : "  <- FAIL");
    }
    printf("two-source kernel logged: %d / %d / %d\n", engaged[0], engaged[1], engaged[2]);
    // the unsupported cases: only the MLA in-place-type pairs may be missing on a device without the MMA kernel
    if (n_bad || n_run == 0 || engaged[0] != 1 || engaged[1] != 1 || engaged[2] != 1) {
        printf("FAIL: %d / %d cases differ (%d unsupported), engagement %d / %d / %d\n", n_bad, n_run, n_unsup, engaged[0], engaged[1], engaged[2]);
        return 1;
    }
    printf("OK: %d / %d two-source cases byte-identical (in place == staged == one node over the concatenated cells), %d unsupported\n",
        n_run, n_run, n_unsup);
    return 0;
}
