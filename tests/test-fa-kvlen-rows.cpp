// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// flash attention of a small batch of query rows (a verify window) at the shape of Qwen3.8-Flash-Next's QSA attention
// (head size 256, 24 query heads over 2 KV heads, q8_0 or f16 K / V in the KV cache layout), with the KV length read on the device
// (ggml_flash_attn_ext_set_kv_len, the n_kv bucket of LLAMA_KV_BUCKET_DEV) and without it, against a double precision reference of
// the visible cells. Every output must be finite and within the tolerance of the reference: row t of T sees the cells [0, L - T + 1 + t)
// of a padded node of P cells (the causal mask of a window that ends at cell L - 1), the cells past L hold zeros (a cleared cache), the
// device length is L rounded up to 256 cells. The 8 GB RTX PRO 6000 token-0 bug: T = 2, P = 2048, q8_0, L = 2 (the seq_rm
// probe) or 643 (a verify window after a 640 token prompt) take the vector kernel at the live length, which read row j of the mask at
// j*ne11 (the live cells) instead of its row stride (the padded cells): row 1 read row 0's masked tail, saw no cell and wrote NaN.
// usage: test-fa-kvlen-rows [-v]. No GPU: the CPU backend (checks the reference only).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int D    = 256;
static constexpr int H    = 24;
static constexpr int HKV  = 2;
static constexpr int NSEL = 2051;

static bool g_verbose = false;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

struct kv_store {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         k   = nullptr; // [D*HKV, P_max], the cache layout
    ggml_tensor *         v   = nullptr;
    ggml_tensor *         q   = nullptr; // [D, H, T_max] f32
    std::vector<float>    kf, vf, qf;    // the values the device holds (dequantized)
};

static void make_store(kv_store & st, ggml_backend_t be, ggml_type type, int64_t P, int64_t L, int T_max, uint32_t seed) {
    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    st.ctx = ggml_init(ip);
    st.k = ggml_new_tensor_2d(st.ctx, type, D*HKV, P);
    st.v = ggml_new_tensor_2d(st.ctx, type, D*HKV, P);
    st.q = ggml_new_tensor_3d(st.ctx, GGML_TYPE_F32, D, H, T_max);
    st.buf = ggml_backend_alloc_ctx_tensors(st.ctx, be);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int w = 0; w < 2; ++w) {
        std::vector<float> f((size_t) D*HKV*P, 0.0f);
        for (int64_t c = 0; c < L; ++c) {
            for (int64_t i = 0; i < D*HKV; ++i) {
                f[(size_t) c*D*HKV + i] = u(rng);
            }
        }
        ggml_tensor * t = w == 0 ? st.k : st.v;
        std::vector<uint8_t> raw(ggml_nbytes(t));
        if (type == GGML_TYPE_F16) {
            ggml_fp16_t * h = (ggml_fp16_t *) raw.data();
            for (size_t i = 0; i < f.size(); ++i) {
                h[i] = ggml_fp32_to_fp16(f[i]);
                f[i] = ggml_fp16_to_fp32(h[i]);
            }
        } else {
            ggml_quantize_chunk(type, f.data(), raw.data(), 0, P, D*HKV, nullptr);
            const auto * tr = ggml_get_type_traits(type);
            tr->to_float(raw.data(), f.data(), (int64_t) f.size());
        }
        ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
        (w == 0 ? st.kf : st.vf) = std::move(f);
    }
    st.qf.resize((size_t) D*H*T_max);
    for (auto & x : st.qf) {
        x = 4.0f*u(rng);
    }
    ggml_backend_tensor_set(st.q, st.qf.data(), 0, st.qf.size()*sizeof(float));
}

static void free_store(kv_store & st) {
    ggml_backend_buffer_free(st.buf);
    ggml_free(st.ctx);
    st = kv_store();
}

struct fa_graph {
    ggml_context *        ctx  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_cgraph *         gf   = nullptr;
    ggml_tensor *         mask = nullptr;
    ggml_tensor *         kvl  = nullptr;
    ggml_tensor *         out  = nullptr;

    void free_all() {
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
    }
};

// FA of T rows over the first n_kv cells of the store, as llm_graph_context::build_attn_mha builds it (permuted views)
static fa_graph build_fa(ggml_backend_t be, const kv_store & st, int64_t n_kv, int T, bool sparse, bool with_len) {
    fa_graph g;
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_tensor * q = ggml_view_3d(g.ctx, st.q, D, H, T, st.q->nb[1], st.q->nb[2], 0);
    q = ggml_permute(g.ctx, q, 0, 2, 1, 3); // [D, T, H]
    const size_t rs = ggml_row_size(st.k->type, D);
    ggml_tensor * k = ggml_view_3d(g.ctx, st.k, D, n_kv, HKV, st.k->nb[1], rs, 0);
    ggml_tensor * v = ggml_view_3d(g.ctx, st.v, D, n_kv, HKV, st.v->nb[1], rs, 0);
    g.mask = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, n_kv, T);
    ggml_set_input(g.mask);
    g.out = ggml_flash_attn_ext(g.ctx, q, k, v, g.mask, 1.0f/sqrtf((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_n_kv_max(g.out, sparse ? NSEL : 0);
    ggml_prec_set_acc(g.out, GGML_PREC_F32);
    if (with_len) {
        g.kvl = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 1);
        ggml_set_input(g.kvl);
        ggml_flash_attn_ext_set_kv_len(g.out, g.kvl);
    }
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

// row t sees the cells [0, L - T + 1 + t)
static std::vector<ggml_fp16_t> causal_mask(int64_t n_cols, int64_t L, int T) {
    std::vector<ggml_fp16_t> m((size_t) n_cols*T, ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < T; ++t) {
        for (int64_t c = 0; c < L - T + 1 + t; ++c) {
            m[(size_t) t*n_cols + c] = ggml_fp32_to_fp16(0.0f);
        }
    }
    return m;
}

static std::vector<float> run(ggml_backend_t be, fa_graph & g, const std::vector<ggml_fp16_t> & m, int32_t L) {
    ggml_backend_tensor_set(g.mask, m.data(), 0, m.size()*sizeof(ggml_fp16_t));
    if (g.kvl) {
        ggml_flash_attn_ext_kv_len_set_host(g.kvl, L);
        ggml_backend_tensor_set(g.kvl, &L, 0, sizeof(L));
    }
    if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(1);
    }
    std::vector<float> o(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, o.data(), 0, ggml_nbytes(g.out));
    return o;
}

// the reference [D, H, T] of the visible cells, double precision
static std::vector<float> reference(const kv_store & st, int64_t L, int T) {
    std::vector<float> o((size_t) D*H*T, 0.0f);
    const float scale = 1.0f/sqrtf((float) D);
    for (int t = 0; t < T; ++t) {
        const int64_t n_vis = L - T + 1 + t;
        for (int h = 0; h < H; ++h) {
            const int hk = h / (H/HKV);
            const float * q = st.qf.data() + (size_t) t*D*H + (size_t) h*D;
            std::vector<double> s(n_vis);
            double mx = -INFINITY;
            for (int64_t c = 0; c < n_vis; ++c) {
                const float * k = st.kf.data() + (size_t) c*D*HKV + (size_t) hk*D;
                double acc = 0.0;
                for (int i = 0; i < D; ++i) {
                    acc += (double) q[i]*k[i];
                }
                s[c] = acc*scale;
                mx = std::max(mx, s[c]);
            }
            double sum = 0.0;
            for (auto & x : s) {
                x = exp(x - mx);
                sum += x;
            }
            for (int i = 0; i < D; ++i) {
                double acc = 0.0;
                for (int64_t c = 0; c < n_vis; ++c) {
                    acc += s[c]*st.vf[(size_t) c*D*HKV + (size_t) hk*D + i];
                }
                o[(size_t) t*D*H + (size_t) h*D + i] = (float) (acc/sum);
            }
        }
    }
    return o;
}

// non-finite outputs, and the worst |got - ref| over max |ref| of a row
static std::string compare(const std::vector<float> & got, const std::vector<float> & ref, int T, int & n_nonfinite, double & worst) {
    n_nonfinite = 0;
    worst = 0.0;
    std::string rows;
    for (int t = 0; t < T; ++t) {
        int64_t nf = 0;
        double mx_ref = 1e-6, mx_err = 0.0;
        for (size_t i = (size_t) t*D*H; i < (size_t) (t + 1)*D*H; ++i) {
            if (!std::isfinite(got[i])) {
                nf++;
                continue;
            }
            mx_ref = std::max(mx_ref, (double) fabsf(ref[i]));
            mx_err = std::max(mx_err, (double) fabsf(got[i] - ref[i]));
        }
        n_nonfinite += (int) nf;
        worst = std::max(worst, mx_err/mx_ref);
        if (nf > 0) {
            rows += " row " + std::to_string(t) + ": " + std::to_string(nf) + " non-finite";
        }
    }
    return rows;
}

int main(int argc, char ** argv) {
    g_verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    const bool gpu = be != nullptr;
    if (!gpu) {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    printf("test-fa-kvlen-rows: backend %s\n", ggml_backend_name(be));

    const double tol = 2e-2; // f16 K / V products and P, q8_0 K / V dequantized the same way on both sides
    int n_cases = 0, n_bad = 0;
    const int64_t P = 2048;
    for (const ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_F16 }) {
        // lengths: the seq_rm probe (2 cells), the RTX PRO 6000 reproducer window (643), a pool boundary, a tile boundary, a full bucket
        for (const int64_t L : { (int64_t) 2, (int64_t) 3, (int64_t) 9, (int64_t) 256, (int64_t) 643, (int64_t) 700, (int64_t) 1024,
                                 (int64_t) 1500, (int64_t) 2048 }) {
            kv_store st;
            make_store(st, be, type, P, L, 8, (uint32_t) (L*7 + type));
            for (int T = 1; T <= 8; ++T) {
                if (T > L) {
                    continue;
                }
                const std::vector<float> ref = reference(st, L, T);
                // the node of the live cells (L rounded up to 256, the plain padding, which is also the device length llama sets:
                // llama_kv_cache::get_n_kv before the bucket), the padded node with the device length, the padded node without it
                const int64_t n_live = std::min<int64_t>(P, (L + 255)/256*256);
                for (int mode = 0; mode < 3; ++mode) {
                    for (const bool sparse : { false, true }) {
                        const int64_t n_kv = mode == 0 ? n_live : P;
                        fa_graph g = build_fa(be, st, n_kv, T, sparse, mode == 1);
                        const std::vector<float> got = run(be, g, causal_mask(n_kv, L, T), (int32_t) n_live);
                        g.free_all();
                        int nf = 0;
                        double worst = 0.0;
                        const std::string rows = compare(got, ref, T, nf, worst);
                        const bool bad = nf > 0 || !(worst <= tol);
                        n_cases++;
                        n_bad += bad;
                        // every case on a GPU (a crash names the case it was in: the line before it is the last one that ran)
                        if (bad || g_verbose || gpu) {
                            printf("%s %s L %4" PRId64 " T %d %s n_kv %4" PRId64 "%s: rel err %.3g%s\n", bad ? "FAIL" : "ok  ",
                                ggml_type_name(type), L, T, mode == 0 ? "live  " : mode == 1 ? "devlen" : "padded", n_kv,
                                sparse ? " n_kv_max 2051" : "", worst, rows.c_str());
                            fflush(stdout);
                        }
                    }
                }
            }
            free_store(st);
        }
    }
    printf("test-fa-kvlen-rows: %d cases, %d bad\n", n_cases, n_bad);
    printf("test-fa-kvlen-rows: %s\n", n_bad == 0 ? "PASS" : "FAIL");
    ggml_backend_free(be);
    return n_bad == 0 ? 0 : 1;
}
