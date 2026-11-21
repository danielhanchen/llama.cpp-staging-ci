// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// Bit identity of a flash attention node padded to P cells with its live length L (ggml_flash_attn_ext_set_kv_len) against
// the node of L cells, on the GPU backend, at the shape of Qwen3.8-Flash-Next's QSA attention (head size 256, 24 query heads over 2 KV
// heads, f16 K / V in the KV cache layout, 1..8 query rows), dense (n_kv_max 0) and sparse (n_kv_max = the QSA selection, 2051 visible
// cells per row), with and without the verify flag. One padded graph per (rows, P, mode) is evaluated at a sequence of lengths, so its
// CUDA graph is captured and then replayed with the length read on the device; every output is compared byte for byte with a fresh
// graph of L cells. GGML_CUDA_GRAPH_CAPTURE_LOG counts the captures of the padded graphs: with the device length they follow the
// launch signature (a few per graph), not the length. Negative control: the padded node without its length must differ somewhere
// (the dense stream-k split follows the cells). GGML_CUDA_FA_KV_LEN=0 (the kill switch) must pass too, with a capture per length.
// the same with a q8_0 K / V cache (the MMA kernel reads it in place, so its padded node takes the device length too).
// No GPU: SKIP, exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_captures = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "CUDA graph capture") != nullptr) {
        n_captures++;
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static constexpr int D    = 256;
static constexpr int H    = 24;
static constexpr int HKV  = 2;
static constexpr int NSEL = 2051;

struct kv_store {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         k   = nullptr; // [D*HKV, P_max] f16, the cache layout
    ggml_tensor *         v   = nullptr;
    ggml_tensor *         q   = nullptr; // [D, H, T_max] f32
};

// the visible cells of query row t at live length L: causal over the last T cells, plus (sparse) a fixed random selection
static void fill_mask(std::vector<ggml_fp16_t> & m, int64_t n_cols, int64_t L, int T, bool sparse, uint32_t seed) {
    m.assign((size_t) n_cols*T, ggml_fp32_to_fp16(-INFINITY));
    std::mt19937 rng(seed);
    for (int t = 0; t < T; ++t) {
        const int64_t n_vis = L - (T - 1 - t); // the row sees cells [0, n_vis)
        if (!sparse || n_vis <= NSEL) {
            for (int64_t c = 0; c < n_vis; ++c) {
                m[(size_t) t*n_cols + c] = ggml_fp32_to_fp16(0.0f);
            }
            continue;
        }
        // the tail cells and a random choice of the others, NSEL in all
        int64_t n_set = 0;
        for (int64_t c = n_vis - 3; c < n_vis; ++c, ++n_set) {
            m[(size_t) t*n_cols + c] = ggml_fp32_to_fp16(0.0f);
        }
        std::uniform_int_distribution<int64_t> pick(0, n_vis - 4);
        while (n_set < NSEL) {
            const int64_t c = pick(rng);
            if (ggml_fp16_to_fp32(m[(size_t) t*n_cols + c]) != 0.0f) {
                m[(size_t) t*n_cols + c] = ggml_fp32_to_fp16(0.0f);
                n_set++;
            }
        }
    }
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

// FA over n_kv cells of the store; with_len: + a KV length input
static fa_graph build_fa(ggml_backend_t be, const kv_store & st, int64_t n_kv, int T, bool sparse, bool verify, bool with_len) {
    fa_graph g;
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_tensor * q = ggml_view_3d(g.ctx, st.q, D, H, T, st.q->nb[1], st.q->nb[2], 0);
    q = ggml_permute(g.ctx, q, 0, 2, 1, 3); // [D, T, H]
    ggml_tensor * k = ggml_view_3d(g.ctx, st.k, D, n_kv, HKV, st.k->nb[1], ggml_row_size(st.k->type, D), 0);
    ggml_tensor * v = ggml_view_3d(g.ctx, st.v, D, n_kv, HKV, st.v->nb[1], ggml_row_size(st.v->type, D), 0);
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
    if (verify) {
        g.out->flags |= GGML_TENSOR_FLAG_VERIFY;
    }
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

static std::vector<uint8_t> run(ggml_backend_t be, fa_graph & g, const std::vector<ggml_fp16_t> & m, int32_t L) {
    ggml_backend_tensor_set(g.mask, m.data(), 0, m.size()*sizeof(ggml_fp16_t));
    if (g.kvl) {
        ggml_flash_attn_ext_kv_len_set_host(g.kvl, L);
        ggml_backend_tensor_set(g.kvl, &L, 0, sizeof(L));
    }
    if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(1);
    }
    std::vector<uint8_t> o(ggml_nbytes(g.out));
    ggml_backend_tensor_get(g.out, o.data(), 0, o.size());
    return o;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    setenv("GGML_CUDA_GRAPH_CAPTURE_LOG", "1", 0);
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (be == nullptr) {
        printf("test-fa-kvlen-exact: SKIP (no GPU)\n");
        return 0;
    }
    const bool kill = getenv("GGML_CUDA_FA_KV_LEN") && atoi(getenv("GGML_CUDA_FA_KV_LEN")) == 0;
    bool ok_all = true;
    for (const ggml_type kv_type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
    n_captures = 0;
    const int64_t P_max = 16384;
    const int     T_max = 8;
    kv_store st;
    {
        ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
        st.ctx = ggml_init(ip);
        st.k = ggml_new_tensor_2d(st.ctx, kv_type, D*HKV, P_max);
        st.v = ggml_new_tensor_2d(st.ctx, kv_type, D*HKV, P_max);
        st.q = ggml_new_tensor_3d(st.ctx, GGML_TYPE_F32, D, H, T_max);
        st.buf = ggml_backend_alloc_ctx_tensors(st.ctx, be);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> kvf((size_t) D*HKV*P_max);
        std::vector<uint8_t> kvb(ggml_nbytes(st.k));
        for (ggml_tensor * t : { st.k, st.v }) {
            for (auto & x : kvf) { x = u(rng); }
            ggml_quantize_chunk(kv_type, kvf.data(), kvb.data(), 0, P_max, D*HKV, nullptr);
            ggml_backend_tensor_set(t, kvb.data(), 0, kvb.size());
        }
        std::vector<float> qv((size_t) D*H*T_max);
        for (auto & x : qv) { x = 4.0f*u(rng); }
        ggml_backend_tensor_set(st.q, qv.data(), 0, qv.size()*sizeof(float));
    }

    int n_cases = 0, n_bad = 0, n_ctrl_diff = 0, n_ctrl = 0;
    const int64_t Ps[] = { 2048, 4096, 8192, 16384 };
    for (const bool sparse : { false, true }) {
        for (const bool verify : { false, true }) {
            for (int T = 1; T <= T_max; ++T) {
                for (const int64_t P : Ps) {
                    if (sparse && P < 8192) {
                        continue;
                    }
                    // lengths that grow through the bucket, with a shorter one after them (a new request in the same bucket)
                    std::vector<int64_t> Ls;
                    const int64_t lo = std::max<int64_t>(256, sparse ? P/2 : P/4);
                    for (int64_t L = lo; L <= P; L += std::max<int64_t>(256, (P - lo)/6/256*256)) {
                        Ls.push_back(L);
                        Ls.push_back(L); // twice: a replay at an unchanged length
                    }
                    Ls.push_back(P);
                    Ls.push_back(lo);
                    fa_graph gp = build_fa(be, st, P, T, sparse, verify, true);
                    fa_graph gc = build_fa(be, st, P, T, sparse, verify, false); // negative control: padded, no length
                    int pad_caps = 0;
                    int n_eval = 0;
                    for (const int64_t L : Ls) {
                        if (L < T) {
                            continue;
                        }
                        std::vector<ggml_fp16_t> m_pad, m_ref;
                        const uint32_t seed = (uint32_t) (L*31 + T);
                        fill_mask(m_pad, P, L, T, sparse, seed);
                        fill_mask(m_ref, L, L, T, sparse, seed);
                        const int cap_before = n_captures;
                        const std::vector<uint8_t> o_pad = run(be, gp, m_pad, (int32_t) L);
                        n_eval++;
                        pad_caps += n_captures - cap_before;
                        fa_graph gr = build_fa(be, st, L, T, sparse, verify, false);
                        const std::vector<uint8_t> o_ref = run(be, gr, m_ref, (int32_t) L);
                        gr.free_all();
                        n_cases++;
                        if (o_pad != o_ref) {
                            n_bad++;
                            if (n_bad <= 10) {
                                printf("FAIL sparse %d verify %d T %d P %" PRId64 " L %" PRId64 "\n", sparse, verify, T, P, L);
                            }
                        }
                        if (L < P && !sparse) {
                            const std::vector<uint8_t> o_ctl = run(be, gc, m_pad, (int32_t) L);
                            n_ctrl++;
                            n_ctrl_diff += o_ctl != o_ref;
                        }
                    }
                    printf("sparse %d verify %d T %d P %5" PRId64 ": %2d lengths, captures of the padded graph %d\n", sparse, verify, T, P,
                        n_eval, pad_caps);
                    gp.free_all();
                    gc.free_all();
                }
            }
        }
    }
    printf("test-fa-kvlen-exact: K / V %s: %d cases, %d differ; control (padded without the length) differs in %d of %d dense cases%s\n",
        ggml_type_name(kv_type), n_cases, n_bad, n_ctrl_diff, n_ctrl, kill ? " (GGML_CUDA_FA_KV_LEN=0)" : "");
    const bool ok = n_bad == 0 && n_ctrl_diff > 0;
    ok_all = ok_all && ok;
    ggml_backend_buffer_free(st.buf);
    ggml_free(st.ctx);
    }
    printf("test-fa-kvlen-exact: %s\n", ok_all ? "PASS" : "FAIL");
    ggml_backend_free(be);
    return ok_all ? 0 : 1;
}
