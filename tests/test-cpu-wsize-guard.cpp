// test-cpu-wsize-guard: every node of a CPU graph must stay inside the work buffer ggml_graph_plan() sized for it.
//
// The graph is computed twice: once with a work buffer of plan.work_size + 1 MiB (reference), once with a buffer that ends exactly
// at plan.work_size, followed by a PROT_NONE page. A node that writes or reads past the planned size faults in the second run
// (SIGSEGV, the test fails); otherwise both runs must give the same output bytes. Covers the ops this fork changed or added on
// the CPU: MUL_MAT_ID with -1 ids (dropped slots), the gated delta net (K > 1 snapshots, the replay window), batch-invariant
// nodes (one column at a time), multi-threaded GET_ROWS; and the mainline matmul / flash attention conversions next to them,
// at 1 / 3 / 8 / 16 threads and several token counts.
//
// Method: port of ik_llama.cpp work by moroznah (MIT): a one-node graph with a work buffer of exactly work_size followed by a
// PROT_NONE page, which found an under-sized fused up/gate conversion there.

#include "ggml.h"
#include "ggml-cpu.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

static std::mt19937 g_rng(1234);

static void fill_f32(ggml_tensor * t, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = d(g_rng);
}

static void fill_any(ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) { fill_f32(t); return; }
    std::vector<float> tmp(ggml_nelements(t));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto & x : tmp) x = d(g_rng);
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t *) t->data, ggml_nelements(t));
        return;
    }
    GGML_ASSERT(ggml_is_contiguous(t));
    const int64_t nrows = ggml_nelements(t) / t->ne[0];
    ggml_quantize_chunk(t->type, tmp.data(), t->data, 0, nrows, t->ne[0], nullptr);
}

struct guarded_buf {
    uint8_t * base = nullptr;
    size_t    len  = 0;
    uint8_t * data = nullptr; // ends at the guard page
    explicit guarded_buf(size_t size) {
        const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
        const size_t body = (size + pg - 1) / pg * pg + pg;
        len  = body + pg;
        base = (uint8_t *) mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        GGML_ASSERT(base != MAP_FAILED);
        GGML_ASSERT(mprotect(base + body, pg, PROT_NONE) == 0);
        // end exactly at the guard page; the start keeps 64-byte alignment (rounded down, at most 63 bytes more room)
        uintptr_t start = (uintptr_t) (base + body) - size;
        start &= ~(uintptr_t) 63;
        data = (uint8_t *) start;
    }
    ~guarded_buf() { munmap(base, len); }
};

typedef std::function<ggml_tensor * (ggml_context *)> build_fn;

static bool run_case(const std::string & name, const build_fn & build, int n_threads) {
    ggml_init_params ip = { /*.mem_size =*/ 256u*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    g_rng.seed(1234);
    ggml_tensor * out = build(ctx);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, out);

    ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    const size_t ws = plan.work_size;

    // reference: generous buffer
    std::vector<uint8_t> big(ws + (1u << 20));
    plan.work_data = big.data();
    plan.work_size = big.size();
    if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
        printf("FAIL %-48s nth=%2d: reference compute failed\n", name.c_str(), n_threads);
        ggml_free(ctx);
        return false;
    }
    std::vector<uint8_t> ref(ggml_nbytes(out));
    memcpy(ref.data(), out->data, ref.size());
    memset(out->data, 0, ggml_nbytes(out));

    // guarded: exactly the planned size, a PROT_NONE page after it (a fault here kills the test)
    // negative control: WSIZE_GUARD_SHRINK=N gives the guarded run N bytes less than planned (must fault where the node uses them)
    const char * shr = getenv("WSIZE_GUARD_SHRINK");
    const size_t shrink = shr ? (size_t) atoll(shr) : 0;
    const size_t gsize = ws > shrink ? ws - shrink : 0;
    guarded_buf gb(gsize);
    plan = ggml_graph_plan(gf, n_threads, nullptr);
    plan.work_data = ws > 0 ? gb.data : nullptr;
    plan.work_size = gsize;
    if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
        printf("FAIL %-48s nth=%2d: guarded compute failed\n", name.c_str(), n_threads);
        ggml_free(ctx);
        return false;
    }
    const bool same = memcmp(ref.data(), out->data, ref.size()) == 0;
    printf("%s %-48s nth=%2d work_size=%zu\n", same ? "OK  " : "FAIL", name.c_str(), n_threads, ws);
    ggml_free(ctx);
    return same;
}

int main(int argc, char ** argv) {
    const char * only = argc > 1 ? argv[1] : nullptr; // run only the cases whose name contains this string
    std::vector<std::pair<std::string, build_fn>> cases;

    for (ggml_type wt : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_F16 }) {
        for (int nt : { 1, 7, 33 }) {
            for (bool binv : { false, true }) {
                cases.push_back({ std::string("mul_mat ") + ggml_type_name(wt) + " nt=" + std::to_string(nt) + (binv ? " batch-inv" : ""),
                    [=](ggml_context * ctx) {
                        ggml_tensor * w = ggml_new_tensor_2d(ctx, wt, 256, 48);
                        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, nt);
                        fill_any(w); fill_f32(x);
                        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
                        if (binv) y->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                        return y;
                    } });
                // MUL_MAT_ID, 8 experts, top-3, every third slot dropped (-1)
                cases.push_back({ std::string("mul_mat_id ") + ggml_type_name(wt) + " nt=" + std::to_string(nt) + " ids-1" + (binv ? " batch-inv" : ""),
                    [=](ggml_context * ctx) {
                        const int n_exp = 8, n_used = 3;
                        ggml_tensor * w   = ggml_new_tensor_3d(ctx, wt, 256, 40, n_exp);
                        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, n_used, nt);
                        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, nt);
                        fill_any(w); fill_f32(x);
                        int32_t * id = (int32_t *) ids->data;
                        for (int t = 0; t < nt; ++t) {
                            for (int j = 0; j < n_used; ++j) {
                                id[t*n_used + j] = ((t + j) % 3 == 2) ? -1 : (t*5 + j*3) % n_exp;
                            }
                        }
                        ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
                        if (binv) y->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                        return y;
                    } });
            }
        }
    }

    // multi-threaded GET_ROWS of a quantized table (k19: rows split over the threads above a size threshold)
    for (int nr : { 3, 4096 }) {
        cases.push_back({ "get_rows q4_0 rows=" + std::to_string(nr), [=](ggml_context * ctx) {
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 512, 300);
            ggml_tensor * r = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nr);
            fill_any(w);
            for (int i = 0; i < nr; ++i) ((int32_t *) r->data)[i] = (i*37) % 300;
            return ggml_get_rows(ctx, w, r);
        } });
    }

    // gated delta net: K = 1 and K = 4 snapshots, 2 sequences
    for (int K : { 1, 4 }) {
        for (int T : { 1, 6 }) {
            cases.push_back({ "gated_delta_net K=" + std::to_string(K) + " T=" + std::to_string(T), [=](ggml_context * ctx) {
                const int S = 32, Hk = 2, Hv = 4, NS = 2;
                ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, NS);
                ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, NS);
                ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, T, NS);
                ggml_tensor * g  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, NS);
                ggml_tensor * b  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, NS);
                ggml_tensor * st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, NS);
                fill_f32(q, -0.2f, 0.2f); fill_f32(k, -0.2f, 0.2f); fill_f32(v); fill_f32(g, -0.5f, -0.01f); fill_f32(b, 0.0f, 1.0f);
                fill_f32(st, -0.1f, 0.1f);
                return ggml_gated_delta_net(ctx, q, k, v, g, b, st, K);
            } });
        }
    }
    // the replay window (k16-draft): stored window of 4 tokens, keep 2 / 3, halves 0 / 1
    cases.push_back({ "gated_delta_net_replay W=4 T=4", [=](ggml_context * ctx) {
        const int S = 32, Hk = 2, Hv = 4, NS = 2, T = 4, W = 4;
        ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, NS);
        ggml_tensor * k   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, NS);
        ggml_tensor * v   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, T, NS);
        ggml_tensor * g   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, NS);
        ggml_tensor * b   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, NS);
        ggml_tensor * st  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, NS);
        ggml_tensor * wk  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, 2*W, NS);
        ggml_tensor * wv  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, 2*W, NS);
        ggml_tensor * wgb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, Hv, 2*W, NS);
        ggml_tensor * wp  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, NS);
        fill_f32(q, -0.2f, 0.2f); fill_f32(k, -0.2f, 0.2f); fill_f32(v); fill_f32(g, -0.5f, -0.01f); fill_f32(b, 0.0f, 1.0f);
        fill_f32(st, -0.1f, 0.1f); fill_f32(wk, -0.2f, 0.2f); fill_f32(wv);
        float * pg = (float *) wgb->data;
        for (int64_t i = 0; i < ggml_nelements(wgb); i += 2) { pg[i] = -0.1f; pg[i + 1] = 0.5f; }
        int32_t * p = (int32_t *) wp->data;
        p[0] = 2; p[1] = 0; p[2] = 3; p[3] = 1;
        return ggml_gated_delta_net_replay(ctx, q, k, v, g, b, st, wk, wv, wgb, wp);
    } });

    // flash attention, F16 K / V, GQA 4, with mask
    for (int nq : { 1, 9 }) {
        cases.push_back({ "flash_attn_ext f16 nq=" + std::to_string(nq), [=](ggml_context * ctx) {
            const int D = 64, nkv = 256, Hq = 8, Hkv = 2;
            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, Hq, 1);
            ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, nkv, Hkv, 1);
            ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, nkv, Hkv, 1);
            ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, nq, 1, 1);
            fill_f32(q); fill_any(k); fill_any(v);
            memset(m->data, 0, ggml_nbytes(m));
            return ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 0.0f);
        } });
    }

    int n_fail = 0, n_ok = 0;
    for (int nth : { 1, 3, 8, 16 }) {
        for (auto & c : cases) {
            if (only && c.first.find(only) == std::string::npos) continue;
            if (run_case(c.first, c.second, nth)) ++n_ok; else ++n_fail;
        }
    }
    printf("%d OK, %d FAIL\n", n_ok, n_fail);
    return n_fail == 0 ? 0 : 1;
}
