// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// The CPU one-token MoE up + gate + SwiGLU fusion (ggml_compute_forward_moe_glu1, GGML_CPU_MOE_GLU1) against the same nodes computed
// unfused, byte for byte: mul_mat_id(up, x, ids), mul_mat_id(gate, x, ids), swiglu_split(gate, up) in one graph (fused) vs the two
// products in one graph and the SwiGLU in a second one (the pattern is broken, so the stock nodes run). Weight types Q4_K / Q8_0 /
// Q5_0 / F16 and a mixed pair (up and gate of different vec_dot types), expert rows that are not a multiple of the 64-row blocks,
// a routed slot with id -1, 1 / 3 / 8 threads. The fused path must engage (its log line is counted): a silent fallback fails.
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fused_logs = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "up+gate+swiglu fused") != nullptr) {
        n_fused_logs++;
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static void fill_weights(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    for (auto & v : f) {
        v = nd(rng);
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(t->data, f.data(), n*sizeof(float));
        return;
    }
    const int64_t n_per_row = t->ne[0];
    ggml_quantize_chunk(t->type, f.data(), t->data, 0, n/n_per_row, n_per_row, nullptr);
}

static bool run_case(ggml_type tu, ggml_type tg, int64_t n_embd, int64_t n_ff, int n_expert, int n_used, int n_threads, bool with_neg,
        uint32_t seed) {
    ggml_init_params ip = { (size_t) 256*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(seed);

    ggml_tensor * wu = ggml_new_tensor_3d(ctx, tu, n_embd, n_ff, n_expert);
    ggml_tensor * wg = ggml_new_tensor_3d(ctx, tg, n_embd, n_ff, n_expert);
    fill_weights(wu, rng);
    fill_weights(wg, rng);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        float * xd = (float *) x->data;
        for (int64_t i = 0; i < n_embd; ++i) {
            xd[i] = nd(rng);
        }
    }
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    {
        std::vector<int> perm(n_expert);
        for (int i = 0; i < n_expert; ++i) {
            perm[i] = i;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        int32_t * idd = (int32_t *) ids->data;
        for (int i = 0; i < n_used; ++i) {
            idd[i] = perm[i];
        }
        if (with_neg) {
            idd[n_used/2] = -1;
        }
    }

    // fused: the three nodes in one graph
    ggml_tensor * up   = ggml_mul_mat_id(ctx, wu, x, ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, wg, x, ids);
    ggml_tensor * out  = ggml_swiglu_split(ctx, gate, up);
    ggml_cgraph * g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, out);

    // reference: the two products, then the SwiGLU in a second graph
    ggml_tensor * up_r   = ggml_mul_mat_id(ctx, wu, x, ids);
    ggml_tensor * gate_r = ggml_mul_mat_id(ctx, wg, x, ids);
    ggml_cgraph * g2a = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2a, up_r);
    ggml_build_forward_expand(g2a, gate_r);
    // the SwiGLU reads copies of the products: expanding it over up_r / gate_r would put the products back into its graph and
    // re-form the fused pattern
    ggml_tensor * up_c   = ggml_dup_tensor(ctx, up_r);
    ggml_tensor * gate_c = ggml_dup_tensor(ctx, gate_r);
    ggml_tensor * out_r = ggml_swiglu_split(ctx, gate_c, up_c);
    ggml_cgraph * g2b = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2b, out_r);

    const int logs0 = n_fused_logs;
    ggml_graph_compute_with_ctx(ctx, g1, n_threads);
    const bool engaged = n_fused_logs > logs0;
    const int logs1 = n_fused_logs;
    ggml_graph_compute_with_ctx(ctx, g2a, n_threads);
    memcpy(up_c->data,   up_r->data,   ggml_nbytes(up_r));
    memcpy(gate_c->data, gate_r->data, ggml_nbytes(gate_r));
    ggml_graph_compute_with_ctx(ctx, g2b, n_threads);
    const bool ref_fused = n_fused_logs > logs1;

    const bool same = memcmp(out->data, out_r->data, ggml_nbytes(out)) == 0;
    int64_t n_diff = 0;
    for (int64_t i = 0; i < ggml_nelements(out); ++i) {
        n_diff += memcmp((const float *) out->data + i, (const float *) out_r->data + i, sizeof(float)) != 0;
    }
    printf("  up %-5s gate %-5s n_embd %4lld n_ff %4lld experts %d/%d threads %d neg %d: %s (%lld differ)\n", ggml_type_name(tu),
           ggml_type_name(tg), (long long) n_embd, (long long) n_ff, n_used, n_expert, n_threads, with_neg ? 1 : 0,
           same ? "identical" : "DIFFERENT", (long long) n_diff);
    if (!engaged) {
        printf("  the fused path did not run\n");
    }
    if (ref_fused) {
        printf("  the reference ran the fused path\n");
    }
    ggml_free(ctx);
    return same && engaged && !ref_fused;
}

int main() {
#ifdef _WIN32
    _putenv_s("GGML_CPU_MOE_GLU1_LOG", "1");
#else
    setenv("GGML_CPU_MOE_GLU1_LOG", "1", 1);
#endif
    ggml_log_set(log_cb, nullptr);
    struct cfg { ggml_type tu, tg; int64_t n_embd, n_ff; };
    const cfg cfgs[] = {
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 512, 549 },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 512 },
        { GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, 256, 101 },
        { GGML_TYPE_F16,  GGML_TYPE_F16,  128, 200 },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, 512, 300 },
    };
    bool ok = true;
    uint32_t seed = 1;
    for (const cfg & c : cfgs) {
        for (int nt : { 1, 3, 8 }) {
            for (bool neg : { false, true }) {
                ok = run_case(c.tu, c.tg, c.n_embd, c.n_ff, 16, 6, nt, neg, seed++) && ok;
            }
        }
    }
    if (n_fused_logs == 0) {
        printf("the fused path never ran\n");
        ok = false;
    }
    printf("test-moe-glu1-exact: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
