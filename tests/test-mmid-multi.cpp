// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// k17-ttft: MUL_MAT_ID on the CPU backend is bit-identical to one ggml_vec_dot per (weight row, routed row), for the weight types
// with multi-column kernels (ggml_cpu_vec_dot_multi: an expert routed to 2 or more rows decodes each weight block once for up to 8
// of them). Experts routed to 1, 2..8 and more than 8 rows are all covered; any byte difference fails.
// The upstream tiled kernels (ggml-cpu/tiled) take an expert routed to 8 or more rows of an IQ / K-quant type; they accumulate
// in another order, so they are not bitwise equal to vec_dot (upstream test-tiled-mulmat bounds them). This test turns them off
// (GGML_CPU_TILED_MM=0) unless GGML_CPU_TILED_MM is set; with GGML_CPU_TILED_MM=1 an expert of 8+ rows is checked against a relative
// bound instead (|got - ref| <= 1e-5 * max |ref| of the case), every other expert stays bitwise.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef _WIN32
// POSIX environment helper for the MSVC CRT (getenv sees _putenv_s)
static int setenv(const char * name, const char * value, int /*overwrite*/) { return _putenv_s(name, value); }
#endif

static bool g_tiled = false; // GGML_CPU_TILED_MM=1: experts of 8+ rows are checked against a bound

static bool run_case(ggml_type type, int64_t K, int64_t M, int n_expert, int n_used, int n_tokens, bool shared_src1, int n_threads, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    ggml_init_params ip = { /*.mem_size =*/ 64*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type, K, M, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, shared_src1 ? 1 : n_used, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);

    // weights: quantized from normal floats (valid blocks, finite scales)
    {
        std::vector<float> w((size_t) K*M*n_expert);
        for (float & v : w) {
            v = nd(rng);
        }
        ggml_quantize_init(type);
        const size_t row = ggml_row_size(type, K);
        ggml_quantize_chunk(type, w.data(), as->data, 0, M*n_expert, K, nullptr);
        (void) row;
    }
    for (int64_t i = 0; i < ggml_nelements(b); ++i) {
        ((float *) b->data)[i] = nd(rng)*(1.0f + 3.0f*(i % 97 == 0)); // a few outliers
    }
    // routing: distinct experts per token, skewed so some experts get 1 row and others more than 8
    for (int t = 0; t < n_tokens; ++t) {
        std::vector<int> pool(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            pool[e] = e;
        }
        for (int u = 0; u < n_used; ++u) {
            std::uniform_int_distribution<int> ud(u, n_expert - 1);
            int j = (rng() % 3 == 0) ? u : ud(rng);
            std::swap(pool[u], pool[j]);
            ((int32_t *) ids->data)[t*n_used + u] = pool[u];
        }
    }

    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // reference: the stock loop's arithmetic, one vec_dot per (row, routed row)
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(type);
    const ggml_type vdt = tr->vec_dot_type;
    const ggml_type_traits_cpu * trv = ggml_get_type_traits_cpu(vdt);
    std::vector<uint8_t> q(ggml_row_size(vdt, K));
    std::vector<int> cnt(n_expert, 0);
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            cnt[((int32_t *) ids->data)[t*n_used + u]]++;
        }
    }
    std::vector<float> refs((size_t) n_tokens*n_used*M);
    float ref_max = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const int e = ((int32_t *) ids->data)[t*n_used + u];
            const float * x = (const float *) ((const char *) b->data + (shared_src1 ? 0 : u)*b->nb[1] + t*b->nb[2]);
            trv->from_float(x, q.data(), K);
            for (int64_t r = 0; r < M; ++r) {
                float ref = 0.0f;
                tr->vec_dot((int) K, &ref, 0, (const char *) as->data + e*as->nb[2] + r*as->nb[1], 0, q.data(), 0, 1);
                refs[((size_t) t*n_used + u)*M + r] = ref;
                ref_max = std::max(ref_max, std::fabs(ref));
            }
        }
    }
    int64_t bad = 0, n = 0, n_tol = 0;
    float rel_max = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const int e = ((int32_t *) ids->data)[t*n_used + u];
            const bool tol = g_tiled && cnt[e] >= 8; // the tiled kernels' expert floor (ggml_tiled_min_batch)
            for (int64_t r = 0; r < M; ++r) {
                const float ref = refs[((size_t) t*n_used + u)*M + r];
                const float got = *(const float *) ((const char *) out->data + r*out->nb[0] + u*out->nb[1] + t*out->nb[2]);
                bool ok;
                if (tol) {
                    const float rel = std::fabs(got - ref) / std::max(ref_max, 1e-30f);
                    rel_max = std::max(rel_max, rel);
                    ok = rel <= 1e-5f;
                    n_tol++;
                } else {
                    ok = memcmp(&ref, &got, sizeof(float)) == 0;
                }
                if (!ok) {
                    if (bad < 5) {
                        fprintf(stderr, "  %s e %d t %d u %d r %" PRId64 ": ref %.9g got %.9g\n", ggml_type_name(type), e, t, u, r, ref, got);
                    }
                    bad++;
                }
                n++;
            }
        }
    }
    int rows_per_expert_max = 0;
    for (int c : cnt) {
        rows_per_expert_max = std::max(rows_per_expert_max, c);
    }
    printf("%-8s K %5" PRId64 " M %4" PRId64 " experts %d used %d tokens %3d%s threads %d: %" PRId64 " / %" PRId64 " bitwise (max rows per expert %d) %s",
           ggml_type_name(type), K, M, n_expert, n_used, n_tokens, shared_src1 ? " shared-src1" : "", n_threads, n - bad, n, rows_per_expert_max,
           bad == 0 ? "OK" : "FAIL");
    if (n_tol > 0) {
        printf(" (%" PRId64 " in tiled experts against the bound, max |got - ref| / max |ref| %.3g)", n_tol, rel_max);
    }
    printf("\n");
    ggml_free(ctx);
    return bad == 0;
}

int main() {
    // before ggml_cpu_init and the first graph: the tiled switch is read once per process
    const char * tiled_env = getenv("GGML_CPU_TILED_MM");
    if (tiled_env == NULL) {
        setenv("GGML_CPU_TILED_MM", "0", 1);
    } else {
        g_tiled = atoi(tiled_env) != 0;
    }
    printf("test-mmid-multi: upstream tiled kernels %s\n", g_tiled ? "on (experts of 8+ rows: relative bound 1e-5)" : "off (GGML_CPU_TILED_MM=0)");
    ggml_cpu_init();
    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_MXFP4,
                                GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_Q8_0 };
    bool ok = true;
    uint32_t seed = 1234;
    for (ggml_type t : types) {
        for (int64_t K : { 256, 2048, 2880 }) {
            if (K % ggml_blck_size(t) != 0) {
                continue;
            }
            ok = run_case(t, K, 40, 6, 4, 37, false, 4, seed++) && ok;   // up to ~25 rows per expert
            ok = run_case(t, K, 24, 64, 8, 3, false, 3, seed++) && ok;   // mostly 1 row per expert
            ok = run_case(t, K, 16, 4, 2, 9, true, 2, seed++) && ok;     // one src1 row per token (broadcast)
        }
    }
    printf("%s\n", ok ? "test-mmid-multi: OK" : "test-mmid-multi: FAIL");
    return ok ? 0 : 1;
}
