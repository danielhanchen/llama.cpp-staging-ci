// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
// k15-mmid: a MUL_MAT_ID graph whose expert tensor changes type between F16 and BF16 in place: the same tensor addresses, shapes and
// strides (both types are 2 bytes), so only the source type tells the two graphs apart. The CUDA backend captures the first one into a
// CUDA graph (an eager pass, a capture, a replay); the second must not replay that exec, whose kernels were chosen for the other type.
// This is what a later test-backend-ops graph whose tensors land on the addresses of an earlier one looks like to the backend.
// Each evaluation is checked against the CPU backend. Usage: test-graph-src-type; exit 1 on a mismatch, 0 (skipped) without a GPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct mmid_shape {
    int n_mats, n_used, m, n, k;
    bool skip;
};

struct mmid_graph {
    ggml_context *        ctx = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_tensor *         a = nullptr, * b = nullptr, * ids = nullptr, * out = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    void build(const mmid_shape & s, ggml_type type_a, ggml_backend_t be) {
        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        a   = ggml_new_tensor_3d(ctx, type_a, s.k, s.m, s.n_mats);
        b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.k, s.n_used, s.n);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.n_used, s.n);
        out = ggml_mul_mat_id(ctx, a, b, ids);
        gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    }
    void free_all() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

static void set_as(ggml_tensor * t, ggml_type type, const std::vector<float> & v) {
    std::vector<uint8_t> q(ggml_row_size(type, t->ne[0]) * (ggml_nelements(t) / t->ne[0]));
    ggml_quantize_chunk(type, v.data(), q.data(), 0, ggml_nelements(t) / t->ne[0], t->ne[0], nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static void set_inputs(mmid_graph & g, ggml_type type_a, const std::vector<float> & va, const std::vector<float> & vb,
                       const std::vector<int32_t> & vi) {
    set_as(g.a, type_a, va);
    ggml_backend_tensor_set(g.b, vb.data(), 0, vb.size()*sizeof(float));
    ggml_backend_tensor_set(g.ids, vi.data(), 0, vi.size()*sizeof(int32_t));
}

static std::vector<float> get_out(const mmid_graph & g) {
    std::vector<float> r(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, r.data(), 0, r.size()*sizeof(float));
    return r;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - b[i]) * (double) (a[i] - b[i]);
        den += b[i] * (double) b[i];
    }
    return den > 0.0 ? num / den : num;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_init(d, nullptr);
            break;
        }
    }
    if (gpu == nullptr) {
        printf("no GPU backend: skipped\n");
        return 0;
    }
    // through the registry, so the test links with GGML_BACKEND_DL (the release recipe) too
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        printf("no CPU backend: skipped\n");
        ggml_backend_free(gpu);
        return 0;
    }

    const ggml_type pairs[2][2] = { { GGML_TYPE_F16, GGML_TYPE_BF16 }, { GGML_TYPE_BF16, GGML_TYPE_F16 } };
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    int n_fail = 0, n_cases = 0;

    for (const auto & pr : pairs) {
        for (int n_used : { 1, 4 }) {
            for (bool skip : { false, true }) {
                for (int n : { 4, 8, 33, 129 }) {
                    const mmid_shape s = { 8, n_used, 512, n, 256, skip };
                    std::vector<float> va((size_t) s.k*s.m*s.n_mats), vb((size_t) s.k*s.n_used*s.n);
                    for (float & x : va) { x = u(rng); }
                    for (float & x : vb) { x = u(rng); }
                    std::vector<int32_t> vi((size_t) s.n_used*s.n);
                    for (int t = 0; t < s.n; ++t) {
                        for (int j = 0; j < s.n_used; ++j) {
                            vi[(size_t) t*s.n_used + j] = (t*3 + j*5 + 1) % s.n_mats; // distinct per token (5 and 8 coprime)
                        }
                        if (s.skip && t % 2 == 0) {
                            vi[(size_t) t*s.n_used + s.n_used - 1] = -1;
                        }
                    }

                    mmid_graph g;
                    g.build(s, pr[0], gpu);
                    for (int phase = 0; phase < 2; ++phase) {
                        const ggml_type type_a = pr[phase];
                        if (phase == 1) {
                            g.a->type = type_a; // same size per element, so the same strides and the same buffer
                        }
                        mmid_graph ref;
                        ref.build(s, type_a, cpu);
                        set_inputs(ref, type_a, va, vb, vi);
                        ggml_backend_graph_compute(cpu, ref.gf);
                        const std::vector<float> want = get_out(ref);
                        ref.free_all();

                        set_inputs(g, type_a, va, vb, vi);
                        // phase 0: an eager pass, the capture, a replay; phase 1: what the backend runs for the changed type
                        for (int r = 0; r < (phase == 0 ? 3 : 2); ++r) {
                            ggml_backend_graph_compute(gpu, g.gf);
                            const double e = nmse(get_out(g), want);
                            n_cases++;
                            if (!(e <= 5e-4)) {
                                n_fail++;
                                printf("FAIL %s -> %s n_used=%d skip=%d n=%d phase %d run %d: NMSE %.6f\n", ggml_type_name(pr[0]),
                                       ggml_type_name(pr[1]), s.n_used, (int) s.skip, s.n, phase, r, e);
                            }
                        }
                    }
                    g.free_all();
                }
            }
        }
    }
    printf("%d / %d evaluations match the CPU\n", n_cases - n_fail, n_cases);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return n_fail == 0 ? 0 : 1;
}
