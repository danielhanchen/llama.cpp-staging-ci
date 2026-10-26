// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// test-gdn-replay: ggml_gated_delta_net_replay (the replay commit folded into the next verify window) against the unfused replay path
// it replaces, bitwise, on every backend that implements it: the commit (ggml_gated_delta_net over the kept tokens of the stored window,
// q = k) then the window from the committed state. Checks the attention scores, the output state (= the committed state) and the
// window's inputs written into the other half of the stores. Exit 0: all equal.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct cfg {
    int S, H_k, H_v, n_seqs, n_tok, n_win;
    std::vector<int> keep, par;
};

static void fill(std::vector<float> & x, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto & v : x) {
        v = d(rng);
    }
}

// one graph on backend be: build(ctx) returns the tensors to read back after compute; inputs are set by set(t) after allocation
template <typename B, typename S, typename R>
static void run(ggml_backend_t be, B build, S set, R read) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> outs = build(ctx, gf);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    set();
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(2);
    }
    read(outs);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static int test_one(ggml_backend_t be, const cfg & c, uint32_t seed) {
    std::mt19937 rng(seed);
    const int S = c.S, Hk = c.H_k, Hv = c.H_v, NS = c.n_seqs, T = c.n_tok, W = c.n_win;
    std::vector<float> q(S*Hk*T*NS), k(S*Hk*T*NS), v(S*Hv*T*NS), g(Hv*T*NS), b(Hv*T*NS), s0(S*S*Hv*NS);
    std::vector<float> wk(S*Hk*2*W*NS), wv(S*Hv*2*W*NS), wgb(2*Hv*2*W*NS);
    fill(q, rng, -0.1f, 0.1f); fill(k, rng, -0.1f, 0.1f); fill(v, rng, -1.f, 1.f);
    fill(g, rng, -0.5f, -0.01f); fill(b, rng, 0.05f, 0.95f); fill(s0, rng, -0.3f, 0.3f);
    fill(wk, rng, -0.1f, 0.1f); fill(wv, rng, -1.f, 1.f);
    for (int i = 0; i < (int) wgb.size(); i += 2) {
        wgb[i]     = std::uniform_real_distribution<float>(-0.5f, -0.01f)(rng);
        wgb[i + 1] = std::uniform_real_distribution<float>(0.05f, 0.95f)(rng);
    }
    std::vector<int32_t> wp(2*NS);
    for (int s = 0; s < NS; ++s) {
        wp[2*s] = c.keep[s];
        wp[2*s + 1] = c.par[s];
    }

    // A: the unfused replay path, per sequence (keep differs): commit graph, then the window
    std::vector<float> s1(S*S*Hv*NS), attnA(S*Hv*T*NS);
    for (int s = 0; s < NS; ++s) {
        std::vector<float> st(s0.begin() + (size_t) s*S*S*Hv, s0.begin() + (size_t) (s + 1)*S*S*Hv);
        const int kp = c.keep[s], half = c.par[s];
        if (kp > 0) {
            std::vector<float> ck(S*Hk*kp), cv(S*Hv*kp), cg(Hv*kp), cb(Hv*kp);
            for (int t = 0; t < kp; ++t) {
                const size_t ts = (size_t) half*W + t;
                memcpy(&ck[(size_t) t*S*Hk], &wk[((size_t) s*2*W + ts)*S*Hk], S*Hk*sizeof(float));
                memcpy(&cv[(size_t) t*S*Hv], &wv[((size_t) s*2*W + ts)*S*Hv], S*Hv*sizeof(float));
                for (int h = 0; h < Hv; ++h) {
                    cg[(size_t) t*Hv + h] = wgb[(((size_t) s*2*W + ts)*Hv + h)*2 + 0];
                    cb[(size_t) t*Hv + h] = wgb[(((size_t) s*2*W + ts)*Hv + h)*2 + 1];
                }
            }
            ggml_tensor *tk, *tv, *tg, *tb, *ts0;
            run(be, [&](ggml_context * ctx, ggml_cgraph * gf) {
                tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, kp, 1);
                tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, kp, 1);
                tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, kp, 1);
                tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, kp, 1);
                ts0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, 1);
                ggml_tensor * o = ggml_gated_delta_net(ctx, tk, tk, tv, tg, tb, ts0, 1);
                ggml_tensor * ns = ggml_view_1d(ctx, o, (int64_t) S*S*Hv, ggml_row_size(o->type, (int64_t) S*Hv*kp));
                ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) S*S*Hv);
                ggml_tensor * cp = ggml_cpy(ctx, ns, dst);
                ggml_build_forward_expand(gf, cp);
                return std::vector<ggml_tensor *>{ dst };
            }, [&]() {
                ggml_backend_tensor_set(tk, ck.data(), 0, ggml_nbytes(tk));
                ggml_backend_tensor_set(tv, cv.data(), 0, ggml_nbytes(tv));
                ggml_backend_tensor_set(tg, cg.data(), 0, ggml_nbytes(tg));
                ggml_backend_tensor_set(tb, cb.data(), 0, ggml_nbytes(tb));
                ggml_backend_tensor_set(ts0, st.data(), 0, ggml_nbytes(ts0));
            }, [&](std::vector<ggml_tensor *> & o) {
                ggml_backend_tensor_get(o[0], st.data(), 0, ggml_nbytes(o[0]));
            });
        }
        memcpy(&s1[(size_t) s*S*S*Hv], st.data(), st.size()*sizeof(float));
        // the window of this sequence from the committed state
        ggml_tensor *tq, *tk, *tv, *tg, *tb, *ts;
        std::vector<float> a(S*Hv*T);
        run(be, [&](ggml_context * ctx, ggml_cgraph * gf) {
            tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, 1);
            tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, 1);
            tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, T, 1);
            tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, 1);
            tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, 1);
            ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, 1);
            ggml_tensor * o = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tb, ts, 1);
            ggml_tensor * av = ggml_view_1d(ctx, o, (int64_t) S*Hv*T, 0);
            ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) S*Hv*T);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, av, dst));
            return std::vector<ggml_tensor *>{ dst };
        }, [&]() {
            ggml_backend_tensor_set(tq, &q[(size_t) s*S*Hk*T], 0, ggml_nbytes(tq));
            ggml_backend_tensor_set(tk, &k[(size_t) s*S*Hk*T], 0, ggml_nbytes(tk));
            ggml_backend_tensor_set(tv, &v[(size_t) s*S*Hv*T], 0, ggml_nbytes(tv));
            ggml_backend_tensor_set(tg, &g[(size_t) s*Hv*T], 0, ggml_nbytes(tg));
            ggml_backend_tensor_set(tb, &b[(size_t) s*Hv*T], 0, ggml_nbytes(tb));
            ggml_backend_tensor_set(ts, &s1[(size_t) s*S*S*Hv], 0, ggml_nbytes(ts));
        }, [&](std::vector<ggml_tensor *> & o) {
            ggml_backend_tensor_get(o[0], a.data(), 0, ggml_nbytes(o[0]));
        });
        memcpy(&attnA[(size_t) s*S*Hv*T], a.data(), a.size()*sizeof(float));
    }

    // B: the fused op over all sequences; the output state goes through a cpy (the cache-write fusion on CUDA)
    std::vector<float> attnB(attnA.size()), sB(s1.size()), wkB(wk.size()), wvB(wv.size()), wgbB(wgb.size());
    {
        ggml_tensor *tq, *tk, *tv, *tg, *tb, *ts, *twk, *twv, *twg, *twp;
        run(be, [&](ggml_context * ctx, ggml_cgraph * gf) {
            tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, NS);
            tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, NS);
            tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, T, NS);
            tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, NS);
            tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, NS);
            ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, NS);
            twk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, 2*W, NS);
            twv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, 2*W, NS);
            twg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, Hv, 2*W, NS);
            twp = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, NS);
            ggml_tensor * o = ggml_gated_delta_net_replay(ctx, tq, tk, tv, tg, tb, ts, twk, twv, twg, twp);
            const int64_t D = (int64_t) S*S*Hv;
            ggml_tensor * av = ggml_view_1d(ctx, o, (int64_t) S*Hv*T*NS, 0);
            ggml_tensor * da = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) S*Hv*T*NS);
            ggml_tensor * sv = ggml_view_2d(ctx, o, D, NS, ggml_row_size(o->type, D), ggml_row_size(o->type, (int64_t) S*Hv*T*NS));
            ggml_tensor * dsall = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NS);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, sv, ggml_view_2d(ctx, dsall, D, NS, dsall->nb[1], 0)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, av, da));
            return std::vector<ggml_tensor *>{ da, dsall, twk, twv, twg };
        }, [&]() {
            ggml_backend_tensor_set(tq, q.data(), 0, ggml_nbytes(tq));
            ggml_backend_tensor_set(tk, k.data(), 0, ggml_nbytes(tk));
            ggml_backend_tensor_set(tv, v.data(), 0, ggml_nbytes(tv));
            ggml_backend_tensor_set(tg, g.data(), 0, ggml_nbytes(tg));
            ggml_backend_tensor_set(tb, b.data(), 0, ggml_nbytes(tb));
            ggml_backend_tensor_set(ts, s0.data(), 0, ggml_nbytes(ts));
            ggml_backend_tensor_set(twk, wk.data(), 0, ggml_nbytes(twk));
            ggml_backend_tensor_set(twv, wv.data(), 0, ggml_nbytes(twv));
            ggml_backend_tensor_set(twg, wgb.data(), 0, ggml_nbytes(twg));
            ggml_backend_tensor_set(twp, wp.data(), 0, ggml_nbytes(twp));
        }, [&](std::vector<ggml_tensor *> & o) {
            ggml_backend_tensor_get(o[0], attnB.data(), 0, ggml_nbytes(o[0]));
            ggml_backend_tensor_get(o[1], sB.data(), 0, ggml_nbytes(o[1]));
            ggml_backend_tensor_get(o[2], wkB.data(), 0, ggml_nbytes(o[2]));
            ggml_backend_tensor_get(o[3], wvB.data(), 0, ggml_nbytes(o[3]));
            ggml_backend_tensor_get(o[4], wgbB.data(), 0, ggml_nbytes(o[4]));
        });
    }

    // expected stores: the read half unchanged, the other half = this window's inputs
    std::vector<float> ewk = wk, ewv = wv, ewg = wgb;
    for (int s = 0; s < NS; ++s) {
        for (int t = 0; t < T; ++t) {
            const size_t ts = (size_t) (1 - c.par[s])*W + t;
            memcpy(&ewk[((size_t) s*2*W + ts)*S*Hk], &k[((size_t) s*T + t)*S*Hk], S*Hk*sizeof(float));
            memcpy(&ewv[((size_t) s*2*W + ts)*S*Hv], &v[((size_t) s*T + t)*S*Hv], S*Hv*sizeof(float));
            for (int h = 0; h < Hv; ++h) {
                ewg[(((size_t) s*2*W + ts)*Hv + h)*2 + 0] = g[((size_t) s*T + t)*Hv + h];
                ewg[(((size_t) s*2*W + ts)*Hv + h)*2 + 1] = b[((size_t) s*T + t)*Hv + h];
            }
        }
    }
    auto same = [](const std::vector<float> & x, const std::vector<float> & y) { return memcmp(x.data(), y.data(), x.size()*sizeof(float)) == 0; };
    const bool ok_a = same(attnA, attnB), ok_s = same(s1, sB), ok_w = same(ewk, wkB) && same(ewv, wvB) && same(ewg, wgbB);
    printf("  %-6s S %d H_k %d H_v %d seqs %d tokens %d keep/half", ggml_backend_name(be), S, Hk, Hv, NS, T);
    for (int s = 0; s < NS; ++s) {
        printf(" %d/%d", c.keep[s], c.par[s]);
    }
    printf(": attn %s, state %s, stores %s\n", ok_a ? "bitwise" : "DIFF", ok_s ? "bitwise" : "DIFF", ok_w ? "ok" : "DIFF");
    return ok_a && ok_s && ok_w ? 0 : 1;
}

int main() {
    ggml_backend_load_all();
    std::vector<ggml_backend_t> bes;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        const char * name = ggml_backend_dev_name(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU || strncmp(name, "CUDA", 4) == 0 || strncmp(name, "ROCm", 4) == 0) {
            bes.push_back(ggml_backend_dev_init(dev, nullptr));
            if (type != GGML_BACKEND_DEVICE_TYPE_CPU && bes.size() > 2) {
                ggml_backend_free(bes.back());
                bes.pop_back();
            }
        }
    }
    const std::vector<cfg> cfgs = {
        { 128, 16, 32, 1, 3, 3, { 2 }, { 0 } },
        { 128, 16, 32, 1, 1, 3, { 3 }, { 1 } },
        { 128, 16, 32, 4, 3, 3, { 0, 1, 2, 3 }, { 1, 0, 1, 0 } },
        { 128, 16, 64, 2, 2, 3, { 3, 1 }, { 0, 1 } },
        { 128, 32, 32, 2, 3, 3, { 2, 3 }, { 1, 1 } },
        {  64,  8, 16, 3, 3, 3, { 1, 3, 0 }, { 0, 0, 1 } },
    };
    int fails = 0;
    for (ggml_backend_t be : bes) {
        if (ggml_backend_is_cpu(be)) {
            ggml_backend_cpu_set_n_threads(be, 8);
        }
        for (size_t i = 0; i < cfgs.size(); ++i) {
            fails += test_one(be, cfgs[i], 1234 + (uint32_t) i);
        }
        ggml_backend_free(be);
    }
    printf("test-gdn-replay: %s (%d failed)\n", fails == 0 ? "OK" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
