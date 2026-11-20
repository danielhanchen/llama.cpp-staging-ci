// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// k19-kvwin: the CPU two-source flash attention (ggml_flash_attn_ext_set_src2) is the softmax over the cells of both sources:
//  1. equal quantized types: bitwise equal to one flash attention over the two sources concatenated (the CPU loop takes the first source, then
//     the second, in the order of the concatenation), for decode, verify and prompt row counts, sinks, ALiBi, GQA, two streams;
//  2. mixed types (a q4_0 cache beside a q8_0 ring, the window mode's layout): against a double precision reference on the
//     dequantized K / V (the Q to q8_0 rounding of the dot products bounds the error).

#include "ggml.h"
#include "ggml-fa-src2.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct shape {
    int64_t hs, nh, nr2, ns, kv, kv2, nb;
    bool sinks;
    float max_bias;
};

static void fill_f32(std::vector<float> & v, std::mt19937 & gen, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto & x : v) {
        x = d(gen);
    }
}

// a mask pair like the cache builds: random visibility, every visible cell finite in one of the two sources only
static void make_masks(std::vector<ggml_fp16_t> & m, std::vector<ggml_fp16_t> & m2, const shape & sh, std::mt19937 & gen) {
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    m.assign(sh.kv*sh.nb*sh.ns, ninf);
    m2.assign(sh.kv2*sh.nb*sh.ns, ninf);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    for (int64_t i = 0; i < sh.nb*sh.ns; ++i) {
        for (int64_t c = 0; c < sh.kv; ++c) {
            if (d(gen) < 0.7f) {
                m[i*sh.kv + c] = ggml_fp32_to_fp16(sh.max_bias > 0.0f ? -(float) (gen() % 64) : 0.0f);
            }
        }
        for (int64_t c = 0; c < sh.kv2; ++c) {
            if (d(gen) < 0.5f) {
                m2[i*sh.kv2 + c] = ggml_fp32_to_fp16(sh.max_bias > 0.0f ? -(float) (gen() % 64) : 0.0f);
            }
        }
    }
}

// run: two-source (concat == false) or one source over the concatenation (concat == true, needs equal types)
static std::vector<float> run(const shape & sh, ggml_type tk, ggml_type tv, ggml_type tk2, ggml_type tv2, int concat,
        const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v, const std::vector<float> & k2,
        const std::vector<float> & v2, const std::vector<ggml_fp16_t> & m, const std::vector<ggml_fp16_t> & m2,
        const std::vector<float> & sinks, int n_threads) {
    ggml_init_params ip = { (size_t) 1024*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t nhq = sh.nh*sh.nr2;
    ggml_tensor * tq  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.hs, sh.nb, nhq, sh.ns);
    memcpy(tq->data, q.data(), ggml_nbytes(tq));

    // [hs, n, nh, ns] from f32 data laid out [hs, nh, n, ns] (the cache layout), permuted
    const auto mk = [&](ggml_type t, const std::vector<float> & src, int64_t n) {
        ggml_tensor * x = ggml_new_tensor_4d(ctx, t, sh.hs, sh.nh, n, sh.ns);
        ggml_quantize_chunk(t, src.data(), x->data, 0, sh.nh*n*sh.ns, sh.hs, nullptr);
        return x;
    };
    ggml_tensor * K  = mk(tk,  k,  sh.kv);
    ggml_tensor * V  = mk(tv,  v,  sh.kv);
    ggml_tensor * K2 = mk(tk2, k2, sh.kv2);
    ggml_tensor * V2 = mk(tv2, v2, sh.kv2);
    ggml_tensor * M  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.kv,  sh.nb, 1, sh.ns);
    ggml_tensor * M2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.kv2, sh.nb, 1, sh.ns);
    memcpy(M->data,  m.data(),  ggml_nbytes(M));
    memcpy(M2->data, m2.data(), ggml_nbytes(M2));

    ggml_tensor * S = nullptr;
    if (sh.sinks) {
        S = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nhq);
        memcpy(S->data, sinks.data(), ggml_nbytes(S));
    }

    ggml_tensor * out;
    if (concat == 2) {
        out = ggml_flash_attn_ext(ctx, tq, ggml_permute(ctx, K, 0, 2, 1, 3), ggml_permute(ctx, V, 0, 2, 1, 3), M,
                1.0f/sqrtf((float) sh.hs), sh.max_bias, 0.0f);
    } else if (concat) {
        ggml_tensor * Kc = ggml_concat(ctx, K, K2, 2);
        ggml_tensor * Vc = ggml_concat(ctx, V, V2, 2);
        ggml_tensor * Mc = ggml_concat(ctx, M, M2, 0);
        out = ggml_flash_attn_ext(ctx, tq, ggml_permute(ctx, Kc, 0, 2, 1, 3), ggml_permute(ctx, Vc, 0, 2, 1, 3), Mc,
                1.0f/sqrtf((float) sh.hs), sh.max_bias, 0.0f);
    } else {
        out = ggml_flash_attn_ext(ctx, tq, ggml_permute(ctx, K, 0, 2, 1, 3), ggml_permute(ctx, V, 0, 2, 1, 3), M,
                1.0f/sqrtf((float) sh.hs), sh.max_bias, 0.0f);
        ggml_flash_attn_ext_set_src2(out, ggml_permute(ctx, K2, 0, 2, 1, 3), ggml_permute(ctx, V2, 0, 2, 1, 3), M2);
    }
    ggml_flash_attn_ext_add_sinks(out, S);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    // the CPU backend (also where it is a loadable module): the tensors live in the context's memory
    static ggml_backend_t cpu = nullptr;
    if (cpu == nullptr) {
        ggml_backend_load_all();
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        GGML_ASSERT(cpu != nullptr);
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto set_threads = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) {
        set_threads(cpu, n_threads);
    }
    GGML_ASSERT(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> res(ggml_nelements(out));
    memcpy(res.data(), out->data, ggml_nbytes(out));
    ggml_free(ctx);
    return res;
}

// double precision reference over the dequantized K / V of both sources
static std::vector<float> reference(const shape & sh, ggml_type tk, ggml_type tv, ggml_type tk2, ggml_type tv2,
        const std::vector<float> & q, std::vector<float> k, std::vector<float> v, std::vector<float> k2, std::vector<float> v2,
        const std::vector<ggml_fp16_t> & m, const std::vector<ggml_fp16_t> & m2, const std::vector<float> & sinks) {
    const auto rt = [&](ggml_type t, std::vector<float> & x) {
        std::vector<uint8_t> buf(ggml_row_size(t, sh.hs)*(x.size()/sh.hs));
        ggml_quantize_chunk(t, x.data(), buf.data(), 0, x.size()/sh.hs, sh.hs, nullptr);
        const auto * tr = ggml_get_type_traits(t);
        if (t == GGML_TYPE_F32) {
            return;
        }
        tr->to_float(buf.data(), x.data(), (int64_t) x.size());
    };
    rt(tk, k); rt(tv, v); rt(tk2, k2); rt(tv2, v2);

    const int64_t nhq = sh.nh*sh.nr2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(nhq));
    const float m0 = powf(2.0f, -(sh.max_bias)/n_head_log2);
    const float m1 = powf(2.0f, -(sh.max_bias/2.0f)/n_head_log2);
    std::vector<float> res(sh.hs*nhq*sh.nb*sh.ns);
    for (int64_t s = 0; s < sh.ns; ++s) {
        for (int64_t h = 0; h < nhq; ++h) {
            const float slope = sh.max_bias > 0.0f ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;
            const int64_t hk = h/sh.nr2;
            for (int64_t r = 0; r < sh.nb; ++r) {
                const float * qr = q.data() + ((s*nhq + h)*sh.nb + r)*sh.hs;
                std::vector<double> sc;
                std::vector<const float *> vr;
                const auto add = [&](const std::vector<float> & kk, const std::vector<float> & vv, const std::vector<ggml_fp16_t> & mm, int64_t n) {
                    for (int64_t c = 0; c < n; ++c) {
                        const float mv = ggml_fp16_to_fp32(mm[(s*sh.nb + r)*n + c]);
                        if (mv == -INFINITY) {
                            continue;
                        }
                        const float * kr = kk.data() + ((s*n + c)*sh.nh + hk)*sh.hs;
                        double d = 0.0;
                        for (int64_t i = 0; i < sh.hs; ++i) {
                            d += (double) qr[i]*kr[i];
                        }
                        sc.push_back(d/sqrt((double) sh.hs) + slope*mv);
                        vr.push_back(vv.data() + ((s*n + c)*sh.nh + hk)*sh.hs);
                    }
                };
                add(k, v, m, sh.kv);
                add(k2, v2, m2, sh.kv2);
                double mx = -INFINITY;
                for (double x : sc) {
                    mx = std::max(mx, x);
                }
                if (sh.sinks) {
                    mx = std::max(mx, (double) sinks[h]);
                }
                double sum = sh.sinks ? exp(sinks[h] - mx) : 0.0;
                std::vector<double> acc(sh.hs, 0.0);
                for (size_t j = 0; j < sc.size(); ++j) {
                    const double w = exp(sc[j] - mx);
                    sum += w;
                    for (int64_t i = 0; i < sh.hs; ++i) {
                        acc[i] += w*vr[j][i];
                    }
                }
                // output layout [hs, nhq, nb, ns]
                float * o = res.data() + ((s*sh.nb + r)*nhq + h)*sh.hs;
                for (int64_t i = 0; i < sh.hs; ++i) {
                    o[i] = sum > 0.0 ? (float) (acc[i]/sum) : 0.0f;
                }
            }
        }
    }
    return res;
}

int main() {
    std::mt19937 gen(42);
    int n_fail = 0;
    int n_case = 0;

    const shape shapes[] = {
        {  64, 2, 1, 1,  512, 256,   1, false, 0.0f },
        { 128, 2, 2, 1, 1024, 256,   1, true,  0.0f },
        { 128, 2, 2, 1, 1024, 512,   3, false, 0.0f },
        { 256, 1, 6, 1,  512, 256,  16, true,  0.0f },
        { 128, 2, 2, 2,  512, 256,   8, false, 8.0f },
        {  64, 4, 2, 1,  256, 256, 128, false, 0.0f },
    };
    struct types { ggml_type k, v, k2, v2; };
    const types same[] = {
        { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0 },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
    };
    const types mixed[] = {
        { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q4_0, GGML_TYPE_F16,  GGML_TYPE_Q8_0, GGML_TYPE_F16  },
        { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_F16,  GGML_TYPE_F16  },
        // F16 / F16: the one-source F16 node takes the tiled / split-KV kernels, so it is checked against the reference only
        { GGML_TYPE_F16,  GGML_TYPE_F16,  GGML_TYPE_F16,  GGML_TYPE_F16  },
    };

    for (const auto & sh : shapes) {
        const int64_t nhq = sh.nh*sh.nr2;
        std::vector<float> q(sh.hs*sh.nb*nhq*sh.ns), k(sh.hs*sh.nh*sh.kv*sh.ns), v(k.size()), k2(sh.hs*sh.nh*sh.kv2*sh.ns), v2(k2.size());
        std::vector<float> sinks(nhq);
        fill_f32(q, gen, -1, 1); fill_f32(k, gen, -1, 1); fill_f32(v, gen, -1, 1); fill_f32(k2, gen, -1, 1); fill_f32(v2, gen, -1, 1);
        fill_f32(sinks, gen, -3, 3);
        std::vector<ggml_fp16_t> m, m2;
        make_masks(m, m2, sh, gen);

        for (const auto & t : same) {
            for (int nt : { 1, 4 }) {
                const auto a = run(sh, t.k, t.v, t.k2, t.v2, false, q, k, v, k2, v2, m, m2, sinks, nt);
                const auto b = run(sh, t.k, t.v, t.k2, t.v2, true,  q, k, v, k2, v2, m, m2, sinks, nt);
                ++n_case;
                if (memcmp(a.data(), b.data(), a.size()*sizeof(float)) != 0) {
                    ++n_fail;
                    printf("FAIL equal types %s/%s hs %lld nb %lld kv %lld+%lld threads %d: two-source != concatenation\n",
                        ggml_type_name(t.k), ggml_type_name(t.k2), (long long) sh.hs, (long long) sh.nb, (long long) sh.kv, (long long) sh.kv2, nt);
                }
            }
        }
        // an all-masked second source changes nothing: bitwise the one-source node (any types)
        for (const auto & t : mixed) {
            if (t.k == GGML_TYPE_F16) {
                continue; // (the one-source F16 node takes other kernels)
            }
            std::vector<ggml_fp16_t> m2e(m2.size(), ggml_fp32_to_fp16(-INFINITY));
            const auto a = run(sh, t.k, t.v, t.k2, t.v2, 0, q, k, v, k2, v2, m, m2e, sinks, 4);
            const auto b = run(sh, t.k, t.v, t.k2, t.v2, 2, q, k, v, k2, v2, m, m2e, sinks, 4);
            ++n_case;
            if (memcmp(a.data(), b.data(), a.size()*sizeof(float)) != 0) {
                ++n_fail;
                printf("FAIL empty second source %s + %s hs %lld nb %lld: != one source\n", ggml_type_name(t.k), ggml_type_name(t.k2),
                    (long long) sh.hs, (long long) sh.nb);
            }
        }
        for (const auto & t : mixed) {
            const auto a = run(sh, t.k, t.v, t.k2, t.v2, false, q, k, v, k2, v2, m, m2, sinks, 4);
            const auto r = reference(sh, t.k, t.v, t.k2, t.v2, q, k, v, k2, v2, m, m2, sinks);
            double e = 0.0, n = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                e += (double) (a[i] - r[i])*(a[i] - r[i]);
                n += (double) r[i]*r[i];
            }
            const double nmse = n > 0 ? e/n : e;
            ++n_case;
            if (!(nmse < 5e-4)) {
                ++n_fail;
                printf("FAIL mixed types %s/%s + %s/%s hs %lld nb %lld: nmse %.3g vs the reference\n", ggml_type_name(t.k), ggml_type_name(t.v),
                    ggml_type_name(t.k2), ggml_type_name(t.v2), (long long) sh.hs, (long long) sh.nb, nmse);
            }
        }
    }
    printf("%s: %d / %d cases pass\n", n_fail == 0 ? "OK" : "FAIL", n_case - n_fail, n_case);
    return n_fail == 0 ? 0 : 1;
}
