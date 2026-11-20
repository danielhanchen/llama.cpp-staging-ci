// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// Bit identity of the k19-moebw routed-expert mat-vec changes against the kernels they replace, at the MoE shapes of the models we ship
// (Qwen3.8-Flash-Next UD-IQ4_XS: IQ3_S / IQ4_XS gate+up, IQ4_NL / Q8_0 down; Qwen3.6-35B-A3B and LFM2.5-24B-A2B UD-Q4_K_XL: Q4_K / Q5_K
// gate+up, Q5_K / Q6_K down) and odd ones (rows not a multiple of 4 / 8, K of one block, more than 128 entries, skipped slots),
// 1..8 tokens, with and without the batch invariant (verify) flag:
//   - the K-unrolled multi-token MoE kernels (GGML_CUDA_MOE_UNROLL, GGML_CUDA_MMVQ_MOE_GROUP_UNROLL) vs their kill switches;
//   - with MOE_BW_BASE_EXE=<this test copied into the base build's bin dir>: every output of the base build (the IQ3_S sign change has
//     no runtime switch) vs this build.
// Each mode runs in a child process (this binary with --child FILE; the CUDA switches are read once per process) that dumps every
// output; the parent compares the dumps byte for byte. GGML_CUDA_MOE_UNROLL_LOG=1 makes the backend log every unrolled launch: the
// default child must log launches, the kill-switch child none, so a silent fallback fails the test; a negative control
// (GGML_CUDA_MMVQ_MOE_ONE=1: one token on the multi-token kernel, another K split) must differ. No CUDA device: SKIP, exit 0.
//
// --bench NAME L REPS P T,T,..: microbench of one shape (fn4gu fn4gux fn4gun fn4dn fn4dq8 q4xgu q4xdn q4xdn6 lfmgu lfmdn): L layers of
// random experts (cold in L2), REPS graph computes with rotating routing; per T: distinct experts, md5 of all outputs, median graph
// ms, us per layer and the GB/s of the distinct experts' weight bytes at that time.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

static int n_bw_logs = 0;

// MOE_BW_LOG_MATCH=s: count only the launch lines that contain s (k19-moegrp: "vl8,4" = mul_mat_vec_q_moe_vl<8 threads, 4 entries>)
static const char * log_match = nullptr;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "k19-moebw:") != nullptr) {
        n_bw_logs += log_match == nullptr || strstr(text, log_match) != nullptr;
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct moe_shape {
    const char * name;
    ggml_type    type;
    int          K;
    int          N;
    int          E;
    int          topk;
    bool         fused;   // gate + up + SWIGLU (one fused launch), else a down product (one input row per slot)
};

// md5 (RFC 1321)
struct md5_ctx { uint32_t a, b, c, d; uint64_t len; uint8_t buf[64]; size_t n; };
static uint32_t md5_rol(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }
static void md5_block(md5_ctx & m, const uint8_t * p) {
    static const uint32_t K[64] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,
        0x6b901122,0xfd987193,0xa679438e,0x49b40821,0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,
        0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,
        0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391};
    static const int R[64] = {7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
                              4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21};
    uint32_t w[16];
    for (int i = 0; i < 16; ++i) w[i] = p[4*i] | (p[4*i+1] << 8) | (p[4*i+2] << 16) | ((uint32_t) p[4*i+3] << 24);
    uint32_t a = m.a, b = m.b, c = m.c, d = m.d;
    for (int i = 0; i < 64; ++i) {
        uint32_t f; int g;
        if (i < 16) { f = (b & c) | (~b & d); g = i; }
        else if (i < 32) { f = (d & b) | (~d & c); g = (5*i + 1) % 16; }
        else if (i < 48) { f = b ^ c ^ d; g = (3*i + 5) % 16; }
        else { f = c ^ (b | ~d); g = (7*i) % 16; }
        const uint32_t t = d; d = c; c = b; b = b + md5_rol(a + f + K[i] + w[g], R[i]); a = t;
    }
    m.a += a; m.b += b; m.c += c; m.d += d;
}
static void md5_init(md5_ctx & m) { m.a = 0x67452301; m.b = 0xefcdab89; m.c = 0x98badcfe; m.d = 0x10325476; m.len = 0; m.n = 0; }
static void md5_update(md5_ctx & m, const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *) data; m.len += len;
    while (len--) { m.buf[m.n++] = *p++; if (m.n == 64) { md5_block(m, m.buf); m.n = 0; } }
}
static std::string md5_final(md5_ctx & m) {
    const uint64_t bits = m.len * 8; uint8_t pad = 0x80; md5_update(m, &pad, 1); pad = 0;
    while (m.n != 56) md5_update(m, &pad, 1);
    uint8_t l[8]; for (int i = 0; i < 8; ++i) l[i] = (uint8_t) (bits >> (8*i));
    md5_update(m, l, 8);
    char s[33]; const uint32_t v[4] = { m.a, m.b, m.c, m.d };
    for (int i = 0; i < 16; ++i) snprintf(s + 2*i, 3, "%02x", (v[i/4] >> (8*(i%4))) & 0xff);
    return std::string(s);
}

// E experts of N rows each, every row a copy of one of npool rows quantized from N(0, 1) values (quantizing whole experts is slow)
static std::vector<uint8_t> make_experts(ggml_type type, int K, int N, int E, uint32_t seed) {
    const size_t rb    = ggml_row_size(type, K);
    const int    npool = 192;
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f((size_t) npool*K);
    for (auto & v : f) {
        v = nd(gen);
    }
    std::vector<uint8_t> pool((size_t) npool*rb);
    ggml_quantize_chunk(type, f.data(), pool.data(), 0, npool, K, nullptr);
    std::vector<uint8_t> m((size_t) E*N*rb);
    for (size_t r = 0; r < (size_t) E*N; ++r) {
        memcpy(&m[r*rb], &pool[(size_t) (gen() % npool)*rb], rb);
    }
    return m;
}

// routing of T tokens: each slot takes an expert of the earlier tokens with probability P, else a new random one (distinct per token);
// p_skip: a slot is -1 (skipped) with that probability
static std::vector<int32_t> make_routing(std::mt19937 & gen, int T, int topk, int E, float P, float p_skip, int * n_distinct) {
    std::vector<int32_t> r((size_t) topk*T);
    std::vector<int32_t> prev;
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> mine;
        for (int s = 0; s < topk; ++s) {
            std::vector<int32_t> cand;
            for (int32_t p : prev) {
                if (std::find(mine.begin(), mine.end(), p) == mine.end()) {
                    cand.push_back(p);
                }
            }
            int32_t e;
            if (!cand.empty() && std::uniform_real_distribution<float>(0, 1)(gen) < P) {
                e = cand[gen() % cand.size()];
            } else {
                do { e = gen() % E; } while (std::find(mine.begin(), mine.end(), e) != mine.end());
            }
            mine.push_back(e);
            r[(size_t) t*topk + s] = e;
        }
        for (int32_t e : mine) {
            if (std::find(prev.begin(), prev.end(), e) == prev.end()) {
                prev.push_back(e);
            }
        }
    }
    if (n_distinct) {
        *n_distinct = (int) prev.size();
    }
    if (p_skip > 0.0f) {
        for (auto & e : r) {
            if (std::uniform_real_distribution<float>(0, 1)(gen) < p_skip) {
                e = -1;
            }
        }
    }
    return r;
}

// the weights of L layers (one random image per matrix kind, uploaded to every layer: other addresses, the same DRAM traffic)
struct moe_weights {
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::vector<ggml_tensor *> a, g;

    bool build(ggml_backend_t be, const moe_shape & s, int L, uint32_t seed) {
        ggml_init_params wp = { (size_t) (2*L + 8) * ggml_tensor_overhead(), nullptr, true };
        wctx = ggml_init(wp);
        a.assign(L, nullptr);
        g.assign(L, nullptr);
        for (int l = 0; l < L; ++l) {
            a[l] = ggml_new_tensor_3d(wctx, s.type, s.K, s.N, s.E);
            g[l] = s.fused ? ggml_new_tensor_3d(wctx, s.type, s.K, s.N, s.E) : nullptr;
        }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        if (!wbuf) {
            return false;
        }
        const std::vector<uint8_t> ha = make_experts(s.type, s.K, s.N, s.E, seed + 1);
        const std::vector<uint8_t> hg = s.fused ? make_experts(s.type, s.K, s.N, s.E, seed + 2) : std::vector<uint8_t>();
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_set(a[l], ha.data(), 0, ha.size());
            if (s.fused) {
                ggml_backend_tensor_set(g[l], hg.data(), 0, hg.size());
            }
        }
        return true;
    }

    void free_all() {
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        wbuf = nullptr; wctx = nullptr;
    }
};

// one MoE product graph over the layers of w: shared input; outputs = the fused SWIGLU or the down product per layer
struct moe_graph {
    ggml_context * ctx  = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * x = nullptr;
    std::vector<ggml_tensor *> ids, out;

    bool build(ggml_backend_t be, const moe_shape & s, const moe_weights & w, int T, bool flagged, uint32_t seed) {
        const int L = (int) w.a.size();
        ggml_init_params gp = { 64*(size_t) L*ggml_tensor_overhead() + ggml_graph_overhead_custom(16*L, false), nullptr, true };
        ctx = ggml_init(gp);
        x = s.fused ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, 1, T) : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, s.topk, T);
        ggml_set_input(x);
        gf = ggml_new_graph_custom(ctx, 16*L, false);
        ids.resize(L);
        out.resize(L);
        for (int l = 0; l < L; ++l) {
            ids[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.topk, T);
            ggml_set_input(ids[l]);
            if (s.fused) {
                ggml_tensor * gt = ggml_mul_mat_id(ctx, w.g[l], x, ids[l]);
                ggml_tensor * ut = ggml_mul_mat_id(ctx, w.a[l], x, ids[l]);
                if (flagged) {
                    gt->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                    ut->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                }
                out[l] = ggml_swiglu_split(ctx, gt, ut);
                if (flagged) {
                    out[l]->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                }
            } else {
                out[l] = ggml_mul_mat_id(ctx, w.a[l], x, ids[l]);
                if (flagged) {
                    out[l]->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                }
            }
            ggml_set_output(out[l]);
            ggml_build_forward_expand(gf, out[l]);
        }
        // k19-ampere: MOE_BENCH_VERIFY_TAG=1 tags every node as a speculative verify graph (GGML_TENSOR_FLAG_VERIFY), so the
        // verify-window kernel choices (GGML_CUDA_VERIFY_WIDE) apply to the bench as they do to a server's verify batches
        static const bool vtag = getenv("MOE_BENCH_VERIFY_TAG") != nullptr && atoi(getenv("MOE_BENCH_VERIFY_TAG")) != 0;
        if (vtag) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_graph_node(gf, i)->flags |= GGML_TENSOR_FLAG_VERIFY;
            }
        }
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            return false;
        }
        std::vector<float> xh(ggml_nelements(x));
        std::mt19937 gen(seed + 3);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & v : xh) {
            v = nd(gen);
        }
        ggml_backend_tensor_set(x, xh.data(), 0, xh.size()*sizeof(float));
        return true;
    }

    void free_all() {
        if (ga) ggml_gallocr_free(ga);
        if (ctx) ggml_free(ctx);
        ga = nullptr; ctx = nullptr;
    }
};

static const moe_shape bench_shapes[] = {
    { "fn4gu",  GGML_TYPE_IQ3_S,  2560,  640, 128, 10, true  },
    { "fn4gux", GGML_TYPE_IQ4_XS, 2560,  640, 128, 10, true  },
    { "fn4gun", GGML_TYPE_IQ3_S,  2560,  640, 128, 10, false },   // k19-ampere: the gate or up alone (MoE cache tiers: no fused gate)
    { "fn4dn",  GGML_TYPE_IQ4_NL,  640, 2560, 128, 10, false },
    { "fn4dq8", GGML_TYPE_Q8_0,    640, 2560, 128, 10, false },
    { "q4xgu",  GGML_TYPE_Q4_K,   2048,  512, 256,  8, true  },
    { "q4xdn",  GGML_TYPE_Q5_K,    512, 2048, 256,  8, false },
    { "q4xdn6", GGML_TYPE_Q6_K,    512, 2048, 256,  8, false },
    { "lfmgu",  GGML_TYPE_Q4_K,   2048, 1536,  64,  4, true  },
    { "lfmdn",  GGML_TYPE_Q5_K,   1536, 2048,  64,  4, false },
    // k19-ampmoe: the other Qwen3.8-Flash-Next UD-IQ4_XS (fn4n) routed types
    { "fn4nq5",  GGML_TYPE_Q5_0,    640, 2560, 128, 10, false },
    { "fn4nxx",  GGML_TYPE_IQ3_XXS, 2560,  640, 128, 10, true  },
    { "fn4n2s",  GGML_TYPE_IQ2_S,   2560,  640, 128, 10, true  },
};

static int run_bench(ggml_backend_t be, int argc, char ** argv) {
    // --bench NAME L REPS P T,T,..
    if (argc < 7) {
        fprintf(stderr, "usage: --bench NAME L REPS P T,T,..\n");
        return 1;
    }
    const moe_shape * sp = nullptr;
    for (const auto & s : bench_shapes) {
        if (strcmp(s.name, argv[2]) == 0) {
            sp = &s;
        }
    }
    if (!sp) {
        fprintf(stderr, "unknown shape %s\n", argv[2]);
        return 1;
    }
    const moe_shape & s = *sp;
    const int   L    = atoi(argv[3]);
    const int   REPS = atoi(argv[4]);
    const float P    = atof(argv[5]);
    std::vector<int> Ts;
    {
        std::string ts = argv[6];
        size_t pos = 0;
        while (pos < ts.size()) {
            size_t q = ts.find(',', pos);
            if (q == std::string::npos) q = ts.size();
            Ts.push_back(atoi(ts.substr(pos, q - pos).c_str()));
            pos = q + 1;
        }
    }
    const int NIDS = 16;
    const size_t rb = ggml_row_size(s.type, s.K);
    printf("bench %s type %s K %d N %d E %d topk %d fused %d L %d REPS %d P %.2f env BW=%s CFG=%s\n", s.name, ggml_type_name(s.type),
        s.K, s.N, s.E, s.topk, s.fused, L, REPS, P, getenv("GGML_CUDA_MOE_BW") ? getenv("GGML_CUDA_MOE_BW") : "-",
        getenv("GGML_CUDA_MOE_BW_CFG") ? getenv("GGML_CUDA_MOE_BW_CFG") : "-");
    moe_weights w;
    if (!w.build(be, s, L, 1234)) {
        fprintf(stderr, "weight alloc failed\n");
        return 1;
    }
    for (const int T : Ts) {
        moe_graph mg;
        if (!mg.build(be, s, w, T, false, 1234)) {
            fprintf(stderr, "build failed\n");
            return 1;
        }
        std::mt19937 gen(777 + T);
        std::vector<std::vector<int32_t>> pool((size_t) NIDS*L);
        double distinct = 0;
        for (auto & r : pool) {
            int nd = 0;
            r = make_routing(gen, T, s.topk, s.E, P, 0.0f, &nd);
            distinct += nd;
        }
        distinct /= pool.size();
        auto set_ids = [&](int k) {
            for (int l = 0; l < L; ++l) {
                ggml_backend_tensor_set(mg.ids[l], pool[(size_t) (k % NIDS)*L + l].data(), 0, pool[0].size()*sizeof(int32_t));
            }
        };
        md5_ctx ma; md5_init(ma);
        std::vector<uint8_t> o;
        for (int k = 0; k < NIDS; ++k) {
            set_ids(k);
            ggml_backend_graph_compute(be, mg.gf);
            for (int l = 0; l < L; ++l) {
                o.resize(ggml_nbytes(mg.out[l]));
                ggml_backend_tensor_get(mg.out[l], o.data(), 0, o.size());
                md5_update(ma, o.data(), o.size());
            }
        }
        const std::string sa = md5_final(ma);
        std::vector<double> ms;
        for (int r = 0; r < 5 + REPS; ++r) {
            set_ids(r);
            ggml_backend_synchronize(be);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, mg.gf);
            ggml_backend_synchronize(be);
            const auto t1 = std::chrono::steady_clock::now();
            if (r >= 5) {
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size()/2], lo = ms[ms.size()/10], hi = ms[ms.size()*9/10];
        const double us_layer = 1000.0*med/L;
        const double bytes = distinct*s.N*rb*(s.fused ? 2 : 1);
        printf("RES %s T %d distinct %.1f md5 %s ms %.4f p10 %.4f p90 %.4f us_layer %.2f MB_layer %.2f GBps %.1f\n", s.name, T, distinct,
            sa.substr(0, 12).c_str(), med, lo, hi, us_layer, bytes/1e6, bytes/(us_layer*1e3));
        fflush(stdout);
        mg.free_all();
    }
    w.free_all();
    return 0;
}

// exactness cases
struct moe_case {
    moe_shape s;
    int   T;
    float P;
    float p_skip;
    bool  flagged;
};

static std::vector<moe_case> make_cases() {
    const moe_shape shapes[] = {
        { "iq3s_gu",  GGML_TYPE_IQ3_S,  2560,  640, 24, 10, true  },
        { "iq4xs_gu", GGML_TYPE_IQ4_XS, 2560,  640, 24, 10, true  },
        { "iq4nl_dn", GGML_TYPE_IQ4_NL,  640, 2560, 24, 10, false },
        { "q8_0_dn",  GGML_TYPE_Q8_0,    640, 2560, 24, 10, false },
        { "q8_0_gu",  GGML_TYPE_Q8_0,   2560,  640, 24, 10, true  },
        { "q4k_gu",   GGML_TYPE_Q4_K,   2048,  512, 16,  8, true  },
        { "q5k_gu",   GGML_TYPE_Q5_K,   2048,  512, 16,  8, true  },
        { "q5k_dn",   GGML_TYPE_Q5_K,    512, 2048, 16,  8, false },
        { "q6k_dn",   GGML_TYPE_Q6_K,    512, 2048, 16,  8, false },
        { "q4k_lgu",  GGML_TYPE_Q4_K,   2048, 1536,  8,  4, true  },
        { "q5k_ldn",  GGML_TYPE_Q5_K,   1536, 2048,  8,  4, false },
        { "q6k_ldn",  GGML_TYPE_Q6_K,   1536, 2048,  8,  4, false },
        { "iq4xs_dn", GGML_TYPE_IQ4_XS, 1536, 2048,  8,  4, false },
        { "iq3s_dn",  GGML_TYPE_IQ3_S,  1024,  256, 12,  6, false },
        // odd: rows not a multiple of 8 (N 90 hits a base race: the single-token small-K launch writes rows past N into the next slot),
        // K of one block, top-k 20 (160 entries at 8 tokens)
        { "q4k_odd",  GGML_TYPE_Q4_K,    768,   92, 12,  4, true  },
        { "iq4nl_k1", GGML_TYPE_IQ4_NL,   32,  100, 12,  4, false },
        { "q8_0_k20", GGML_TYPE_Q8_0,    256,   64, 40, 20, false },
    };
    std::vector<moe_case> cs;
    for (const auto & s : shapes) {
        for (int T = 1; T <= 8; ++T) {
            cs.push_back({ s, T, 0.55f, 0.0f, false });
            cs.push_back({ s, T, 0.55f, 0.0f, true  });
        }
        cs.push_back({ s, 1, 0.0f, 0.25f, false });
        cs.push_back({ s, 4, 0.55f, 0.25f, false });
        cs.push_back({ s, 4, 0.0f, 0.0f, true  });
        cs.push_back({ s, 8, 0.9f, 0.1f, true  });
    }
    return cs;
}

static int run_child(ggml_backend_t be, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return 1;
    }
    const std::vector<moe_case> cs = make_cases();
    const int NR = 3;
    uint32_t seed = 100;
    for (const auto & c : cs) {
        moe_weights w;
        moe_graph mg;
        if (!w.build(be, c.s, 1, seed) || !mg.build(be, c.s, w, c.T, c.flagged, seed)) {
            fprintf(stderr, "build failed %s\n", c.s.name);
            return 1;
        }
        std::mt19937 gen(seed + 9);
        std::vector<uint8_t> o;
        for (int k = 0; k < NR; ++k) {
            const std::vector<int32_t> r = make_routing(gen, c.T, c.s.topk, c.s.E, c.P, c.p_skip, nullptr);
            ggml_backend_tensor_set(mg.ids[0], r.data(), 0, r.size()*sizeof(int32_t));
            ggml_backend_graph_compute(be, mg.gf);
            o.resize(ggml_nbytes(mg.out[0]));
            ggml_backend_tensor_get(mg.out[0], o.data(), 0, o.size());
            fwrite(o.data(), 1, o.size(), f);
        }
        mg.free_all();
        w.free_all();
        seed += 17;
    }
    fclose(f);
    printf("CHILD_LOGS %d\n", n_bw_logs);
    return 0;
}

// runs this binary as a child with the given environment; returns its exit code and the logged launches
static int spawn_child(const std::string & exe, const std::vector<std::string> & env, const std::string & out, int * n_logs) {
    std::string cmd;
    for (const auto & e : env) {
        cmd += "env " + e + " ";
    }
    cmd += "GGML_CUDA_MOE_UNROLL_LOG=1 \"" + exe + "\" --child \"" + out + "\"";
#if defined(_WIN32)
    FILE * p = _popen(cmd.c_str(), "r");
#else
    FILE * p = popen(cmd.c_str(), "r");
#endif
    if (!p) {
        return -1;
    }
    char line[512];
    *n_logs = -1;
    while (fgets(line, sizeof(line), p)) {
        int n = 0;
        if (sscanf(line, "CHILD_LOGS %d", &n) == 1) {
            *n_logs = n;
        }
    }
#if defined(_WIN32)
    return _pclose(p);
#else
    return pclose(p);
#endif
}

static std::vector<uint8_t> read_file(const std::string & path) {
    std::vector<uint8_t> d;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return d;
    }
    fseek(f, 0, SEEK_END);
    d.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(d.data(), 1, d.size(), f) != d.size()) {
        d.clear();
    }
    fclose(f);
    return d;
}

int main(int argc, char ** argv) {
    log_match = getenv("MOE_BW_LOG_MATCH");
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && strstr(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "CUDA")) {
            dev = d;
            break;
        }
    }
    if (getenv("MOE_BW_CPU")) {   // harness smoke only (the CUDA switches do nothing there)
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "--child") == 0) {
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        const int rc = run_child(be, argv[2]);
        ggml_backend_free(be);
        return rc;
    }
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0) {
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        printf("device %s\n", ggml_backend_dev_description(dev));
        const int rc = run_bench(be, argc, argv);
        ggml_backend_free(be);
        return rc;
    }

    printf("device %s\n", ggml_backend_dev_description(dev));
    const char * tmp = getenv("TMPDIR");
    const std::string dir = tmp ? tmp : ".";
    const std::string tag = std::to_string((long) getpid());
    // (name, env of the kill-switch child, env of the default child, the default child must log launches)
    // neg: a negative control, the two children must differ (GGML_CUDA_MMVQ_MOE_ONE=1 runs one token on the multi-token kernel, whose
    // lane map differs from the single-token launch's: another K split, other float sums). logs: the head child must log unrolled
    // launches (the base child never does)
    const std::string off = "GGML_CUDA_MOE_UNROLL=1 GGML_CUDA_MMVQ_MOE_GROUP_UNROLL=1 GGML_CUDA_MOE_RPB=2 GGML_CUDA_MOE_BW=0 GGML_CUDA_MOE_VL=0";
    struct mode { const char * name; std::string exe_base; std::vector<std::string> env_base, env_head; bool neg = false; int logs = 1; };   // logs: 1 must, 0 must not, -1 any
    const std::string self = argv[0];
    std::vector<mode> modes = {
        { "neg_moe_one", self, { off }, { off, "GGML_CUDA_MMVQ_MOE_ONE=1" }, true, 0 },
        { "default",     self, { off }, {} },   // the shipped defaults (4 rows per warp for the small-K downs at 2..4 tokens; Ampere and newer: the IQ3_S gate / up with 16 threads per row at 3..8 tokens)
        { "bw1",         self, { off }, { "GGML_CUDA_MOE_BW=1" } },
        { "bw1_r8_w8",   self, { off }, { "GGML_CUDA_MOE_BW=1 GGML_CUDA_MOE_BW_CFG=8,8" } },
        { "bw1_r2_w1",   self, { off }, { "GGML_CUDA_MOE_BW=1 GGML_CUDA_MOE_BW_CFG=2,1" } },
        { "unroll4",     self, { off }, { "GGML_CUDA_MOE_UNROLL=4 GGML_CUDA_MMVQ_MOE_GROUP_UNROLL=3 GGML_CUDA_MOE_BW=0" } },
        { "unroll2",     self, { off }, { "GGML_CUDA_MOE_UNROLL=2 GGML_CUDA_MMVQ_MOE_GROUP_UNROLL=2 GGML_CUDA_MOE_BW=0" } },
        { "rpb4",        self, { off }, { "GGML_CUDA_MOE_RPB=4 GGML_CUDA_MOE_BW=0" } },
        { "rpb8_g0",     self, { off, "GGML_CUDA_MMVQ_MOE_GROUP=0" }, { "GGML_CUDA_MOE_RPB=8 GGML_CUDA_MOE_BW=0 GGML_CUDA_MMVQ_MOE_GROUP=0" } },
        { "group0_u4",   self, { off, "GGML_CUDA_MMVQ_MOE_GROUP=0" }, { "GGML_CUDA_MMVQ_MOE_GROUP=0 GGML_CUDA_MOE_UNROLL=4 GGML_CUDA_MOE_BW=0" } },
        { "reassoc0_bw", self, { off, "GGML_CUDA_VERIFY_REASSOC=0" }, { "GGML_CUDA_VERIFY_REASSOC=0 GGML_CUDA_MOE_BW=1" } },
        // k19-moegrp: c_tpr threads per row, c_g entries per weight decode (GGML_CUDA_MOE_VL=tpr,g; the IQ3_S gate / up), every
        // width; MOE_BW_LOG_MATCH counts only that kernel's launches
        { "vl8_4",       self, { off }, { "GGML_CUDA_MOE_VL=8,4 MOE_BW_LOG_MATCH=vl8,4" } },
        { "vl16_4",      self, { off }, { "GGML_CUDA_MOE_VL=16,4 MOE_BW_LOG_MATCH=vl16,4" } },
        { "vl16_2",      self, { off }, { "GGML_CUDA_MOE_VL=16,2 MOE_BW_LOG_MATCH=vl16,2" } },
        { "reassoc0_vl", self, { off, "GGML_CUDA_VERIFY_REASSOC=0" }, { "GGML_CUDA_VERIFY_REASSOC=0 GGML_CUDA_MOE_VL=8,4" }, false, -1 },
    };
    if (const char * b = getenv("MOE_BW_BASE_EXE")) {   // the base build (no such switches there): its default vs this build's
        modes.push_back({ "xbin_default",  b, {}, {}, false, -1 });   // the defaults of this GPU vs the base build
        modes.push_back({ "xbin_bw1",      b, {}, { "GGML_CUDA_MOE_BW=1" } });
        modes.push_back({ "xbin_vl8_4",    b, {}, { "GGML_CUDA_MOE_VL=8,4 MOE_BW_LOG_MATCH=vl8,4" } });
        modes.push_back({ "xbin_vl16_2",   b, {}, { "GGML_CUDA_MOE_VL=16,2 MOE_BW_LOG_MATCH=vl16,2" } });
        modes.push_back({ "xbin_off",      b, {}, { off }, false, 0 });
        modes.push_back({ "xbin_reassoc0", b, { "GGML_CUDA_VERIFY_REASSOC=0" }, { "GGML_CUDA_VERIFY_REASSOC=0 GGML_CUDA_MOE_BW=1" } });
        modes.push_back({ "xbin_group0",   b, { "GGML_CUDA_MMVQ_MOE_GROUP=0" }, { "GGML_CUDA_MMVQ_MOE_GROUP=0 GGML_CUDA_MOE_RPB=4 GGML_CUDA_MOE_BW=0" } });
    }
    if (const char * m = getenv("MOE_BW_EXACT_MODES")) {   // a subset, e.g. "default,reassoc0"
        std::vector<mode> sel;
        for (const auto & md : modes) {
            if (strstr(m, md.name)) {
                sel.push_back(md);
            }
        }
        modes = sel;
    }
    int n_fail = 0;
    for (const auto & md : modes) {
        const std::string fb = dir + "/moe_bw_" + tag + "_" + md.name + "_b.bin";
        const std::string fh = dir + "/moe_bw_" + tag + "_" + md.name + "_h.bin";
        int lb = -1, lh = -1;
        const int rb = spawn_child(md.exe_base, md.env_base, fb, &lb);
        const int rh = spawn_child(self, md.env_head, fh, &lh);
        const std::vector<uint8_t> db = read_file(fb), dh = read_file(fh);
        remove(fb.c_str());
        remove(fh.c_str());
        size_t ndiff = 0;
        if (db.size() == dh.size()) {
            for (size_t i = 0; i < db.size(); i += 4) {
                ndiff += memcmp(&db[i], &dh[i], 4) != 0;
            }
        }
        md5_ctx m1; md5_init(m1); md5_update(m1, db.data(), db.size());
        md5_ctx m2; md5_init(m2); md5_update(m2, dh.data(), dh.size());
        const bool same = rb == 0 && rh == 0 && !db.empty() && db.size() == dh.size();
        const bool ok   = md.neg ? same && ndiff > 0 && lb == 0 : same && ndiff == 0 && lb == 0 && (md.logs < 0 || (md.logs ? lh > 0 : lh == 0));
        printf("%s %-14s base rc %d logs %d md5 %s | head rc %d logs %d md5 %s | %zu bytes, %zu floats differ\n", ok ? "OK  " : "FAIL",
            md.name, rb, lb, md5_final(m1).substr(0, 12).c_str(), rh, lh, md5_final(m2).substr(0, 12).c_str(), db.size(), ndiff);
        n_fail += !ok;
    }
    printf("%s: %d of %zu modes failed\n", n_fail ? "FAIL" : "PASS", n_fail, modes.size());
    return n_fail ? 1 : 0;
}
