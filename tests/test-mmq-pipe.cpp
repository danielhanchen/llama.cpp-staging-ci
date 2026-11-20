// k19-mmq-pipe: MoE expert GEMM (MUL_MAT_ID, MMQ) exactness + timing at the Flash-Next / Qwen3.6 MoE prompt shapes.
// For each case one MUL_MAT_ID runs on the first GPU backend; prints
//   case <name> md5 <md5 of the output> us <median microseconds per graph compute over the timed reps>
// Run it once per GGML_CUDA_MMQ_PIPE value: the md5 of every case must equal the GGML_CUDA_MMQ_PIPE=0 run (byte-identical output).
// argv[1] (optional): run only cases whose name contains it. TEST_MMQ_PIPE_REPS (default 30) timed reps, 5 untimed warm-up reps.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---- MD5 (RFC 1321) ----
struct md5_ctx {
    uint32_t a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
    uint64_t len = 0;
    uint8_t  buf[64];
    size_t   n = 0;
};

static uint32_t md5_rol(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

static void md5_block(md5_ctx & m, const uint8_t * p) {
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af,
        0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
        0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97,
        0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 };
    static const int R[64] = { 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21 };
    uint32_t w[16];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t) p[4*i] | ((uint32_t) p[4*i + 1] << 8) | ((uint32_t) p[4*i + 2] << 16) | ((uint32_t) p[4*i + 3] << 24);
    }
    uint32_t a = m.a, b = m.b, c = m.c, d = m.d;
    for (int i = 0; i < 64; ++i) {
        uint32_t f;
        int g;
        if (i < 16)      { f = (b & c) | (~b & d); g = i; }
        else if (i < 32) { f = (d & b) | (~d & c); g = (5*i + 1) % 16; }
        else if (i < 48) { f = b ^ c ^ d;          g = (3*i + 5) % 16; }
        else             { f = c ^ (b | ~d);       g = (7*i) % 16; }
        const uint32_t t = d;
        d = c;
        c = b;
        b = b + md5_rol(a + f + K[i] + w[g], R[i]);
        a = t;
    }
    m.a += a; m.b += b; m.c += c; m.d += d;
}

static void md5_update(md5_ctx & m, const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *) data;
    m.len += len;
    while (len > 0) {
        const size_t k = std::min(len, 64 - m.n);
        memcpy(m.buf + m.n, p, k);
        m.n += k; p += k; len -= k;
        if (m.n == 64) {
            md5_block(m, m.buf);
            m.n = 0;
        }
    }
}

static std::string md5_hex(md5_ctx m) {
    const uint64_t bits = m.len*8;
    const uint8_t pad = 0x80;
    md5_update(m, &pad, 1);
    const uint8_t zero = 0;
    while (m.n != 56) {
        md5_update(m, &zero, 1);
    }
    uint8_t lb[8];
    for (int i = 0; i < 8; ++i) {
        lb[i] = (uint8_t) (bits >> (8*i));
    }
    md5_update(m, lb, 8);
    char out[33];
    const uint32_t v[4] = { m.a, m.b, m.c, m.d };
    for (int i = 0; i < 16; ++i) {
        snprintf(out + 2*i, 3, "%02x", (v[i/4] >> (8*(i%4))) & 0xff);
    }
    return out;
}

// ---- cases ----
struct mm_case {
    std::string name;
    ggml_type   type;
    int64_t     K, N, E, topk;
    bool        bcast;     // src1 [K, 1, T] (gate / up) instead of [K, topk, T] (down)
    int64_t     tokens;
    int32_t     op2;       // op param 2 (> 0: an in-place MoE cache bank, the skip_empty / stream-k launch)
    float       skip_frac; // share of slots set to -1
    float       skew;      // routing skew: 0 uniform, larger = a few hot experts
    uint32_t    seed;
};

static void run_case(ggml_backend_t backend, const mm_case & c, const int reps) {
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, c.E);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.bcast ? 1 : c.topk, c.tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.topk, c.tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, ids);
    ((int32_t *) out->op_params)[2] = c.op2;
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    {   // weights: a pool of random rows quantized once, each row of the tensor a random pool row
        const int64_t nrow  = c.N*c.E;
        const int64_t npool = std::min<int64_t>(nrow, 4096);
        const size_t  rs    = ggml_row_size(c.type, c.K);
        std::vector<float> f(c.K*npool);
        for (auto & v : f) {
            v = ud(rng);
        }
        std::vector<uint8_t> pool(rs*npool);
        ggml_quantize_chunk(c.type, f.data(), pool.data(), 0, npool, c.K, nullptr);
        std::vector<uint8_t> q(rs*nrow);
        std::uniform_int_distribution<int64_t> pick(0, npool - 1);
        for (int64_t r = 0; r < nrow; ++r) {
            memcpy(q.data() + r*rs, pool.data() + pick(rng)*rs, rs);
        }
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
    }
    {
        std::vector<float> f(ggml_nelements(x));
        for (auto & v : f) {
            v = ud(rng);
        }
        ggml_backend_tensor_set(x, f.data(), 0, f.size()*sizeof(float));
        std::vector<int32_t> id(c.topk*c.tokens);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        for (int64_t t = 0; t < c.tokens; ++t) {
            std::vector<bool> used(c.E, false);
            for (int64_t s = 0; s < c.topk; ++s) {
                int32_t e;
                do {
                    const float r = u01(rng);
                    e = (int32_t) (c.E * (c.skew > 0.0f ? std::pow(r, 1.0f + c.skew) : r));
                    e = std::min<int32_t>(e, (int32_t) c.E - 1);
                } while (used[e]);
                used[e] = true;
                id[t*c.topk + s] = (s > 0 && u01(rng) < c.skip_frac) ? -1 : e;
            }
        }
        ggml_backend_tensor_set(ids, id.data(), 0, id.size()*sizeof(int32_t));
    }

    // poison the output, so that an element nothing writes differs
    std::vector<uint8_t> o(ggml_nbytes(out), 0x7f);
    ggml_backend_tensor_set(out, o.data(), 0, o.size());
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) {
        printf("case %s compute failed\n", c.name.c_str());
        return;
    }
    ggml_backend_tensor_get(out, o.data(), 0, o.size());
    md5_ctx m;
    md5_update(m, o.data(), o.size());
    const std::string h = md5_hex(m);

    for (int r = 0; r < 5; ++r) {
        ggml_backend_graph_compute(backend, g);
    }
    ggml_backend_synchronize(backend);
    std::vector<double> t(reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, g);
        ggml_backend_synchronize(backend);
        t[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    // the output after the timed reps must equal the first one (the same launch every time)
    std::vector<uint8_t> o2(o.size());
    ggml_backend_tensor_get(out, o2.data(), 0, o2.size());
    std::sort(t.begin(), t.end());
    printf("case %s md5 %s us %.1f min %.1f max %.1f repeat %s\n", c.name.c_str(), h.c_str(), t[reps/2], t[0], t[reps - 1],
        o == o2 ? "same" : "DIFFERENT");
    fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        printf("no GPU device\n");
        return 2;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("device %s pipe %s\n", ggml_backend_dev_description(dev), getenv("GGML_CUDA_MMQ_PIPE") ? getenv("GGML_CUDA_MMQ_PIPE") : "(default)");
    const char * only = argc > 1 ? argv[1] : nullptr;
    const int reps = getenv("TEST_MMQ_PIPE_REPS") ? std::max(1, atoi(getenv("TEST_MMQ_PIPE_REPS"))) : 30;

    std::vector<mm_case> cs;
    // Flash-Next (qwen4exp): 512 experts, 10 used, n_embd 2560, n_ff_exp 640; -ub 2048 prompt ubatches (40 columns per expert, J 40)
    cs.push_back({ "fn_gate_iq3s_2048",      GGML_TYPE_IQ3_S,  2560,  640, 512, 10, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_gate_iq3s_2048_s2",   GGML_TYPE_IQ3_S,  2560,  640, 512, 10, true,  2048, 0,   0.0f,  0.0f, 2 });
    cs.push_back({ "fn_down_iq4nl_2048",     GGML_TYPE_IQ4_NL,  640, 2560, 512, 10, false, 2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_gate_iq4xs_2048",     GGML_TYPE_IQ4_XS, 2560,  640, 512, 10, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_gate_q4k_2048",       GGML_TYPE_Q4_K,   2560,  640, 512, 10, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_gate_q8_0_2048",      GGML_TYPE_Q8_0,   2560,  640, 512, 10, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_gate_iq3s_skip_skew", GGML_TYPE_IQ3_S,  2560,  640, 512, 10, true,  2048, 0,   0.05f, 1.0f, 3 });
    cs.push_back({ "fn_gate_iq3s_inplace",   GGML_TYPE_IQ3_S,  2560,  640, 512, 10, true,  2048, 384, 0.0f,  0.5f, 4 }); // stream-k
    cs.push_back({ "fn_down_iq4nl_inplace",  GGML_TYPE_IQ4_NL,  640, 2560, 512, 10, false, 2048, 384, 0.0f,  0.5f, 4 }); // stream-k
    cs.push_back({ "fn_gate_iq3s_1638",      GGML_TYPE_IQ3_S,  2560,  640, 512, 10, true,  1638, 0,   0.0f,  0.0f, 5 }); // J 32
    cs.push_back({ "fn_gate_iq3s_1229",      GGML_TYPE_IQ3_S,  2560,  640, 512, 10, true,  1229, 0,   0.0f,  0.0f, 6 }); // J 24: base
    // the UD-IQ4_XS Flash-Next expert types on the pipe (GGML_CUDA_MMQ_PIPE_EXT)
    cs.push_back({ "fn_gate_iq3xxs_2048",    GGML_TYPE_IQ3_XXS, 2560, 640, 512, 10, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_gate_iq2s_2048",      GGML_TYPE_IQ2_S,  2560,  640, 512, 10, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_down_q5_0_2048",      GGML_TYPE_Q5_0,    640, 2560, 512, 10, false, 2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_down_q5_0_2044",      GGML_TYPE_Q5_0,    640, 2560, 512, 10, false, 2044, 0,   0.0f,  0.0f, 2 }); // last ubatch
    cs.push_back({ "fn_gate_iq3xxs_skew",    GGML_TYPE_IQ3_XXS, 2560, 640, 512, 10, true,  2048, 0,   0.05f, 1.0f, 3 });
    cs.push_back({ "fn_gate_iq3xxs_inplace", GGML_TYPE_IQ3_XXS, 2560, 640, 512, 10, true,  2048, 384, 0.0f,  0.5f, 4 }); // stream-k
    cs.push_back({ "fn_gate_iq3xxs_1229",    GGML_TYPE_IQ3_XXS, 2560, 640, 512, 10, true,  1229, 0,   0.0f,  0.0f, 6 }); // J 24
    // Qwen3.6 MoE (35B-A3B class): 256 experts, 8 used, n_embd 2048, n_ff_exp 512; -ub 2048 (64 columns per expert, J 64)
    cs.push_back({ "q36_gate_q4k_2048",      GGML_TYPE_Q4_K,   2048,  512, 256,  8, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "q36_down_q6k_2048",      GGML_TYPE_Q6_K,    512, 2048, 256,  8, false, 2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "q36_gate_iq4xs_2048",    GGML_TYPE_IQ4_XS, 2048,  512, 256,  8, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "q36_down_iq4xs_2048",    GGML_TYPE_IQ4_XS,  512, 2048, 256,  8, false, 2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "q36_down_iq4nl_2048",    GGML_TYPE_IQ4_NL,  512, 2048, 256,  8, false, 2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "q36_gate_q8_0_2048",     GGML_TYPE_Q8_0,   2048,  512, 256,  8, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "q36_gate_q5k_2048",      GGML_TYPE_Q5_K,   2048,  512, 256,  8, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "q36_gate_iq3s_2048",     GGML_TYPE_IQ3_S,  2048,  512, 256,  8, true,  2048, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "q36_gate_q4k_inplace",   GGML_TYPE_Q4_K,   2048,  512, 256,  8, true,  2048, 128, 0.0f,  0.5f, 4 }); // stream-k
    // a 4096-token ubatch (J 128 at Qwen3.6: 128 columns per expert)
    cs.push_back({ "q36_gate_q4k_4096",      GGML_TYPE_Q4_K,   2048,  512, 256,  8, true,  4096, 0,   0.0f,  0.0f, 1 });
    cs.push_back({ "fn_gate_iq3s_6554",      GGML_TYPE_IQ3_S,  2560,  640, 512, 10, true,  6554, 0,   0.0f,  0.0f, 1 }); // J 128

    for (const auto & c : cs) {
        if (only && c.name.find(only) == std::string::npos) {
            continue;
        }
        run_case(backend, c, reps);
    }
    ggml_backend_free(backend);
    return 0;
}
