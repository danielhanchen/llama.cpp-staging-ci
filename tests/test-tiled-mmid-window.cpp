// Tiled CPU MUL_MAT_ID: hash of the raw tiled output per case, and optional per-case timing.
// The output bytes must not depend on how many row groups share one window
// (GGML_CPU_TILED_PAIR), so runs of this tool under different settings or builds are
// compared hash for hash (or with --dump, byte for byte).
//
//   test-tiled-mmid-window [--set exact|bench] [--threads N] [--reps N] [--dump FILE]

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

static void cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_n_threads)(ggml_backend_t, int) =
        (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (!set_n_threads) {
        fprintf(stderr, "ggml_backend_set_n_threads not available\n");
        exit(1);
    }
    set_n_threads(backend, n_threads);
}

static double now_s(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double) t.QuadPart / (double) freq.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
#endif
}

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint32_t rng_state = 1;
static uint32_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

struct mmid_case {
    const char * name;
    int64_t K;        // ne00
    int64_t R;        // ne01, output rows per expert
    int64_t E;        // experts
    int64_t k;        // experts per token
    int64_t T;        // tokens
    ggml_type type;
    int routing;      // 0 random top-k, 1 skewed (half of the tokens prefer the low experts), 2 one hot expert
};

// one case: build, fill, compute tiled, hash, optionally time
static void run_case(ggml_backend_t backend, const mmid_case & c, int reps, FILE * dump) {
    rng_state = 0x9E3779B9u ^ (uint32_t) (c.K * 31 + c.R * 17 + c.E * 7 + c.T + c.routing);

    struct ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * src0  = ggml_new_tensor_3d(ctx, c.type, c.K, c.R, c.E);
    struct ggml_tensor * src1  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, 1, c.T);
    struct ggml_tensor * ids_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.k, c.T);
    struct ggml_tensor * dst   = ggml_mul_mat_id(ctx, src0, src1, ids_t);
    struct ggml_cgraph * gf    = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // weights: quantize one expert at a time from random f32
    {
        std::vector<float> w((size_t) c.K * c.R);
        std::vector<uint8_t> q(ggml_row_size(c.type, c.K) * c.R);
        std::vector<float> im;
        const float * imatrix = NULL;
        if (ggml_quantize_requires_imatrix(c.type)) {
            im.assign((size_t) c.K, 1.0f);
            imatrix = im.data();
        }
        for (int64_t e = 0; e < c.E; e++) {
            for (auto & x : w) {
                x = ((float) (rng() & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
            }
            ggml_quantize_chunk(c.type, w.data(), q.data(), 0, c.R, c.K, imatrix);
            ggml_backend_tensor_set(src0, q.data(), (size_t) e * src0->nb[2], q.size());
        }
    }
    {
        std::vector<float> x((size_t) c.K * c.T);
        for (auto & v : x) {
            v = ((float) (rng() & 0xFFFF) / 65535.0f - 0.5f) * 4.0f;
        }
        ggml_backend_tensor_set(src1, x.data(), 0, ggml_nbytes(src1));
    }
    {
        std::vector<int32_t> ids((size_t) c.k * c.T);
        std::vector<int32_t> perm(c.E);
        for (int64_t t = 0; t < c.T; t++) {
            for (int64_t e = 0; e < c.E; e++) perm[e] = (int32_t) e;
            // partial Fisher-Yates over a pool: random, skewed (low half), or fixed hot expert 0
            int64_t pool = c.E;
            if (c.routing == 1 && (t & 1) == 0) pool = std::max<int64_t>(c.k, c.E / 4);
            for (int64_t i = 0; i < c.k; i++) {
                const int64_t j = i + (int64_t) (rng() % (uint32_t) (pool - i));
                std::swap(perm[i], perm[j]);
            }
            if (c.routing == 2) {
                // expert 0 always first: it gets every token
                for (int64_t i = 0; i < c.k; i++) {
                    if (perm[i] == 0) { std::swap(perm[i], perm[0]); break; }
                }
                if (perm[0] != 0) perm[0] = 0;
            }
            for (int64_t i = 0; i < c.k; i++) ids[t * c.k + i] = perm[i];
        }
        ggml_backend_tensor_set(ids_t, ids.data(), 0, ggml_nbytes(ids_t));
    }

    ggml_backend_graph_compute(backend, gf);
    std::vector<uint8_t> out(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, out.data(), 0, out.size());
    const uint64_t h = fnv1a(out.data(), out.size());
    if (dump) {
        fwrite(out.data(), 1, out.size(), dump);
    }

    printf("CASE %-28s K=%-5lld R=%-5lld E=%-4lld k=%lld T=%-5lld %-7s route=%d hash=%016llx",
           c.name, (long long) c.K, (long long) c.R, (long long) c.E, (long long) c.k, (long long) c.T,
           ggml_type_name(c.type), c.routing, (unsigned long long) h);
    if (reps > 0) {
        std::vector<double> ts;
        ggml_backend_graph_compute(backend, gf); // warm
        for (int i = 0; i < reps; i++) {
            const double t0 = now_s();
            ggml_backend_graph_compute(backend, gf);
            ts.push_back(now_s() - t0);
        }
        std::sort(ts.begin(), ts.end());
        printf(" med_us=%.1f min_us=%.1f", ts[ts.size() / 2] * 1e6, ts[0] * 1e6);
    }
    printf("\n");
    fflush(stdout);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * set = "exact";
    const char * dump_path = NULL;
    const char * only = NULL;
    int n_threads = 8;
    int reps = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--set") && i + 1 < argc) {
            set = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--reps") && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
            only = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--set exact|bench] [--threads N] [--reps N] [--dump FILE] [--only NAME]\n", argv[0]);
            return 1;
        }
    }

    std::vector<mmid_case> cases;
    if (!strcmp(set, "bench")) {
        // small MoE shapes: granite-3.1-1b-a400m (1024 / 512, 32 x top 8) and
        // Qwen3-30B-A3B (2048 / 768, 128 x top 8), prompt ubatch 512 and 2048
        for (int64_t T : { 512, 2048 }) {
            const bool big = T == 2048;
            cases.push_back({ big ? "granite_gateup_T2048" : "granite_gateup_T512", 1024,  512,  32, 8, T, GGML_TYPE_Q4_K, 0 });
            cases.push_back({ big ? "granite_down_T2048"   : "granite_down_T512",    512, 1024,  32, 8, T, GGML_TYPE_Q4_K, 0 });
            cases.push_back({ big ? "granite_down6_T2048"  : "granite_down6_T512",   512, 1024,  32, 8, T, GGML_TYPE_Q6_K, 0 });
            cases.push_back({ big ? "qwen3a3b_gateup_T2048": "qwen3a3b_gateup_T512",2048,  768, 128, 8, T, GGML_TYPE_Q4_K, 0 });
            cases.push_back({ big ? "qwen3a3b_down_T2048"  : "qwen3a3b_down_T512",   768, 2048, 128, 8, T, GGML_TYPE_Q4_K, 0 });
            cases.push_back({ big ? "qwen3a3b_down6_T2048" : "qwen3a3b_down6_T512",  768, 2048, 128, 8, T, GGML_TYPE_Q6_K, 0 });
        }
    } else {
        // routes per expert around the narrow / standard / window edges, ragged R, skewed routing
        const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K,
                                    GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_M };
        for (ggml_type t : types) {
            cases.push_back({ "mix_R300_route1",        512,  300,  8, 2,  200, t, 1 });
        }
        for (int64_t T : { 8, 9, 16, 17, 255, 256, 300 }) {
            cases.push_back({ "hot_R256",               512,  256,  4, 2,    T, GGML_TYPE_Q4_K, 2 });
            cases.push_back({ "hot_R1000_K768",         768, 1000,  4, 2,    T, GGML_TYPE_Q6_K, 2 });
        }
        cases.push_back({ "R65",                        512,   65,  4, 2,  128, GGML_TYPE_Q4_K, 0 });
        cases.push_back({ "R129",                       512,  129,  4, 2,  128, GGML_TYPE_Q4_K, 0 });
        cases.push_back({ "R200",                       512,  200,  4, 2,  128, GGML_TYPE_Q5_K, 0 });
        cases.push_back({ "R1024_down",                 512, 1024, 32, 8,  512, GGML_TYPE_Q4_K, 0 });
        cases.push_back({ "R1024_down_skew",            512, 1024, 32, 8,  512, GGML_TYPE_Q4_K, 1 });
        cases.push_back({ "R2048_K768",                 768, 2048, 16, 8,  300, GGML_TYPE_Q4_K, 1 });
        cases.push_back({ "R768_K2048",                2048,  768, 16, 8,  300, GGML_TYPE_Q4_K, 0 });
    }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        fprintf(stderr, "failed to initialize the CPU backend\n");
        return 1;
    }
    cpu_set_n_threads(backend, n_threads);

    FILE * dump = dump_path ? fopen(dump_path, "wb") : NULL;
    for (const auto & c : cases) {
        if (only && strcmp(only, c.name) != 0) continue;
        run_case(backend, c, reps, dump);
    }
    if (dump) fclose(dump);
    ggml_backend_free(backend);
    return 0;
}
