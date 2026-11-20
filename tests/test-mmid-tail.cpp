// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// k19-exactdet: single column MUL_MAT_ID launches whose last row block is partial (small K: rows per block = warps; rows not a
// multiple of it) must not write past an expert's rows. Before the fix the row guard of mul_mat_vec_q compared against the token stride,
// so the last block of slot s wrote its rows past N into slot s+1's first rows (two blocks writing one value: the bytes depended on which
// wrote last, run to run) and the last slot of the last token wrote past dst. Checks, per case (types, fused gate + up + SWIGLU or the
// up product alone, 1..4 tokens, plain and as a batch invariant verify batch):
//   - sentinel: the tensor allocated right after dst (dst bytes a multiple of the buffer alignment) keeps its bytes;
//   - reference: the CUDA output against the CPU backend's, within one RMS of the output (a row of another expert is far off);
//   - repeat: REPS more computes of the same graph give the same bytes as the first.
// The default run also runs itself in the two exact verify modes (GGML_CUDA_VERIFY_REASSOC=0, and with GGML_CUDA_MMVQ_EXACT_IDS_MERGE=0)
// and with every single column launch lever on / off / at its other rows per block, plain and exact (TAIL_NO_LEVERS=1: not).
// --bench: median us of the fused gate + up product at a Qwen3.6-35B-A3B expert shape and at the 66-row shape (1 and 4 tokens).
// No CUDA device: SKIP, exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static int n_engaged = 0;   // the shared-memory table / grid launches' INFO lines (GGML_CUDA_MMVQ_IQ_TAB_LOG=1 in the lever children)

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, " engaged (") != nullptr || strstr(text, " in shared memory: ") != nullptr) {
        n_engaged++;
        if (getenv("TAIL_PRINT_LOGS")) {
            fputs(text, stdout);
        }
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct tcase {
    ggml_type type;
    int  K, N, E, topk, T;
    bool fused, flagged;
};

static const float SENT = -12345.0f;
static const int   NSENT = 256;

struct tgraph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * out = nullptr, * sent = nullptr;

    bool build(ggml_backend_t be, const tcase & c, int L, uint32_t seed) {
        const int nw = c.fused ? 2 : 1;
        ggml_init_params p = { (size_t) (16*L + 8)*ggml_tensor_overhead() + ggml_graph_overhead_custom(16*L, false), nullptr, true };
        ctx = ggml_init(p);
        std::vector<ggml_tensor *> a(L), g(L, nullptr), ids(L);
        for (int l = 0; l < L; ++l) {
            a[l] = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, c.E);
            if (c.fused) {
                g[l] = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, c.E);
            }
            ids[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.topk, c.T);
        }
        ggml_tensor * x = c.fused ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, 1, c.T) : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.topk, c.T);
        gf = ggml_new_graph_custom(ctx, 16*L, false);
        // tensors are placed in creation order: the sentinel right after the last output
        for (int l = 0; l < L; ++l) {
            ggml_tensor * u  = ggml_mul_mat_id(ctx, a[l], x, ids[l]);
            ggml_tensor * gt = c.fused ? ggml_mul_mat_id(ctx, g[l], x, ids[l]) : nullptr;
            ggml_tensor * o  = c.fused ? ggml_swiglu_split(ctx, gt, u) : u;
            if (c.flagged) {
                u->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                o->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                if (gt) {
                    gt->flags |= GGML_TENSOR_FLAG_BATCH_INVARIANT;
                }
            }
            ggml_build_forward_expand(gf, o);
            out = o;
        }
        sent = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NSENT);
        if (c.flagged) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_graph_node(gf, i)->flags |= GGML_TENSOR_FLAG_VERIFY;
            }
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) {
            return false;
        }
        std::mt19937 gen(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        // weight rows: copies of 64 pool rows quantized from N(0, 1) once per (type, K) (imatrix all ones: IQ types need one)
        const size_t rb = ggml_row_size(c.type, c.K);
        static std::map<std::pair<int, int>, std::vector<uint8_t>> pools;
        std::vector<uint8_t> & pool = pools[{ (int) c.type, c.K }];
        if (pool.empty()) {
            std::mt19937 pg(4321 + c.K + 7*(int) c.type);
            std::vector<float> f((size_t) 64*c.K), imat(c.K, 1.0f);
            for (auto & v : f) {
                v = nd(pg);
            }
            pool.resize(64*rb);
            ggml_quantize_chunk(c.type, f.data(), pool.data(), 0, 64, c.K, imat.data());
        }
        for (int l = 0; l < L; ++l) {
            for (int w = 0; w < nw; ++w) {
                ggml_tensor * t = w == 0 ? a[l] : g[l];
                std::vector<uint8_t> q(ggml_nbytes(t));
                for (int64_t r = 0; r < (int64_t) c.N*c.E; ++r) {
                    memcpy(&q[(size_t) r*rb], &pool[(size_t) (gen() % 64)*rb], rb);
                }
                ggml_backend_tensor_set(t, q.data(), 0, q.size());
            }
            std::vector<int32_t> ih((size_t) c.topk*c.T);
            for (int t = 0; t < c.T; ++t) {
                std::vector<int32_t> perm(c.E);
                for (int e = 0; e < c.E; ++e) {
                    perm[e] = e;
                }
                std::shuffle(perm.begin(), perm.end(), gen);
                // the highest expert in the last slot of the last token: its rows past N would be read past the weights
                if (t == c.T - 1) {
                    std::iter_swap(std::find(perm.begin(), perm.end(), c.E - 1), perm.begin() + (c.topk - 1));
                }
                for (int k = 0; k < c.topk; ++k) {
                    ih[(size_t) t*c.topk + k] = perm[k];
                }
            }
            ggml_backend_tensor_set(ids[l], ih.data(), 0, ih.size()*sizeof(int32_t));
        }
        std::vector<float> xh(ggml_nelements(x));
        for (auto & v : xh) {
            v = nd(gen);
        }
        ggml_backend_tensor_set(x, xh.data(), 0, xh.size()*sizeof(float));
        reset_sentinel();
        return true;
    }

    void reset_sentinel() {
        std::vector<float> s(NSENT, SENT);
        ggml_backend_tensor_set(sent, s.data(), 0, s.size()*sizeof(float));
    }

    int sentinel_bad() const {
        std::vector<float> s(NSENT);
        ggml_backend_tensor_get(sent, s.data(), 0, s.size()*sizeof(float));
        int n = 0;
        for (float v : s) {
            n += memcmp(&v, &SENT, 4) != 0;
        }
        return n;
    }

    std::vector<float> get_out() const {
        std::vector<float> o(ggml_nelements(out));
        ggml_backend_tensor_get(out, o.data(), 0, o.size()*sizeof(float));
        return o;
    }

    void free_all() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
    }
};

static std::vector<tcase> make_cases() {
    struct shp { int K, N, E, topk; bool fused; };
    // N 66 / 70: not a multiple of 4; dst bytes N*topk*4 a multiple of 128 (topk 16 / 32: the sentinel follows dst directly)
    const shp shapes[] = {
        { 1024, 66, 32, 16, true  },
        { 1024, 66, 32, 16, false },
        { 2048, 70, 40, 32, true  },
        { 1024, 64, 32, 16, true  },   // control: rows a multiple of 4
        { 1024, 68, 32, 16, true  },   // rows a multiple of 4, not of 8 (8 rows per block levers)
        { 4096, 70, 16, 16, true  },   // K of 16 super-blocks: no small-K launch, the codebook / grid launches with 2 / 4 / 8 rows
        { 4096, 72, 16, 16, true  },   // same, rows a multiple of 8
        {  512, 68, 32, 16, false },   // k19-q4xt: K of one warp pass (warp per row launch), rows a multiple of 4, not of 8
        {  512, 72, 32, 16, true  },   // same, rows a multiple of 8
    };
    const ggml_type types[] = { GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
                                GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q5_K };
    std::vector<tcase> cs;
    for (const ggml_type t : types) {
        for (const auto & s : shapes) {
            for (int T = 1; T <= 4; ++T) {
                cs.push_back({ t, s.K, s.N, s.E, s.topk, T, s.fused, false });
                if (T > 1) {
                    cs.push_back({ t, s.K, s.N, s.E, s.topk, T, s.fused, true });
                }
            }
        }
    }
    return cs;
}

static int run_bench(ggml_backend_t be, int reps) {
    struct bs { const char * name; ggml_type type; int K, N, E, topk, T; bool flagged; };
    const bs shapes[] = {
        { "q36gu_t1",    GGML_TYPE_IQ2_XXS, 2048, 512, 128,  8, 1, false },
        { "q36gu_t4x",   GGML_TYPE_IQ2_XXS, 2048, 512, 128,  8, 4, true  },
        { "q36gu1s_t1",  GGML_TYPE_IQ1_S,   2048, 512, 128,  8, 1, false },
        { "q36gu1s_t4x", GGML_TYPE_IQ1_S,   2048, 512, 128,  8, 4, true  },
        { "r66_t1",      GGML_TYPE_IQ2_XXS, 1024,  66,  32, 16, 1, false },
        { "r66_t4x",     GGML_TYPE_IQ2_XXS, 1024,  66,  32, 16, 4, true  },
        { "q4k66_t4x",   GGML_TYPE_Q4_K,    1024,  66,  32, 16, 4, true  },
    };
    const int L = 4;
    for (const auto & s : shapes) {
        tgraph g;
        if (!g.build(be, { s.type, s.K, s.N, s.E, s.topk, s.T, true, s.flagged }, L, 77)) {
            fprintf(stderr, "build failed\n");
            return 1;
        }
        std::vector<double> ms;
        for (int r = 0; r < 5 + reps; ++r) {
            ggml_backend_synchronize(be);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, g.gf);
            ggml_backend_synchronize(be);
            const auto t1 = std::chrono::steady_clock::now();
            if (r >= 5) {
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }
        std::sort(ms.begin(), ms.end());
        printf("RES %-12s %-8s us %.2f p10 %.2f p90 %.2f\n", s.name, ggml_type_name(s.type), 1000.0*ms[ms.size()/2]/L,
            1000.0*ms[ms.size()/10]/L, 1000.0*ms[ms.size()*9/10]/L);
        fflush(stdout);
        g.free_all();
    }
    return 0;
}

int main(int argc, char ** argv) {
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
    if (getenv("TAIL_CPU")) {   // harness smoke only
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    ggml_backend_t be  = ggml_backend_dev_init(dev, nullptr);
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0) {
        printf("device %s\n", ggml_backend_dev_description(dev));
        const int rc = run_bench(be, argc > 2 ? atoi(argv[2]) : 200);
        ggml_backend_free(be);
        return rc;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    const int reps = argc >= 3 && strcmp(argv[1], "--reps") == 0 ? atoi(argv[2]) : 20;
    // --one: the first case only (compute-sanitizer)
    const bool one = argc >= 2 && strcmp(argv[1], "--one") == 0;
    printf("device %s, %zu cases, %d repeats\n", ggml_backend_dev_description(dev), make_cases().size(), reps);
    int n_fail = 0, n_case = 0, n_sent = 0, n_ref = 0, n_rep = 0;
    uint32_t seed = 1000;
    for (const auto & c : make_cases()) {
        seed += 13;
        tgraph g, r;
        if (!g.build(be, c, 1, seed) || !r.build(cpu, c, 1, seed)) {
            printf("FAIL build %s\n", ggml_type_name(c.type));
            return 1;
        }
        ggml_backend_graph_compute(be, g.gf);
        ggml_backend_synchronize(be);
        const int sbad = g.sentinel_bad();
        const std::vector<float> o = g.get_out();
        ggml_backend_graph_compute(cpu, r.gf);
        const std::vector<float> ref = r.get_out();
        double ss = 0.0, md = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            ss += (double) ref[i]*ref[i];
        }
        const double rms = std::sqrt(ss/ref.size());
        size_t nbad_ref = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = std::fabs((double) o[i] - ref[i]);
            md = std::max(md, d);
            // CPU vs CUDA (q8_K vs q8_1 activations, other sums) stays within 0.26 rms on T4; a row of another expert is 2..8 rms off
            nbad_ref += !(d <= 1.0*rms);   // NaN counts
        }
        size_t nrep = 0;
        for (int k = 0; k < reps && !one; ++k) {
            ggml_backend_graph_compute(be, g.gf);
            const std::vector<float> o2 = g.get_out();
            for (size_t i = 0; i < o.size(); ++i) {
                nrep += memcmp(&o[i], &o2[i], 4) != 0;
            }
        }
        const bool ok = sbad == 0 && nbad_ref == 0 && nrep == 0;
        n_case++;
        n_fail += !ok;
        n_sent += sbad > 0;
        n_ref  += nbad_ref > 0;
        n_rep  += nrep > 0;
        if (!ok || getenv("TAIL_VERBOSE")) {
            printf("%s %-8s K %d N %d E %d topk %d T %d fused %d flagged %d: sentinel %d floats changed, %zu off the CPU reference (max |d| %.3g, rms %.3g), %zu repeat floats differ\n",
                ok ? "ok  " : "FAIL", ggml_type_name(c.type), c.K, c.N, c.E, c.topk, c.T, (int) c.fused, (int) c.flagged, sbad, nbad_ref, md, rms, nrep);
            fflush(stdout);
        }
        g.free_all();
        r.free_all();
        if (one) {
            break;
        }
    }
    const char * mode = getenv("TAIL_MODE") ? getenv("TAIL_MODE") : "default";
    printf("%s [%s]: %d of %d cases failed (sentinel %d, reference %d, repeat %d), %d engagement lines\n", n_fail ? "FAIL" : "PASS", mode, n_fail,
        n_case, n_sent, n_ref, n_rep, n_engaged);
    fflush(stdout);
    ggml_backend_free(cpu);
    ggml_backend_free(be);
    int rc = n_fail ? 1 : 0;
#if !defined(_WIN32)
    // the exact verify modes (the switches are read once per process: a child each): GGML_CUDA_VERIFY_REASSOC=0 runs a verify batch
    // of MUL_MAT_ID as one single column launch with the tokens as samples, GGML_CUDA_MMVQ_EXACT_IDS_MERGE=0 as one launch per token
    if (getenv("TAIL_MODE") == nullptr && !one) {
        std::vector<std::string> envs = { "TAIL_MODE=exact GGML_CUDA_VERIFY_REASSOC=0", "TAIL_MODE=exact_per_token GGML_CUDA_VERIFY_REASSOC=0 GGML_CUDA_MMVQ_EXACT_IDS_MERGE=0" };
        // every single column launch lever on (default) / off / at its other rows per block, plain and in the exact verify mode
        const char * levers[][2] = {
            { "iqtab0",    "GGML_CUDA_MMVQ_IQ_TAB=0" },                // k19-dsv4p IQ1_S / IQ2_XXS codebook off
            { "iqtab2",    "GGML_CUDA_MMVQ_IQ_TAB=2" },                // and forced on (any table)
            { "idsrows1",  "GGML_CUDA_MMVQ_TURING_IDS_ROWS=1" },       // its MUL_MAT_ID rows per block 1 / 4 (default 2)
            { "idsrows4",  "GGML_CUDA_MMVQ_TURING_IDS_ROWS=4" },
            { "gsm0",      "GGML_CUDA_MMVQ_GRID_SMEM=0" },             // k19-q38t IQ3 grid in shared memory off
            { "gids0",     "GGML_CUDA_MMVQ_GRID_IDS=0" },              // k19-iqsmem IQ3 MUL_MAT_ID rows: off / 2 / 4 / 8
            { "gids2",     "GGML_CUDA_MMVQ_GRID_IDS=2" },
            { "gids4",     "GGML_CUDA_MMVQ_GRID_IDS=4" },
            { "gids8",     "GGML_CUDA_MMVQ_GRID_IDS=8" },
            { "gidsw0",    "GGML_CUDA_MMVQ_GRID_IDS_WARPS=0" },        // its 2 warps off
            { "gids8w0",   "GGML_CUDA_MMVQ_GRID_IDS=8 GGML_CUDA_MMVQ_GRID_IDS_WARPS=0" },
            { "iq2tab0",   "GGML_CUDA_MMVQ_IQ2_TAB=0" },               // k19-iqsmem IQ2_XS / IQ2_S codebook off
            { "iq2r4",     "GGML_CUDA_MMVQ_IQ2_TAB_ROWS=4" },          // its dense rows 4 / 8
            { "iq2r8",     "GGML_CUDA_MMVQ_IQ2_TAB_ROWS=8" },
            { "wpr0",      "GGML_CUDA_MMVQ_IDS_WPR=0" },               // k19-q4xt warp per row MUL_MAT_ID launch off
            { "wprw2r1",   "GGML_CUDA_MMVQ_IDS_WPR_W=2 GGML_CUDA_MMVQ_IDS_WPR_ROWS=1" },   // its warps x rows (default 4 x 2)
            { "wprw2r2",   "GGML_CUDA_MMVQ_IDS_WPR_W=2 GGML_CUDA_MMVQ_IDS_WPR_ROWS=2" },
            { "wprw4r4",   "GGML_CUDA_MMVQ_IDS_WPR_W=4 GGML_CUDA_MMVQ_IDS_WPR_ROWS=4" },
            { "alloff",    "GGML_CUDA_MMVQ_IDS_WPR=0 GGML_CUDA_MMVQ_IQ_TAB=0 GGML_CUDA_MMVQ_IQ2_TAB=0 GGML_CUDA_MMVQ_GRID_IDS=0 GGML_CUDA_MMVQ_GRID_SMEM=0" },
        };
        if (!getenv("TAIL_NO_LEVERS")) {
            envs.push_back("TAIL_MODE=default_log GGML_CUDA_MMVQ_IQ_TAB_LOG=1");
            for (const auto & l : levers) {
                envs.push_back(std::string("TAIL_MODE=") + l[0] + " GGML_CUDA_MMVQ_IQ_TAB_LOG=1 " + l[1]);
                envs.push_back(std::string("TAIL_MODE=") + l[0] + "_exact GGML_CUDA_MMVQ_IQ_TAB_LOG=1 GGML_CUDA_VERIFY_REASSOC=0 " + l[1]);
            }
        }
        for (const auto & e : envs) {
            const std::string cmd = "env " + e + " \"" + argv[0] + "\" --reps " + std::to_string(std::min(reps, e.find("_LOG") != std::string::npos ? 20 : reps));
            rc |= system(cmd.c_str()) != 0;
        }
    }
#endif
    return rc;
}
