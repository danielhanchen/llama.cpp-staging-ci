// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// DeepSeek-V4.1 candidate scoring. A candidate consumer layer used to score every compressed row with the fused lightning
// indexer and then add a block mask built from the candidate source's block ids (0 inside a candidate block, -inf elsewhere: fill +
// set_rows + repeat + cont + add, exactly as deepseek41.cpp builds it with LLAMA_DSV41_CAND_SCORE=0). Now the indexer node takes the
// block ids (src[4], block size in op_params[0]) and scores only the rows of those blocks. This test runs both graphs on one device
// and compares, byte for byte, the score rows AND the top-k ids that follow (ties and -inf included):
//   - heads 4 (CUDA tile kernel from 8 tokens, vec kernel below), 32 and 64 (wmma for F16 / quantized keys, vec for F32 / BF16);
//   - key types F16, Q8_0, F32, BF16; 1 / 2 streams; partial last block; repeated key rows (exact score ties across the candidate
//     boundary); a causal -inf mask (a query sees a prefix) plus random -inf rows; candidates that hold fewer finite rows than top-k
//     (so -inf ties decide the tail of the top-k); the newest block always a candidate (as cand_pin pins it).
//   - negative control: the candidate graph with ONE candidate block replaced by a non-candidate block that has finite scores must
//     differ from the mask graph (else the comparison could not see a wrong candidate set) -> FAIL if equal.
// Usage: (no args) the CPU; --gpu the first CUDA / ROCm device (exit 77 = SKIP without one); --bench adds per-graph timings;
// --gallocr allocates with a bare graph allocator (buffer reuse without the backend graph_optimize rules: diagnostic only).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct tcase {
    int       n_head;
    int       n_tok;
    int       n_kv;
    int       n_cand;   // candidate blocks per token
    int       bs;       // rows per block
    int       n_stream;
    ggml_type tk;       // key type
    int       top_k;
    bool      causal;   // query t sees rows < n_kv - n_tok + t + 1 (else random -inf rows only)
    std::string str() const {
        char b[200];
        snprintf(b, sizeof(b), "heads %2d tok %3d kv %6d cand %4d x %d streams %d K %-4s top-k %3d %s", n_head, n_tok, n_kv, n_cand, bs,
                n_stream, ggml_type_name(tk), top_k, causal ? "causal" : "dense");
        return b;
    }
};

static std::vector<tcase> cases(bool gpu) {
    std::vector<tcase> v;
    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_BF16 };
    int i = 0;
    for (int nh : { 4, 32, 64 }) {
        for (ggml_type t : types) {
            for (int nt : { 1, 3, 9, 33 }) {
                for (int ns : { 1, 2 }) {
                    const int  kv  = (i % 3 == 0) ? 4093 : (i % 3 == 1 ? 8192 : 2051); // partial last block in two of three
                    const int  nb  = (kv + 7) / 8;
                    const int  nc  = std::min(nb, (i % 2) ? 64 : 160);
                    const bool cau = (i % 4) != 3;
                    i++;
                    if (!gpu && nh == 64 && nt == 33) {
                        continue; // CPU time
                    }
                    v.push_back({ nh, nt, kv, nc, 8, ns, t, (i % 2) ? 512 : 64, cau });
                }
            }
        }
    }
    // the real shape: 32 heads, 2048 candidate blocks of 8 over 122880 rows, top-512, one decode token and a short prompt chunk
    v.push_back({ 32, 1, 122880, 2048, 8, 1, GGML_TYPE_F16, 512, true });
    v.push_back({ 32, gpu ? 64 : 4, 40000, 2048, 8, 1, GGML_TYPE_F16, 512, true });
    v.push_back({ 32, 2, 40000, 2048, 8, 1, GGML_TYPE_Q8_0, 512, true });
    // block size 4 / 16 (not a divisor of the 32-key tile, or larger than it)
    v.push_back({ 32, 5, 3001, 100, 3, 1, GGML_TYPE_F16, 64, true });
    v.push_back({ 32, 5, 3001, 40, 16, 1, GGML_TYPE_F16, 64, true });
    v.push_back({ 4, 12, 3001, 40, 16, 1, GGML_TYPE_F16, 64, true });
    return v;
}

struct inputs {
    std::vector<float>   q, w, k;
    std::vector<uint16_t> m;  // f16 mask
    std::vector<int32_t> cand, cand_neg;
};

static inputs make_inputs(const tcase & c, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    inputs in;
    const int D = 128;
    in.q.resize((size_t) D * c.n_head * c.n_tok * c.n_stream);
    for (auto & x : in.q) x = nd(rng);
    in.w.resize((size_t) c.n_head * c.n_tok * c.n_stream);
    for (auto & x : in.w) x = nd(rng) * 0.1f; // mixed signs, as the prescaled indexer weights
    in.k.resize((size_t) D * c.n_kv * c.n_stream);
    for (auto & x : in.k) x = nd(rng);
    // repeated key rows: every 7th row copies a row 61 rows earlier (exact score ties, across block / candidate boundaries)
    for (int s = 0; s < c.n_stream; ++s) {
        for (int r = 61; r < c.n_kv; r += 7) {
            memcpy(&in.k[((size_t) s * c.n_kv + r) * D], &in.k[((size_t) s * c.n_kv + r - 61) * D], D * sizeof(float));
        }
    }
    const int n_blk = (c.n_kv + c.bs - 1) / c.bs;
    const uint16_t h0 = ggml_fp32_to_fp16(0.0f), hinf = ggml_fp32_to_fp16(-INFINITY);
    in.m.resize((size_t) c.n_kv * c.n_tok * c.n_stream);
    in.cand.resize((size_t) c.n_cand * c.n_tok * c.n_stream);
    in.cand_neg = in.cand;
    std::vector<int> perm(n_blk);
    for (int s = 0; s < c.n_stream; ++s) {
        for (int t = 0; t < c.n_tok; ++t) {
            // visible rows: a prefix; the first token of a causal case sees only a few blocks (fewer finite candidates than top-k)
            const int n_vis = !c.causal ? c.n_kv : (t == 0 && c.n_tok > 1 ? std::min(c.n_kv, 3 * c.bs + 1) : c.n_kv - (c.n_tok - 1 - t) * 5);
            uint16_t * mr = &in.m[((size_t) s * c.n_tok + t) * c.n_kv];
            for (int r = 0; r < c.n_kv; ++r) {
                mr[r] = (r < n_vis && (rng() % 17) != 0) ? h0 : hinf;
            }
            // candidates: the newest visible block + distinct random blocks, in random order (as the source's top-k)
            const int newest = (n_vis - 1) / c.bs;
            for (int b = 0; b < n_blk; ++b) perm[b] = b;
            std::shuffle(perm.begin(), perm.end(), rng);
            std::vector<int32_t> sel;
            sel.push_back(newest);
            for (int b : perm) {
                if ((int) sel.size() == c.n_cand) break;
                if (b != newest) sel.push_back(b);
            }
            std::shuffle(sel.begin(), sel.end(), rng);
            int32_t * cr = &in.cand[((size_t) s * c.n_tok + t) * c.n_cand];
            std::copy(sel.begin(), sel.end(), cr);
            // negative control: replace the candidate in slot 0 by a visible non-candidate block (if any)
            int32_t * nr = &in.cand_neg[((size_t) s * c.n_tok + t) * c.n_cand];
            std::copy(sel.begin(), sel.end(), nr);
            for (int b = 0; b < n_vis / c.bs; ++b) {
                if (std::find(sel.begin(), sel.end(), b) == sel.end()) {
                    nr[0] = b;
                    break;
                }
            }
        }
    }
    return in;
}

static bool g_gallocr = false; // --gallocr: a bare graph allocator (buffer reuse) instead of one buffer per tensor (diagnostic)

struct result {
    std::vector<float>   score;
    std::vector<int32_t> ids;
    double us = 0.0;
};

// mode 0: lightning indexer + the block mask graph of deepseek41.cpp (LLAMA_DSV41_CAND_SCORE=0); 1: candidate scoring; 2: candidate
// scoring with the negative-control ids
static result run_case(ggml_backend_t backend, const tcase & c, const inputs & in, int mode, int n_timed) {
    const int D = 128;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, c.n_head, c.n_tok, c.n_stream);
    ggml_tensor * kf   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, c.n_kv, c.n_stream);
    ggml_tensor * w    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.n_head, c.n_tok, 1, c.n_stream);
    ggml_tensor * m    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_tok, 1, c.n_stream);
    ggml_tensor * cand = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, c.n_cand, c.n_tok, 1, c.n_stream);
    for (ggml_tensor * t : { q, kf, w, m, cand }) {
        ggml_set_input(t);
    }
    // the key cache in its type (converted on the device by a copy, as the cache would hold it)
    ggml_tensor * k = c.tk == GGML_TYPE_F32 ? kf : ggml_cpy(ctx, kf, ggml_new_tensor_4d(ctx, c.tk, D, 1, c.n_kv, c.n_stream));

    const int64_t n_comp = c.n_kv;
    const int64_t bs     = c.bs;
    const int64_t n_blk  = (n_comp + bs - 1) / bs;
    const int64_t nc     = c.n_tok;
    const int64_t n_s    = c.n_stream;

    ggml_tensor * score = ggml_lightning_indexer(ctx, q, k, w, m);
    if (mode == 0) {
        // deepseek41.cpp, use_blk: 0 inside the candidate blocks, -inf elsewhere
        ggml_tensor * bsel  = ggml_cont(ctx, cand);
        const int64_t k_b   = bsel->ne[0];
        ggml_tensor * bm    = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, n_blk, nc*n_s, 1), -INFINITY);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, k_b, nc*n_s, 1), 0.0f);
        bm = ggml_set_rows(ctx, bm, zeros, ggml_reshape_3d(ctx, bsel, k_b, nc*n_s, 1));
        bm = ggml_reshape_2d(ctx, ggml_repeat_4d(ctx, bm, bs, n_blk, nc*n_s, 1), bs*n_blk, nc*n_s);
        bm = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, bm, n_comp, nc*n_s, bm->nb[1], 0)), n_comp, nc, 1, n_s);
        score = ggml_add(ctx, score, bm);
    } else {
        score->src[4]       = cand;
        score->op_params[0] = (int32_t) bs;
    }
    ggml_tensor * ids = ggml_top_k(ctx, score, std::min<int64_t>(n_comp, c.top_k));
    ggml_set_output(score);
    ggml_set_output(ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, score);
    ggml_build_forward_expand(gf, ids);

    result r;
    if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, -1)) ||
        !ggml_backend_supports_op(backend, mode == 0 ? score->src[0] : score)) {
        // reported by the caller as an unsupported case
        ggml_free(ctx);
        return r;
    }
    // every tensor gets its own memory: a graph allocator reusing buffers needs the backend's graph_optimize allocation rules
    // (ggml_backend_sched applies them, a bare gallocr does not), and the comparison is about the op, not about buffer reuse
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    if (g_gallocr) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            fprintf(stderr, "alloc failed\n");
            exit(1);
        }
    } else {
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "alloc failed\n");
            exit(1);
        }
    }
    ggml_backend_tensor_set(q, in.q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(kf, in.k.data(), 0, ggml_nbytes(kf));
    ggml_backend_tensor_set(w, in.w.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(m, in.m.data(), 0, ggml_nbytes(m));
    ggml_backend_tensor_set(cand, (mode == 2 ? in.cand_neg : in.cand).data(), 0, ggml_nbytes(cand));
    ggml_backend_graph_compute(backend, gf);
    if (n_timed > 0) {
        std::vector<double> t;
        for (int it = 0; it < n_timed; ++it) {
            ggml_backend_synchronize(backend);
            const auto t0 = std::chrono::high_resolution_clock::now();
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            t.push_back(std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        r.us = t[t.size() / 2];
    }
    r.score.resize(ggml_nelements(score));
    r.ids.resize(ggml_nelements(ids));
    ggml_backend_tensor_get(score, r.score.data(), 0, ggml_nbytes(score));
    ggml_backend_tensor_get(ids, r.ids.data(), 0, ggml_nbytes(ids));
    if (galloc) {
        ggml_gallocr_free(galloc);
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return r;
}

static bool same(const result & a, const result & b) {
    return a.ids == b.ids && a.score.size() == b.score.size() &&
           memcmp(a.score.data(), b.score.data(), a.score.size() * sizeof(float)) == 0;
}

static std::string first_diff(const result & a, const result & b) {
    char buf[400];
    size_t n_diff = 0, first = SIZE_MAX, last = 0, n_zero_a = 0;
    for (size_t i = 0; i < a.score.size() && i < b.score.size(); ++i) {
        if (memcmp(&a.score[i], &b.score[i], sizeof(float)) != 0) {
            n_diff++;
            first = std::min(first, i);
            last  = i;
            n_zero_a += a.score[i] == 0.0f;
        }
    }
    if (n_diff > 0) {
        snprintf(buf, sizeof(buf), "%zu scores differ in [%zu, %zu] (%zu of them 0 in the first), ids %s; score[%zu] %a / %a", n_diff, first,
                last, n_zero_a, a.ids == b.ids ? "equal" : "differ", first, a.score[first], b.score[first]);
        return buf;
    }
    for (size_t i = 0; i < a.score.size() && i < b.score.size(); ++i) {
        if (memcmp(&a.score[i], &b.score[i], sizeof(float)) != 0) {
            snprintf(buf, sizeof(buf), "score[%zu] %a / %a", i, a.score[i], b.score[i]);
            return buf;
        }
    }
    for (size_t i = 0; i < a.ids.size() && i < b.ids.size(); ++i) {
        if (a.ids[i] != b.ids[i]) {
            snprintf(buf, sizeof(buf), "ids[%zu] %d / %d", i, a.ids[i], b.ids[i]);
            return buf;
        }
    }
    return "sizes differ";
}

// the case really exercises ties and -inf in the top-k (counted on the mask graph's output)
static void stats(const tcase & c, const result & r, long & n_inf_sel, long & n_tie_sel) {
    const int kk = (int) std::min<int64_t>(c.n_kv, c.top_k);
    for (int row = 0; row < c.n_tok * c.n_stream; ++row) {
        std::vector<float> sel;
        for (int j = 0; j < kk; ++j) {
            const float v = r.score[(size_t) row * c.n_kv + r.ids[(size_t) row * kk + j]];
            n_inf_sel += std::isinf(v);
            sel.push_back(v);
        }
        std::sort(sel.begin(), sel.end());
        for (int j = 1; j < kk; ++j) {
            n_tie_sel += std::isfinite(sel[j]) && sel[j] == sel[j - 1];
        }
    }
}

int main(int argc, char ** argv) {
    bool want_gpu = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        want_gpu |= strcmp(argv[i], "--gpu") == 0;
        bench    |= strcmp(argv[i], "--bench") == 0;
        g_gallocr |= strcmp(argv[i], "--gallocr") == 0;
    }
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t di = 0; di < ggml_backend_dev_count() && !dev; ++di) {
        ggml_backend_dev_t d = ggml_backend_dev_get(di);
        const auto type = ggml_backend_dev_type(d);
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        if (want_gpu ? ((type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                        (strcmp(reg, "CUDA") == 0 || strcmp(reg, "ROCm") == 0))
                     : type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            dev = d;
        }
    }
    if (!dev) {
        printf("SKIP: no %s device\n", want_gpu ? "CUDA / ROCm" : "CPU");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        printf("SKIP: %s did not initialise\n", ggml_backend_dev_name(dev));
        return 77;
    }
    printf("device %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    const auto cs = cases(want_gpu);
    int n_bad = 0, n_run = 0, n_unsup = 0, n_neg_ok = 0, n_neg = 0;
    long n_inf_sel = 0, n_tie_sel = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const inputs in = make_inputs(cs[i], 100 + (int) i);
        const int nt = bench ? 20 : 0;
        const result a = run_case(backend, cs[i], in, 0, nt);
        const result b = run_case(backend, cs[i], in, 1, nt);
        if (a.score.empty() || b.score.empty()) {
            printf("  %s: unsupported on this device\n", cs[i].str().c_str());
            n_unsup++;
            continue;
        }
        n_run++;
        const bool ok = same(a, b);
        stats(cs[i], a, n_inf_sel, n_tie_sel);
        // negative control on every fourth case (each run is a full graph)
        bool neg_ok = true;
        if (i % 4 == 0) {
            const result n = run_case(backend, cs[i], in, 2, 0);
            neg_ok = !same(a, n);
            n_neg++;
            n_neg_ok += neg_ok;
        }
        n_bad += !ok;
        if (!ok || !neg_ok || bench) {
            printf("  %s: %s%s", cs[i].str().c_str(), ok ? "equal" : "DIFF <- FAIL",
                    neg_ok ? "" : " | negative control EQUAL <- FAIL (a wrong candidate set is not seen)");
            if (!ok) {
                printf(" first: %s", first_diff(a, b).c_str());
            }
            if (nt > 0) {
                printf(" | us mask %9.1f cand %9.1f (%.2fx)", a.us, b.us, a.us / std::max(b.us, 1e-9));
            }
            printf("\n");
        }
    }
    ggml_backend_free(backend);
    printf("  %d cases run (%d unsupported), %d differ; negative control differs in %d / %d; selected -inf %ld, selected finite ties %ld\n",
            n_run, n_unsup, n_bad, n_neg_ok, n_neg, n_inf_sel, n_tie_sel);
    if (n_bad != 0 || n_run == 0 || n_neg_ok != n_neg || n_inf_sel == 0 || n_tie_sel == 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("OK: candidate scoring == lightning indexer + block mask, byte for byte (scores and top-k ids)\n");
    return 0;
}
