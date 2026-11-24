// build_sampling's batched verify-window top-k (one ggml_top_k over n consecutive rows of the padded logits + one
// get_rows of every row's top logits, then a slice per row) vs what each row's chain built before (a top-k over that row alone + a
// get_rows), byte for byte: the candidate ids, their order and the logits. Every backend of the build (CPU, CUDA / ROCm when present)
// runs every case; the gate is what build_sampling batches (cparams.topk_batch: the CPU any width, CUDA past 32768 columns and at most
// 15 rows per launch): a difference inside it fails, outside it is reported only. Each case also runs the per-row graph twice (the
// device's own repeatability, inside the gate it must hold). Values: a normal draw, a coarse
// grid (many ties, also at the k-th), a masked suffix (-inf) and signed zeros. Rows 2 .. 17 (17: the CUDA one-block path for 16+ rows),
// k 1 / 10 / 20 / 40 / 256, columns 4097 .. 262144 (151936 / 248320: Qwen vocabularies).
// Control (GPU only): 3 rows of 2048 columns, where a single row takes the sorted CUB path and several rows the column-order path;
// the comparison must see that difference (it is why CUDA engages only past 4096 columns).
// With --bench, also the median us of the two graphs (n single-row top-ks + get_rows vs one batched) per device, 30 runs each.

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
    int nrows;
    int ncols;
    int k;
    int dist; // 0 normal, 1 coarse grid (ties), 2 normal with a -inf masked suffix, 3 signed zeros + grid
    std::string str() const {
        char b[128];
        snprintf(b, sizeof(b), "rows %2d cols %7d k %4d dist %d", nrows, ncols, k, dist);
        return b;
    }
};

static std::vector<tcase> cases(bool gpu) {
    std::vector<tcase> v;
    int i = 0;
    for (int nc : { 4097, 32768, 32769, 65536, 69632, 151936, 248320, 262144 }) {
        for (int k : { 1, 10, 20, 40, 256 }) {
            for (int nr : { 2, 3, 5, 8, 9, 17 }) {
                if (!gpu && (i++ % 3) != 0) {
                    continue; // the CPU: a third of the cases (its top-k is a per-row partial_sort, the slow part of the test)
                }
                v.push_back({ nr, nc, k, (nr + k + nc) % 4 });
            }
        }
    }
    if (gpu) {
        // the bench shapes: Qwen3.6 vocabulary, the server's k = 256 and Studio's top-k 20, 2..8 verify rows
        for (int nr : { 2, 3, 4, 5, 6, 7, 8 }) {
            v.push_back({ nr, 248320, 256, 0 });
            v.push_back({ nr, 248320, 20, 0 });
        }
    }
    return v;
}

static std::vector<float> make_data(const tcase & c, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // the padded logits: the window starts at row 1 (as a verify window after a prompt row) and a padding row follows
    const int n_tot = c.nrows + 2;
    std::vector<float> x((size_t) n_tot * c.ncols);
    for (int r = 0; r < n_tot; ++r) {
        const int n_vis = c.dist == 2 ? c.ncols - (int) (rng() % (c.ncols / 3 + 1)) : c.ncols;
        for (int i = 0; i < c.ncols; ++i) {
            float v = nd(rng);
            if (c.dist == 1 || c.dist == 3) {
                v = std::round(v * 4.0f) / 4.0f;
            }
            if (c.dist == 3 && v == 0.0f && (rng() & 1)) {
                v = -0.0f;
            }
            if (c.dist == 2 && i >= n_vis) {
                v = -INFINITY;
            }
            x[(size_t) r * c.ncols + i] = v;
        }
    }
    return x;
}

struct result {
    std::vector<int32_t> ids;    // k x nrows
    std::vector<float>   logits; // k x nrows
    double us = 0.0;
};

// batched: the graph of build_sampling with cparams.topk_batch; else each row's chain as before (llama_sampler_top_k_backend_apply)
static result run_case(ggml_backend_t backend, const tcase & c, const std::vector<float> & x, bool batched, int n_timed) {
    ggml_init_params ip = { (size_t) (16 + 8 * c.nrows) * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ncols, c.nrows + 2);
    ggml_set_input(all);
    ggml_tensor * logits_t = ggml_pad(ctx, all, 0, 1, 0, 0); // as build_sampling's logits_t (a computed tensor, not the input)
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> ids(c.nrows);
    std::vector<ggml_tensor *> lgs(c.nrows);
    const int row0 = 1;
    if (batched) {
        ggml_tensor * xw = ggml_view_2d(ctx, logits_t, c.ncols, c.nrows, logits_t->nb[1], row0 * logits_t->nb[1]);
        ggml_tensor * tk = ggml_top_k(ctx, xw, c.k);
        ggml_tensor * tl = ggml_get_rows(ctx, ggml_reshape_3d(ctx, xw, 1, c.ncols, c.nrows), tk);
        for (int i = 0; i < c.nrows; ++i) {
            ids[i] = ggml_view_1d(ctx, tk, c.k, i * tk->nb[1]);
            lgs[i] = ggml_view_2d(ctx, tl, 1, c.k, tl->nb[1], i * tl->nb[2]);
        }
    } else {
        for (int i = 0; i < c.nrows; ++i) {
            ggml_tensor * row = ggml_view_1d(ctx, logits_t, c.ncols, (row0 + i) * logits_t->nb[1]);
            ggml_tensor * l1  = ggml_reshape_1d(ctx, row, c.ncols);
            ids[i] = ggml_top_k(ctx, l1, c.k);
            lgs[i] = ggml_get_rows(ctx, ggml_reshape_2d(ctx, l1, 1, c.ncols), ids[i]);
        }
    }
    for (int i = 0; i < c.nrows; ++i) {
        ggml_set_output(ids[i]);
        ggml_set_output(lgs[i]);
        ggml_build_forward_expand(gf, ids[i]);
        ggml_build_forward_expand(gf, lgs[i]);
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    ggml_backend_tensor_set(all, x.data(), 0, ggml_nbytes(all));
    ggml_backend_graph_compute(backend, gf);
    result r;
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
    r.ids.resize((size_t) c.k * c.nrows);
    r.logits.resize((size_t) c.k * c.nrows);
    for (int i = 0; i < c.nrows; ++i) {
        ggml_backend_tensor_get(ids[i], r.ids.data()    + (size_t) i * c.k, 0, c.k * sizeof(int32_t));
        ggml_backend_tensor_get(lgs[i], r.logits.data() + (size_t) i * c.k, 0, c.k * sizeof(float));
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return r;
}

static bool same(const result & a, const result & b) {
    return a.ids == b.ids && a.logits.size() == b.logits.size() &&
           memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0;
}

// the first difference: row, position, id and logit of each side
static std::string first_diff(const tcase & c, const result & a, const result & b) {
    for (size_t i = 0; i < a.ids.size() && i < b.ids.size(); ++i) {
        if (a.ids[i] != b.ids[i] || memcmp(&a.logits[i], &b.logits[i], sizeof(float)) != 0) {
            char buf[160];
            snprintf(buf, sizeof(buf), "row %d pos %d: id %d / %d logit %g / %g", (int) (i / c.k), (int) (i % c.k), a.ids[i], b.ids[i],
                    a.logits[i], b.logits[i]);
            return buf;
        }
    }
    return "sizes differ";
}

// where build_sampling batches on this device (llama_context: cparams.topk_batch_min_cols / topk_batch_max_rows)
static bool in_gate(const tcase & c, bool gpu) {
    return !gpu || (c.ncols > 32768 && c.nrows <= 15);
}

int main(int argc, char ** argv) {
    const bool bench = argc > 1 && strcmp(argv[1], "--bench") == 0;
    ggml_backend_load_all();
    int n_bad = 0;
    int n_run = 0;
    int n_gpu = 0;
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        const auto type = ggml_backend_dev_type(dev);
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        const bool gpu = type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        if (gpu && strcmp(reg, "CUDA") != 0 && strcmp(reg, "ROCm") != 0) {
            continue;
        }
        if (!gpu && type != GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (gpu && n_gpu++ > 0) {
            continue; // the first GPU is enough
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            continue;
        }
        printf("device %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        const auto cs = cases(gpu);
        int n_dev_bad = 0;
        int n_out_gate = 0;
        for (size_t i = 0; i < cs.size(); ++i) {
            const std::vector<float> x = make_data(cs[i], 11 + (int) i);
            const int nt = bench && gpu ? 30 : 0;
            const result a  = run_case(backend, cs[i], x, false, nt);
            const result b  = run_case(backend, cs[i], x, true,  nt);
            const result a2 = run_case(backend, cs[i], x, false, 0); // the per-row graph again: is the device's top-k repeatable here?
            const bool ok   = same(a, b);
            const bool rep  = same(a, a2);
            const bool gate = in_gate(cs[i], gpu);
            n_dev_bad += gate && (!ok || !rep);
            n_out_gate += !gate && !ok;
            n_run++;
            if (!ok || !rep || bench) {
                printf("  %s: %s%s%s", cs[i].str().c_str(), ok ? "equal" : "DIFF", rep ? "" : " (per-row not repeatable)",
                        !gate ? " [outside the gate: not batched on this device]" : (ok && rep ? "" : "  <- FAIL"));
                if (!ok) {
                    printf(" first: %s", first_diff(cs[i], a, b).c_str());
                }
                if (nt > 0) {
                    printf(" | us per-row %8.1f batched %8.1f", a.us, b.us);
                }
                printf("\n");
            }
        }
        if (gpu && strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0) {
            // control: past the gate's other side the order of a single row and of several rows differ, and the check sees it
            // (CUDA only: ROCm gfx1151 returns the per-row order for several rows too, so there is no order change to see)
            const tcase cc = { 3, 2048, 20, 0 };
            const std::vector<float> x = make_data(cc, 5);
            const bool ok = !same(run_case(backend, cc, x, false, 0), run_case(backend, cc, x, true, 0));
            printf("  control %s (single row sorted, several rows column order): %s\n", cc.str().c_str(),
                    ok ? "differs as expected" : "EQUAL <- the comparison cannot see an order change (FAIL)");
            n_dev_bad += !ok;
        }
        printf("  %s: %zu cases, %d differ inside the gate (%d differ outside it, not batched there)\n", ggml_backend_dev_name(dev), cs.size(),
                n_dev_bad, n_out_gate);
        n_bad += n_dev_bad;
        ggml_backend_free(backend);
    }
    if (n_bad != 0 || n_run == 0) {
        printf("FAIL: %d cases differ (%d run)\n", n_bad, n_run);
        return 1;
    }
    printf("OK: %d cases byte-identical (ids, order, logits)%s\n", n_run, n_gpu ? "" : "; no CUDA / ROCm device (CPU only)");
    return 0;
}
