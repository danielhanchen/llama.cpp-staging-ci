// GET_ROWS results gathered ahead on the CPU backend (ggml_backend_cpu_get_rows_ahead) give the bytes of the plain gather: rows whose
// index matches are copied from the result, the others converted as before. Quantized tables of several types; results that match
// fully, partly (other ids, negative and out-of-range ids, a shorter result), a replaced and a dropped result. The count of rows taken
// (ggml_backend_cpu_get_rows_ahead_taken) must equal the matching rows of each case, so a silent fallback to the plain gather fails.
// Without the CPU backend's proc address it prints SKIP and exits 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using rows_ahead_fn_t = bool (*)(const ggml_tensor *, const int32_t *, int64_t, int);
using rows_taken_fn_t = int64_t (*)(void);

static rows_ahead_fn_t g_ahead = nullptr;
static rows_taken_fn_t g_taken = nullptr;

// out = table[ids] on the CPU backend (4 threads)
static std::vector<uint8_t> gather(ggml_backend_t be, ggml_tensor * table, const std::vector<int32_t> & ids, int64_t ne1_ids) {
    ggml_init_params ip = { 4*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t_ids = ne1_ids > 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, (int64_t) ids.size()/ne1_ids, ne1_ids) :
                                        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) ids.size());
    ggml_tensor * out = ggml_get_rows(ctx, table, t_ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(t_ids, ids.data(), 0, ggml_nbytes(t_ids));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }
    std::vector<uint8_t> res(ggml_nbytes(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (reg) {
        g_ahead = (rows_ahead_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_rows_ahead");
        g_taken = (rows_taken_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_rows_ahead_taken");
    }
    if (!g_ahead || !g_taken) {
        printf("SKIP: the CPU backend has no ggml_backend_cpu_get_rows_ahead\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) {
        printf("SKIP: no CPU backend\n");
        return 0;
    }
    using set_threads_fn_t = void (*)(ggml_backend_t, int);
    if (auto st = (set_threads_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")) {
        st(be, 4);
    }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t nr = 1000;
    const int64_t n  = 300;
    int n_fail = 0;
    int n_case = 0;

    for (const ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K }) {
        const int64_t nc = 256;
        ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * table = ggml_new_tensor_2d(ctx, type, nc, nr);
        ggml_set_name(table, "table");
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<float> src(nc*nr);
        for (auto & v : src) {
            v = nd(rng);
        }
        std::vector<uint8_t> q(ggml_nbytes(table));
        ggml_quantize_chunk(type, src.data(), q.data(), 0, nr, nc, nullptr);
        ggml_backend_tensor_set(table, q.data(), 0, q.size());

        std::vector<int32_t> ids(n);
        for (auto & id : ids) {
            id = (int32_t) (rng() % nr);
        }
        g_ahead(table, nullptr, 0, 0);
        const int64_t t_ref0 = g_taken();
        const std::vector<uint8_t> ref = gather(be, table, ids, 1);
        const bool ref_clean = g_taken() == t_ref0;

        auto run = [&](const char * what, const std::vector<int32_t> * ahead, int64_t ne1_ids, int64_t expect_taken) {
            n_case++;
            if (ahead) {
                if (!g_ahead(table, ahead->data(), (int64_t) ahead->size(), 3)) {
                    printf("FAIL %-8s %-34s: the result was refused\n", ggml_type_name(type), what);
                    n_fail++;
                    return;
                }
            }
            const int64_t t0 = g_taken();
            const std::vector<uint8_t> got = gather(be, table, ids, ne1_ids);
            const int64_t taken = g_taken() - t0;
            const bool same = got.size() == ref.size() && memcmp(got.data(), ref.data(), ref.size()) == 0;
            const bool ok = same && taken == expect_taken && ref_clean;
            printf("%s %-8s %-34s: bytes %s, rows taken %lld (expected %lld)\n", ok ? "OK  " : "FAIL", ggml_type_name(type), what,
                same ? "equal" : "DIFFER", (long long) taken, (long long) expect_taken);
            n_fail += !ok;
        };

        // the same ids: every row from the result
        run("same ids", &ids, 1, n);
        // other ids every third row, negative and out-of-range ids: those rows converted as before
        {
            std::vector<int32_t> a = ids;
            int64_t match = 0;
            for (int64_t i = 0; i < n; ++i) {
                if (i % 3 == 1) {
                    a[i] = (a[i] + 1) % (int32_t) nr;
                } else if (i % 7 == 2) {
                    a[i] = -1;
                } else if (i % 11 == 5) {
                    a[i] = (int32_t) (nr + i);
                }
                match += a[i] == ids[i];
            }
            run("partly other ids", &a, 1, match);
        }
        // a shorter result: its rows, then the rest as before
        {
            std::vector<int32_t> a(ids.begin(), ids.begin() + n/2);
            run("half as many rows", &a, 1, n/2);
        }
        // a result replaced before the gather: only the second counts
        {
            std::vector<int32_t> a(n, 0);
            g_ahead(table, a.data(), n, 3);
            run("replaced result", &ids, 1, n);
        }
        // dropped
        g_ahead(table, nullptr, 0, 0);
        run("dropped result", nullptr, 1, 0);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    printf("%d of %d cases %s\n", n_case - n_fail, n_case, n_fail ? "passed: FAIL" : "passed: PASS");
    return n_fail ? 1 : 0;
}
