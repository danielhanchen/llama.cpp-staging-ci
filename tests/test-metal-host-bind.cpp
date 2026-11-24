// the Metal host bank binding (proc address "ggml_backend_metal_alias_host", used by the MoE copy callback on unified memory): a
// Metal tensor that holds zeros is bound to host memory at page and non-page offsets, a graph that reads it (the tensor and a view
// of it) must return the host bytes, the next graph (nothing bound) must read the zeros again, and the tensor itself is never
// written. The default arm sets GGML_METAL_HOST_BANK_BIND=1 (the bind is off by default);
// with --off (run under GGML_METAL_HOST_BANK_BIND=0 or unset) the bind must be refused and every graph reads the zeros.
// SKIP without a Metal device; exit 1 on any difference.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>

using alias_host_t = bool (*)(ggml_backend_t, const ggml_tensor *, const void *, size_t);

// out = cpy(bank) and view_out = cpy(view of bank at view_off elements), both computed on be; returns the two outputs
static void run_graph(ggml_backend_t be, ggml_tensor * bank, int64_t n, int64_t view_off, int64_t view_n,
        std::vector<float> & out, std::vector<float> & out_view) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * dst  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * dstv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, view_n);
    ggml_tensor * v    = ggml_view_1d(ctx, bank, view_n, view_off*sizeof(float));
    ggml_tensor * c0   = ggml_cpy(ctx, bank, dst);
    ggml_tensor * c1   = ggml_cpy(ctx, v, dstv);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c0);
    ggml_build_forward_expand(gf, c1);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }
    ggml_backend_synchronize(be);
    out.resize(n);
    out_view.resize(view_n);
    ggml_backend_tensor_get(dst,  out.data(),      0, n*sizeof(float));
    ggml_backend_tensor_get(dstv, out_view.data(), 0, view_n*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static int run(int argc, char ** argv) {
    const bool off = argc > 1 && strcmp(argv[1], "--off") == 0;

    // the bind is off by default: the default arm enables it, --off keeps whatever the environment says (=0 or unset)
    if (!off) {
        setenv("GGML_METAL_HOST_BANK_BIND", "1", 1);
    }

    ggml_backend_load_all();

    int n_dev   = 0;
    int n_cases = 0;
    int n_bind  = 0;
    int n_bad   = 0;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        if (strncmp(ggml_backend_dev_name(dev), "MTL", 3) != 0) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto fn = (alias_host_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_alias_host");
        if (fn == nullptr) {
            printf("%s: no ggml_backend_metal_alias_host: FAIL\n", ggml_backend_dev_name(dev));
            n_bad++;
            continue;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (be == nullptr) {
            continue;
        }
        n_dev++;

        const size_t page    = (size_t) sysconf(_SC_PAGESIZE);
        const size_t map_len = (size_t) 40 << 20;
        uint8_t * host = (uint8_t *) mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (host == MAP_FAILED) {
            fprintf(stderr, "mmap failed\n");
            return 1;
        }
        std::mt19937 rng(7);

        for (int64_t n : { (int64_t) 1000, (int64_t) 65536, (int64_t) (8 << 20) / 4 + 3 }) {
            for (size_t src_off : { (size_t) 0, (size_t) 4, (size_t) 100, page + 4 }) {
                ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * bank = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
                ggml_set_name(bank, "bank");
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
                ggml_backend_buffer_clear(buf, 0);

                float * src = (float *) (host + src_off);
                for (int64_t i = 0; i < n; ++i) {
                    src[i] = (float) (int32_t) (rng() % 2000001) - 1000000.0f;
                }
                const int64_t view_off = n/3;
                const int64_t view_n   = n - view_off;

                const bool bound = fn(be, bank, src, n*sizeof(float));
                n_bind += bound;
                std::vector<float> o0, o0v, o1, o1v;
                run_graph(be, bank, n, view_off, view_n, o0, o0v);
                // nothing bound for the next graph
                run_graph(be, bank, n, view_off, view_n, o1, o1v);

                int64_t bad = 0;
                if (off && bound) {
                    bad++;
                    printf("  bound under GGML_METAL_HOST_BANK_BIND=0\n");
                }
                for (int64_t i = 0; i < n; ++i) {
                    const float want0 = bound ? src[i] : 0.0f;
                    bad += memcmp(&o0[i], &want0, sizeof(float)) != 0;
                    bad += o1[i] != 0.0f;
                }
                for (int64_t i = 0; i < view_n; ++i) {
                    const float want0 = bound ? src[view_off + i] : 0.0f;
                    bad += memcmp(&o0v[i], &want0, sizeof(float)) != 0;
                    bad += o1v[i] != 0.0f;
                }
                // the bound tensor itself still holds the zeros
                std::vector<float> own(n);
                ggml_backend_tensor_get(bank, own.data(), 0, n*sizeof(float));
                for (int64_t i = 0; i < n; ++i) {
                    bad += own[i] != 0.0f;
                }
                n_cases++;
                printf("  n %8lld src_off %5zu: %s, %lld differ%s\n", (long long) n, src_off, bound ? "bound" : "not bound", (long long) bad, "");
                if (bad != 0) {
                    n_bad++;
                }
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
        }
        munmap(host, map_len);
        ggml_backend_free(be);
    }

    if (n_dev == 0) {
        printf("SKIP: no Metal device\n");
        return 0;
    }
    printf("%s: %d cases, %d bound, %d with differences%s\n", n_bad == 0 && (off ? n_bind == 0 : true) ? "OK" : "FAIL", n_cases, n_bind, n_bad,
            off ? " (GGML_METAL_HOST_BANK_BIND=0)" : "");
    // with the bind on, a unified-memory device must bind every case (the hosted runners are unified memory)
    if (!off && n_bind == 0) {
        printf("FAIL: nothing was bound\n");
        return 1;
    }
    return n_bad == 0 ? 0 : 1;
}
#endif

int main(int argc, char ** argv) {
#if defined(__APPLE__)
    return run(argc, argv);
#else
    (void) argc;
    (void) argv;
    printf("SKIP: no Metal device\n");
    return 0;
#endif
}
