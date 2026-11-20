// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// k19-ikfix: the CUDA mat-vec (MUL_MAT at 1..8 columns, MUL_MAT_ID at 1..4 tokens) must not read past the last row of a weight
// matrix when the last row block is partial (rows per block > 1: 2..8 columns, small K at one column, the MoE multi-token kernel; rows
// not a multiple of it). The weights of each case end right before an unmapped page: a read past their last row is an illegal address
// (the CUDA error aborts the test), so it fails without compute-sanitizer. Per case:
//   - guard: the product with the weights at the end of a mapped VMM range (an unmapped range follows);
//   - reference: the CUDA output against the CPU backend's (NMSE);
//   - bytes: the same product with the weights in an ordinary CUDA buffer gives the same bytes as the guarded one.
// --hash: ordinary buffers only, one FNV-1a hash of every case's CUDA output bytes (base vs head: the fix keeps the bytes).
// No CUDA device: SKIP, exit 0. No VMM (driver API missing): the guard is off (said so), the rest still runs.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
static bool guard_init() { return false; }
static void * guard_tail(size_t) { return nullptr; }
#else
#include <dlfcn.h>

// the driver API through dlopen (no CUDA headers or link dependency in the test); the struct layouts are those of cuda.h
typedef unsigned long long cu_ptr;
typedef unsigned long long cu_handle;
struct cu_location { int type; int id; };
struct cu_alloc_prop {
    int type; int handle_types; cu_location location; void * win32_meta;
    struct { unsigned char compression; unsigned char rdma; unsigned short usage; unsigned char reserved[4]; } flags;
};
struct cu_access { cu_location location; int flags; };

static int (*p_reserve)(cu_ptr *, size_t, size_t, cu_ptr, unsigned long long);
static int (*p_create)(cu_handle *, size_t, const cu_alloc_prop *, unsigned long long);
static int (*p_map)(cu_ptr, size_t, size_t, cu_handle, unsigned long long);
static int (*p_access)(cu_ptr, size_t, const cu_access *, size_t);
static int (*p_gran)(size_t *, const cu_alloc_prop *, int);
static int (*p_ctx_dev)(int *);

static bool guard_init() {
    void * h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        return false;
    }
    p_reserve = (decltype(p_reserve)) dlsym(h, "cuMemAddressReserve");
    p_create  = (decltype(p_create))  dlsym(h, "cuMemCreate");
    p_map     = (decltype(p_map))     dlsym(h, "cuMemMap");
    p_access  = (decltype(p_access))  dlsym(h, "cuMemSetAccess");
    p_gran    = (decltype(p_gran))    dlsym(h, "cuMemGetAllocationGranularity");
    p_ctx_dev = (decltype(p_ctx_dev)) dlsym(h, "cuCtxGetDevice");
    return p_reserve && p_create && p_map && p_access && p_gran && p_ctx_dev;
}

// device memory of at least `size` bytes whose end is followed by a reserved, unmapped range (never freed: a test process)
static void * guard_tail(size_t size) {
    int dev = 0;
    if (p_ctx_dev(&dev) != 0) {
        return nullptr;
    }
    cu_alloc_prop prop = {};
    prop.type = 1;            // CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = 1;   // CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev;
    size_t g = 0;
    if (p_gran(&g, &prop, 0) != 0 || g == 0) {
        return nullptr;
    }
    const size_t mapped = (size + g - 1)/g*g;
    cu_ptr va = 0;
    cu_handle hd = 0;
    if (p_reserve(&va, mapped + 2*g, 0, 0, 0) != 0 || p_create(&hd, mapped, &prop, 0) != 0 || p_map(va, mapped, 0, hd, 0) != 0) {
        return nullptr;
    }
    cu_access acc = {};
    acc.location = prop.location;
    acc.flags = 3;            // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    if (p_access(va, mapped, &acc, 1) != 0) {
        return nullptr;
    }
    return (void *) (va + mapped);
}
#endif

struct tcase {
    ggml_type type;
    int K, N, ncols;
    int E;      // 0: MUL_MAT, else MUL_MAT_ID with E experts (the last one selected in every token's last slot)
};

static std::vector<tcase> make_cases() {
    std::vector<tcase> v;
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q2_K,
                                GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS };
    for (ggml_type t : types) {
        for (int K : { 512, 1024, 4096 }) {
            for (int N : { 33, 67 }) {
                for (int nc : { 1, 2, 3, 5, 8 }) {
                    v.push_back({ t, K, N, nc, 0 });
                }
                for (int nt : { 1, 2, 4 }) {
                    v.push_back({ t, K, N, nt, 4 });
                }
            }
        }
    }
    return v;
}

static const int TOPK = 2;

// one product: weights a [K, N(, E)] (placed by the caller), x, ids; out
struct tgraph {
    ggml_context * ctx = nullptr, * wctx = nullptr;
    ggml_backend_buffer_t buf = nullptr, wbuf = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * a = nullptr, * out = nullptr;

    // where: 0 = an ordinary buffer of the backend, 1 = the end of a guarded range (CUDA)
    bool build(ggml_backend_t be, const tcase & c, int where, uint32_t seed) {
        ggml_init_params p = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(p);
        wctx = ggml_init(p);
        a = c.E ? ggml_new_tensor_3d(wctx, c.type, c.K, c.N, c.E) : ggml_new_tensor_2d(wctx, c.type, c.K, c.N);
        ggml_tensor * ids = nullptr, * x = nullptr;
        if (c.E) {
            x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, TOPK, c.ncols);
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, TOPK, c.ncols);
            out = ggml_mul_mat_id(ctx, a, x, ids);
        } else {
            x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.ncols);
            out = ggml_mul_mat(ctx, a, x);
        }
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) {
            return false;
        }
        if (where == 0) {
            wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
            if (!wbuf) {
                return false;
            }
        } else {
            // the weights end within 16 bytes of the unmapped range (a row is >= 66 bytes here); a's buffer is a CUDA buffer for the
            // copies, its data is the guarded memory
            const size_t nb = ggml_nbytes(a);
            char * end = (char *) guard_tail(nb + 64);
            if (!end) {
                return false;
            }
            a->data   = (void *) (((uintptr_t) end - nb) & ~(uintptr_t) 15);
            a->buffer = buf;
        }

        std::mt19937 gen(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        const int64_t nrows = (int64_t) c.N*(c.E ? c.E : 1);
        std::vector<float> f((size_t) nrows*c.K), imat(c.K, 1.0f);
        for (auto & v : f) {
            v = nd(gen);
        }
        std::vector<uint8_t> q(ggml_nbytes(a));
        ggml_quantize_chunk(c.type, f.data(), q.data(), 0, nrows, c.K, imat.data());
        ggml_backend_tensor_set(a, q.data(), 0, q.size());
        if (c.E) {
            std::vector<int32_t> ih((size_t) TOPK*c.ncols);
            for (int t = 0; t < c.ncols; ++t) {
                ih[(size_t) t*TOPK + 0] = (int32_t) (gen() % (c.E - 1));
                ih[(size_t) t*TOPK + 1] = c.E - 1;   // the last expert: its last row block ends at the end of the weights
            }
            ggml_backend_tensor_set(ids, ih.data(), 0, ih.size()*sizeof(int32_t));
        }
        std::vector<float> xh(ggml_nelements(x));
        for (auto & v : xh) {
            v = nd(gen);
        }
        ggml_backend_tensor_set(x, xh.data(), 0, xh.size()*sizeof(float));
        return true;
    }

    std::vector<float> get_out() const {
        std::vector<float> o(ggml_nelements(out));
        ggml_backend_tensor_get(out, o.data(), 0, o.size()*sizeof(float));
        return o;
    }

    ~tgraph() {
        ggml_backend_buffer_free(wbuf);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_free(wctx);
    }
};

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i])*1099511628211ull;
    }
    return h;
}

static std::string cname(const tcase & c) {
    char s[128];
    snprintf(s, sizeof(s), "%s %s K=%d N=%d %s=%d", c.E ? "MUL_MAT_ID" : "MUL_MAT", ggml_type_name(c.type), c.K, c.N,
        c.E ? "tokens" : "cols", c.ncols);
    return s;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && strstr(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "CUDA")) {
            dev = d;
            break;
        }
    }
    const bool cpu_smoke = getenv("LASTROW_CPU") != nullptr;   // harness smoke only: the CPU backend as the device, no guard
    if (cpu_smoke) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    const bool hash_only = argc >= 2 && strcmp(argv[1], "--hash") == 0;
    ggml_backend_t be  = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    const bool guard = !hash_only && !cpu_smoke && guard_init();
    const auto cases = make_cases();
    printf("device %s, %zu cases, guard %s\n", ggml_backend_dev_description(dev), cases.size(), hash_only ? "off (--hash)" : guard ? "on" : cpu_smoke ? "off (CPU smoke)" : "OFF (no driver VMM)");
    fflush(stdout);
    int n_fail = 0;
    uint64_t h_all = 1469598103934665603ull;
    uint32_t seed = 77;
    for (const auto & c : cases) {
        seed += 11;
        tgraph g0;
        if (!g0.build(be, c, 0, seed)) {
            printf("FAIL build %s\n", cname(c).c_str());
            return 1;
        }
        ggml_backend_graph_compute(be, g0.gf);
        const std::vector<float> o0 = g0.get_out();
        const uint64_t h = fnv(o0.data(), o0.size()*sizeof(float));
        h_all = fnv(&h, sizeof(h), h_all);
        if (hash_only) {
            printf("%016llx %s\n", (unsigned long long) h, cname(c).c_str());
            continue;
        }
        bool same = true;
        if (guard) {
            tgraph g1;
            if (!g1.build(be, c, 1, seed)) {
                printf("FAIL guard build %s\n", cname(c).c_str());
                return 1;
            }
            ggml_backend_graph_compute(be, g1.gf);
            ggml_backend_synchronize(be);   // an illegal address aborts here (base)
            const std::vector<float> o1 = g1.get_out();
            same = memcmp(o0.data(), o1.data(), o0.size()*sizeof(float)) == 0;
        }
        tgraph r;
        if (!r.build(cpu, c, 0, seed)) {
            printf("FAIL cpu build %s\n", cname(c).c_str());
            return 1;
        }
        ggml_backend_graph_compute(cpu, r.gf);
        const std::vector<float> ref = r.get_out();
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            num += ((double) o0[i] - ref[i])*((double) o0[i] - ref[i]);
            den += (double) ref[i]*ref[i];
        }
        const double nmse = den > 0.0 ? num/den : num;
        const bool ok = same && nmse < 5e-3 && std::isfinite(nmse);
        if (!ok) {
            n_fail++;
            printf("FAIL %s: nmse %.3g, guarded bytes %s\n", cname(c).c_str(), nmse, same ? "same" : "DIFFER");
        }
    }
    printf("hash %016llx\n", (unsigned long long) h_all);
    if (!hash_only) {
        printf("%s: %zu cases, %d failed (guard %s)\n", n_fail ? "FAIL" : "OK", cases.size(), n_fail, guard ? "on" : "off");
    }
    ggml_backend_free(cpu);
    ggml_backend_free(be);
    return n_fail ? 1 : 0;
}
