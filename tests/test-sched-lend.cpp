// test-sched-lend: the scheduler's host weight prefetch with a lender (e.g. the MoE cache) on a mock device with a fixed capacity
//
// graph A (a "large ubatch"): every host weight copy is ASYNC and placed in lent memory, so no staging arena exists afterwards
// graph B (a "medium ubatch", e.g. the last ubatch of a prompt): the same weights with only SYNC copies
// GGML_SCHED_LEND_SYNC unset (=1): when the graph allocator cannot grow by B's copies (free memory less than the growth + 512 MiB of
// headroom) the lender places them too and the graph allocator does not grow; when it can, the old path is kept (the lender is not asked);
// with GGML_SCHED_LEND_SYNC=0 (the old behaviour) the graph allocator has to hold every copy of B, which the device cannot hold;
// GGML_SCHED_LEND_SYNC=2 always offers them (a lender that places none reverts to the old path)
// (cudaMalloc failed on a 1725 MiB compute buffer after a 16k prompt at -ub 512 on RTX PRO 6000)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
// POSIX environment helpers for the MSVC CRT (getenv sees _putenv_s; an empty value removes the variable)
static int setenv(const char * name, const char * value, int /*overwrite*/) { return _putenv_s(name, value); }
static int unsetenv(const char * name) { return _putenv_s(name, ""); }
#endif

// aligned host memory for the mock device buffers
static void * aligned_alloc_mock(size_t align, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void * mem = NULL;
    return posix_memalign(&mem, align, size) == 0 ? mem : NULL;
#endif
}
static void aligned_free_mock(void * mem) {
#ifdef _WIN32
    _aligned_free(mem);
#else
    free(mem);
#endif
}

//
// mock "GPU" device: host memory behind a non-host buffer type, a capacity, events that do nothing (everything is synchronous)
//

struct mock_device_ctx {
    size_t cap  = 0; // device memory
    size_t used = 0; // live buffers
    size_t peak = 0;
    int    n_failed = 0; // allocations refused (out of memory)
};

static mock_device_ctx g_dev_ctx;

struct mock_buffer_ctx {
    void * mem;
    size_t size;
};

static const size_t k_align = 64;

static void mock_buffer_free(ggml_backend_buffer_t buffer) {
    mock_buffer_ctx * bc = (mock_buffer_ctx *) buffer->context;
    g_dev_ctx.used -= bc->size;
    aligned_free_mock(bc->mem);
    delete bc;
}
static void * mock_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((mock_buffer_ctx *) buffer->context)->mem;
}
static enum ggml_status mock_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}
static void mock_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * t, uint8_t v, size_t off, size_t size) {
    memset((char *) t->data + off, v, size);
}
static void mock_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t off, size_t size) {
    memcpy((char *) t->data + off, data, size);
}
static void mock_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t off, size_t size) {
    memcpy(data, (const char *) t->data + off, size);
}
static void mock_buffer_clear(ggml_backend_buffer_t buffer, uint8_t v) {
    mock_buffer_ctx * bc = (mock_buffer_ctx *) buffer->context;
    memset(bc->mem, v, bc->size);
}

static const ggml_backend_buffer_i mock_buffer_i = {
    /* .free_buffer   = */ mock_buffer_free,
    /* .get_base      = */ mock_buffer_get_base,
    /* .init_tensor   = */ mock_buffer_init_tensor,
    /* .memset_tensor = */ mock_buffer_memset_tensor,
    /* .set_tensor    = */ mock_buffer_set_tensor,
    /* .get_tensor    = */ mock_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ NULL,
    /* .clear         = */ mock_buffer_clear,
    /* .reset         = */ NULL,
};

static const char * mock_buft_get_name(ggml_backend_buffer_type_t) { return "MockGPU"; }
static ggml_backend_buffer_t mock_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (g_dev_ctx.used + size > g_dev_ctx.cap) {
        g_dev_ctx.n_failed++;
        fprintf(stderr, "mock: allocating %zu bytes: out of device memory (%zu of %zu used)\n", size, g_dev_ctx.used, g_dev_ctx.cap);
        return NULL;
    }
    void * mem = aligned_alloc_mock(k_align, std::max<size_t>(size, k_align));
    if (mem == NULL) {
        return NULL;
    }
    g_dev_ctx.used += size;
    g_dev_ctx.peak  = std::max(g_dev_ctx.peak, g_dev_ctx.used);
    return ggml_backend_buffer_init(buft, mock_buffer_i, new mock_buffer_ctx{ mem, size }, size);
}
static size_t mock_buft_get_alignment(ggml_backend_buffer_type_t) { return k_align; }
static bool   mock_buft_is_host(ggml_backend_buffer_type_t) { return false; }

static ggml_backend_buffer_type g_buft = {
    /* .iface = */ {
        /* .get_name         = */ mock_buft_get_name,
        /* .alloc_buffer     = */ mock_buft_alloc_buffer,
        /* .alloc_buffer_n   = */ NULL,
        /* .get_alignment    = */ mock_buft_get_alignment,
        /* .get_max_size     = */ NULL,
        /* .get_alloc_size   = */ NULL,
        /* .get_alloc_size_n = */ NULL,
        /* .is_host          = */ mock_buft_is_host,
    },
    /* .device  = */ NULL,
    /* .context = */ NULL,
};

// backend
static const char * mock_backend_get_name(ggml_backend_t) { return "MockGPU"; }
static void mock_backend_free(ggml_backend_t backend) { delete backend; }
static void mock_backend_set_tensor_async(ggml_backend_t, ggml_tensor * t, const void * data, size_t off, size_t size) {
    memcpy((char *) t->data + off, data, size);
}
static void mock_backend_get_tensor_async(ggml_backend_t, const ggml_tensor * t, void * data, size_t off, size_t size) {
    memcpy(data, (const char *) t->data + off, size);
}
static void mock_backend_synchronize(ggml_backend_t) {}
static enum ggml_status mock_backend_graph_compute(ggml_backend_t, ggml_cgraph *) {
    return GGML_STATUS_SUCCESS; // the test is about memory, not results
}
static void mock_backend_event_record(ggml_backend_t, ggml_backend_event_t) {}
static void mock_backend_event_wait(ggml_backend_t, ggml_backend_event_t) {}

static const ggml_backend_i mock_backend_i = {
    /* .get_name            = */ mock_backend_get_name,
    /* .free                = */ mock_backend_free,
    /* .set_tensor_async    = */ mock_backend_set_tensor_async,
    /* .get_tensor_async    = */ mock_backend_get_tensor_async,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ mock_backend_synchronize,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ mock_backend_graph_compute,
    /* .event_record        = */ mock_backend_event_record,
    /* .event_wait          = */ mock_backend_event_wait,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid g_guid = { 0x6d, 0x6f, 0x63, 0x6b, 0x67, 0x70, 0x75, 0x00, 0x6c, 0x65, 0x6e, 0x64, 0x00, 0x00, 0x00, 0x01 };

// device
static const char * mock_dev_get_name(ggml_backend_dev_t) { return "MockGPU0"; }
static const char * mock_dev_get_description(ggml_backend_dev_t) { return "mock GPU with a capacity"; }
static void mock_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *total = g_dev_ctx.cap;
    *free  = g_dev_ctx.cap - g_dev_ctx.used;
}
static enum ggml_backend_dev_type mock_dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void mock_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));
    props->name        = mock_dev_get_name(dev);
    props->description = mock_dev_get_description(dev);
    props->type        = GGML_BACKEND_DEVICE_TYPE_GPU;
    mock_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps.async                = true;
    props->caps.host_buffer          = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events               = true;
}
static ggml_backend_t mock_dev_init_backend(ggml_backend_dev_t dev, const char *) {
    return new ggml_backend{ &g_guid, mock_backend_i, dev, NULL };
}
static ggml_backend_buffer_type_t mock_dev_get_buffer_type(ggml_backend_dev_t) { return &g_buft; }
static bool mock_dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static bool mock_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &g_buft; }
// the matrix multiplications with host weights run on the device (their weights are copied), the rest stays on the CPU
static bool mock_dev_offload_op(ggml_backend_dev_t, const ggml_tensor * op) { return op->op == GGML_OP_MUL_MAT; }
static ggml_backend_event_t mock_dev_event_new(ggml_backend_dev_t dev) { return new ggml_backend_event{ dev, NULL }; }
static void mock_dev_event_free(ggml_backend_dev_t, ggml_backend_event_t ev) { delete ev; }
static void mock_dev_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t) {}

static const char * mock_reg_get_name(ggml_backend_reg_t) { return "MockGPU"; } // not CUDA: needs GGML_SCHED_PREFETCH=2
static size_t mock_reg_get_device_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t mock_reg_get_device(ggml_backend_reg_t, size_t);

static ggml_backend_reg g_reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface = */ {
        /* .get_name         = */ mock_reg_get_name,
        /* .get_device_count = */ mock_reg_get_device_count,
        /* .get_device       = */ mock_reg_get_device,
        /* .get_proc_address = */ NULL,
    },
    /* .context = */ NULL,
};

static ggml_backend_device g_dev = {
    /* .iface = */ {
        /* .get_name             = */ mock_dev_get_name,
        /* .get_description      = */ mock_dev_get_description,
        /* .get_memory           = */ mock_dev_get_memory,
        /* .get_type             = */ mock_dev_get_type,
        /* .get_props            = */ mock_dev_get_props,
        /* .init_backend         = */ mock_dev_init_backend,
        /* .get_buffer_type      = */ mock_dev_get_buffer_type,
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ mock_dev_supports_op,
        /* .supports_buft        = */ mock_dev_supports_buft,
        /* .offload_op           = */ mock_dev_offload_op,
        /* .event_new            = */ mock_dev_event_new,
        /* .event_free           = */ mock_dev_event_free,
        /* .event_synchronize    = */ mock_dev_event_synchronize,
    },
    /* .reg     = */ &g_reg,
    /* .context = */ NULL,
};

static ggml_backend_dev_t mock_reg_get_device(ggml_backend_reg_t, size_t) { return &g_dev; }

//
// the lender: a preallocated "cache" buffer of the device; copies whose ordinals are less than `depth` apart get different slots
//

struct lender {
    ggml_backend_buffer_t buf = NULL;
    size_t slot = 0;   // bytes per slot (the largest copy)
    int    n_slots = 0;
    int    n_calls = 0;
    int    n_placed = 0;      // copies placed by the last non-dry call
    int    n_asked = 0;       // copies asked for by the last non-dry call
    int    last_depth = 0;
    int    n_take_back = 0;
    bool   overlap = false;   // two copies live together got overlapping bytes
    bool   refuse_sync = false; // place nothing for a depth 1 request (the SYNC copies of B): the scheduler must fall back to the old path
};

static void lend_cb(ggml_backend_t, ggml_backend_sched_lend_copy * copies, int n_copies, int depth, bool dry, void * user_data) {
    lender * l = (lender *) user_data;
    if (n_copies == 0) {
        l->n_take_back += dry ? 0 : 1;
        return;
    }
    int n_asked = 0;
    int n_placed = 0;
    const bool refuse = l->refuse_sync && depth == 1;
    for (int i = 0; i < n_copies; i++) {
        ggml_backend_sched_lend_copy & c = copies[i];
        if (c.src == NULL) {
            continue; // the graph allocation (LEND=2 style): not lent here
        }
        n_asked++;
        if (refuse) {
            continue;
        }
        const int s = c.ordinal % depth;
        if (s < l->n_slots && c.size <= l->slot) {
            c.buffer = l->buf;
            c.offset = (size_t) s*l->slot;
            n_placed++;
        }
    }
    for (int i = 0; i < n_copies; i++) {
        for (int j = i + 1; j < n_copies; j++) {
            const ggml_backend_sched_lend_copy & a = copies[i];
            const ggml_backend_sched_lend_copy & b = copies[j];
            if (a.buffer == NULL || b.buffer == NULL || std::abs(a.ordinal - b.ordinal) >= depth) {
                continue;
            }
            if (a.offset < b.offset + b.size && b.offset < a.offset + a.size) {
                l->overlap = true;
            }
        }
    }
    if (!dry) {
        l->n_calls++;
        l->n_asked    = n_asked;
        l->n_placed   = n_placed;
        l->last_depth = depth;
    }
}

static enum ggml_backend_sched_prefetch_mode g_mode = GGML_BACKEND_SCHED_PREFETCH_ASYNC;

static enum ggml_backend_sched_prefetch_mode prefetch_cb(const ggml_tensor *, const ggml_tensor *, ggml_cgraph *, void *) {
    return g_mode;
}

//
// the model: n_layer host weights W (K x K f32) in a CPU buffer, h = W_l * h + b_l (the add with a CPU weight keeps a CPU split between
// the device splits, so every weight copy belongs to its own split)
//

static const int K       = 256;
static const int N_LAYER = 8;

struct model {
    ggml_context * ctx = NULL;
    ggml_backend_buffer_t buf = NULL;
    std::vector<ggml_tensor *> w, b;
};

static model make_model() {
    model m;
    ggml_init_params ip = { ggml_tensor_overhead()*(2*N_LAYER + 4), NULL, true };
    m.ctx = ggml_init(ip);
    for (int il = 0; il < N_LAYER; il++) {
        m.w.push_back(ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, K, K));
        m.b.push_back(ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, K));
    }
    m.buf = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(m.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(m.buf, 0);
    return m;
}

struct graph {
    ggml_context * ctx = NULL;
    ggml_cgraph  * gf  = NULL;
};

static graph make_graph(const model & m, int n_tokens) {
    graph g;
    ggml_init_params ip = { ggml_tensor_overhead()*(4*N_LAYER + 16) + ggml_graph_overhead(), NULL, true };
    g.ctx = ggml_init(ip);
    ggml_tensor * h = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, n_tokens);
    ggml_set_input(h);
    for (int il = 0; il < N_LAYER; il++) {
        h = ggml_mul_mat(g.ctx, m.w[il], h);
        h = ggml_add(g.ctx, h, m.b[il]);
    }
    ggml_set_output(h);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, h);
    return g;
}

struct result {
    bool   ok_a = false;
    bool   ok_b = false;
    size_t gbuf_a = 0;   // graph allocator buffer of the device after A
    size_t gbuf_b = 0;   // after B
    int    lend_b_asked = 0;
    int    lend_b_placed = 0;
    int    lend_b_depth = 0;
    int    lend_a_placed = 0;
    bool   overlap = false;
    int    n_failed = 0;
    size_t peak = 0;
};

static result run(const model & m, ggml_backend_t cpu, size_t cap, size_t copy_bytes, bool refuse_sync = false) {
    result r;
    g_dev_ctx.cap = cap;
    g_dev_ctx.n_failed = 0;
    g_dev_ctx.peak = g_dev_ctx.used;

    ggml_backend_t dev = ggml_backend_dev_init(&g_dev, NULL);

    // the cache: 2 slots of one copy each (depth 2 for the async copies of A)
    lender l;
    l.slot    = copy_bytes;
    l.n_slots = 2;
    l.refuse_sync = refuse_sync;
    l.buf     = ggml_backend_buft_alloc_buffer(&g_buft, l.slot*l.n_slots);
    GGML_ASSERT(l.buf != NULL);

    ggml_backend_t backends[2] = { dev, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_set_prefetch_callback(sched, prefetch_cb, NULL);
    ggml_backend_sched_set_lend_callback(sched, lend_cb, &l);

    // A: a large ubatch, async copies, all in lent memory
    {
        graph g = make_graph(m, 64);
        g_mode = GGML_BACKEND_SCHED_PREFETCH_ASYNC;
        r.ok_a = ggml_backend_sched_alloc_graph(sched, g.gf);
        r.lend_a_placed = l.n_placed;
        if (r.ok_a) {
            r.ok_a = ggml_backend_sched_graph_compute(sched, g.gf) == GGML_STATUS_SUCCESS;
        }
        r.gbuf_a = ggml_backend_sched_get_buffer_size(sched, dev);
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }
    // B: a medium ubatch (e.g. the last one of a prompt), only SYNC copies
    if (r.ok_a) {
        graph g = make_graph(m, 16);
        g_mode = GGML_BACKEND_SCHED_PREFETCH_SYNC;
        l.n_asked = l.n_placed = l.last_depth = 0;
        r.ok_b = ggml_backend_sched_alloc_graph(sched, g.gf);
        r.lend_b_asked  = l.n_asked;
        r.lend_b_placed = l.n_placed;
        r.lend_b_depth  = l.last_depth;
        if (r.ok_b) {
            r.ok_b = ggml_backend_sched_graph_compute(sched, g.gf) == GGML_STATUS_SUCCESS;
        }
        r.gbuf_b = ggml_backend_sched_get_buffer_size(sched, dev);
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }
    r.overlap  = l.overlap;
    r.n_failed = g_dev_ctx.n_failed;
    r.peak     = g_dev_ctx.peak;

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(l.buf);
    ggml_backend_free(dev);
    return r;
}

static void print(const char * name, const result & r) {
    printf("%-22s A ok=%d (lent %d copies) graph buffer %zu | B ok=%d graph buffer %zu, lender asked %d placed %d depth %d | overlap=%d oom=%d peak %zu\n",
        name, r.ok_a, r.lend_a_placed, r.gbuf_a, r.ok_b, r.gbuf_b, r.lend_b_asked, r.lend_b_placed, r.lend_b_depth, r.overlap, r.n_failed, r.peak);
}

int main() {
    setenv("GGML_SCHED_PREFETCH", "2", 1);      // any backend with async copies and events (the mock is not CUDA)
    setenv("GGML_SCHED_PREFETCH_ADAPT", "0", 1); // fixed depth 2

    // the CPU backend can be a dynamically loaded module (GGML_BACKEND_DL), so resolve it through the registry
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        printf("test-sched-lend: skipped, no CPU backend\n");
        return 0;
    }
    model m = make_model();
    const size_t copy_bytes = GGML_PAD(ggml_nbytes(m.w[0]), k_align);

    // 1. calibration: unlimited device, old behaviour: how large B's graph allocation gets with every copy in it
    setenv("GGML_SCHED_LEND_SYNC", "0", 1);
    result cal = run(m, cpu, SIZE_MAX/2, copy_bytes);
    print("calibration (LEND_SYNC=0)", cal);
    GGML_ASSERT(cal.ok_a && cal.ok_b);
    GGML_ASSERT(cal.lend_a_placed == N_LAYER);
    // B's graph allocation holds its copies (the allocator reuses the bytes of a copy once its split is done: at least one copy more
    // than A, which has smaller activations of its own), else this test does not model the OOM
    GGML_ASSERT(cal.gbuf_b > cal.gbuf_a + copy_bytes/2);

    // the device holds the cache and A's graph allocation (and B's activations), but not B's copies in the graph allocator
    const size_t cache = 2*copy_bytes;
    const size_t cap   = cache + cal.gbuf_a + (cal.gbuf_b - cal.gbuf_a)/2;
    printf("copy %zu bytes, cache %zu, A graph %zu, B graph with copies %zu -> device capacity %zu\n", copy_bytes, cache, cal.gbuf_a, cal.gbuf_b, cap);

    int n_fail = 0;
    auto check = [&](bool cond, const char * what) {
        printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
        n_fail += cond ? 0 : 1;
    };

    // 2. old behaviour at that capacity: B cannot be allocated
    setenv("GGML_SCHED_LEND_SYNC", "0", 1);
    result base = run(m, cpu, cap, copy_bytes);
    print("base (LEND_SYNC=0)", base);
    check(base.ok_a, "base: A allocates with every copy lent");
    check(!base.ok_b && base.n_failed > 0, "base: B fails to allocate (the copies need a new graph allocation, the OOM)");

    // 3. fix: B's SYNC copies are lent with depth 1, the graph allocator does not grow by them
    unsetenv("GGML_SCHED_LEND_SYNC");
    result fix = run(m, cpu, cap, copy_bytes);
    print("fix (default)", fix);
    check(fix.ok_a, "fix: A allocates with every copy lent");
    check(fix.ok_b, "fix: B allocates and computes");
    check(fix.n_failed == 0, "fix: no device allocation failed");
    check(fix.lend_b_asked == N_LAYER && fix.lend_b_placed == N_LAYER, "fix: the lender was asked for and placed every copy of B");
    check(fix.lend_b_depth == 1, "fix: SYNC copies are offered with depth 1");
    check(fix.gbuf_b < fix.gbuf_a + copy_bytes, "fix: the graph allocator did not grow by the copies of B");
    check(!fix.overlap, "fix: no live copies overlap");
    check(fix.peak <= cap, "fix: device memory stayed within the capacity");

    // 4. a device with room for the growth: the old path, the lender is not asked for B's copies
    result fits = run(m, cpu, SIZE_MAX/2, copy_bytes);
    print("fix, growth fits", fits);
    check(fits.ok_a && fits.ok_b && fits.n_failed == 0, "fits: A and B allocate and compute");
    check(fits.lend_b_asked == 0 && fits.lend_b_placed == 0, "fits: the lender is not asked for B's SYNC copies");
    check(fits.gbuf_b == cal.gbuf_b, "fits: B's graph allocation equals the old behaviour");

    // 5. GGML_SCHED_LEND_SYNC=2 and a lender that places none of B's copies: the scheduler reverts to the old path (graph allocator holds them)
    setenv("GGML_SCHED_LEND_SYNC", "2", 1);
    result rev = run(m, cpu, SIZE_MAX/2, copy_bytes, /*refuse_sync =*/ true);
    print("LEND_SYNC=2, lender refuses B", rev);
    check(rev.ok_a && rev.ok_b, "revert: A and B allocate and compute");
    check(rev.lend_b_asked == N_LAYER && rev.lend_b_placed == 0 && rev.lend_b_depth == 1, "revert: the lender was asked (depth 1) and placed nothing");
    check(rev.gbuf_b == cal.gbuf_b, "revert: B's graph allocation equals the old behaviour");

    ggml_backend_buffer_free(m.buf);
    ggml_free(m.ctx);
    ggml_backend_free(cpu);

    printf("%s\n", n_fail == 0 ? "test-sched-lend: OK" : "test-sched-lend: FAILED");
    return n_fail == 0 ? 0 : 1;
}
