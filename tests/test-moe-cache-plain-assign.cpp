// k19-q4xcache: the hybrid assign at fetch share 1 with ggml_moe_cache_assign_lookup_counts (+ its fetch) stands in for the plain
// device lookup (ggml_moe_cache_lookup) of a verify window whose host path is off (LLAMA_MOE_OVERLAP_SMALL_PLAIN). On a GPU backend
// that supports GGML_OP_MOE_CACHE_LOOKUP, two copies of one cache state run the same sequence of windows (1..8 tokens, CLOCK and
// decayed counts, route tail skip off / rank 7): one through the lookup, one through the assign. After every window the slots, the
// copy list, the slot map, the state (hand, epoch, counters, keys, reference bits, pins), the scores, the use counters and the bank
// bytes must be identical. Negative control: the assign without ggml_moe_cache_assign_lookup_counts writes the use counters.
// Skipped (exit 0) without such a backend.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// defined in ggml.c (k19-q4xcache), declared by its users
extern "C" { GGML_API void ggml_moe_cache_assign_lookup_counts(struct ggml_tensor * op); }

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { if (g_fail++ < 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static const int HDR       = 8;
static const int N_EXPERT  = 128;
static const int NE0       = 8;
static const int N_LAYER   = 2;
static const int IL        = 1;
static const int N_SLOTS   = 72; // at least the distinct ids of an 8-token window (64)
static const int ROW_BYTES = 64;

struct cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * map = nullptr, * state = nullptr, * counts = nullptr, * scores = nullptr, * bank = nullptr, * src = nullptr;

    void init(ggml_backend_t be) {
        ctx    = ggml_init({ 16*ggml_tensor_overhead(), nullptr, true });
        map    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_EXPERT, N_LAYER);
        state  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, HDR + 3*N_SLOTS);
        counts = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_EXPERT, N_LAYER + 1);
        scores = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_EXPERT, N_LAYER);
        bank   = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, ROW_BYTES, 1, N_SLOTS);
        src    = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, ROW_BYTES, 1, N_EXPERT); // the "host" experts (device memory here)
        buf    = ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<int32_t> m((size_t) N_EXPERT*N_LAYER, -1);
        std::vector<int32_t> st((size_t) HDR + 3*N_SLOTS, 0);
        for (int i = 0; i < N_SLOTS; ++i) st[HDR + i] = -1;
        std::vector<int32_t> z((size_t) N_EXPERT*(N_LAYER + 1), 0);
        ggml_backend_tensor_set(map, m.data(), 0, ggml_nbytes(map));
        ggml_backend_tensor_set(state, st.data(), 0, ggml_nbytes(state));
        ggml_backend_tensor_set(counts, z.data(), 0, ggml_nbytes(counts));
        ggml_backend_tensor_set(scores, z.data(), 0, ggml_nbytes(scores));
        ggml_backend_tensor_memset(bank, 0, 0, ggml_nbytes(bank));
        std::vector<int8_t> s((size_t) ROW_BYTES*N_EXPERT);
        for (int e = 0; e < N_EXPERT; ++e) for (int b = 0; b < ROW_BYTES; ++b) s[(size_t) e*ROW_BYTES + b] = (int8_t) (e*7 + b);
        ggml_backend_tensor_set(src, s.data(), 0, ggml_nbytes(src));
    }
    void free_() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    template <typename T> static std::vector<T> get(const ggml_tensor * t) {
        std::vector<T> v(ggml_nbytes(t)/sizeof(T));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        return v;
    }
};

// one window through the lookup (form 0) or the assign at share 1 with (form 1) / without (form 2) the lookup's counters
static std::vector<int32_t> step(ggml_backend_t be, cache & c, const std::vector<int32_t> & ids_h, int n_tok, int form, bool decay, int rank, bool advance) {
    ggml_context * ictx = ggml_init({ 4*ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * ids = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, NE0, n_tok);
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors(ictx, be);
    ggml_backend_tensor_set(ids, ids_h.data(), 0, ggml_nbytes(ids));

    ggml_context * gctx = ggml_init({ 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
    ggml_cgraph * gf = ggml_new_graph(gctx);
    const void * host[1] = { c.src->data };
    ggml_tensor * banks[1] = { c.bank };
    ggml_tensor * op = nullptr;
    if (form == 0) {
        op = ggml_moe_cache_lookup(gctx, ids, c.map, c.state, banks, host, 1, IL);
    } else {
        op = ggml_moe_cache_assign(gctx, ids, c.map, c.state, c.counts, 1.0f, IL);
        if (form == 1) {
            ggml_moe_cache_assign_lookup_counts(op);
        }
    }
    if (decay) {
        ggml_moe_cache_set_scores(op, c.scores, 69901, advance);
    }
    ggml_moe_cache_set_tail_skip(op, rank, false, c.counts);
    ggml_build_forward_expand(gf, op);
    ggml_tensor * fetched = nullptr;
    if (form != 0) {
        fetched = ggml_moe_cache_fetch(gctx, op, banks, host, 1, false);
        ggml_build_forward_expand(gf, fetched);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    if (!ggml_backend_supports_op(be, op)) {
        printf("test-moe-cache-plain-assign: backend does not support the lookup, skipped\n");
        exit(0);
    }
    ggml_backend_graph_compute(be, gf);
    std::vector<int32_t> out = cache::get<int32_t>(op);
    const int n_ids = NE0*n_tok;
    // slots, copy count, copied experts and their slots (the rows the lookup and the assign share)
    std::vector<int32_t> r(out.begin(), out.begin() + n_ids + 1);
    const int n_fill = out[n_ids];
    for (int j = 0; j < n_fill; ++j) r.push_back(out[(size_t) n_ids + NE0 + j]);
    for (int j = 0; j < n_fill; ++j) r.push_back(out[(size_t) 2*n_ids + NE0 + j]);
    if (fetched != nullptr) {
        const std::vector<int32_t> f = cache::get<int32_t>(fetched);
        for (int i = 0; i < n_ids; ++i) CHECK(f[i] == out[i], "fetch result %d: %d vs %d", i, f[i], out[i]);
    }
    ggml_gallocr_free(ga);
    ggml_free(gctx);
    ggml_backend_buffer_free(ibuf);
    ggml_free(ictx);
    return r;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (be == nullptr) {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (be == nullptr) {
        printf("test-moe-cache-plain-assign: no GPU backend, skipped\n");
        return 0;
    }
    printf("test-moe-cache-plain-assign: backend %s\n", ggml_backend_name(be));
    std::mt19937 rng(4321);
    int n_windows = 0, n_copies = 0, n_ctrl_diff = 0;
    for (int seq = 0; seq < 24; ++seq) {
        const bool decay = seq % 2 == 1;
        const int  rank  = (seq / 2) % 3 == 2 ? 7 : 0;
        cache a, b, ctl;
        a.init(be); b.init(be); ctl.init(be);
        for (int w = 0; w < 60; ++w) {
            const int n_tok = w % 5 == 4 ? 8 : 1 + (int) (rng() % 4);
            std::vector<int32_t> ids((size_t) NE0*n_tok);
            for (int t = 0; t < n_tok; ++t) {
                std::vector<int> perm(N_EXPERT);
                for (int e = 0; e < N_EXPERT; ++e) perm[e] = e;
                for (int j = 0; j < NE0; ++j) {
                    // a skewed pick so windows share hot experts, distinct within a token
                    const int span = rng() % 4 == 0 ? N_EXPERT - j : std::max(1, 24 - j);
                    std::swap(perm[j], perm[std::min(j + (int) (rng() % (uint32_t) span), N_EXPERT - 1)]);
                    ids[(size_t) t*NE0 + j] = perm[j];
                }
            }
            const bool adv = w % 3 == 0;
            const std::vector<int32_t> ra = step(be, a, ids, n_tok, 0, decay, rank, adv);
            const std::vector<int32_t> rb = step(be, b, ids, n_tok, 1, decay, rank, adv);
            step(be, ctl, ids, n_tok, 2, decay, rank, adv);
            n_windows++;
            n_copies += ra[(size_t) NE0*n_tok];
            CHECK(ra == rb, "seq %d window %d (%d tokens, decay %d, rank %d): slots / copies differ", seq, w, n_tok, decay, rank);
            CHECK(cache::get<int32_t>(a.map) == cache::get<int32_t>(b.map), "seq %d window %d: map differs", seq, w);
            CHECK(cache::get<int32_t>(a.state) == cache::get<int32_t>(b.state), "seq %d window %d: state differs", seq, w);
            CHECK(cache::get<int32_t>(a.scores) == cache::get<int32_t>(b.scores), "seq %d window %d: scores differ", seq, w);
            CHECK(cache::get<int32_t>(a.counts) == cache::get<int32_t>(b.counts), "seq %d window %d: use counters differ", seq, w);
            CHECK(cache::get<int8_t>(a.bank) == cache::get<int8_t>(b.bank), "seq %d window %d: bank bytes differ", seq, w);
            n_ctrl_diff += cache::get<int32_t>(a.counts) != cache::get<int32_t>(ctl.counts);
        }
        a.free_(); b.free_(); ctl.free_();
    }
    // negative control: without the flag the assign writes the use counters, so they must differ from the lookup's
    CHECK(n_ctrl_diff > 0, "negative control: the assign without ggml_moe_cache_assign_lookup_counts never changed the use counters");
    printf("test-moe-cache-plain-assign: %d windows, %d experts copied, negative control differs in %d windows, %d failures\n",
        n_windows, n_copies, n_ctrl_diff, g_fail);
    ggml_backend_free(be);
    return g_fail == 0 ? 0 : 1;
}
