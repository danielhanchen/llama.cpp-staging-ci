// route tail skip of the device MoE cache lookup (ggml_moe_cache_set_tail_skip): on a GPU backend that supports
// GGML_OP_MOE_CACHE_LOOKUP, random windows of 1..8 tokens are looked up with the skip off and at ranks 1..7, in the three forms the
// graph builds (frozen assign, assign with a fetch share, plain lookup with copies), and checked against a host reference of the rule:
// a missing expert (not cached before the lookup) every entry of which has a rank of at least the threshold gets slot -1 and host id
// -1, is not admitted, fetched or counted as a miss, its map entry stays -1, and the skip counter (stat[4..5] of the use counters)
// counts its entries; every other entry is as with the skip off. Skipped (exit 0) without such a backend.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { if (g_fail++ < 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static const int HDR = 8;

struct setup {
    int n_expert, ne0, n_tok, n_layer, n_slots, il;
    std::vector<int32_t> ids, map, state, counts;
};

static setup make_setup(std::mt19937 & rng, int n_tok) {
    setup s;
    s.n_expert = 32;
    s.ne0      = 8;
    s.n_tok    = n_tok;
    s.n_layer  = 2;
    s.n_slots  = 80; // at least the distinct ids of a window
    s.il       = 1;
    s.ids.resize((size_t) s.ne0*n_tok);
    // a few hot experts so the windows share some of them, ids distinct within a token
    for (int t = 0; t < n_tok; ++t) {
        std::vector<int> perm(s.n_expert);
        for (int e = 0; e < s.n_expert; ++e) perm[e] = e;
        for (int j = 0; j < s.ne0; ++j) {
            const int pick = j + (int) (rng() % (uint32_t) (rng() % 3 == 0 ? s.n_expert - j : std::max(1, 12 - j)));
            std::swap(perm[j], perm[std::min(pick, s.n_expert - 1)]);
            s.ids[(size_t) t*s.ne0 + j] = perm[j];
        }
    }
    s.map.assign((size_t) s.n_expert*s.n_layer, -1);
    s.state.assign((size_t) HDR + 3*s.n_slots, 0);
    int32_t * key_of = s.state.data() + HDR;
    for (int i = 0; i < s.n_slots; ++i) key_of[i] = -1;
    // about a third of the experts of the layer cached, in random slots
    int next = 0;
    for (int e = 0; e < s.n_expert; ++e) {
        if (rng() % 3 == 0 && next < s.n_slots/2) {
            const int slot = next++;
            s.map[(size_t) s.il*s.n_expert + e] = slot;
            key_of[slot] = s.il*s.n_expert + e;
        }
    }
    s.state[6] = next; // filled slots
    s.counts.assign((size_t) s.n_expert*(s.n_layer + 1), 0);
    return s;
}

// host reference: skipped entries
static std::vector<bool> ref_skip(const setup & s, int rank) {
    std::vector<bool> sk(s.ids.size(), false);
    if (rank <= 0) return sk;
    for (size_t i = 0; i < s.ids.size(); ++i) {
        const int e = s.ids[i];
        if (s.map[(size_t) s.il*s.n_expert + e] >= 0 || (int) (i % s.ne0) < rank) continue;
        bool all = true;
        for (size_t j = 0; j < s.ids.size(); ++j) {
            if (s.ids[j] == e && (int) (j % s.ne0) < rank) all = false;
        }
        sk[i] = all;
    }
    return sk;
}

struct result {
    std::vector<int32_t> out, map, state, counts, bank;
};

// form 0: lookup with copies, 1: assign (fetch share 1, then the fetch), 2: frozen assign
static result run(ggml_backend_t be, const setup & s, int form, int rank) {
    const int64_t row_bytes = 64; // one bank, an "expert" of 64 bytes
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.ne0, s.n_tok);
    ggml_tensor * map    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.n_expert, s.n_layer);
    ggml_tensor * state  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, HDR + 3*s.n_slots);
    ggml_tensor * counts = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.n_expert, s.n_layer + 1);
    ggml_tensor * bank   = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, row_bytes, 1, s.n_slots + 1);
    ggml_tensor * src    = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, row_bytes, 1, s.n_expert); // the "host" experts (device memory here)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(ids, s.ids.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(map, s.map.data(), 0, ggml_nbytes(map));
    ggml_backend_tensor_set(state, s.state.data(), 0, ggml_nbytes(state));
    ggml_backend_tensor_set(counts, s.counts.data(), 0, ggml_nbytes(counts));
    ggml_backend_tensor_memset(bank, 0, 0, ggml_nbytes(bank));
    std::vector<int8_t> srcd((size_t) row_bytes*s.n_expert);
    for (int e = 0; e < s.n_expert; ++e) for (int b = 0; b < row_bytes; ++b) srcd[(size_t) e*row_bytes + b] = (int8_t) (e + 1);
    ggml_backend_tensor_set(src, srcd.data(), 0, ggml_nbytes(src));

    ggml_context * gctx = ggml_init({ 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
    ggml_cgraph * gf = ggml_new_graph(gctx);
    const void * host[1] = { src->data };
    ggml_tensor * banks[1] = { bank };
    ggml_tensor * op = nullptr;
    ggml_tensor * out = nullptr;
    if (form == 0) {
        op = ggml_moe_cache_lookup(gctx, ids, map, state, banks, host, 1, s.il);
        ggml_moe_cache_set_tail_skip(op, rank, false, counts);
        out = op;
    } else {
        op = ggml_moe_cache_assign(gctx, ids, map, state, counts, form == 2 ? 0.0f : 1.0f, s.il);
        if (form == 2) ggml_moe_cache_assign_freeze(op);
        ggml_moe_cache_set_tail_skip(op, rank, false, counts);
        out = op;
    }
    ggml_build_forward_expand(gf, out);
    ggml_tensor * fetched = nullptr;
    if (form == 1) {
        fetched = ggml_moe_cache_fetch(gctx, op, banks, host, 1, false);
        ggml_build_forward_expand(gf, fetched);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    if (!ggml_backend_supports_op(be, op)) {
        printf("backend does not support the lookup\n");
        exit(0);
    }
    ggml_backend_graph_compute(be, gf);
    result r;
    r.out.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, r.out.data(), 0, ggml_nbytes(out));
    r.map.resize(s.map.size());       ggml_backend_tensor_get(map, r.map.data(), 0, ggml_nbytes(map));
    r.state.resize(s.state.size());   ggml_backend_tensor_get(state, r.state.data(), 0, ggml_nbytes(state));
    r.counts.resize(s.counts.size()); ggml_backend_tensor_get(counts, r.counts.data(), 0, ggml_nbytes(counts));
    std::vector<int8_t> bk(ggml_nbytes(bank)); ggml_backend_tensor_get(bank, bk.data(), 0, ggml_nbytes(bank));
    r.bank.resize(s.n_slots + 1);
    for (int i = 0; i <= s.n_slots; ++i) r.bank[i] = bk[(size_t) i*row_bytes];
    if (form == 1) {
        std::vector<int32_t> f(ggml_nelements(fetched));
        ggml_backend_tensor_get(fetched, f.data(), 0, ggml_nbytes(fetched));
        for (size_t i = 0; i < f.size(); ++i) CHECK(f[i] == r.out[i], "fetch result %zu: %d vs %d", i, f[i], r.out[i]);
    }
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    ggml_free(ctx);
    return r;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (be == nullptr) {
        printf("test-moe-cache-tail-skip: no GPU backend, skipped\n");
        return 0;
    }
    printf("test-moe-cache-tail-skip: backend %s\n", ggml_backend_name(be));
    std::mt19937 rng(1234);
    int n_cases = 0, n_skipped = 0;
    for (int iter = 0; iter < 200; ++iter) {
        const int n_tok = 1 + iter % 8;
        const setup s = make_setup(rng, n_tok);
        const int n_ids = s.ne0*n_tok;
        for (int form = 0; form < 3; ++form) {
            const result r0 = run(be, s, form, 0);
            for (int rank = 1; rank <= 7; ++rank) {
                const std::vector<bool> sk = ref_skip(s, rank);
                const result r = run(be, s, form, rank);
                int n_sk = 0;
                std::set<int> skipped_e;
                for (int i = 0; i < n_ids; ++i) {
                    if (sk[i]) { n_sk++; skipped_e.insert(s.ids[i]); }
                }
                n_cases++;
                n_skipped += n_sk;
                const uint64_t stat = (uint64_t) (uint32_t) r.counts[(size_t) s.n_layer*s.n_expert + 4] |
                    (uint64_t) (uint32_t) r.counts[(size_t) s.n_layer*s.n_expert + 5] << 32;
                CHECK(stat == (uint64_t) n_sk, "form %d rank %d: skip counter %llu, expected %d", form, rank, (unsigned long long) stat, n_sk);
                for (int i = 0; i < n_ids; ++i) {
                    const int e = s.ids[i];
                    const int cached = s.map[(size_t) s.il*s.n_expert + e];
                    if (sk[i]) {
                        CHECK(r.out[i] == -1, "form %d rank %d: skipped entry %d has slot %d", form, rank, i, r.out[i]);
                        CHECK(r.map[(size_t) s.il*s.n_expert + e] == -1, "form %d rank %d: skipped expert %d in the map", form, rank, e);
                        if (form != 0) {
                            CHECK(r.out[(size_t) (3*n_tok + 1)*s.ne0 + i] == -1, "form %d rank %d: skipped entry %d left to the host", form, rank, i);
                            CHECK(r.out[(size_t) (4*n_tok + 1)*s.ne0 + i] == -1, "form %d rank %d: skipped entry %d hit slot", form, rank, i);
                        }
                    } else if (skipped_e.empty()) {
                        CHECK(r.out == r0.out, "form %d rank %d: nothing skipped but the result differs", form, rank);
                    } else {
                        // a kept entry: a hit keeps its slot, a miss is admitted / left to the host as in the unskipped rule
                        if (cached >= 0) {
                            CHECK(r.out[i] == cached, "form %d rank %d: hit %d slot %d vs %d", form, rank, i, r.out[i], cached);
                        } else if (form == 2) {
                            CHECK(r.out[i] == -1 && r.out[(size_t) (3*n_tok + 1)*s.ne0 + i] == e, "form 2 rank %d: miss %d not on the host", rank, i);
                        } else {
                            CHECK(r.out[i] >= 0 && r.bank[r.out[i]] == (int8_t) (e + 1), "form %d rank %d: miss %d slot %d bank %d", form, rank, i, r.out[i],
                                r.out[i] >= 0 ? r.bank[r.out[i]] : -999);
                        }
                    }
                }
                if (form != 2) {
                    // copies: the distinct missing experts that are not skipped
                    std::set<int> miss;
                    for (int i = 0; i < n_ids; ++i) if (!sk[i] && s.map[(size_t) s.il*s.n_expert + s.ids[i]] < 0) miss.insert(s.ids[i]);
                    CHECK(r.out[(size_t) n_ids] == (int) miss.size(), "form %d rank %d: %d copies, expected %zu", form, rank, r.out[(size_t) n_ids], miss.size());
                    const uint64_t misses = (uint64_t) (uint32_t) r.state[4] | (uint64_t) (uint32_t) r.state[5] << 32;
                    CHECK(misses == miss.size(), "form %d rank %d: miss counter %llu", form, rank, (unsigned long long) misses);
                } else {
                    const uint64_t host = (uint64_t) (uint32_t) r.counts[(size_t) s.n_layer*s.n_expert] |
                        (uint64_t) (uint32_t) r.counts[(size_t) s.n_layer*s.n_expert + 1] << 32;
                    int n_miss = 0;
                    for (int i = 0; i < n_ids; ++i) n_miss += !sk[i] && s.map[(size_t) s.il*s.n_expert + s.ids[i]] < 0;
                    CHECK(host == (uint64_t) n_miss, "form 2 rank %d: host counter %llu vs %d", rank, (unsigned long long) host, n_miss);
                }
            }
        }
    }
    printf("test-moe-cache-tail-skip: %d cases, %d skipped entries, %d failures\n", n_cases, n_skipped, g_fail);
    ggml_backend_free(be);
    return g_fail == 0 ? 0 : 1;
}
