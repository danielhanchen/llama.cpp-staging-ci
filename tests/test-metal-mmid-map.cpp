// host-side models of the Metal MUL_MAT_ID and selection-mask changes, run on any host (no Metal device needed):
//  1. the CPU reference of kernel_mul_mm_id_map0 (ggml-metal-mmid.h) against a direct count: every (token, slot) entry of
//     an in-range expert lands in exactly one tile of its expert, tiles hold at most 32 rows, the tile count stays within
//     the bound the host sizes the dispatch and the scratch by (ggml_metal_mmid_max_tiles), the map is a function of the ids
//  2. the map-share key and range test of ggml_metal_op_mul_mat_id
//  3. the simdgroup block assignment of kernel_mul_mm_id (non-tensor path) with GGML_METAL_MMID_SGSPLIT / _BSKIP on and
//     off: every live output block is computed by exactly one simdgroup, as one accumulator over the full K chain, and
//     every block written back is computed; prints the multiply-accumulates per K step (busiest simdgroup / total)
//  4. the compute buffer of the selection mask: the ggml_sel_mask node against the chain it replaces (graph allocator
//     sizes on the CPU buffer type)

#include "../ggml/src/ggml-metal/ggml-metal-mmid.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sel-mask.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { n_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static void test_map0(std::mt19937 & rng) {
    int n_cases = 0;
    for (int64_t ne02 : { 1, 4, 28, 128, 512, 1024 }) {
        for (int64_t ne20 : { 1, 2, 4, 8, 10 }) {
            for (int64_t ne21 : { 1, 9, 16, 32, 33, 64, 300, 512 }) {
                for (int mode = 0; mode < 4; ++mode) {
                    // 0: distinct ids per token, 1: repeated ids (mostly expert 0), 2: all on one expert, 3: some ids out of range
                    std::vector<int32_t> ids(ne20*ne21);
                    for (int64_t t = 0; t < ne21; ++t) {
                        std::vector<int32_t> perm(ne02);
                        for (int64_t e = 0; e < ne02; ++e) {
                            perm[e] = (int32_t) e;
                        }
                        std::shuffle(perm.begin(), perm.end(), rng);
                        for (int64_t s = 0; s < ne20; ++s) {
                            int32_t v = 0;
                            switch (mode) {
                                case 0: v = perm[s % ne02]; break;
                                case 1: v = rng() % 4 == 0 ? (int32_t) (rng() % ne02) : 0; break;
                                case 2: v = (int32_t) (ne02 - 1); break;
                                case 3: v = rng() % 5 == 0 ? (int32_t) (ne02 + rng() % 3) : (int32_t) (rng() % ne02); break;
                            }
                            ids[t*ne20 + s] = v;
                        }
                    }

                    std::vector<uint32_t> tpe;
                    std::vector<int32_t>  hids;
                    ggml_metal_mmid_map0_ref(ids.data(), ne20, ne21, ne02, tpe, hids);

                    const int64_t n_tiles = tpe[0];
                    const int64_t bound   = ggml_metal_mmid_max_tiles(ne02, ne20, ne21);
                    CHECK(n_tiles <= bound, "tiles %lld > bound %lld (ne02 %lld ne20 %lld ne21 %lld mode %d)",
                        (long long) n_tiles, (long long) bound, (long long) ne02, (long long) ne20, (long long) ne21, mode);
                    // scratch sizes of the host (extra_tpe / extra_ids)
                    CHECK((int64_t) tpe.size() <= 1 + 3*bound, "tpe size");
                    CHECK((int64_t) hids.size() <= ne20*ne21 + ne02, "hids size %zu", hids.size());

                    std::vector<int> seen(ne20*ne21, 0);
                    int64_t prev_e = -1;
                    for (int64_t i = 0; i < n_tiles; ++i) {
                        const uint32_t e = tpe[1 + 3*i], first = tpe[1 + 3*i + 1], rows = tpe[1 + 3*i + 2];
                        CHECK(rows >= 1 && rows <= GGML_METAL_MMID_NTILE, "rows %u", rows);
                        CHECK((int64_t) e >= prev_e, "experts out of order");
                        prev_e = e;
                        for (uint32_t r = 0; r < rows; ++r) {
                            const int32_t ent = hids[first + r];
                            CHECK(ent >= 0 && ent < ne20*ne21, "entry %d", ent);
                            if (ent >= 0 && ent < ne20*ne21) {
                                CHECK(ids[ent] == (int32_t) e, "entry %d routes to %d, tile expert %u", ent, ids[ent], e);
                                seen[ent]++;
                            }
                        }
                    }
                    for (int64_t k = 0; k < ne20*ne21; ++k) {
                        const bool in_range = ids[k] >= 0 && ids[k] < ne02;
                        CHECK(seen[k] == (in_range ? 1 : 0), "entry %lld seen %d times (id %d)", (long long) k, seen[k], ids[k]);
                    }

                    // the map is a function of the ids: a second build is identical
                    std::vector<uint32_t> tpe2;
                    std::vector<int32_t>  hids2;
                    ggml_metal_mmid_map0_ref(ids.data(), ne20, ne21, ne02, tpe2, hids2);
                    CHECK(tpe == tpe2 && hids == hids2, "map not deterministic");

                    n_cases++;
                }
            }
        }
    }
    printf("map0 reference: %d cases\n", n_cases);
}

static void test_share_key() {
    int buf_a = 0, buf_b = 0;
    ggml_metal_mmid_map_key k0;
    k0.ids_buf = &buf_a; k0.ids_data = (const void *) 0x1000; k0.ne20 = 8; k0.ne21 = 512; k0.nb20 = 4; k0.nb21 = 32; k0.ne02 = 128;
    ggml_metal_mmid_map_key k1 = k0;
    CHECK(k0 == k1, "equal keys");
    k1.ids_buf = &buf_b;    CHECK(!(k0 == k1), "buffer"); k1 = k0;
    k1.ids_data = (const void *) 0x2000; CHECK(!(k0 == k1), "address"); k1 = k0;
    k1.ne20 = 4;  CHECK(!(k0 == k1), "ne20"); k1 = k0;
    k1.ne21 = 511; CHECK(!(k0 == k1), "ne21"); k1 = k0;
    k1.nb21 = 64; CHECK(!(k0 == k1), "nb21"); k1 = k0;
    k1.ne02 = 64; CHECK(!(k0 == k1), "ne02"); k1 = k0;

    CHECK( ggml_metal_mmid_ranges_overlap(&buf_a, 0, 10, &buf_a, 9, 20), "overlap");
    CHECK(!ggml_metal_mmid_ranges_overlap(&buf_a, 0, 10, &buf_a, 10, 20), "touching");
    CHECK(!ggml_metal_mmid_ranges_overlap(&buf_a, 0, 10, &buf_b, 0, 10), "other buffer");
    CHECK( ggml_metal_mmid_ranges_overlap(&buf_a, 5, 6, &buf_a, 0, 100), "inside");
    printf("map share key / ranges: ok\n");
}

// kernel_mul_mm_id, non-tensor path: the (A block g, B block b) output blocks each simdgroup accumulates, and the ones it
// writes to threadgroup memory. A block g covers A rows 8g .. 8g+7 (0..7), B block b token rows 8b .. 8b+7 (0..3).
struct mmid_block { int sg, g, b; };

static void mmid_blocks(int nr1, bool sgsplit, bool bskip, std::vector<mmid_block> & mma, std::vector<mmid_block> & stored) {
    const int NR1H = 16;
    const bool has_hi = nr1 > NR1H;
    mma.clear();
    stored.clear();
    for (int sg = 0; sg < 4; ++sg) {
        const bool split4    = sgsplit && !has_hi;
        const bool sg_active = split4 || has_hi || sg < 2;
        const int  r1b       = split4 ? 0 : NR1H*(sg >> 1);
        const bool b1_live   = !bskip || r1b + 8 < nr1;
        if (split4) {
            for (int i = 0; i < 2; ++i) {
                mma.push_back({ sg, 2*sg + i, 0 });
                if (b1_live) {
                    mma.push_back({ sg, 2*sg + i, 1 });
                }
            }
            for (int i = 0; i < (b1_live ? 4 : 2); ++i) {
                stored.push_back({ sg, 2*sg + (i % 2), i / 2 });
            }
        } else if (sg_active) {
            for (int i = 0; i < 4; ++i) {
                mma.push_back({ sg, 4*(sg % 2) + i, 2*(sg / 2) + 0 });
            }
            if (b1_live) {
                for (int i = 4; i < 8; ++i) {
                    mma.push_back({ sg, 4*(sg % 2) + (i % 4), 2*(sg / 2) + 1 });
                }
            }
            for (int i = 0; i < (b1_live ? 8 : 4); ++i) {
                stored.push_back({ sg, 4*(sg % 2) + (i % 4), 2*(sg / 2) + i / 4 });
            }
        }
    }
}

static void test_mmid_blocks() {
    printf("kernel_mul_mm_id multiply-accumulates per K step of 8 (busiest simdgroup / all), by valid rows of the tile:\n");
    printf("  nr1 | before  | bskip   | sgsplit | both\n");
    for (int nr1 = 1; nr1 <= 32; ++nr1) {
        char line[256];
        int  off = snprintf(line, sizeof(line), "  %3d", nr1);
        for (int f = 0; f < 4; ++f) {
            const bool sgsplit = f & 2, bskip = f & 1;
            std::vector<mmid_block> mma, stored;
            mmid_blocks(nr1, sgsplit, bskip, mma, stored);

            // every live output block (B rows < nr1) once, as one accumulator
            for (int b = 0; b < 4; ++b) {
                for (int g = 0; g < 8; ++g) {
                    int n = 0;
                    for (const auto & m : mma) {
                        n += m.g == g && m.b == b;
                    }
                    const bool live = 8*b < nr1;
                    if (live) {
                        CHECK(n == 1, "nr1 %d flags %d: live block (g %d, b %d) computed %d times", nr1, f, g, b, n);
                    } else {
                        CHECK(n <= 1, "nr1 %d flags %d: dead block (g %d, b %d) computed %d times", nr1, f, g, b, n);
                    }
                    int ns = 0;
                    for (const auto & s : stored) {
                        ns += s.g == g && s.b == b;
                    }
                    if (live) {
                        CHECK(ns == 1, "nr1 %d flags %d: live block (g %d, b %d) stored %d times", nr1, f, g, b, ns);
                    }
                }
            }
            // a stored block is computed by the simdgroup that stores it
            for (const auto & s : stored) {
                bool found = false;
                for (const auto & m : mma) {
                    found |= m.sg == s.sg && m.g == s.g && m.b == s.b;
                }
                CHECK(found, "nr1 %d flags %d: sg %d stores (g %d, b %d) it did not compute", nr1, f, s.sg, s.g, s.b);
            }
            int per_sg[4] = { 0, 0, 0, 0 };
            for (const auto & m : mma) {
                per_sg[m.sg]++;
            }
            const int busiest = *std::max_element(per_sg, per_sg + 4);
            off += snprintf(line + off, sizeof(line) - off, " | %2d / %2d", busiest, (int) mma.size());
        }
        if (nr1 <= 17 || nr1 == 24 || nr1 == 25 || nr1 == 32) {
            printf("%s\n", line);
        }
    }
}

// compute buffer of the selection mask node vs the chain (as tests/test-backend-ops.cpp test_sel_mask / llama-graph)
static size_t sel_mask_buffer(int64_t kv, int64_t nb, int64_t n_sel, bool chain) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * kq  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb, 1, 1);
    ggml_tensor * idx = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_sel, nb, 1);
    ggml_set_input(kq);
    ggml_set_input(idx);
    ggml_tensor * out;
    if (!chain) {
        out = ggml_sel_mask(ctx, idx, kq);
    } else {
        ggml_tensor * mask_all = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv + n_sel, 1, 1, 1), -INFINITY);
        mask_all = ggml_repeat_4d(ctx, mask_all, kv + n_sel, nb, 1, 1);
        mask_all = ggml_reshape_3d(ctx, mask_all, 1, kv + n_sel, nb);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_sel, 1, 1, 1), 0.0f);
        zeros = ggml_repeat_4d(ctx, zeros, n_sel, nb, 1, 1);
        zeros = ggml_reshape_3d(ctx, zeros, 1, n_sel, nb);
        ggml_tensor * sel = ggml_set_rows(ctx, mask_all, zeros, idx);
        const size_t row = sel->nb[2];
        sel = ggml_view_4d(ctx, sel, kv, nb, 1, 1, row, row*nb, row*nb, 0);
        out = ggml_add(ctx, sel, kq);
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    size_t size = 0;
    if (ggml_gallocr_reserve(ga, gf)) {
        size = ggml_gallocr_get_buffer_size(ga, 0);
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return size;
}

static void test_sel_mask_scratch() {
    printf("selection mask compute buffer (n_sel 2051), node vs chain:\n");
    for (int64_t kv : { 8192, 65536, 262144 }) {
        for (int64_t nb : { 512, 2048 }) {
            const size_t node  = sel_mask_buffer(kv, nb, 2051, false);
            const size_t chain = sel_mask_buffer(kv, nb, 2051, true);
            CHECK(node > 0 && chain > 0 && node < chain, "kv %lld nb %lld: node %zu chain %zu", (long long) kv, (long long) nb, node, chain);
            printf("  kv %6lld ub %4lld: node %8.1f MiB, chain %8.1f MiB, saved %8.1f MiB\n", (long long) kv, (long long) nb,
                node/1048576.0, chain/1048576.0, (double) (chain - node)/1048576.0);
        }
    }
}

int main() {
    std::mt19937 rng(42);

    test_map0(rng);
    test_share_key();
    test_mmid_blocks();
    test_sel_mask_scratch();

    if (n_fail > 0) {
        printf("FAILED: %d checks\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
