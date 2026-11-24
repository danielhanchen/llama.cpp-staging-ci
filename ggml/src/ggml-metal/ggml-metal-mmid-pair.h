#pragma once

// paired routes of the mat-vec MUL_MAT_ID path (GGML_METAL_MMID_PAIR=1)
//
// A route is one (token, slot) pair of the ids, numbered p = token*n_used + slot. Routes of the same
// expert are paired in increasing p: the 1st with the 2nd, the 3rd with the 4th, ... (an odd last one
// stays single). Every pair or single is one entry (expert, route A, route B or -1), and the entries are
// ordered by route A. Layout of the list: [0] = number of entries, then 3 int32 per entry.
//
// This is the reference of kernel_mul_mv_id_pair_map (kernels/mul_mv.metal), written the way the
// kernel computes it: one thread per route, each with
//   rank    = routes q < p of the same expert
//   partner = the first route q > p of the same expert, or -1
//   leader  = rank is even (emits the entry of its pair or single)
//   pos     = leaders q < p (the entry index)
// The kernel must produce exactly this list (tests/test-metal-mmid-pair-model.cpp checks it against an
// independent sequential pairing).

#include <stdint.h>

// at most this many routes per op take the paired path (one threadgroup of the map kernel, a thread per route)
#define GGML_METAL_MMID_PAIR_MAX_ROUTES 256

// the most tokens of a mat-vec MUL_MAT_ID that take the paired path
#define GGML_METAL_MMID_PAIR_MAX_TOKENS 16

// the size in int32 of the list of n routes
static inline int64_t ggml_metal_mmid_pair_list_len(int64_t n_routes) {
    return 1 + 3*n_routes;
}

// ids: n_tokens rows of n_used int32, row r at ids + r*row_stride (in int32); out: ggml_metal_mmid_pair_list_len(n) int32
// returns the number of entries. The phases are those of the kernel (expert per route; rank / partner / leader per route;
// entry index = leaders before the route), with the per-thread arrays as local ones.
static inline int32_t ggml_metal_mmid_pair_build(const int32_t * ids, int64_t row_stride, int32_t n_used, int32_t n_tokens, int32_t * out) {
    const int32_t n = n_used*n_tokens;

    if (n > GGML_METAL_MMID_PAIR_MAX_ROUTES) {
        out[0] = -1;
        return -1;
    }

    int32_t eid    [GGML_METAL_MMID_PAIR_MAX_ROUTES];
    int32_t lead   [GGML_METAL_MMID_PAIR_MAX_ROUTES];
    int32_t partner[GGML_METAL_MMID_PAIR_MAX_ROUTES];

    for (int32_t p = 0; p < n; ++p) {
        eid[p] = ids[(p/n_used)*row_stride + p%n_used];
    }

    for (int32_t p = 0; p < n; ++p) {
        int32_t rank = 0;

        partner[p] = -1;

        for (int32_t q = 0; q < n; ++q) {
            rank      += q < p && eid[q] == eid[p];
            partner[p] = q > p && eid[q] == eid[p] && partner[p] < 0 ? q : partner[p];
        }

        lead[p] = rank % 2 == 0;
    }

    int32_t n_entries = 0;

    for (int32_t p = 0; p < n; ++p) {
        if (!lead[p]) {
            continue;
        }

        int32_t pos = 0;
        for (int32_t q = 0; q < p; ++q) {
            pos += lead[q];
        }

        out[1 + 3*pos + 0] = eid[p];
        out[1 + 3*pos + 1] = p;
        out[1 + 3*pos + 2] = partner[p];

        n_entries++;
    }

    out[0] = n_entries;

    return n_entries;
}
