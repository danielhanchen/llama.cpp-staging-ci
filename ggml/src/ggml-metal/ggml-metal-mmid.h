#pragma once

// host side of the kernel_mul_mm_id expert map (kernel_mul_mm_id_map0): the tile bound, the key under which one map is
// shared by the MUL_MAT_ID nodes of an encoder, and a CPU reference of the map the kernel builds (plain C++, no Metal:
// tests/test-metal-mmid-map.cpp runs it on any host)

#include <cstdint>
#include <vector>

// rows per entry tile, NR1 of kernel_mul_mm_id and NTILE of kernel_mul_mm_id_map0
static constexpr int64_t GGML_METAL_MMID_NTILE = 32;

// upper bound on the number of entry tiles emitted by kernel_mul_mm_id_map0: the full tiles over all entries plus at
// most one partial tile per expert
static inline int64_t ggml_metal_mmid_max_tiles(int64_t ne02, int64_t ne20, int64_t ne21) {
    const int64_t n_rows = ne20*ne21;
    const int64_t tiles  = (n_rows + GGML_METAL_MMID_NTILE - 1)/GGML_METAL_MMID_NTILE + (ne02 < n_rows ? ne02 : n_rows) - 1;

    return n_rows < tiles ? n_rows : tiles;
}

// the map depends on the ids and their shape only (not on src0 / src1): two MUL_MAT_ID nodes with equal keys read
// identical maps, as long as neither the ids nor the scratch holding the map were written in between
struct ggml_metal_mmid_map_key {
    const void * ids_buf  = nullptr; // buffer of the ids (base tensor)
    const void * ids_data = nullptr; // address of the ids
    int64_t ne20 = 0; // n_expert_used
    int64_t ne21 = 0; // n_tokens
    int64_t nb20 = 0;
    int64_t nb21 = 0;
    int64_t ne02 = 0; // n_expert

    bool operator==(const ggml_metal_mmid_map_key & o) const {
        return ids_buf == o.ids_buf && ids_data == o.ids_data && ne20 == o.ne20 && ne21 == o.ne21 &&
               nb20 == o.nb20 && nb21 == o.nb21 && ne02 == o.ne02;
    }
};

// [a0, a1) of buffer a and [b0, b1) of buffer b share a byte
static inline bool ggml_metal_mmid_ranges_overlap(const void * a, uint64_t a0, uint64_t a1, const void * b, uint64_t b0, uint64_t b1) {
    return a == b && a0 < b1 && b0 < a1;
}

// CPU reference of kernel_mul_mm_id_map0. ids: [ne21][ne20] expert ids (row stride ne20). Output tpe: tile count, then
// (expert, first entry, valid rows) per tile; hids: the entries ((token)*ne20 + slot) of every expert in token order, the
// first matching slot of a token first, then its other matching slots (duplicate ids), each expert's slice followed by one
// padding slot (the kernel's dummy store; left at -1 here)
static inline void ggml_metal_mmid_map0_ref(const int32_t * ids, int64_t ne20, int64_t ne21, int64_t ne02,
        std::vector<uint32_t> & tpe, std::vector<int32_t> & hids) {
    std::vector<uint32_t> cnt(ne02, 0);
    for (int64_t t = 0; t < ne21; ++t) {
        for (int64_t s = 0; s < ne20; ++s) {
            const int32_t e = ids[t*ne20 + s];
            if (e >= 0 && e < ne02) {
                cnt[e]++;
            }
        }
    }

    std::vector<uint32_t> bas(ne02, 0);
    tpe.assign(1, 0);
    uint32_t base = 0;
    for (int64_t e = 0; e < ne02; ++e) {
        bas[e] = base;
        for (uint32_t k = 0; k < cnt[e]; k += GGML_METAL_MMID_NTILE) {
            tpe.push_back((uint32_t) e);
            tpe.push_back(base + k);
            tpe.push_back(cnt[e] - k < GGML_METAL_MMID_NTILE ? cnt[e] - k : (uint32_t) GGML_METAL_MMID_NTILE);
            tpe[0]++;
        }
        base += cnt[e] + 1;
    }

    hids.assign(base, -1);
    for (int64_t e = 0; e < ne02; ++e) {
        uint32_t ient = bas[e];
        for (int64_t t = 0; t < ne21; ++t) {
            int64_t first = -1;
            for (int64_t s = 0; s < ne20; ++s) {
                if (ids[t*ne20 + s] == e) {
                    if (first < 0) {
                        first = s;
                    }
                }
            }
            if (first < 0) {
                continue;
            }
            hids[ient++] = (int32_t) (t*ne20 + first);
            for (int64_t s = 0; s < ne20; ++s) {
                if (ids[t*ne20 + s] == e && s != first) {
                    hids[ient++] = (int32_t) (t*ne20 + s);
                }
            }
        }
    }
}
