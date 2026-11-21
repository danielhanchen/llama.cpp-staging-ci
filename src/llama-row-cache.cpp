#include "llama-row-cache.h"

#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>

#if defined(_WIN32)
#include <process.h>
#define llama_rc_getpid _getpid
#else
#include <unistd.h>
#define llama_rc_getpid getpid
#endif

// profile file (little endian):
//   char     magic[8] = "LLPLERC2"
//   uint32   row_bytes, ggml type
//   uint64   n_rows (table), table nbytes, fingerprint (llama_row_cache::fingerprint), n_entries
//   uint32   ids[n_entries]      strictly ascending, < n_rows (the rows above a part are a suffix: only that is read)
//   uint32   counts[n_entries]   uses (the rank: a budget keeps the highest counts)
//   uint32   sums[n_entries]     llama_row_cache::row_sum of each row (every row kept is checked)
//   uint8    rows[n_entries][row_bytes]
static const char k_magic[8] = { 'L', 'L', 'P', 'L', 'E', 'R', 'C', '2' };

struct prof_header {
    char     magic[8];
    uint32_t row_bytes;
    uint32_t type;
    uint64_t n_rows;
    uint64_t nbytes;
    uint64_t fingerprint;
    uint64_t n_entries;
};
static_assert(sizeof(prof_header) == 48, "profile header layout");

static constexpr uint32_t k_empty = UINT32_MAX;

static int prof_seek(FILE * f, uint64_t off) {
#if defined(_WIN32)
    return _fseeki64(f, (__int64) off, SEEK_SET);
#else
    return fseeko(f, (off_t) off, SEEK_SET);
#endif
}

int llama_row_cache_mode() {
    // read at each context creation (the rule runs once per model)
    return getenv("LLAMA_PLE_ROW_CACHE") ? atoi(getenv("LLAMA_PLE_ROW_CACHE")) : 1;
}

// FNV-1a 64 over the first and last 64 KiB of the table and 62 4 KiB samples spread evenly between them
uint64_t llama_row_cache::fingerprint(const ggml_tensor * table) {
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix_bytes = [&](const uint8_t * p, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 0x100000001b3ull;
        }
    };
    const size_t nb = ggml_nbytes(table);
    const size_t w  = std::min<size_t>(nb, 65536);
    const size_t ws = std::min<size_t>(nb, 4096);
    const auto * d  = (const uint8_t *) table->data;
    mix_bytes(d, w);
    for (size_t j = 1; j <= 62; ++j) {
        mix_bytes(d + (size_t) ((uint64_t) j * (uint64_t) (nb - ws) / 63), ws);
    }
    mix_bytes(d + nb - w, w);
    return h;
}

// a checksum of one row: 8-byte little-endian words (the last one zero padded)
uint32_t llama_row_cache::row_sum(const uint8_t * p, size_t n) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t) n;
    for (size_t i = 0; i < n; i += 8) {
        uint64_t w = 0;
        memcpy(&w, p + i, std::min<size_t>(8, n - i));
        h = (h ^ w) * 0xff51afd7ed558ccdull;
        h ^= h >> 31;
    }
    return (uint32_t) (h ^ (h >> 32));
}

// per slot: row bytes + id + count + ref, plus the index (key + value) at a power-of-two size of 2-4x the slots
size_t llama_row_cache::slot_bytes(size_t row_bytes) {
    return row_bytes + 4 + 4 + 1 + 4*8;
}

llama_row_cache::llama_row_cache(const ggml_tensor * table, size_t dyn_bytes, size_t static_max_bytes, int64_t skip_below) :
        table(table), row_bytes(ggml_row_size(table->type, table->ne[0])), n_rows(table->ne[1]), skip_below(skip_below) {
    GGML_ASSERT(n_rows < (int64_t) k_empty);
    verify   = getenv("LLAMA_PLE_ROW_CACHE_VERIFY") != nullptr && atoi(getenv("LLAMA_PLE_ROW_CACHE_VERIFY")) != 0;
    n_shards = 64;
    shards   = std::vector<shard>(n_shards);
    n_dyn_slots  = dyn_bytes / slot_bytes(row_bytes);
    n_static_max = static_max_bytes / slot_bytes(row_bytes);
    alloc(std::vector<uint32_t>(n_shards, 0));
}

llama_row_cache::~llama_row_cache() = default;

int64_t llama_row_cache::find(const shard & s, uint32_t key) {
    if (s.keys.empty()) {
        return -1;
    }
    for (uint32_t i = (uint32_t) mix(key) & s.mask;; i = (i + 1) & s.mask) {
        const uint32_t k = s.keys[i];
        if (k == key) {
            return s.vals[i];
        }
        if (k == k_empty) {
            return -1;
        }
    }
}

void llama_row_cache::insert_key(shard & s, uint32_t key, uint32_t slot) {
    for (uint32_t i = (uint32_t) mix(key) & s.mask;; i = (i + 1) & s.mask) {
        if (s.keys[i] == k_empty) {
            s.keys[i] = key;
            s.vals[i] = slot;
            return;
        }
    }
}

// backward-shift deletion of linear probing
void llama_row_cache::erase(shard & s, uint32_t key) {
    uint32_t i = (uint32_t) mix(key) & s.mask;
    while (s.keys[i] != key) {
        if (s.keys[i] == k_empty) {
            return;
        }
        i = (i + 1) & s.mask;
    }
    uint32_t j = i;
    for (;;) {
        j = (j + 1) & s.mask;
        if (s.keys[j] == k_empty) {
            break;
        }
        const uint32_t home = (uint32_t) mix(s.keys[j]) & s.mask;
        // move j into the hole at i unless its home lies cyclically in (i, j]
        const bool in_range = i <= j ? (home > i && home <= j) : (home > i || home <= j);
        if (!in_range) {
            s.keys[i] = s.keys[j];
            s.vals[i] = s.vals[j];
            i = j;
        }
    }
    s.keys[i] = k_empty;
}

size_t llama_row_cache::load_profile(const std::string & path) {
    // the shards are sized here (once): static rows of the profile first, the rest of the budget dynamic
    std::vector<uint32_t> ids, cnts;
    std::vector<uint32_t> keep; // indices into ids of the rows kept, in file order
    FILE * f = path.empty() ? nullptr : fopen(path.c_str(), "rb");
    const char * why = nullptr;
    prof_header h {};
    if (!path.empty() && !f) {
        why = "not found";
    }
    std::vector<uint32_t> sums;
    if (f) {
        uint64_t fsize = 0;
        std::error_code ec;
        fsize = (uint64_t) std::filesystem::file_size(path, ec);
        if (ec) {
            fsize = 0;
        }
        if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, k_magic, 8) != 0) {
            why = "not a PLE row profile";
        } else if (h.row_bytes != row_bytes || h.type != (uint32_t) table->type || h.n_rows != (uint64_t) n_rows ||
                   h.nbytes != ggml_nbytes(table)) {
            why = "made for another table (shape / type)";
        } else if (h.n_entries > (uint64_t) n_rows || fsize < sizeof(h) || (fsize - sizeof(h)) / (12ull + row_bytes) < h.n_entries) {
            why = "corrupt or truncated";
        } else if (h.fingerprint != fingerprint(table)) {
            why = "made for another table (fingerprint)";
        } else {
            ids.resize(h.n_entries);
            cnts.resize(h.n_entries);
            sums.resize(h.n_entries);
            if (fread(ids.data(), 4, ids.size(), f) != ids.size() || fread(cnts.data(), 4, cnts.size(), f) != cnts.size() ||
                fread(sums.data(), 4, sums.size(), f) != sums.size()) {
                why = "truncated";
                ids.clear();
            }
        }
    }
    const int64_t t0 = ggml_time_us();
    for (size_t i = 0; i < ids.size() && !why; ++i) {
        if ((i > 0 && ids[i] <= ids[i - 1]) || (int64_t) ids[i] >= n_rows) {
            why = "corrupt (ids not ascending or out of range)";
        }
    }
    if (!ids.empty() && !why) {
        // rows at or above skip_below, the highest counts first while the static budget lasts, kept in file order
        const size_t first = std::lower_bound(ids.begin(), ids.end(), (uint32_t) std::max<int64_t>(0, skip_below)) - ids.begin();
        for (size_t i = first; i < ids.size(); ++i) {
            keep.push_back((uint32_t) i);
        }
        if (keep.size() > n_static_max) {
            std::nth_element(keep.begin(), keep.begin() + n_static_max, keep.end(), [&](uint32_t a, uint32_t b) {
                return cnts[a] != cnts[b] ? cnts[a] > cnts[b] : a < b;
            });
            keep.resize(n_static_max);
            std::sort(keep.begin(), keep.end());
        }
    }

    // size the shards
    std::vector<uint32_t> n_stat(n_shards, 0);
    for (uint32_t i : keep) {
        n_stat[(size_t) (mix(ids[i]) >> 58) % n_shards]++;
    }
    alloc(n_stat);

    // the row bytes of the kept entries, read sequentially
    size_t n_loaded = 0;
    if (!keep.empty()) {
        const uint64_t data0 = sizeof(h) + 12ull*h.n_entries;
        std::vector<uint8_t> buf(8u << 20);
        const size_t per = buf.size() / row_bytes;
        size_t k = 0;
        for (uint64_t e0 = keep.front(); e0 < h.n_entries && k < keep.size() && !why; e0 += per) {
            const uint64_t e1 = std::min<uint64_t>(h.n_entries, e0 + per);
            if (keep[k] >= e1) {
                continue;
            }
            if (prof_seek(f, data0 + e0*row_bytes) != 0 || fread(buf.data(), row_bytes, (size_t) (e1 - e0), f) != (size_t) (e1 - e0)) {
                why = "truncated rows";
                break;
            }
            for (; k < keep.size() && keep[k] < e1; ++k) {
                const uint32_t i = keep[k];
                if (row_sum(buf.data() + (size_t) (i - e0)*row_bytes, row_bytes) != sums[i]) {
                    why = "corrupt (row checksum)";
                    break;
                }
                shard & s = shard_of(ids[i]);
                const uint32_t slot = s.n_used++;
                GGML_ASSERT(slot < s.n_static);
                memcpy(s.data.get() + (size_t) slot*row_bytes, buf.data() + (size_t) (i - e0)*row_bytes, row_bytes);
                s.slot_row[slot] = ids[i];
                s.slot_cnt[slot] = std::max<uint32_t>(1, cnts[i]);
                s.slot_ref[slot] = 1;
                insert_key(s, ids[i], slot);
                n_loaded++;
            }
        }
        // spot check against the mapping: every row of a small profile, else 64 random ones (a stale profile drops every row)
        if (!why && n_loaded == keep.size()) {
            std::vector<uint32_t> probe;
            if (keep.size() <= 4096) {
                for (uint32_t i : keep) {
                    probe.push_back(ids[i]);
                }
            } else {
                std::mt19937_64 rng(keep.size());
                for (int j = 0; j < 64; ++j) {
                    probe.push_back(ids[keep[rng() % keep.size()]]);
                }
            }
            std::vector<uint8_t> tmp(row_bytes);
            for (uint32_t r : probe) {
                const auto * src = (const uint8_t *) table->data + (size_t) r*table->nb[1];
                if (!peek(r, tmp.data()) || memcmp(tmp.data(), src, row_bytes) != 0) {
                    why = "rows differ from the table (stale)";
                    break;
                }
            }
        }
        if (why) {
            // all dynamic again
            alloc(std::vector<uint32_t>(n_shards, 0));
            n_loaded = 0;
        }
    }
    if (f) {
        fclose(f);
    }
    n_static_rows = n_loaded;
    if (!path.empty()) {
        if (why) {
            LLAMA_LOG_WARN("%s: PLE row profile %s: %s; no rows preloaded\n", __func__, path.c_str(), why);
        } else {
            LLAMA_LOG_INFO("%s: PLE row profile %s: %zu of %" PRIu64 " rows preloaded (%.1f MiB) in %.2f s\n", __func__, path.c_str(),
                n_loaded, h.n_entries, n_loaded*(double) row_bytes/1048576.0, (ggml_time_us() - t0)/1e6);
        }
    }
    return n_loaded;
}

void llama_row_cache::alloc(const std::vector<uint32_t> & n_stat) {
    const size_t n_dyn = (n_dyn_slots + n_shards - 1) / n_shards;
    mem_bytes = 0;
    n_slots_total = 0;
    for (size_t si = 0; si < n_shards; ++si) {
        shard & s = shards[si];
        s.n_static = n_stat[si];
        s.n_slots  = (uint32_t) (n_stat[si] + n_dyn);
        size_t tab = 16;
        while (tab < 2ull*s.n_slots) {
            tab <<= 1;
        }
        s.mask = (uint32_t) (tab - 1);
        // fresh vectors: a smaller layout gives the old capacity back
        std::vector<uint32_t>(tab, k_empty).swap(s.keys);
        std::vector<uint32_t>(tab, 0).swap(s.vals);
        s.data.reset();
        s.data.reset(new uint8_t[std::max<size_t>(1, (size_t) s.n_slots * row_bytes)]);
        std::vector<uint32_t>(s.n_slots, k_empty).swap(s.slot_row);
        std::vector<uint32_t>(s.n_slots, 0).swap(s.slot_cnt);
        std::vector<uint8_t>(s.n_slots, 0).swap(s.slot_ref);
        s.n_used = 0;
        s.hand   = s.n_static;
        mem_bytes += tab*8 + (size_t) s.n_slots*(row_bytes + 9);
        n_slots_total += s.n_slots;
    }
}

bool llama_row_cache::peek(int64_t r, uint8_t * out) const {
    if (r < 0 || r >= n_rows) {
        return false;
    }
    shard & s = shard_of(r);
    std::lock_guard<std::mutex> lock(s.mtx);
    const int64_t slot = find(s, (uint32_t) r);
    if (slot < 0) {
        return false;
    }
    memcpy(out, s.data.get() + (size_t) slot*row_bytes, row_bytes);
    return true;
}

bool llama_row_cache::get(int64_t r, uint8_t * out) {
    if (r < 0 || r >= n_rows) {
        return false;
    }
    st.n_lookups.fetch_add(1, std::memory_order_relaxed);
    shard & s = shard_of(r);
    std::lock_guard<std::mutex> lock(s.mtx);
    const int64_t slot = find(s, (uint32_t) r);
    if (slot < 0) {
        return false;
    }
    memcpy(out, s.data.get() + (size_t) slot*row_bytes, row_bytes);
    s.slot_ref[slot] = 1;
    // (with the recorder the reads are counted there: a static row keeps its loaded count)
    if (!rec_on && s.slot_cnt[slot] != UINT32_MAX) {
        s.slot_cnt[slot]++;
    }
    st.n_hits.fetch_add(1, std::memory_order_relaxed);
    if ((uint32_t) slot < s.n_static) {
        st.n_hits_static.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

bool llama_row_cache::contains(int64_t r) const {
    if (r < 0 || r >= n_rows) {
        return false;
    }
    shard & s = shard_of(r);
    std::lock_guard<std::mutex> lock(s.mtx);
    return find(s, (uint32_t) r) >= 0;
}

void llama_row_cache::put(int64_t r, const uint8_t * src) {
    if (r < skip_below || r >= n_rows) {
        return;
    }
    shard & s = shard_of(r);
    std::lock_guard<std::mutex> lock(s.mtx);
    if (s.n_slots <= s.n_static || find(s, (uint32_t) r) >= 0) {
        return;
    }
    uint32_t slot;
    if (s.n_used < s.n_slots) {
        slot = s.n_used++;
    } else {
        // clock over the dynamic slots
        for (;;) {
            if (s.hand >= s.n_slots) {
                s.hand = s.n_static;
            }
            if (s.slot_ref[s.hand]) {
                s.slot_ref[s.hand] = 0;
                s.hand++;
                continue;
            }
            slot = s.hand++;
            break;
        }
        if (s.slot_row[slot] != k_empty) {
            erase(s, s.slot_row[slot]);
            st.n_evicts.fetch_add(1, std::memory_order_relaxed);
        }
    }
    memcpy(s.data.get() + (size_t) slot*row_bytes, src, row_bytes);
    s.slot_row[slot] = (uint32_t) r;
    s.slot_cnt[slot] = 1;
    s.slot_ref[slot] = 0;
    insert_key(s, (uint32_t) r, slot);
    st.n_inserts.fetch_add(1, std::memory_order_relaxed);
}

size_t llama_row_cache::filter_uncached(int32_t * rows, size_t n) {
    size_t o = 0;
    for (size_t i = 0; i < n; ++i) {
        if (rows[i] >= 0 && rows[i] >= skip_below && contains(rows[i])) {
            continue;
        }
        rows[o++] = rows[i];
    }
    st.n_filtered.fetch_add(n - o, std::memory_order_relaxed);
    return o;
}

void llama_row_cache::mask_cached(int32_t * rows, size_t n) {
    size_t m = 0;
    for (size_t i = 0; i < n; ++i) {
        if (rows[i] >= 0 && rows[i] >= skip_below && contains(rows[i])) {
            rows[i] = -1;
            m++;
        }
    }
    st.n_filtered.fetch_add(m, std::memory_order_relaxed);
}

bool llama_row_cache::save_profile(const std::string & path) const {
    struct ent { uint32_t row, cnt; const uint8_t * p; };
    std::vector<ent> es;
    for (const auto & s : shards) {
        std::lock_guard<std::mutex> lock(s.mtx);
        for (uint32_t q = 0; q < s.n_used && q < s.n_slots; ++q) {
            if (s.slot_row[q] == k_empty) {
                continue;
            }
            // carried over counts of the static rows decay by half per session (k19-autoprof: with the recorder, the reads of this launch
            // are its counts: the static rows carry their loaded count halved, the dynamic slots are left to the recorder)
            if (rec_on && q >= s.n_static) {
                continue;
            }
            const uint32_t c = q < s.n_static ? std::max<uint32_t>(1, s.slot_cnt[q] / 2) : s.slot_cnt[q];
            es.push_back({ s.slot_row[q], c, s.data.get() + (size_t) q*row_bytes });
        }
        for (size_t j = 0; j < s.rec_row.size(); ++j) {
            es.push_back({ s.rec_row[j], s.rec_cnt[j], s.rec_data.data() + j*row_bytes });
        }
    }
    std::sort(es.begin(), es.end(), [](const ent & a, const ent & b) { return a.row < b.row; });
    // a row both preloaded and recorded: one entry, the counts added
    {
        size_t o = 0;
        for (size_t i = 0; i < es.size(); ++i) {
            if (o > 0 && es[o - 1].row == es[i].row) {
                es[o - 1].cnt = (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) es[o - 1].cnt + es[i].cnt);
                continue;
            }
            es[o++] = es[i];
        }
        es.resize(o);
    }
    if (rec_on && es.empty()) {
        return false; // k19-autoprof: nothing read in this launch, the file (if any) stays
    }
    // a temporary of this process only, created exclusively (an existing file or link of that name is not followed), then renamed over
    // the profile (std::filesystem::rename replaces an existing file on Windows too)
    const std::string tmp = path + ".tmp." + std::to_string((long long) llama_rc_getpid());
    // k19-autoprof2: the directory may not exist yet (the self-learned profile's directory is made by the first write into it; where no
    // MoE profile or CPU-miss calibration was written before, e.g. host-lookup tiers, the PLE profile was never saved: "cannot write")
    {
        std::error_code ec;
        const std::filesystem::path pp(path);
        if (pp.has_parent_path()) {
            std::filesystem::create_directories(pp.parent_path(), ec);
        }
    }
    FILE * f = fopen(tmp.c_str(), "wbx");
    if (!f) {
        LLAMA_LOG_WARN("%s: cannot write %s\n", __func__, tmp.c_str());
        return false;
    }
    prof_header h {};
    memcpy(h.magic, k_magic, 8);
    h.row_bytes   = (uint32_t) row_bytes;
    h.type        = (uint32_t) table->type;
    h.n_rows      = (uint64_t) n_rows;
    h.nbytes      = ggml_nbytes(table);
    h.fingerprint = fingerprint(table);
    h.n_entries   = es.size();
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1;
    for (const auto & e : es) {
        ok = ok && fwrite(&e.row, 4, 1, f) == 1;
    }
    for (const auto & e : es) {
        ok = ok && fwrite(&e.cnt, 4, 1, f) == 1;
    }
    for (const auto & e : es) {
        const uint32_t sum = row_sum(e.p, row_bytes);
        ok = ok && fwrite(&sum, 4, 1, f) == 1;
    }
    for (const auto & e : es) {
        ok = ok && fwrite(e.p, row_bytes, 1, f) == 1;
    }
    ok = fclose(f) == 0 && ok;
    if (ok) {
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }
    LLAMA_LOG_INFO("%s: PLE row profile %s: %zu rows written%s\n", __func__, path.c_str(), es.size(), ok ? "" : " (FAILED)");
    return ok;
}

void llama_row_cache::enable_recorder(size_t max_bytes) {
    rec_on        = true;
    rec_max_bytes = max_bytes;
}

size_t llama_row_cache::n_recorded_rows() const {
    size_t n = 0;
    for (const auto & s : shards) {
        std::lock_guard<std::mutex> lock(s.mtx);
        n += s.rec_row.size();
    }
    return n;
}

void llama_row_cache::record(int64_t r, const uint8_t * bytes) {
    if (!rec_on || r < 0 || r >= n_rows) {
        return;
    }
    st.n_recorded.fetch_add(1, std::memory_order_relaxed);
    shard & s = shard_of(r);
    const uint32_t key = (uint32_t) r;
    std::lock_guard<std::mutex> lock(s.mtx);
    if (!s.rec_keys.empty()) {
        for (uint32_t i = (uint32_t) mix(key) & s.rec_mask;; i = (i + 1) & s.rec_mask) {
            const uint32_t k = s.rec_keys[i];
            if (k == key) {
                uint32_t & c = s.rec_cnt[s.rec_vals[i]];
                c += c != UINT32_MAX ? 1 : 0;
                return;
            }
            if (k == k_empty) {
                break;
            }
        }
    }
    // a new row: entry (row, count, bytes) + its share of the index (at most half full)
    const size_t per = row_bytes + 8 + 16;
    if (rec_bytes.load(std::memory_order_relaxed) + per > rec_max_bytes) {
        st.n_rec_full.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    rec_bytes.fetch_add(per, std::memory_order_relaxed);
    if (2*(s.rec_row.size() + 1) > s.rec_keys.size()) {
        const size_t tab = std::max<size_t>(64, 2*s.rec_keys.size());
        std::vector<uint32_t>(tab, k_empty).swap(s.rec_keys);
        std::vector<uint32_t>(tab, 0).swap(s.rec_vals);
        s.rec_mask = (uint32_t) (tab - 1);
        for (uint32_t j = 0; j < (uint32_t) s.rec_row.size(); ++j) {
            uint32_t i = (uint32_t) mix(s.rec_row[j]) & s.rec_mask;
            while (s.rec_keys[i] != k_empty) {
                i = (i + 1) & s.rec_mask;
            }
            s.rec_keys[i] = s.rec_row[j];
            s.rec_vals[i] = j;
        }
    }
    const uint32_t j = (uint32_t) s.rec_row.size();
    s.rec_row.push_back(key);
    s.rec_cnt.push_back(1);
    s.rec_data.insert(s.rec_data.end(), bytes, bytes + row_bytes);
    uint32_t i = (uint32_t) mix(key) & s.rec_mask;
    while (s.rec_keys[i] != k_empty) {
        i = (i + 1) & s.rec_mask;
    }
    s.rec_keys[i] = key;
    s.rec_vals[i] = j;
}

std::string llama_row_cache::describe() const {
    char buf[512];
    snprintf(buf, sizeof(buf), "lookups = %" PRIu64 ", hits = %" PRIu64 " (static %" PRIu64 "), inserts = %" PRIu64 ", evictions = %" PRIu64
        ", hint rows dropped = %" PRIu64 ", static rows = %zu, slots = %zu, %.1f MiB",
        st.n_lookups.load(), st.n_hits.load(), st.n_hits_static.load(), st.n_inserts.load(), st.n_evicts.load(),
        st.n_filtered.load(), n_static_rows, n_slots_total, mem_bytes/1048576.0);
    std::string s = buf;
    if (rec_on) {
        snprintf(buf, sizeof(buf), ", routing profile recorder: %" PRIu64 " row reads, %zu rows (%.1f MiB), %" PRIu64 " new rows past the cap",
            st.n_recorded.load(), n_recorded_rows(), rec_bytes.load()/1048576.0, st.n_rec_full.load());
        s += buf;
    }
    if (verify) {
        snprintf(buf, sizeof(buf), ", verified hits = %" PRIu64 ", mismatched = %" PRIu64, st.n_verified.load(), st.n_verify_bad.load());
        s += buf;
    }
    return s;
}

// the row source of the CPU gather: a cached row is copied out, any other row is copied from the mapping (its page faults here, as it
// would in the gather) and cached
static const void * llama_row_cache_source(void * ud, const ggml_tensor * t, int64_t row, void * tmp) {
    auto * rc = (llama_row_cache *) ud;
    const auto * src = (const uint8_t *) t->data + (size_t) row*t->nb[1];
    const bool rec = rc->recording(); // k19-autoprof: the row (and the bytes the gather reads) to the recorder
    if (row < rc->get_skip_below()) {
        if (rec) {
            rc->record(row, src);
        }
        return src; // the part the page cache holds (k19-disktt)
    }
    if (rc->get(row, (uint8_t *) tmp)) {
        if (rc->verify) {
            rc->st.n_verified.fetch_add(1, std::memory_order_relaxed);
            if (memcmp(tmp, src, rc->get_row_bytes()) != 0) {
                rc->st.n_verify_bad.fetch_add(1, std::memory_order_relaxed);
                if (rec) {
                    rc->record(row, src);
                }
                return src; // the mapping's bytes win
            }
        }
        if (rec) {
            rc->record(row, (const uint8_t *) tmp);
        }
        return tmp;
    }
    rc->put(row, src);
    if (rec) {
        rc->record(row, src);
    }
    return src;
}

bool llama_row_cache_attach(llama_row_cache * rc) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (!reg) {
        return false;
    }
    using set_fn_t = bool (*)(const ggml_tensor *, ggml_cpu_row_source_t, void *);
    auto fn = (set_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_row_source");
    if (!fn) {
        return false;
    }
    return fn(rc->table, llama_row_cache_source, rc);
}

void llama_row_cache_detach(const ggml_tensor * table) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (!reg) {
        return;
    }
    using set_fn_t = bool (*)(const ggml_tensor *, ggml_cpu_row_source_t, void *);
    auto fn = (set_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_row_source");
    if (fn) {
        fn(table, nullptr, nullptr);
    }
}
