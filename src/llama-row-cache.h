#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

// a row-granular host cache in front of a lazily mapped embedding table (the Qwen3.8-Flash-Next PLE n-gram table:
// 320M rows of 90 bytes, 16 random rows per token; DeepSeek-V4.1's Engram table goes through the same loader path). A row reached
// through the mapping faults its whole 4 KiB page, so page-cache RAM spent on the table holds ~45x fewer useful bytes than the same
// RAM spent on packed rows. The cache keeps
//   - static rows: the hottest rows of a saved per-model profile (row ids ranked by use + their bytes), read sequentially at load
//     and never evicted, and
//   - dynamic rows: rows the gather faulted, under clock eviction.
// The graph is unchanged: the CPU GET_ROWS of the table asks the cache for each row (ggml_backend_cpu_set_row_source) and dequantizes
// the cached copy, or the mapping's bytes on a miss, with the same function: the bytes are the table's bytes, so the outputs are
// bit-identical.
// Rows below `skip_below` (the part of the table the page cache already holds) are neither preloaded nor inserted.
struct llama_row_cache {
    struct stats {
        std::atomic<uint64_t> n_lookups  {0};
        std::atomic<uint64_t> n_hits     {0};
        std::atomic<uint64_t> n_hits_static {0};
        std::atomic<uint64_t> n_inserts  {0};
        std::atomic<uint64_t> n_evicts   {0};
        std::atomic<uint64_t> n_filtered {0}; // hint / prefetch rows dropped because they were cached
        std::atomic<uint64_t> n_verified {0}; // LLAMA_PLE_ROW_CACHE_VERIFY=1: hits compared with the mapping
        std::atomic<uint64_t> n_verify_bad {0};
        std::atomic<uint64_t> n_recorded {0};   // row reads counted by the recorder
        std::atomic<uint64_t> n_rec_full {0};   // ... of new rows past the recorder's memory cap (not kept)
    };

    // table: the lazily mapped table (host buffer, matrix); dyn_bytes: memory (slots + index) of the dynamic rows; static_max_bytes:
    // at most this much for the profile's rows (only what load_profile keeps is allocated)
    llama_row_cache(const ggml_tensor * table, size_t dyn_bytes, size_t static_max_bytes, int64_t skip_below);
    ~llama_row_cache();

    // reads a profile written by save_profile (or scripts that use the same format): checks the header against the table,
    // keeps the most used rows >= skip_below that fit the static budget, reads their bytes sequentially (from the first row >=
    // skip_below: the ids are ascending), and spot-checks up to 64 of
    // them (every one when the profile is small) against the mapping; a mismatch drops them all. Returns the rows kept.
    size_t load_profile(const std::string & path);
    // writes every cached row (static and dynamic) with its uses (static counts carried over, halved) to path (tmp + rename); with the
    // recorder on, the recorded rows (their reads of this launch) instead of the dynamic slots' counts
    bool save_profile(const std::string & path) const;

    // a recorder of the rows the gather reads (every row of the table, also below skip_below and rows the cache holds):
    // a count per row and a copy of its bytes taken at its first read (the bytes the gather reads anyway), at most max_bytes of memory
    // (rows first seen past it are not kept); written by save_profile. The cache itself may have no slots (record only): the gather
    // then reads every row from the mapping as without a cache
    void enable_recorder(size_t max_bytes);
    bool recording() const { return rec_on; }
    void record(int64_t r, const uint8_t * bytes);
    size_t n_recorded_rows() const;

    // copies row `r` into out and returns true when cached (counts a use)
    bool get(int64_t r, uint8_t * out);
    // the same without counting a use
    bool peek(int64_t r, uint8_t * out) const;
    // caches row `r` from src (row_bytes bytes; no-op below skip_below or when present)
    void put(int64_t r, const uint8_t * src);
    // true when row `r` is cached (no use counted)
    bool contains(int64_t r) const;
    // drops the cached rows from `rows` (in place, order kept); returns the new count
    size_t filter_uncached(int32_t * rows, size_t n);
    // sets the cached rows of `rows` to -1 (prefetcher jobs skip negative ids)
    void mask_cached(int32_t * rows, size_t n);

    size_t  get_row_bytes()  const { return row_bytes; }
    int64_t get_skip_below() const { return skip_below; }
    size_t  n_static()  const { return n_static_rows; }
    size_t  capacity()  const { return n_slots_total; }
    size_t  bytes()     const { return mem_bytes; }
    std::string describe() const;

    const ggml_tensor * table;
    stats st;
    bool verify = false; // LLAMA_PLE_ROW_CACHE_VERIFY=1

    // FNV-1a 64 of the first and last 64 KiB of the table and 62 evenly spread 4 KiB samples (the profile's key, with shape and type)
    static uint64_t fingerprint(const ggml_tensor * table);
    // checksum of a row's bytes in a profile
    static uint32_t row_sum(const uint8_t * p, size_t n);
    // memory per slot (row copy, metadata, index share)
    static size_t slot_bytes(size_t row_bytes);

private:
    struct shard {
        mutable std::mutex mtx;
        std::vector<uint32_t> keys;   // open addressing (linear probing), EMPTY = UINT32_MAX
        std::vector<uint32_t> vals;   // slot index within the shard
        uint32_t mask = 0;
        std::unique_ptr<uint8_t[]> data; // slots * row_bytes, not initialized (pages are committed as slots fill)
        std::vector<uint32_t> slot_row;
        std::vector<uint32_t> slot_cnt;
        std::vector<uint8_t>  slot_ref;
        uint32_t n_slots   = 0;
        uint32_t n_static  = 0;   // slots [0, n_static) are static
        uint32_t n_used    = 0;   // slots in use (static + dynamic, filled from the front)
        uint32_t hand      = 0;   // clock hand over the dynamic slots
        // recorder: open addressing row -> entry
        std::vector<uint32_t> rec_keys;
        std::vector<uint32_t> rec_vals;
        uint32_t rec_mask = 0;
        std::vector<uint32_t> rec_row;
        std::vector<uint32_t> rec_cnt;
        std::vector<uint8_t>  rec_data;
    };

    shard & shard_of(int64_t r) const { return shards[(size_t) (mix(r) >> 58) % n_shards]; }
    static uint64_t mix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; }
    static int64_t find(const shard & s, uint32_t key);
    static void    erase(shard & s, uint32_t key);
    static void    insert_key(shard & s, uint32_t key, uint32_t slot);
    void alloc(const std::vector<uint32_t> & n_stat);

    size_t  row_bytes;
    int64_t n_rows;
    int64_t skip_below;
    size_t  n_shards;
    mutable std::vector<shard> shards;
    size_t  n_slots_total = 0;
    size_t  n_dyn_slots   = 0;
    size_t  n_static_max  = 0;
    size_t  n_static_rows = 0;
    size_t  mem_bytes     = 0;
    bool    rec_on        = false;
    size_t  rec_max_bytes = 0;
    std::atomic<size_t> rec_bytes {0};
};

// makes rc the row source of the CPU GET_ROWS of rc->table (ggml_backend_cpu_set_row_source); false where the CPU backend lacks it
bool llama_row_cache_attach(llama_row_cache * rc);
// removes the row source of table
void llama_row_cache_detach(const ggml_tensor * table);

// LLAMA_PLE_ROW_CACHE: 0 off (the kill switch: base graph and behaviour), unset/1 auto, 2 forced on (test hook: also where the whole
// table is resident or the file is in RAM)
int llama_row_cache_mode();
