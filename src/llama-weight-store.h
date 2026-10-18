#pragma once

// weight store: the host weights that are mapped from the model files are uploaded to the devices from a pinned staging ring that is
// filled with large parallel file reads (pread / O_DIRECT) instead of the page faults of copies from the mapping
// exact: the bytes are the same, only where they are read from changes (LLAMA_WEIGHT_STREAM_VERIFY=1 compares them with the mapping)

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// a model file whose tensors may be streamed
struct llama_weight_file {
    std::string     path;               // empty: the file is read through fd
    int             fd       = -1;      // POSIX: a duplicate of the loader's descriptor, used when the path is unknown (owned)
    const uint8_t * map_addr = nullptr; // address of file offset 0 in the model mapping, nullptr if the file is not mapped
    size_t          map_size = 0;
    size_t          size     = 0;       // file size
};

// the bytes of a tensor in a model file
struct llama_weight_extent {
    uint16_t file = 0;
    uint64_t offs = 0;    // offset of the tensor data in the file
    uint64_t size = 0;
    bool     lazy = false; // read on demand by the host (e.g. per-layer token embeddings), never uploaded whole
};

// file extents of the host weights mapped from the model files, captured when the model is loaded (the loader that knows the file
// offsets is released after loading)
struct llama_weight_extents {
    std::vector<llama_weight_file> files;
    std::unordered_map<const ggml_tensor *, llama_weight_extent> tensors;

    ~llama_weight_extents();

    // file and offset of bytes [p, p + size) if they are in a mapped model file
    bool resolve(const void * p, size_t size, uint16_t & file, uint64_t & offs) const;

    // the extent of src (also a view of a recorded tensor), nullptr if it is not a mapped host weight
    const llama_weight_extent * find(const ggml_tensor * src, uint64_t * offs = nullptr) const;

    // bytes of the mapped host weights that are not lazy (the bytes a prompt streams to the devices)
    size_t bytes_streamable() const;
};

enum llama_weight_stream_mode {
    LLAMA_WEIGHT_STREAM_OFF  = 0,
    LLAMA_WEIGHT_STREAM_AUTO = 1, // only when the mapped host weights do not fit in the available RAM
    LLAMA_WEIGHT_STREAM_ON   = 2,
};

enum llama_weight_stream_io {
    LLAMA_WEIGHT_STREAM_IO_AUTO     = 0, // prefetched: resident pages from the mapping, the rest unbuffered; demand reads buffered
    LLAMA_WEIGHT_STREAM_IO_BUFFERED = 1,
    LLAMA_WEIGHT_STREAM_IO_DIRECT   = 2, // O_DIRECT (Linux), F_NOCACHE (macOS); buffered where unsupported
};

struct llama_weight_store_params {
    int    io        = LLAMA_WEIGHT_STREAM_IO_AUTO;
    size_t ring_size = 0; // bytes of pinned staging (prefetch ring + demand area), 0 = default
    int    n_threads = 0; // IO threads, 0 = default
    size_t chunk     = 0; // bytes per read of the prefetch ring, 0 = default (16 MiB)
    bool   verify    = false; // compare every read with the mapping (LLAMA_WEIGHT_STREAM_VERIFY=1)
    bool   wrap      = true;  // read the next graph's copies ahead when the plan is repeated (LLAMA_WEIGHT_STREAM_WRAP=0 off)
    size_t ram_avail = 0; // available RAM for the auto IO mode, 0 = query
};

// a host tier in front of the model files (the RAM expert cache of llama_moe_cache): the reads ahead of a plan skip the bytes it holds,
// the uploads copy those from it, and the uploads from the ring offer it the bytes they read (all calls on the main thread)
class llama_weight_tier {
public:
    virtual ~llama_weight_tier() = default;

    struct run {
        size_t off;
        size_t size;
        const uint8_t * ptr;    // the bytes in the tier, nullptr if it does not hold them
        bool  device = false;   // planning only: the device holds these bytes, the uploads do not ask for them (not read ahead)
    };
    // the runs of bytes [off, off + size) of src, in order and covering them; lease: the held runs stay until the next new_plan
    virtual void runs(const ggml_tensor * src, size_t off, size_t size, bool lease, std::vector<run> & out) = 0;
    // a plan is built: the leases of the previous one end
    virtual void new_plan() = 0;
    // changes when the held bytes change (a plan is built again then)
    virtual uint64_t generation() const = 0;
    // bytes [off, off + n) of src were uploaded from p (ring or demand staging), the tier may keep a copy
    virtual void offer(const ggml_tensor * src, size_t off, size_t n, const uint8_t * p) = 0;
    // uploads from held runs were queued on backend since the last call
    virtual void uploaded(ggml_backend_t backend) = 0;
};

// test hook: replaces the positional read of the IO threads (returns bytes read or -errno)
typedef int64_t (*llama_weight_store_read_fn)(int fd, void * buf, size_t n, uint64_t off, bool direct, void * user_data);

class llama_weight_store {
public:
    llama_weight_store(std::shared_ptr<const llama_weight_extents> extents, const llama_weight_store_params & params);
    ~llama_weight_store();

    // mode from the context parameter and LLAMA_WEIGHT_STREAM (0 = off whatever the parameter says, 1 = on): true if the store is used
    // for a model whose mapped host weights take `bytes` with `avail` bytes of RAM available
    static bool enabled(int mode, size_t bytes, size_t avail);

    // available RAM (cgroup-aware on Linux)
    static size_t ram_available();

    // true if src is a mapped host weight the store can read
    bool covers(const ggml_tensor * src) const;

    // the host weights the prefetched copies of the graph that starts computing read, in the order they are issued
    // (ggml_backend_sched prefetch plan callback); n = 0 drops the reads ahead
    void plan(const ggml_tensor * const * srcs, int n);

    // upload bytes [off, off + size) of src to dst at dst_off on backend: from the prefetch ring when src is planned, else read on demand
    // false: src is not covered (or no pinned memory), the caller copies from src->data
    bool upload(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, size_t off, size_t dst_off, size_t size);

    struct region {
        const ggml_tensor * src;
        size_t              src_off;
        ggml_tensor *       dst;
        size_t              dst_off;
        size_t              size;
    };

    // demand uploads: the reads of all regions are issued together, then the uploads; false if a source is not covered (nothing uploaded)
    bool upload_batch(ggml_backend_t backend, const region * regions, size_t n);

    // planned reads: same as upload for every region of the same src (regions sorted by src_off), demand reads if src is not planned
    bool upload_runs(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<std::pair<size_t, size_t>> & runs);

    // reads into host memory (the slots of the RAM expert cache), all issued together; area is 4 KiB aligned and holds at least
    // size + 8 KiB; the bytes land at area + lead (set here); direct: unbuffered where supported (the tier is the cache)
    struct host_read {
        const ggml_tensor * src;
        size_t    src_off;
        size_t    size;
        uint8_t * area;
        size_t    lead; // out
    };
    bool read_host(host_read * reads, size_t n, bool direct);

    // the same reads, speculative (e.g. the experts a layer ahead is expected to route): queued behind every demand read and read
    // ahead, returns at once with a ticket (0: a source is not covered, nothing is read); the areas must stay valid until the ticket
    // is done or waited for (the store's destructor also waits for them)
    uint64_t read_host_async(host_read * reads, size_t n, bool direct);
    bool     host_done(uint64_t ticket); // true once its reads are complete (the ticket is then collected), never blocks
    void     host_wait(uint64_t ticket); // its reads that did not start move in front of the demand reads, then waits (collects it)

    // the tier in front of the model files, nullptr = none
    void set_tier(llama_weight_tier * tier);

    struct stats {
        uint64_t tier_skipped    = 0; // bytes the reads ahead skipped (held by the tier)
        uint64_t tier_uploaded   = 0; // bytes uploaded from the tier
        uint64_t host_reads      = 0; // read_host regions
        uint64_t host_read_bytes = 0;
        uint64_t plan_calls      = 0;
        uint64_t plan_reads      = 0; // reads of the prefetch ring
        uint64_t plan_bytes      = 0; // bytes read into the ring
        uint64_t plan_uploaded   = 0; // bytes uploaded from the ring
        uint64_t plan_discarded  = 0; // bytes read ahead and never uploaded
        uint64_t demand_regions  = 0;
        uint64_t demand_bytes    = 0;
        uint64_t direct_bytes    = 0;
        uint64_t buffered_bytes  = 0;
        uint64_t fallback_bytes  = 0; // copied from the mapping after a read error
        uint64_t io_errors       = 0;
        uint64_t short_reads     = 0;
        uint64_t direct_disabled = 0; // files whose unbuffered reads failed (EINVAL) and that are read buffered
        uint64_t verify_regions  = 0;
        uint64_t verify_bytes    = 0;
        uint64_t verify_mismatch = 0;
        uint64_t wait_us         = 0; // main thread waiting for reads
        uint64_t not_covered     = 0;
        uint64_t cached_bytes    = 0; // prefetched bytes copied from the resident pages of the mapping (auto)
        uint64_t spec_reads      = 0; // read_host_async reads done
        uint64_t spec_bytes      = 0; // read_host_async bytes issued
        uint64_t spec_boosted    = 0; // tickets waited for before all of their reads started
        uint64_t spec_wait_us    = 0; // main thread waiting in host_wait
    };
    stats get_stats() const;
    void  log_stats() const;

    // false: the next graphs are not read ahead after the current plan (e.g. the ubatch ends a prompt, generation follows)
    void set_wrap(bool wrap);

    // times the prefetch ring was allocated (it is released while graphs prefetch nothing, LLAMA_WEIGHT_STREAM_KEEP_RING=1 keeps it)
    uint64_t ring_allocs() const;

    void set_read_fn(llama_weight_store_read_fn fn, void * user_data); // tests

    // bytes of pinned staging, 0 before the first upload
    size_t pinned_size() const;

    // bytes of the mapped host weights the store can read (not lazy)
    size_t bytes_mapped() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
