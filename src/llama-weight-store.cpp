#include "llama-weight-store.h"

#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

//
// extents
//

llama_weight_extents::~llama_weight_extents() {
#ifndef _WIN32
    for (auto & f : files) {
        if (f.fd >= 0) {
            close(f.fd);
        }
    }
#endif
}

bool llama_weight_extents::resolve(const void * p, size_t size, uint16_t & file, uint64_t & offs) const {
    const uint8_t * q = (const uint8_t *) p;
    for (size_t i = 0; i < files.size(); ++i) {
        const llama_weight_file & f = files[i];
        if (f.map_addr != nullptr && q >= f.map_addr && q + size <= f.map_addr + std::min(f.map_size, f.size)) {
            file = (uint16_t) i;
            offs = (uint64_t) (q - f.map_addr);
            return true;
        }
    }
    return false;
}

const llama_weight_extent * llama_weight_extents::find(const ggml_tensor * src, uint64_t * offs) const {
    size_t view_offs = 0;
    const ggml_tensor * t = src;
    while (t->view_src != nullptr) {
        view_offs += t->view_offs;
        t = t->view_src;
    }
    const auto it = tensors.find(t);
    if (it == tensors.end()) {
        return nullptr;
    }
    if (offs != nullptr) {
        *offs = it->second.offs + view_offs;
    }
    return &it->second;
}

size_t llama_weight_extents::bytes_streamable() const {
    size_t n = 0;
    for (const auto & [t, e] : tensors) {
        if (!e.lazy) {
            n += e.size;
        }
    }
    return n;
}

//
// platform IO
//

namespace llama_weight_store_io {

static constexpr size_t k_align = 4096; // unbuffered reads: offset, size and address aligned to the logical block size (<= 4 KiB)

static size_t round_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}

struct file_state {
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    int fd_buf = -1;
    int fd_dir = -1;
#endif
    std::atomic<bool> dir_ok { true };
    std::mutex        mtx;  // opening the unbuffered descriptor
    bool              dir_tried = false;
};

static bool open_buffered(file_state & fs, const llama_weight_file & f) {
#ifdef _WIN32
    if (f.path.empty()) {
        return false;
    }
    fs.h = CreateFileA(f.path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    return fs.h != INVALID_HANDLE_VALUE;
#else
    if (!f.path.empty()) {
        fs.fd_buf = open(f.path.c_str(), O_RDONLY | O_CLOEXEC);
    }
    if (fs.fd_buf < 0 && f.fd >= 0) {
        fs.fd_buf = dup(f.fd);
    }
    if (fs.fd_buf < 0) {
        return false;
    }
#if defined(POSIX_FADV_RANDOM) && !defined(__APPLE__)
    // the reads are large and positioned, the kernel read-ahead would only add bytes nobody asked for
    posix_fadvise(fs.fd_buf, 0, 0, POSIX_FADV_RANDOM);
#endif
    return true;
#endif
}

// the unbuffered descriptor, opened on first use; -1 (and dir_ok = false) if the file system does not support it
static int fd_direct(file_state & fs, const llama_weight_file & f, std::atomic<uint64_t> & n_disabled) {
#ifdef _WIN32
    GGML_UNUSED(f);
    GGML_UNUSED(n_disabled);
    // TODO: FILE_FLAG_NO_BUFFERING + OVERLAPPED (sector-aligned buffers, the views of the file closed first)
    fs.dir_ok = false;
    return -1;
#else
    std::lock_guard<std::mutex> lock(fs.mtx);
    if (!fs.dir_tried) {
        fs.dir_tried = true;
#if defined(__linux__) && defined(O_DIRECT)
        std::string path = f.path;
        if (path.empty() && f.fd >= 0) {
            path = "/proc/self/fd/" + std::to_string(f.fd);
        }
        if (!path.empty()) {
            fs.fd_dir = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        }
#elif defined(__APPLE__)
        if (!f.path.empty()) {
            fs.fd_dir = open(f.path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fs.fd_dir >= 0 && fcntl(fs.fd_dir, F_NOCACHE, 1) != 0) {
                close(fs.fd_dir);
                fs.fd_dir = -1;
            }
        }
#endif
        if (fs.fd_dir < 0) {
            fs.dir_ok = false;
            n_disabled++;
        }
    }
    return fs.fd_dir;
#endif
}

static void close_file(file_state & fs) {
#ifdef _WIN32
    if (fs.h != INVALID_HANDLE_VALUE) {
        CloseHandle(fs.h);
    }
#else
    if (fs.fd_buf >= 0) {
        close(fs.fd_buf);
    }
    if (fs.fd_dir >= 0) {
        close(fs.fd_dir);
    }
#endif
}

// positional read: bytes read or -errno
static int64_t pread_os(file_state & fs, bool direct, void * buf, size_t n, uint64_t off) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    OVERLAPPED ov = {};
    ov.Offset     = (DWORD) (off & 0xffffffffu);
    ov.OffsetHigh = (DWORD) (off >> 32);
    DWORD got = 0;
    const DWORD want = (DWORD) std::min<size_t>(n, 1u << 30);
    if (!ReadFile(fs.h, buf, want, &got, &ov)) {
        const DWORD err = GetLastError();
        return err == ERROR_HANDLE_EOF ? 0 : -EIO;
    }
    return (int64_t) got;
#else
    const int fd = direct ? fs.fd_dir : fs.fd_buf;
    const ssize_t r = pread(fd, buf, std::min<size_t>(n, (size_t) 1 << 30), (off_t) off);
    return r < 0 ? -(int64_t) errno : (int64_t) r;
#endif
}

// page residency of a mapped range (mincore), false if unknown
static bool residency(const uint8_t * addr, size_t n, unsigned char * vec) {
#if defined(__APPLE__)
    return mincore(const_cast<uint8_t *>(addr), n, (char *) vec) == 0;
#elif defined(__linux__)
    return mincore(const_cast<uint8_t *>(addr), n, vec) == 0;
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(n);
    GGML_UNUSED(vec);
    return false;
#endif
}

static size_t page_size() {
#ifdef _WIN32
    return 4096;
#else
    static const size_t pg = (size_t) std::max(4096L, sysconf(_SC_PAGESIZE));
    return pg;
#endif
}

static bool mixed_supported() {
#if defined(__linux__) || defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

} // namespace llama_weight_store_io

//
// store
//

using namespace llama_weight_store_io;

namespace {

enum job_state { JOB_PENDING = 0, JOB_READING = 1, JOB_READY = 2 };

enum plan_io_mode { PLAN_IO_BUFFERED = 0, PLAN_IO_DIRECT = 1, PLAN_IO_MIXED = 2 };

// a read of the prefetch ring: bytes [off, off + len) of src
struct ws_job {
    uint64_t seq = 0;
    uint64_t gen = 0;
    const ggml_tensor * src = nullptr;
    size_t   off  = 0;
    size_t   len  = 0;
    uint16_t file = 0;
    uint64_t foff = 0;

    bool   allocated = false;
    size_t ring_off  = 0; // aligned start of the read in the ring, the bytes are at ring_off + lead
    size_t lead      = 0;
    size_t charge    = 0; // ring bytes held until the job is freed (with the gap left at the end of the ring)

    int  state    = JOB_PENDING;
    bool consumed = false; // the main thread will not upload from it anymore
    bool used     = false; // uploaded from at least once

    ggml_backend_event_t ev     = nullptr; // recorded after the uploads from it
    ggml_backend_dev_t   ev_dev = nullptr;
};

struct ws_plan_entry {
    const ggml_tensor * src;
    size_t   off;
    size_t   len;
    uint16_t file;
    uint64_t foff;
};

// a demand read of the IO threads
struct ws_task {
    uint16_t  file;
    uint64_t  foff;
    size_t    len;
    size_t    lead;
    uint8_t * base;   // aligned, the bytes go to base + lead
    bool      direct;
    std::atomic<int> * remaining;
};

} // namespace

struct llama_weight_store::impl {
    std::shared_ptr<const llama_weight_extents> ext;
    llama_weight_store_params params;

    std::vector<std::unique_ptr<file_state>> files;

    // pinned staging
    ggml_backend_buffer_type_t pinned_buft = nullptr;
    ggml_backend_buffer_t ring_buf = nullptr;
    ggml_backend_buffer_t dem_buf  = nullptr;
    size_t   ring_size_cfg = 0;
    bool     pinned_failed = false;
    bool     ring_failed   = false;
    bool     keep_ring     = false; // LLAMA_WEIGHT_STREAM_KEEP_RING=1: the ring is never released
    int      idle_plans    = 0;
    std::atomic<bool> wrap_hint { true };
    uint8_t * ring_base  = nullptr;
    size_t    ring_cap   = 0;
    uint8_t * dem_base   = nullptr;
    size_t    dem_half   = 0;
    std::vector<std::pair<ggml_backend_dev_t, ggml_backend_event_t>> dem_pending[2]; // uploads still reading a half of the demand area
    int       dem_next = 0;

    // event pool per device
    std::unordered_map<ggml_backend_dev_t, std::vector<ggml_backend_event_t>> ev_pool;
    std::vector<ggml_backend_event_t> ev_all; // freed at the end

    // ring state (mtx)
    mutable std::mutex mtx;
    std::condition_variable cv_io;
    std::condition_variable cv_main;
    std::deque<ws_job> jobs;
    uint64_t base_seq  = 0; // seq of jobs.front()
    uint64_t next_seq  = 0; // seq of the next appended job
    uint64_t alloc_seq = 0; // next job to allocate
    uint64_t cursor    = 0; // first job the main thread may still upload from
    uint64_t gen_consume = 0;
    uint64_t gen_last    = 0;
    uint64_t gen_counter = 0;
    size_t   ring_head = 0;
    size_t   ring_used = 0;
    bool     syncing   = false;
    bool     stop      = false;
    int      plan_io = PLAN_IO_BUFFERED; // reads of the prefetch ring (see plan)
    std::vector<const ggml_tensor *> cur_plan;
    std::vector<ws_plan_entry>       plan_entries;
    std::deque<ws_task *> demand;
    // speculative reads into host memory (read_host_async): taken only when no demand read waits and the ring has nothing to read
    struct spec_req {
        std::vector<ws_task> tasks;
        std::atomic<int>     remaining{0};
        size_t               bytes = 0;
    };
    std::deque<ws_task *> spec;
    std::unordered_map<uint64_t, std::unique_ptr<spec_req>> spec_reqs; // tickets not collected yet (main thread)
    uint64_t spec_next = 1;

    std::vector<std::thread> threads;

    // statistics (atomics: updated by the IO threads)
    std::atomic<uint64_t> s_plan_reads{0}, s_plan_bytes{0}, s_plan_uploaded{0}, s_plan_discarded{0}, s_demand_regions{0}, s_demand_bytes{0},
        s_direct_bytes{0}, s_buffered_bytes{0}, s_fallback_bytes{0}, s_io_errors{0}, s_short_reads{0}, s_direct_disabled{0},
        s_verify_regions{0}, s_verify_bytes{0}, s_verify_mismatch{0}, s_wait_us{0}, s_not_covered{0}, s_plan_calls{0}, s_ring_allocs{0}, s_cached_bytes{0};
    std::atomic<int> logged_errors{0};

    llama_weight_store_read_fn read_fn = nullptr;
    void * read_fn_ud = nullptr;

    llama_weight_tier * tier = nullptr;
    uint64_t plan_tier_gen = 0; // generation of the tier when the plan was built
    std::atomic<uint64_t> s_tier_skipped{0}, s_tier_uploaded{0}, s_host_reads{0}, s_host_read_bytes{0}, s_device_skipped{0};
    std::atomic<uint64_t> s_spec_reads{0}, s_spec_bytes{0}, s_spec_boosted{0}, s_spec_wait_us{0}, s_spec_cancelled_bytes{0};

    impl(std::shared_ptr<const llama_weight_extents> e, const llama_weight_store_params & p) : ext(std::move(e)), params(p) {
        if (params.chunk == 0) {
            params.chunk = 16u << 20;
        }
        params.chunk = round_up(params.chunk, k_align);
        if (params.n_threads <= 0) {
            const int hw = (int) std::max(1u, std::thread::hardware_concurrency());
            params.n_threads = std::max(2, std::min(8, hw - 1));
        }
        if (params.ram_avail == 0) {
            params.ram_avail = llama_weight_store::ram_available();
        }
        if (const char * e = getenv("LLAMA_WEIGHT_STREAM_KEEP_RING")) {
            keep_ring = atoi(e) != 0;
        }
        for (const auto & f : ext->files) {
            files.emplace_back(new file_state());
            if (!open_buffered(*files.back(), f)) {
                LLAMA_LOG_WARN("%s: cannot open %s for reading, its weights are copied from the mapping\n", __func__, f.path.c_str());
            }
        }
        for (int i = 0; i < params.n_threads; ++i) {
            threads.emplace_back([this]() { io_loop(); });
        }
    }

    ~impl() {
        {
            // the speculative reads write memory of the caller: every one is done before the store goes
            std::unique_lock<std::mutex> lock(mtx);
            while (!spec.empty()) {
                demand.push_back(spec.front());
                spec.pop_front();
            }
            cv_io.notify_all();
            cv_main.wait(lock, [&]() {
                for (const auto & [id, r] : spec_reqs) {
                    if (r->remaining.load() != 0) {
                        return false;
                    }
                }
                return true;
            });
            spec_reqs.clear();
            stop = true;
        }
        cv_io.notify_all();
        for (auto & t : threads) {
            t.join();
        }
        for (auto * ev : ev_all) {
            ggml_backend_event_free(ev);
        }
        if (ring_buf != nullptr) {
            ggml_backend_buffer_free(ring_buf);
        }
        if (dem_buf != nullptr) {
            ggml_backend_buffer_free(dem_buf);
        }
        for (auto & f : files) {
            close_file(*f);
        }
    }

    // pinned staging from the host buffer type of the device of backend (plain host memory for a CPU backend: tests)
    // the demand area stays, the prefetch ring is released while graphs prefetch nothing (see plan) and allocated again for the next prompt
    bool ensure_pinned(ggml_backend_t backend) {
        if (dem_buf != nullptr) {
            return true; // the ring is allocated again by the next prefetching plan (upload_planned reads on demand without it)
        }
        if (pinned_failed) {
            return false;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (buft == nullptr && dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft = ggml_backend_cpu_buffer_type();
        }
        if (buft == nullptr) {
            pinned_failed = true;
            LLAMA_LOG_WARN("%s: %s has no pinned host memory, the weights are copied from the mapping\n", __func__, ggml_backend_name(backend));
            return false;
        }
        size_t total = params.ring_size;
        if (total == 0) {
            // default: a sixteenth of the available RAM, at most 4 GiB, at least 512 MiB
            total = std::max<size_t>(512u << 20, std::min<size_t>((size_t) 4 << 30, params.ram_avail / 16));
        }
        size_t dem = std::min<size_t>(256u << 20, total / 4);
        dem = std::max<size_t>(dem, 2*params.chunk);
        size_t ring = total > dem ? total - dem : 0;
        ring = std::max<size_t>(ring, 4*params.chunk);
        pinned_buft = buft;
        ring_size_cfg = round_up(ring, k_align);
        dem = round_up(dem, 2*k_align);
        dem_buf = ggml_backend_buft_alloc_buffer(buft, dem + k_align);
        if (dem_buf == nullptr) {
            pinned_failed = true;
            LLAMA_LOG_WARN("%s: failed to allocate %.1f MiB of pinned memory, the weights are copied from the mapping\n", __func__, dem/1048576.0);
            return false;
        }
        dem_base = (uint8_t *) round_up((size_t) ggml_backend_buffer_get_base(dem_buf), k_align);
        dem_half = dem / 2;
        LLAMA_LOG_INFO("%s: %s: %.1f MiB ring + %.1f MiB demand staging (%s), %d IO threads, %zu MiB reads, io %s\n", __func__,
                ggml_backend_name(backend), ring_size_cfg/1048576.0, dem/1048576.0, ggml_backend_buft_name(buft), params.n_threads, params.chunk >> 20,
                params.io == LLAMA_WEIGHT_STREAM_IO_DIRECT ? "direct" : params.io == LLAMA_WEIGHT_STREAM_IO_BUFFERED ? "buffered" : "auto");
        return alloc_ring();
    }

    bool alloc_ring() {
        if (ring_buf != nullptr) {
            return true;
        }
        if (pinned_buft == nullptr || ring_failed) {
            return false;
        }
        ring_buf = ggml_backend_buft_alloc_buffer(pinned_buft, ring_size_cfg + k_align);
        if (ring_buf == nullptr) {
            ring_failed = true;
            LLAMA_LOG_WARN("%s: failed to allocate %.1f MiB of pinned memory for the prefetch ring, prompts read on demand\n", __func__, ring_size_cfg/1048576.0);
            return false;
        }
        {
            // published to the IO threads, which read ring_cap under mtx
            std::lock_guard<std::mutex> lock(mtx);
            ring_base = (uint8_t *) round_up((size_t) ggml_backend_buffer_get_base(ring_buf), k_align);
            ring_cap  = ring_size_cfg;
            ring_head = 0;
            ring_used = 0;
        }
        cv_io.notify_all();
        s_ring_allocs++;
        return true;
    }

    // main thread, graphs without prefetched copies: the ring goes back to the page cache once nothing uses it
    void release_ring_if_idle() {
        if (ring_buf == nullptr || keep_ring) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!jobs.empty() || ring_used != 0) {
                return;
            }
            ring_cap  = 0;
            ring_base = nullptr;
        }
        ggml_backend_buffer_free(ring_buf);
        ring_buf = nullptr;
    }

    // an event of dev from the pool (nullptr if the device has no events: the caller synchronizes the backend)
    ggml_backend_event_t ev_get(ggml_backend_dev_t dev) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto & pool = ev_pool[dev];
            if (!pool.empty()) {
                ggml_backend_event_t ev = pool.back();
                pool.pop_back();
                return ev;
            }
        }
        ggml_backend_event_t ev = ggml_backend_event_new(dev);
        if (ev != nullptr) {
            std::lock_guard<std::mutex> lock(mtx);
            ev_all.push_back(ev);
        }
        return ev;
    }

    // reads [foff, foff + len) of file into base + lead; buffered after an unbuffered failure; false if the file cannot be read
    bool read_range(uint16_t file, uint64_t foff, size_t len, uint8_t * base, size_t lead, bool direct) {
        file_state & fs = *files[file];
        const llama_weight_file & f = ext->files[file];
        auto rd = [&](bool dir, void * buf, size_t n, uint64_t off) -> int64_t {
            if (read_fn != nullptr) {
#ifdef _WIN32
                const int fd = -1;
#else
                const int fd = dir ? fs.fd_dir : fs.fd_buf;
#endif
                return read_fn(fd, buf, n, off, dir, read_fn_ud);
            }
            return pread_os(fs, dir, buf, n, off);
        };

        size_t done = 0; // bytes of [foff, foff + len) in place
        if (direct && fs.dir_ok.load() && fd_direct(fs, f, s_direct_disabled) >= 0) {
            const uint64_t start = foff - lead;
            const size_t   want  = lead + len;
            const size_t   total = round_up(want, k_align);
            size_t got = 0;
            while (got < want) {
                const int64_t r = rd(true, base + got, total - got, start + got);
                if (r == -EINTR) {
                    continue;
                }
                if (r == -EINVAL) {
                    // the file system or the device refuses this alignment: buffered from now on
                    if (fs.dir_ok.exchange(false)) {
                        s_direct_disabled++;
                        LLAMA_LOG_WARN("%s: unbuffered reads of %s failed (EINVAL), reading it buffered\n", __func__, f.path.c_str());
                    }
                    break;
                }
                if (r < 0) {
                    s_io_errors++;
                    log_error("unbuffered read", f, start + got, total - got, (int) -r);
                    break;
                }
                if (r == 0) {
                    break; // end of file before the bytes: buffered read reports it
                }
                got += (size_t) r;
                if (got < want && got % k_align != 0) {
                    s_short_reads++;
                    break; // the next unbuffered read would be unaligned
                }
                if (got < want) {
                    s_short_reads++;
                }
            }
            if (got >= want) {
                s_direct_bytes += len;
                return true;
            }
            done = got > lead ? got - lead : 0;
        }

        int retries = 1;
        const size_t done0 = done;
        while (done < len) {
            const int64_t r = rd(false, base + lead + done, len - done, foff + done);
            if (r == -EINTR) {
                continue;
            }
            if (r <= 0) {
                if (r < 0) {
                    s_io_errors++;
                    log_error("read", f, foff + done, len - done, (int) -r);
                } else {
                    log_error("read (end of file)", f, foff + done, len - done, 0);
                }
                if (retries-- > 0) {
                    continue;
                }
                return false;
            }
            if ((size_t) r < len - done) {
                s_short_reads++;
            }
            done += (size_t) r;
        }
        s_buffered_bytes += len - done0;
        s_direct_bytes   += done0;
        return true;
    }
    void log_error(const char * what, const llama_weight_file & f, uint64_t off, size_t n, int err) {
        if (logged_errors.fetch_add(1) < 8) {
            LLAMA_LOG_WARN("%s: %s of %zu bytes at %llu of %s failed: %s\n", __func__, what, n, (unsigned long long) off, f.path.c_str(),
                    err ? strerror(err) : "short");
        }
    }

    // read + fallback + verify: the bytes [foff, foff + len) of file end at base + lead, whatever happens
    void read_exact(uint16_t file, uint64_t foff, size_t len, uint8_t * base, size_t lead, bool direct) {
        const llama_weight_file & f = ext->files[file];
        const uint8_t * map = f.map_addr != nullptr ? f.map_addr + foff : nullptr;
        if (!read_range(file, foff, len, base, lead, direct)) {
            // the mapping has the same bytes (the page faults read them)
            GGML_ASSERT(map != nullptr && "weight store: read failed and the file is not mapped");
            memcpy(base + lead, map, len);
            s_fallback_bytes += len;
        }
        if (params.verify && map != nullptr) {
            s_verify_regions++;
            s_verify_bytes += len;
            if (memcmp(base + lead, map, len) != 0) {
                if (s_verify_mismatch.fetch_add(1) < 8) {
                    LLAMA_LOG_ERROR("%s: VERIFY: %zu bytes at %llu of %s differ from the mapping\n", __func__, len, (unsigned long long) foff, f.path.c_str());
                }
                memcpy(base + lead, map, len);
            }
        }
    }

    // the resident pages of [foff, foff + len) are copied from the mapping, the runs of non-resident pages are read unbuffered
    // (buffered if the residency is unknown)
    void read_mixed(uint16_t file, uint64_t foff, size_t len, uint8_t * base, size_t lead) {
        const llama_weight_file & f = ext->files[file];
        const size_t   pg    = page_size();
        const uint64_t start = foff - lead; // 4 KiB aligned, at base
        const uint64_t q0    = start / pg * pg; // residency granularity: the system pages
        const uint64_t dend  = foff + len;
        const uint64_t q1    = round_up(dend, pg);
        const size_t   npg   = (size_t) ((q1 - q0) / pg);
        std::vector<unsigned char> & vec = tls_vec();
        vec.resize(npg);
        if (f.map_addr == nullptr || q1 > round_up(f.map_size, pg) || !residency(f.map_addr + q0, (size_t) (q1 - q0), vec.data())) {
            read_exact(file, foff, len, base, lead, false);
            return;
        }
        for (size_t p0 = 0; p0 < npg; ) {
            const bool res = (vec[p0] & 1) != 0;
            size_t p1 = p0 + 1;
            while (p1 < npg && ((vec[p1] & 1) != 0) == res) {
                p1++;
            }
            const uint64_t c0 = std::max<uint64_t>(q0 + (uint64_t) p0*pg, foff);
            const uint64_t c1 = std::min<uint64_t>(q0 + (uint64_t) p1*pg, dend);
            if (c1 > c0) {
                if (res) {
                    // may fault if the page was evicted since: the mapping reads it, the bytes are the same
                    memcpy(base + (c0 - start), f.map_addr + c0, (size_t) (c1 - c0));
                    s_cached_bytes += c1 - c0;
                } else {
                    // unbuffered from the 4 KiB boundary at or before c0 (inside the ring slot, which starts at start)
                    const uint64_t a0 = std::max<uint64_t>(c0 / k_align * k_align, start);
                    if (!read_range(file, c0, (size_t) (c1 - c0), base + (a0 - start), (size_t) (c0 - a0), true)) {
                        memcpy(base + (c0 - start), f.map_addr + c0, (size_t) (c1 - c0));
                        s_fallback_bytes += c1 - c0;
                    }
                }
            }
            p0 = p1;
        }
        if (params.verify) {
            s_verify_regions++;
            s_verify_bytes += len;
            if (memcmp(base + lead, f.map_addr + foff, len) != 0) {
                if (s_verify_mismatch.fetch_add(1) < 8) {
                    LLAMA_LOG_ERROR("%s: VERIFY: %zu bytes at %llu of %s differ from the mapping\n", __func__, len, (unsigned long long) foff, f.path.c_str());
                }
                memcpy(base + lead, f.map_addr + foff, len);
            }
        }
    }

    static std::vector<unsigned char> & tls_vec() {
        static thread_local std::vector<unsigned char> v;
        return v;
    }

    bool file_direct_ok(uint16_t file) const {
        return files[file]->dir_ok.load();
    }

    ws_job & job_at(uint64_t seq) {
        return jobs[(size_t) (seq - base_seq)];
    }

    // ring allocation in job order (FIFO), under mtx
    bool try_alloc(ws_job & j) {
        const size_t lead  = (size_t) (j.foff % k_align);
        const size_t alloc = round_up(lead + j.len, k_align);
        if (ring_used == 0) {
            ring_head = 0;
        }
        // the job wraps to the start when it does not fit before the end (also when the previous one ends exactly there: the gap is 0
        // then, and placing the job at ring_head = ring_cap read past the ring)
        const bool wrap = ring_head + alloc > ring_cap;
        const size_t gap = wrap ? ring_cap - ring_head : 0;
        if (ring_used + gap + alloc > ring_cap) {
            return false;
        }
        j.ring_off  = wrap ? 0 : ring_head;
        GGML_ASSERT(j.ring_off + alloc <= ring_cap);
        j.lead      = lead;
        j.charge    = gap + alloc;
        j.allocated = true;
        ring_head   = j.ring_off + alloc;
        ring_used  += j.charge;
        return true;
    }

    // frees the consumed jobs at the front of the ring; may release mtx to wait for the uploads from a job (lock is held on return)
    void free_front(std::unique_lock<std::mutex> & lock) {
        while (!jobs.empty() && !syncing) {
            ws_job & j = jobs.front();
            // a consumed job that was never allocated is PENDING, one being read waits for its read
            if (!j.consumed || j.state == JOB_READING) {
                return;
            }
            if (j.ev != nullptr) {
                // the uploads that read the job were queued by the main thread before it marked the job consumed
                ggml_backend_event_t ev = j.ev;
                syncing = true;
                lock.unlock();
                ggml_backend_event_synchronize(ev);
                lock.lock();
                syncing = false;
                ws_job & jf = jobs.front();
                ev_pool[jf.ev_dev].push_back(jf.ev);
                jf.ev = nullptr;
                continue;
            }
            if (!j.used && j.allocated && j.state == JOB_READY) {
                s_plan_discarded += j.len;
            }
            if (j.allocated) {
                ring_used -= j.charge;
            }
            jobs.pop_front();
            base_seq++;
            if (alloc_seq < base_seq) {
                alloc_seq = base_seq;
            }
            if (cursor < base_seq) {
                cursor = base_seq;
            }
            cv_main.notify_all();
            cv_io.notify_all(); // the other IO threads may allocate the space
        }
    }

    void append_generation(uint64_t gen) {
        for (const ws_plan_entry & e : plan_entries) {
            ws_job j;
            j.seq  = next_seq++;
            j.gen  = gen;
            j.src  = e.src;
            j.off  = e.off;
            j.len  = e.len;
            j.file = e.file;
            j.foff = e.foff;
            jobs.push_back(j);
        }
    }

    void io_loop() {
        std::unique_lock<std::mutex> lock(mtx);
        while (true) {
            if (stop) {
                return;
            }
            if (!demand.empty()) {
                ws_task * t = demand.front();
                demand.pop_front();
                lock.unlock();
                read_exact(t->file, t->foff, t->len, t->base, t->lead, t->direct && file_direct_ok(t->file));
                lock.lock();
                if (t->remaining->fetch_sub(1) == 1) {
                    cv_main.notify_all();
                }
                continue;
            }
            free_front(lock);
            if (stop) {
                return;
            }
            // the jobs dropped before they were allocated are skipped
            while (alloc_seq < next_seq && job_at(alloc_seq).consumed && !job_at(alloc_seq).allocated) {
                job_at(alloc_seq).state = JOB_READY;
                alloc_seq++;
            }
            // the next graph with the same plan: its reads follow
            if (alloc_seq == next_seq && params.wrap && wrap_hint.load() && !plan_entries.empty() && gen_last == gen_consume && ring_cap > 0) {
                gen_last = ++gen_counter;
                append_generation(gen_last);
            }
            if (alloc_seq < next_seq && ring_cap > 0) {
                ws_job & j = job_at(alloc_seq);
                if (try_alloc(j)) {
                    j.state = JOB_READING;
                    alloc_seq++;
                    const uint16_t file = j.file;
                    const uint64_t foff = j.foff;
                    const size_t   len  = j.len;
                    uint8_t *      base = ring_base + j.ring_off;
                    const size_t   lead = j.lead;
                    const uint64_t seq  = j.seq;
                    const int  pio      = plan_io;
                    lock.unlock();
                    if (pio == PLAN_IO_MIXED && file_direct_ok(file)) {
                        read_mixed(file, foff, len, base, lead);
                    } else {
                        read_exact(file, foff, len, base, lead, pio == PLAN_IO_DIRECT && file_direct_ok(file));
                    }
                    s_plan_reads++;
                    s_plan_bytes += len;
                    lock.lock();
                    job_at(seq).state = JOB_READY;
                    cv_main.notify_all();
                    continue;
                }
            }
            if (!spec.empty()) {
                ws_task * t = spec.front();
                spec.pop_front();
                lock.unlock();
                read_exact(t->file, t->foff, t->len, t->base, t->lead, t->direct && file_direct_ok(t->file));
                s_spec_reads++;
                lock.lock();
                if (t->remaining->fetch_sub(1) == 1) {
                    cv_main.notify_all();
                }
                continue;
            }
            cv_io.wait(lock);
        }
    }

    // the main thread records an event after its uploads from the jobs (synchronizes the backend if the device has none)
    void record_jobs(ggml_backend_t backend, const std::vector<uint64_t> & seqs, std::unique_lock<std::mutex> & lock) {
        if (seqs.empty()) {
            return;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        lock.unlock();
        std::vector<ggml_backend_event_t> evs;
        bool synced = false;
        for (size_t i = 0; i < seqs.size(); ++i) {
            ggml_backend_event_t ev = nullptr;
            {
                std::lock_guard<std::mutex> lg(mtx);
                ev = job_at(seqs[i]).ev;
            }
            if (ev == nullptr) {
                ev = ev_get(dev);
            }
            if (ev != nullptr) {
                ggml_backend_event_record(ev, backend);
            } else if (!synced) {
                ggml_backend_synchronize(backend);
                synced = true;
            }
            evs.push_back(ev);
        }
        lock.lock();
        for (size_t i = 0; i < seqs.size(); ++i) {
            ws_job & j = job_at(seqs[i]);
            j.ev     = evs[i];
            j.ev_dev = dev;
        }
    }

    // uploads from the prefetch ring; returns the bytes uploaded from the start of the range, 0 if it is not planned in the current
    // generation (the rest of a range that the plan covers only in part, e.g. bytes a tier held when it was planned, is read on demand)
    size_t upload_planned(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, size_t off, size_t dst_off, size_t size) {
        std::unique_lock<std::mutex> lock(mtx);
        if (ring_cap == 0) {
            return 0; // no ring (allocation failed): read on demand
        }
        const size_t off0 = off;
        uint64_t s = std::max(cursor, base_seq);
        while (s < next_seq) {
            const ws_job & j = job_at(s);
            if (j.gen == gen_consume && j.src == src && !j.consumed && j.off + j.len > off) {
                break;
            }
            if (j.gen > gen_consume) {
                s = next_seq;
                break;
            }
            s++;
        }
        if (s >= next_seq || job_at(s).off > off) {
            return 0;
        }
        // the jobs before it are skipped (copied from elsewhere, e.g. the MoE cache, or not needed)
        for (uint64_t k = std::max(cursor, base_seq); k < s; ++k) {
            job_at(k).consumed = true;
        }
        cursor = s;
        cv_io.notify_all();

        const size_t end = off + size;
        std::vector<uint64_t> touched;
        std::vector<uint64_t> done;
        for (uint64_t k = s; k < next_seq && off < end; ++k) {
            ws_job & j = job_at(k);
            if (j.src != src || j.gen != gen_consume) {
                break;
            }
            if (j.off + j.len <= off) {
                j.consumed = true;
                continue;
            }
            if (j.off > off) {
                break; // a gap in the plan
            }
            if (j.state != JOB_READY) {
                // the uploads of the jobs finished so far are recorded first, so the ring can move on
                record_jobs(backend, done, lock);
                for (uint64_t d : done) {
                    job_at(d).consumed = true;
                }
                done.clear();
                cv_io.notify_all();
                const auto t0 = std::chrono::steady_clock::now();
                cv_main.wait(lock, [&]() { return job_at(k).state == JOB_READY; });
                s_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            }
            ws_job & jr = job_at(k);
            const size_t n = std::min(end, jr.off + jr.len) - off;
            const uint8_t * p = ring_base + jr.ring_off + jr.lead + (off - jr.off);
            jr.used = true;
            const size_t d_off = dst_off + (size - (end - off));
            lock.unlock();
            ggml_backend_tensor_set_async(backend, dst, p, d_off, n);
            if (tier != nullptr) {
                tier->offer(src, off, n, p);
            }
            lock.lock();
            s_plan_uploaded += n;
            off += n;
            if (job_at(k).off + job_at(k).len <= end) {
                done.push_back(k);
            } else {
                touched.push_back(k);
            }
        }
        std::vector<uint64_t> all = done;
        all.insert(all.end(), touched.begin(), touched.end());
        record_jobs(backend, all, lock);
        for (uint64_t d : done) {
            job_at(d).consumed = true;
        }
        // cursor: first job not consumed
        while (cursor < next_seq && job_at(cursor).consumed) {
            cursor++;
        }
        cv_io.notify_all();
        return off - off0;
    }

    // an upload through the tier: the held runs are copied from it, the others from the ring or read on demand (appended to rest)
    void upload_tiered(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, size_t off, size_t dst_off, size_t size, std::vector<region> & rest) {
        std::vector<llama_weight_tier::run> & tr = tls_runs();
        tr.clear();
        tier->runs(src, off, size, false, tr);
        bool held = false;
        for (const auto & r : tr) {
            const size_t d = dst_off + (r.off - off);
            if (r.ptr != nullptr) {
                ggml_backend_tensor_set_async(backend, dst, r.ptr, d, r.size);
                s_tier_uploaded += r.size;
                held = true;
                continue;
            }
            const size_t n = upload_planned(backend, src, dst, r.off, d, r.size);
            if (n < r.size) {
                rest.push_back({ src, r.off + n, dst, d + n, r.size - n });
            }
        }
        if (held) {
            tier->uploaded(backend);
        }
    }

    static std::vector<llama_weight_tier::run> & tls_runs() {
        static thread_local std::vector<llama_weight_tier::run> v;
        return v;
    }

    // reads into host memory, see llama_weight_store::read_host
    bool read_host(host_read * reads, size_t n, bool direct) {
        std::vector<ws_task> tasks;
        tasks.reserve(n);
        std::atomic<int> remaining((int) n);
        for (size_t i = 0; i < n; ++i) {
            uint64_t offs = 0;
            const llama_weight_extent * e = ext->find(reads[i].src, &offs);
            if (e == nullptr || reads[i].src_off + reads[i].size > ggml_nbytes(reads[i].src)) {
                s_not_covered++;
                return false;
            }
            const uint64_t foff = offs + reads[i].src_off;
            reads[i].lead = (size_t) (foff % k_align);
            tasks.push_back({ e->file, foff, reads[i].size, reads[i].lead, reads[i].area, direct, &remaining });
        }
        if (n == 0) {
            return true;
        }
        std::unique_lock<std::mutex> lock(mtx);
        for (auto & t : tasks) {
            demand.push_back(&t);
        }
        cv_io.notify_all();
        const auto t0 = std::chrono::steady_clock::now();
        cv_main.wait(lock, [&]() { return remaining.load() == 0; });
        s_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        s_host_reads += n;
        for (size_t i = 0; i < n; ++i) {
            s_host_read_bytes += reads[i].size;
        }
        return true;
    }

    // speculative reads into host memory, see llama_weight_store::read_host_async
    uint64_t read_host_async(host_read * reads, size_t n, bool direct) {
        auto req = std::make_unique<spec_req>();
        req->tasks.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            uint64_t offs = 0;
            const llama_weight_extent * e = ext->find(reads[i].src, &offs);
            if (e == nullptr || reads[i].src_off + reads[i].size > ggml_nbytes(reads[i].src)) {
                s_not_covered++;
                return 0;
            }
            const uint64_t foff = offs + reads[i].src_off;
            reads[i].lead = (size_t) (foff % k_align);
            req->tasks.push_back({ e->file, foff, reads[i].size, reads[i].lead, reads[i].area, direct, &req->remaining });
            req->bytes += reads[i].size;
        }
        if (n == 0) {
            return 0;
        }
        req->remaining.store((int) n);
        std::lock_guard<std::mutex> lock(mtx);
        const uint64_t id = spec_next++;
        for (auto & t : req->tasks) {
            spec.push_back(&t);
        }
        s_spec_bytes += req->bytes;
        spec_reqs.emplace(id, std::move(req));
        cv_io.notify_all();
        return id;
    }

    bool host_done(uint64_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        const auto it = spec_reqs.find(id);
        if (it == spec_reqs.end()) {
            return true;
        }
        if (it->second->remaining.load() != 0) {
            return false;
        }
        spec_reqs.erase(it);
        return true;
    }

    bool host_cancel(uint64_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        const auto it = spec_reqs.find(id);
        if (it == spec_reqs.end()) {
            return false;
        }
        spec_req & r = *it->second;
        // only if every read is still queued (remaining counts completions only: a read in progress is off the queue)
        const ws_task * b = r.tasks.data();
        const ws_task * e = b + r.tasks.size();
        size_t n = 0;
        for (const ws_task * t : spec) {
            n += t >= b && t < e;
        }
        if (n != r.tasks.size()) {
            return false; // a read started (or all are done): it completes
        }
        spec.erase(std::remove_if(spec.begin(), spec.end(), [&](const ws_task * t) { return t >= b && t < e; }), spec.end());
        s_spec_cancelled_bytes += r.bytes;
        spec_reqs.erase(it);
        return true;
    }

    void host_wait(uint64_t id) {
        std::unique_lock<std::mutex> lock(mtx);
        const auto it = spec_reqs.find(id);
        if (it == spec_reqs.end()) {
            return;
        }
        spec_req & r = *it->second;
        if (r.remaining.load() != 0) {
            // its reads that did not start go in front of the demand reads: the caller needs them now
            const ws_task * b = r.tasks.data();
            const ws_task * e = b + r.tasks.size();
            bool moved = false;
            for (auto q = spec.begin(); q != spec.end(); ) {
                if (*q >= b && *q < e) {
                    demand.push_front(*q);
                    q = spec.erase(q);
                    moved = true;
                } else {
                    ++q;
                }
            }
            if (moved) {
                s_spec_boosted++;
                cv_io.notify_all();
            }
            const auto t0 = std::chrono::steady_clock::now();
            cv_main.wait(lock, [&]() { return r.remaining.load() == 0; });
            s_spec_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        }
        spec_reqs.erase(id);
    }

    // demand reads: all regions read in parallel into a half of the demand area, then uploaded
    bool upload_demand(ggml_backend_t backend, const region * regions, size_t n) {
        std::vector<ws_task> tasks;
        struct item { const region * r; uint16_t file; uint64_t foff; };
        std::vector<item> items;
        items.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            uint64_t offs = 0;
            const llama_weight_extent * e = ext->find(regions[i].src, &offs);
            if (e == nullptr || regions[i].src_off + regions[i].size > ggml_nbytes(regions[i].src)) {
                s_not_covered++;
                return false;
            }
            items.push_back({ &regions[i], e->file, offs + regions[i].src_off });
        }
        // with a host tier the demand reads are the experts it does not keep: unbuffered unless buffered reads were asked for
        const bool direct = params.io == LLAMA_WEIGHT_STREAM_IO_DIRECT || (tier != nullptr && params.io == LLAMA_WEIGHT_STREAM_IO_AUTO);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);

        size_t i = 0;
        size_t split_off = 0; // progress inside a region larger than a half
        while (i < items.size()) {
            const int h = dem_next;
            dem_next ^= 1;
            for (auto & [edev, ev] : dem_pending[h]) {
                ggml_backend_event_synchronize(ev);
                std::lock_guard<std::mutex> lock(mtx);
                ev_pool[edev].push_back(ev);
            }
            dem_pending[h].clear();
            uint8_t * half = dem_base + (size_t) h*dem_half;
            // the regions that fit in the half (a region larger than the half is split)
            struct piece { size_t item; size_t off; size_t len; size_t pos; size_t lead; };
            std::vector<piece> pieces;
            size_t pos = 0;
            while (i < items.size()) {
                const item & it = items[i];
                const size_t rem  = it.r->size - split_off;
                const size_t lead = (size_t) ((it.foff + split_off) % k_align);
                const size_t room = (dem_half - pos) / k_align * k_align; // pos stays aligned
                if (room <= lead) {
                    break;
                }
                // reads of at most one chunk, so large regions are read by several threads
                const size_t len = std::min({ rem, params.chunk, room - lead });
                if (len < rem && len < params.chunk && !pieces.empty()) {
                    break; // the rest of the region goes to the next half
                }
                pieces.push_back({ i, split_off, len, pos, lead });
                pos += round_up(lead + len, k_align);
                split_off += len;
                if (split_off == it.r->size) {
                    split_off = 0;
                    i++;
                }
            }
            GGML_ASSERT(!pieces.empty());
            std::atomic<int> remaining((int) pieces.size());
            tasks.clear();
            tasks.reserve(pieces.size());
            for (const piece & p : pieces) {
                const item & it = items[p.item];
                tasks.push_back({ it.file, it.foff + p.off, p.len, p.lead, half + p.pos, direct, &remaining });
            }
            {
                std::unique_lock<std::mutex> lock(mtx);
                for (auto & t : tasks) {
                    demand.push_back(&t);
                }
                cv_io.notify_all();
                const auto t0 = std::chrono::steady_clock::now();
                cv_main.wait(lock, [&]() { return remaining.load() == 0; });
                s_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            }
            for (const piece & p : pieces) {
                const region & r = *items[p.item].r;
                ggml_backend_tensor_set_async(backend, r.dst, half + p.pos + p.lead, r.dst_off + p.off, p.len);
                if (tier != nullptr) {
                    tier->offer(r.src, r.src_off + p.off, p.len, half + p.pos + p.lead);
                }
                s_demand_bytes += p.len;
            }
            s_demand_regions += pieces.size();
            ggml_backend_event_t ev = ev_get(dev);
            if (ev != nullptr) {
                ggml_backend_event_record(ev, backend);
                dem_pending[h].push_back({ dev, ev });
            } else {
                ggml_backend_synchronize(backend);
            }
        }
        return true;
    }
};

llama_weight_store::llama_weight_store(std::shared_ptr<const llama_weight_extents> extents, const llama_weight_store_params & params)
    : pimpl(new impl(std::move(extents), params)) {}

llama_weight_store::~llama_weight_store() {
    log_stats();
}

bool llama_weight_store::enabled(int mode, size_t bytes, size_t avail) {
    const char * env = getenv("LLAMA_WEIGHT_STREAM");
    if (env != nullptr) {
        if (atoi(env) == 0) {
            return false; // kill switch: the copies read the mapping as before
        }
        mode = LLAMA_WEIGHT_STREAM_ON;
    }
    if (bytes == 0) {
        return false;
    }
    switch (mode) {
        case LLAMA_WEIGHT_STREAM_ON:   return true;
        case LLAMA_WEIGHT_STREAM_AUTO: return avail > 0 && (double) bytes > 0.8*(double) avail;
        default:                       return false;
    }
}

size_t llama_weight_store::ram_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX st = {};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return (size_t) st.ullAvailPhys;
    }
    return 0;
#elif defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
        return (size_t) mem;
    }
    return 0;
#else
    size_t avail = 0;
    {
        std::ifstream f("/proc/meminfo");
        std::string key;
        size_t val = 0;
        std::string unit;
        while (f >> key >> val >> unit) {
            if (key == "MemAvailable:") {
                avail = val*1024;
                break;
            }
        }
    }
    // cgroup v2: the limit minus what is used, the page cache of the group counts as available
    std::ifstream fmax("/sys/fs/cgroup/memory.max");
    std::string smax;
    if (fmax >> smax && smax != "max") {
        const size_t lim = (size_t) strtoull(smax.c_str(), nullptr, 10);
        size_t cur = 0;
        std::ifstream fcur("/sys/fs/cgroup/memory.current");
        fcur >> cur;
        size_t file = 0;
        std::ifstream fst("/sys/fs/cgroup/memory.stat");
        std::string k;
        size_t v = 0;
        while (fst >> k >> v) {
            if (k == "file") {
                file = v;
                break;
            }
        }
        const size_t used = cur > file ? cur - file : 0;
        const size_t cg   = lim > used ? lim - used : 0;
        avail = avail == 0 ? cg : std::min(avail, cg);
    }
    return avail;
#endif
}

bool llama_weight_store::covers(const ggml_tensor * src) const {
    if (src->buffer == nullptr || !ggml_backend_buffer_is_host(src->buffer)) {
        return false;
    }
    const llama_weight_extent * e = pimpl->ext->find(src);
    return e != nullptr && !e->lazy;
}

void llama_weight_store::plan(const ggml_tensor * const * srcs, int n) {
    impl & d = *pimpl;
    if (n == 0) {
        // generation: after two graphs without prefetched copies the ring is released (the reads ahead are done or dropped by then)
        if (++d.idle_plans >= 2) {
            d.release_ring_if_idle();
        }
    } else {
        d.idle_plans = 0;
        if (d.dem_buf != nullptr) {
            d.alloc_ring(); // released during the generation before this prompt
        }
    }
    std::unique_lock<std::mutex> lock(d.mtx);
    d.s_plan_calls++;
    // the plan skips the bytes the tier holds: built again when they change
    const bool same = n > 0 && (size_t) n == d.cur_plan.size() && std::equal(srcs, srcs + n, d.cur_plan.begin()) &&
        (d.tier == nullptr || d.tier->generation() == d.plan_tier_gen);
    // the jobs not uploaded so far are dropped (a repeated plan keeps the jobs of its next generation)
    for (uint64_t k = std::max(d.cursor, d.base_seq); k < d.next_seq; ++k) {
        ws_job & j = d.job_at(k);
        if (!same || j.gen <= d.gen_consume) {
            j.consumed = true;
        }
    }
    if (!same) {
        d.cur_plan.assign(srcs, srcs + n);
        d.plan_entries.clear();
        if (d.tier != nullptr) {
            d.tier->new_plan();
        }
        std::vector<llama_weight_tier::run> tr;
        size_t total = 0;
        for (int i = 0; i < n; ++i) {
            uint64_t offs = 0;
            const llama_weight_extent * e = covers(srcs[i]) ? d.ext->find(srcs[i], &offs) : nullptr;
            if (e == nullptr) {
                continue; // copied from its memory
            }
            const size_t nb = ggml_nbytes(srcs[i]);
            tr.clear();
            if (d.tier != nullptr) {
                d.tier->runs(srcs[i], 0, nb, true, tr);
            } else {
                tr.push_back({ 0, nb, nullptr });
            }
            for (const auto & r : tr) {
                if (r.ptr != nullptr) {
                    d.s_tier_skipped += r.size;
                    continue; // uploaded from the tier
                }
                if (r.device) {
                    d.s_device_skipped += r.size;
                    continue; // copied on the device (if an upload asks for them after all, they are read on demand)
                }
                for (size_t o = r.off; o < r.off + r.size; o += d.params.chunk) {
                    d.plan_entries.push_back({ srcs[i], o, std::min(d.params.chunk, r.off + r.size - o), e->file, offs + o });
                }
            }
            total += nb;
        }
        d.plan_tier_gen = d.tier != nullptr ? d.tier->generation() : 0;
        // auto: the pages in the page cache are copied from the mapping (their bytes are not read again: on a 345 MB/s disk O_DIRECT
        // re-read them every ubatch, 1.4-1.6x slower prompts than the mapping), the others are read unbuffered (a scan larger than the
        // page cache would evict what the host computes from: buffered reads cost the generation after the prompt up to 40%)
        // buffered where residency or unbuffered reads are not available
        GGML_UNUSED(total);
        d.plan_io = d.params.io == LLAMA_WEIGHT_STREAM_IO_DIRECT ? PLAN_IO_DIRECT :
                    d.params.io == LLAMA_WEIGHT_STREAM_IO_BUFFERED ? PLAN_IO_BUFFERED : (mixed_supported() ? PLAN_IO_MIXED : PLAN_IO_BUFFERED);
        if (d.plan_entries.empty()) {
            d.cursor = d.next_seq;
            d.cv_io.notify_all();
            return;
        }
        d.gen_consume = d.gen_last = ++d.gen_counter;
        d.append_generation(d.gen_consume);
    } else {
        d.gen_consume++;
        if (d.gen_last < d.gen_consume) {
            d.gen_counter = d.gen_last = d.gen_consume;
            d.append_generation(d.gen_consume);
        }
    }
    uint64_t s = std::max(d.cursor, d.base_seq);
    while (s < d.next_seq && d.job_at(s).gen < d.gen_consume) {
        s++;
    }
    d.cursor = s;
    d.cv_io.notify_all();
}

bool llama_weight_store::upload(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, size_t off, size_t dst_off, size_t size) {
    if (size == 0) {
        return true;
    }
    if (!covers(src) || off + size > ggml_nbytes(src)) {
        pimpl->s_not_covered++;
        return false;
    }
    if (!pimpl->ensure_pinned(backend)) {
        return false;
    }
    if (pimpl->tier != nullptr) {
        std::vector<region> rest;
        pimpl->upload_tiered(backend, src, dst, off, dst_off, size, rest);
        return rest.empty() || pimpl->upload_demand(backend, rest.data(), rest.size());
    }
    const size_t n = pimpl->upload_planned(backend, src, dst, off, dst_off, size);
    if (n == size) {
        return true;
    }
    const region r = { src, off + n, dst, dst_off + n, size - n };
    return pimpl->upload_demand(backend, &r, 1);
}

bool llama_weight_store::upload_runs(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<std::pair<size_t, size_t>> & runs) {
    if (runs.empty()) {
        return true;
    }
    if (!covers(src) || !pimpl->ensure_pinned(backend)) {
        return false;
    }
    std::vector<region> rest;
    for (const auto & [off, size] : runs) {
        if (off + size > ggml_nbytes(src)) {
            return false;
        }
    }
    for (const auto & [off, size] : runs) {
        if (size == 0) {
            continue;
        }
        if (pimpl->tier != nullptr) {
            pimpl->upload_tiered(backend, src, dst, off, off, size, rest);
            continue;
        }
        const size_t n = rest.empty() ? pimpl->upload_planned(backend, src, dst, off, off, size) : 0;
        if (n < size) {
            rest.push_back({ src, off + n, dst, off + n, size - n });
        }
    }
    return rest.empty() || pimpl->upload_demand(backend, rest.data(), rest.size());
}

bool llama_weight_store::upload_batch(ggml_backend_t backend, const region * regions, size_t n) {
    if (n == 0) {
        return true;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!covers(regions[i].src)) {
            pimpl->s_not_covered++;
            return false;
        }
    }
    if (!pimpl->ensure_pinned(backend)) {
        return false;
    }
    return pimpl->upload_demand(backend, regions, n);
}

bool llama_weight_store::read_host(host_read * reads, size_t n, bool direct) {
    for (size_t i = 0; i < n; ++i) {
        if (!covers(reads[i].src)) {
            pimpl->s_not_covered++;
            return false;
        }
    }
    return pimpl->read_host(reads, n, direct);
}

uint64_t llama_weight_store::read_host_async(host_read * reads, size_t n, bool direct) {
    for (size_t i = 0; i < n; ++i) {
        if (!covers(reads[i].src)) {
            pimpl->s_not_covered++;
            return 0;
        }
    }
    return pimpl->read_host_async(reads, n, direct);
}

bool llama_weight_store::host_done(uint64_t ticket) {
    return ticket == 0 || pimpl->host_done(ticket);
}

bool llama_weight_store::host_cancel(uint64_t ticket) {
    return ticket != 0 && pimpl->host_cancel(ticket);
}

void llama_weight_store::host_wait(uint64_t ticket) {
    if (ticket != 0) {
        pimpl->host_wait(ticket);
    }
}

void llama_weight_store::set_tier(llama_weight_tier * tier) {
    pimpl->tier = tier;
    pimpl->cur_plan.clear(); // the next plan is built with it
}

llama_weight_store::stats llama_weight_store::get_stats() const {
    const impl & d = *pimpl;
    stats s;
    s.tier_skipped    = d.s_tier_skipped;
    s.tier_uploaded   = d.s_tier_uploaded;
    s.host_reads      = d.s_host_reads;
    s.host_read_bytes = d.s_host_read_bytes;
    s.plan_calls      = d.s_plan_calls;
    s.plan_reads      = d.s_plan_reads;
    s.plan_bytes      = d.s_plan_bytes;
    s.plan_uploaded   = d.s_plan_uploaded;
    s.plan_discarded  = d.s_plan_discarded;
    s.demand_regions  = d.s_demand_regions;
    s.demand_bytes    = d.s_demand_bytes;
    s.direct_bytes    = d.s_direct_bytes;
    s.buffered_bytes  = d.s_buffered_bytes;
    s.fallback_bytes  = d.s_fallback_bytes;
    s.io_errors       = d.s_io_errors;
    s.short_reads     = d.s_short_reads;
    s.direct_disabled = d.s_direct_disabled;
    s.verify_regions  = d.s_verify_regions;
    s.verify_bytes    = d.s_verify_bytes;
    s.verify_mismatch = d.s_verify_mismatch;
    s.wait_us         = d.s_wait_us;
    s.not_covered     = d.s_not_covered;
    s.cached_bytes    = d.s_cached_bytes;
    s.spec_reads      = d.s_spec_reads;
    s.spec_bytes      = d.s_spec_bytes;
    s.spec_boosted    = d.s_spec_boosted;
    s.spec_wait_us    = d.s_spec_wait_us;
    s.spec_cancelled_bytes = d.s_spec_cancelled_bytes;
    return s;
}

void llama_weight_store::log_stats() const {
    const stats s = get_stats();
    const double mib = 1.0/1048576.0;
    LLAMA_LOG_INFO("%s: plans %llu, ring reads %llu = %.1f MiB (uploaded %.1f MiB, discarded %.1f MiB), demand %llu regions = %.1f MiB, "
            "direct %.1f MiB, buffered %.1f MiB, fallback %.1f MiB, io errors %llu, short reads %llu, direct off %llu, wait %.1f ms, not covered %llu, cached %.1f MiB\n",
            __func__, (unsigned long long) s.plan_calls, (unsigned long long) s.plan_reads, s.plan_bytes*mib, s.plan_uploaded*mib, s.plan_discarded*mib,
            (unsigned long long) s.demand_regions, s.demand_bytes*mib, s.direct_bytes*mib, s.buffered_bytes*mib, s.fallback_bytes*mib,
            (unsigned long long) s.io_errors, (unsigned long long) s.short_reads, (unsigned long long) s.direct_disabled, s.wait_us/1000.0,
            (unsigned long long) s.not_covered, s.cached_bytes*mib);
    if (s.tier_skipped > 0 || s.tier_uploaded > 0 || s.host_reads > 0) {
        LLAMA_LOG_INFO("%s: host tier: %.1f MiB skipped by the reads ahead, %.1f MiB uploaded from it, %llu reads into it = %.1f MiB, %.1f MiB skipped as held by the device\n", __func__,
                s.tier_skipped*mib, s.tier_uploaded*mib, (unsigned long long) s.host_reads, s.host_read_bytes*mib, pimpl->s_device_skipped.load()*mib);
    }
    if (s.spec_bytes > 0) {
        LLAMA_LOG_INFO("%s: speculative reads into the host tier: %llu reads, %.1f MiB, %llu waited for before they started, %.1f ms waiting, %.1f MiB cancelled before they started\n", __func__,
                (unsigned long long) s.spec_reads, s.spec_bytes*mib, (unsigned long long) s.spec_boosted, s.spec_wait_us/1000.0, s.spec_cancelled_bytes*mib);
    }
    if (pimpl->params.verify) {
        LLAMA_LOG_INFO("%s: VERIFY %llu regions, %.1f MiB, %llu mismatches\n", __func__, (unsigned long long) s.verify_regions, s.verify_bytes*mib,
                (unsigned long long) s.verify_mismatch);
    }
}

void llama_weight_store::set_wrap(bool wrap) {
    pimpl->wrap_hint = wrap;
}

uint64_t llama_weight_store::ring_allocs() const {
    return pimpl->s_ring_allocs;
}

void llama_weight_store::set_read_fn(llama_weight_store_read_fn fn, void * user_data) {
    pimpl->read_fn    = fn;
    pimpl->read_fn_ud = user_data;
}

size_t llama_weight_store::bytes_mapped() const {
    return pimpl->ext->bytes_streamable();
}

size_t llama_weight_store::pinned_size() const {
    return (pimpl->ring_buf ? ggml_backend_buffer_get_size(pimpl->ring_buf) : 0) + (pimpl->dem_buf ? ggml_backend_buffer_get_size(pimpl->dem_buf) : 0);
}
