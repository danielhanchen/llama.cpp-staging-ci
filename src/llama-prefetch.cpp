#include "llama-prefetch.h"

#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <string>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/resource.h>
    #include <sys/time.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

#if defined(__linux__)
    #include <sys/vfs.h>
#endif

#if defined(__linux__) && !defined(MADV_POPULATE_READ)
    #define MADV_POPULATE_READ 22 // Linux 5.14, older headers lack it
#endif

const char * llama_ple_prefetch_mode_name(enum llama_ple_prefetch_mode mode) {
    switch (mode) {
        case LLAMA_PLE_PREFETCH_AUTO:     return "auto";
        case LLAMA_PLE_PREFETCH_OFF:      return "off";
        case LLAMA_PLE_PREFETCH_MADVISE:  return "madvise";
        case LLAMA_PLE_PREFETCH_POPULATE: return "populate";
        case LLAMA_PLE_PREFETCH_PREAD:    return "pread";
    }
    return "unknown";
}

enum llama_ple_prefetch_mode llama_ple_prefetch_resolve(enum llama_ple_prefetch_mode requested) {
    // the default when nothing asks for a mode
    constexpr enum llama_ple_prefetch_mode def = LLAMA_PLE_PREFETCH_OFF;

    enum llama_ple_prefetch_mode mode = requested;

    // LLAMA_PLE_PREFETCH=0 is the kill switch, a mode name overrides the params (for tools that do not parse the flag)
    if (const char * env = getenv("LLAMA_PLE_PREFETCH")) {
        const std::string v = env;
        /**/ if (v == "0" || v == "off")  { mode = LLAMA_PLE_PREFETCH_OFF;      }
        else if (v == "madvise")          { mode = LLAMA_PLE_PREFETCH_MADVISE;  }
        else if (v == "populate")         { mode = LLAMA_PLE_PREFETCH_POPULATE; }
        else if (v == "pread")            { mode = LLAMA_PLE_PREFETCH_PREAD;    }
        else if (v == "auto" || v == "1") { mode = requested;                   }
        else {
            LLAMA_LOG_WARN("%s: unknown LLAMA_PLE_PREFETCH=%s, ignored\n", __func__, env);
        }
    }

    return mode == LLAMA_PLE_PREFETCH_AUTO ? def : mode;
}

int llama_prefetch_reopen_fd(int fd, size_t offs, size_t size) {
#if defined(__linux__)
    if (fd < 0) {
        return -1;
    }
    // a new open file description: no O_DIRECT from the loader, its own read-ahead state
    const std::string path = "/proc/self/fd/" + std::to_string(fd);
    const int res = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (res >= 0) {
        (void) posix_fadvise(res, (off_t) offs, (off_t) size, POSIX_FADV_RANDOM);
    }
    return res;
#elif defined(__APPLE__)
    if (fd < 0) {
        return -1;
    }
    char path[PATH_MAX];
    if (fcntl(fd, F_GETPATH, path) == -1) {
        return -1;
    }
    GGML_UNUSED(offs);
    GGML_UNUSED(size);
    return open(path, O_RDONLY | O_CLOEXEC);
#else
    GGML_UNUSED(fd);
    GGML_UNUSED(offs);
    GGML_UNUSED(size);
    return -1;
#endif
}

void llama_prefetch_close_fd(int fd) {
#if !defined(_WIN32)
    if (fd >= 0) {
        close(fd);
    }
#else
    GGML_UNUSED(fd);
#endif
}

static int64_t prefetch_majflt_thread() {
#if defined(__linux__)
    struct rusage ru;
    if (getrusage(RUSAGE_THREAD, &ru) == 0) {
        return ru.ru_majflt;
    }
#endif
    return 0;
}

llama_row_prefetcher::llama_row_prefetcher(enum llama_ple_prefetch_mode mode_, int n_threads, const ggml_tensor * tensor, int fd, size_t file_offs)
    : mode(mode_), fd(fd), file_offs(file_offs) {
    GGML_ASSERT(mode != LLAMA_PLE_PREFETCH_OFF && mode != LLAMA_PLE_PREFETCH_AUTO);
    GGML_ASSERT(ggml_is_matrix(tensor) && tensor->data);

    data       = (const char *) tensor->data;
    row_bytes  = ggml_row_size(tensor->type, tensor->ne[0]);
    row_stride = tensor->nb[1];
    n_rows_max = tensor->ne[1];

#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    page = info.dwPageSize;
    // the Windows backend issues one PrefetchVirtualMemory per job (populate touches the pages afterwards)
    if (mode == LLAMA_PLE_PREFETCH_PREAD) {
        this->mode = LLAMA_PLE_PREFETCH_POPULATE;
    }
#else
    page = (size_t) sysconf(_SC_PAGESIZE);
    if (mode == LLAMA_PLE_PREFETCH_PREAD && fd < 0) {
        LLAMA_LOG_WARN("%s: no file to pread the table from, using populate\n", __func__);
        this->mode = LLAMA_PLE_PREFETCH_POPULATE;
    }
#endif
    if (page == 0) {
        page = 4096;
    }

    if (n_threads <= 0) {
        n_threads = this->mode == LLAMA_PLE_PREFETCH_MADVISE ? 1 : 8;
    }
    n_threads = std::min(n_threads, 64);

    workers.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        workers.emplace_back([this, i] { worker_main(i); });
    }
}

llama_row_prefetcher::~llama_row_prefetcher() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    cv_done.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

void llama_row_prefetcher::submit(std::vector<int32_t> rows) {
    if (rows.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.size() >= max_pending) {
            // the workers are behind: the oldest pending ubatch is the least useful one
            queue.pop_front();
            st.n_dropped++;
        }
        queue.push_back({ std::move(rows), ggml_time_us() });
    }
    cv.notify_all();
}

void llama_row_prefetcher::build_runs(const std::vector<int32_t> & rows, std::vector<run> & out) const {
    out.clear();
    out.reserve(rows.size());
    for (const int32_t r : rows) {
        if (r < 0 || r >= n_rows_max) {
            continue;
        }
        const uintptr_t a     = (uintptr_t) (data + (size_t) r * row_stride);
        const uintptr_t first = a / page * page;
        const uintptr_t last  = (a + row_bytes + page - 1) / page * page;
        out.push_back({ first, last });
    }
    std::sort(out.begin(), out.end(), [](const run & a, const run & b) { return a.begin < b.begin; });

    // merge overlapping and adjacent page ranges
    size_t n = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (n > 0 && out[i].begin <= out[n - 1].end) {
            out[n - 1].end = std::max(out[n - 1].end, out[i].end);
        } else {
            out[n++] = out[i];
        }
    }
    out.resize(n);
}

void llama_row_prefetcher::work_runs(int id) {
    GGML_UNUSED(id);

    std::vector<char> buf;
    if (mode == LLAMA_PLE_PREFETCH_PREAD) {
        buf.resize(step_bytes);
    }

    constexpr size_t chunk = 16;

    for (;;) {
        const size_t i0 = run_next.fetch_add(chunk);
        if (i0 >= runs.size()) {
            break;
        }
        const size_t i1 = std::min(i0 + chunk, runs.size());
        for (size_t i = i0; i < i1; ++i) {
            const run & r = runs[i];
            for (uintptr_t b = r.begin; b < r.end; b += step_bytes) {
                const size_t len = std::min((size_t) (r.end - b), step_bytes);
                switch (mode) {
                    case LLAMA_PLE_PREFETCH_POPULATE:
                        {
#if defined(__linux__)
                            if (populate_ok) {
                                if (madvise((void *) b, len, MADV_POPULATE_READ) == 0) {
                                    break;
                                }
                                if (errno == EINVAL) {
                                    // kernel < 5.14: WILLNEED was issued already, the gather takes the soft faults
                                    populate_ok = false;
                                    LLAMA_LOG_WARN("%s: MADV_POPULATE_READ is not supported, using madvise\n", __func__);
                                } else {
                                    st.n_errors++;
                                }
                            }
#else
                            // no populate advice: fault the pages in from the worker
                            for (uintptr_t p = b; p < b + len; p += page) {
                                (void) *(const volatile char *) p;
                            }
#endif
                        } break;
                    case LLAMA_PLE_PREFETCH_PREAD:
                        {
#if !defined(_WIN32)
                            // the mapping starts at a page boundary of the file, so the page runs map to file offsets
                            const int64_t off = (int64_t) file_offs + (int64_t) (b - (uintptr_t) data);
                            if (off < 0) {
                                st.n_errors++;
                                break;
                            }
                            const ssize_t n = pread(fd, buf.data(), len, (off_t) off);
                            if (n < 0) {
                                st.n_errors++;
                            }
#endif
                        } break;
                    default:
                        break;
                }
            }
        }
    }
}

void llama_row_prefetcher::worker_main(int id) {
    uint64_t seen_phase = 0;

    if (id > 0) {
        // helper: run the parallel phases the leader publishes
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return stop || phase != seen_phase; });
                if (stop) {
                    return;
                }
                seen_phase = phase;
            }
            const int64_t f0 = prefetch_majflt_thread();
            work_runs(id);
            st.n_majflt += prefetch_majflt_thread() - f0;
            {
                std::lock_guard<std::mutex> lock(mtx);
                n_active--;
            }
            cv_done.notify_all();
        }
    }

    // leader: take the jobs in order
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop) {
                return;
            }
            j = std::move(queue.front());
            queue.pop_front();
        }

        std::vector<run> rs;
        build_runs(j.rows, rs);

        size_t n_bytes = 0;
        for (const auto & r : rs) {
            n_bytes += r.end - r.begin;
        }

        const int64_t f0 = prefetch_majflt_thread();

#if defined(_WIN32)
        {
            using prefetch_virtual_memory_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            static const auto pPrefetchVirtualMemory = (prefetch_virtual_memory_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
            if (pPrefetchVirtualMemory && !rs.empty()) {
                std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
                entries.reserve(rs.size());
                for (const auto & r : rs) {
                    entries.push_back({ (PVOID) r.begin, (SIZE_T) (r.end - r.begin) });
                }
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0)) {
                    st.n_errors++;
                }
            }
        }
#else
        if (mode == LLAMA_PLE_PREFETCH_MADVISE || mode == LLAMA_PLE_PREFETCH_POPULATE) {
            // queue the reads of all runs first, so that they are in flight together
            for (const auto & r : rs) {
                for (uintptr_t b = r.begin; b < r.end; b += step_bytes) {
                    const size_t len = std::min((size_t) (r.end - b), step_bytes);
                    if (madvise((void *) b, len, MADV_WILLNEED) != 0) {
                        st.n_errors++;
                    }
                }
            }
        }
#endif

#if defined(__linux__)
        const bool populate_par = populate_ok;
#else
        const bool populate_par = true; // the helpers touch the pages
#endif
        const bool parallel = mode == LLAMA_PLE_PREFETCH_PREAD || (mode == LLAMA_PLE_PREFETCH_POPULATE && populate_par);

        const size_t n_runs = rs.size();

        if (parallel && !rs.empty()) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                runs = std::move(rs);
                run_next = 0;
                n_active = (int) workers.size() - 1;
                phase++;
            }
            cv.notify_all();

            work_runs(0);

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_done.wait(lock, [&] { return stop || n_active == 0; });
                if (stop && n_active > 0) {
                    return;
                }
            }
        }

        st.n_majflt += prefetch_majflt_thread() - f0;

        st.n_jobs++;
        st.n_rows    += j.rows.size();
        st.n_runs    += n_runs;
        st.n_bytes   += n_bytes;
        st.t_busy_us += ggml_time_us() - j.t_submit_us;
    }
}

// ----------------------------------------------------------------------------------------------------------------

llama_ple_prompt_reader::llama_ple_prompt_reader(const ggml_tensor * tensor, int fd, size_t file_offs, int n_threads)
    : fd(fd), file_offs(file_offs) {
    GGML_ASSERT(ggml_is_matrix(tensor) && tensor->data);
    data       = (const char *) tensor->data;
    row_bytes  = ggml_row_size(tensor->type, tensor->ne[0]);
    row_stride = tensor->nb[1];
    n_rows_max = tensor->ne[1];
#if !defined(_WIN32)
    const long ps = sysconf(_SC_PAGESIZE);
    if (ps > 0) {
        page = (size_t) ps;
    }
#endif
    sync  = getenv("LLAMA_PLE_PROMPT_READ_SYNC") != nullptr && atoi(getenv("LLAMA_PLE_PROMPT_READ_SYNC")) != 0;
    probe = getenv("LLAMA_PLE_PROMPT_READ_STATS") != nullptr && atoi(getenv("LLAMA_PLE_PROMPT_READ_STATS")) != 0;
    n_threads = std::max(1, std::min(n_threads, 256));
    workers.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        workers.emplace_back([this] { worker_main(); });
    }
}

llama_ple_prompt_reader::~llama_ple_prompt_reader() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
        queue.clear();
    }
    cv.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

void llama_ple_prompt_reader::worker_main() {
#if !defined(_WIN32)
    std::vector<char> buf;
    for (;;) {
        std::pair<uint64_t, uint64_t> r;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop) {
                return;
            }
            r = queue.front();
            queue.pop_front();
            n_busy++;
        }
        if (buf.size() < r.second) {
            buf.resize(r.second);
        }
        const ssize_t n = pread(fd, buf.data(), r.second, (off_t) r.first);
        st.n_reads++;
        if (n < 0) {
            st.n_errors++;
        }
        bool drained;
        {
            std::lock_guard<std::mutex> lock(mtx);
            n_busy--;
            drained = queue.empty() && n_busy == 0;
        }
        if (drained) {
            cv_done.notify_all();
        }
    }
#endif
}

void llama_ple_prompt_reader::read(const int32_t * rows, size_t n) {
    const int64_t t0 = ggml_time_us();
    // page-aligned file ranges of the rows (the tensor data starts at file_offs; a row may straddle two pages), merged when adjacent
    std::vector<std::pair<uint64_t, uint64_t>> rs;
    rs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const int32_t r = rows[i];
        if (r < 0 || r >= n_rows_max) {
            continue;
        }
        const uint64_t a     = (uint64_t) file_offs + (uint64_t) r * row_stride;
        const uint64_t first = a / page * page;
        const uint64_t last  = (a + row_bytes + page - 1) / page * page;
        rs.emplace_back(first, last);
    }
    std::sort(rs.begin(), rs.end());
    size_t m = 0;
    uint64_t n_pages = 0;
    constexpr uint64_t max_run = 1u << 20; // a long run is split, so that the threads share it
    for (size_t i = 0; i < rs.size(); ++i) {
        auto r = rs[i];
        if (m > 0) {
            auto & c = rs[m - 1];
            if (r.second <= c.second) {
                continue; // inside the previous run
            }
            if (r.first <= c.second) {
                if (r.second - c.first <= max_run) {
                    c.second = r.second;
                    continue;
                }
                r.first = c.second; // the rest only
            }
        }
        rs[m++] = r;
    }
    rs.resize(m);
#if !defined(_WIN32)
    if (probe) {
        // diagnostic: how many of these pages the page cache holds right now (the mapping starts at file offset 0)
        std::vector<unsigned char> vec;
        uint64_t res = 0, all = 0;
        for (const auto & r : rs) {
            const size_t np = (size_t) ((r.second - r.first) / page);
            vec.resize(np);
            void * addr = const_cast<char *>(data - file_offs + r.first);
#if defined(__linux__)
            const int mc = mincore(addr, (size_t) (r.second - r.first), vec.data());
#else
            // char * on macOS and the BSDs
            const int mc = mincore(addr, (size_t) (r.second - r.first), (char *) vec.data());
#endif
            if (mc == 0) {
                for (size_t q = 0; q < np; ++q) {
                    res += vec[q] & 1;
                }
                all += np;
            }
        }
        st.n_resident += res;
        st.n_probed   += all;
    }
#endif
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto & r : rs) {
            n_pages += (r.second - r.first) / page;
            queue.emplace_back(r.first, r.second - r.first);
        }
    }
    cv.notify_all();
    if (sync) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_done.wait(lock, [&] { return stop || (queue.empty() && n_busy == 0); });
    }
    st.n_calls++;
    st.n_rows  += n;
    st.n_pages += n_pages;
    st.t_us    += ggml_time_us() - t0;
}

size_t llama_ple_prompt_read_min_rows() {
    static const size_t v = getenv("LLAMA_PLE_PROMPT_READ_MIN_ROWS") ? (size_t) std::max(1, atoi(getenv("LLAMA_PLE_PROMPT_READ_MIN_ROWS"))) : 256;
    return v;
}

llama_ple_prompt_reader * llama_ple_prompt_reader_make(const ggml_tensor * tensor, int fd, size_t file_offs) {
#if defined(_WIN32)
    GGML_UNUSED(tensor);
    GGML_UNUSED(fd);
    GGML_UNUSED(file_offs);
    return nullptr;
#else
    // opt-in (LLAMA_PLE_PROMPT_READ=1): on A100 / L4 VM disks random reads are capped at ~3k/s however many are in flight,
    // so the pool bought 0.8 s of a 22 s 8k prompt there (one launch), and its blocking form cost the L4 8k +29 s
    if (const char * e = getenv("LLAMA_PLE_PROMPT_READ"); e == nullptr || atoi(e) == 0) {
        return nullptr;
    }
    if (tensor == nullptr || fd < 0 || !ggml_is_matrix(tensor) || tensor->data == nullptr) {
        return nullptr;
    }
#if defined(__linux__)
    // a table in tmpfs / ramfs is RAM already: its faults are minor
    struct statfs sfs;
    if (fstatfs(fd, &sfs) == 0 && ((unsigned long) sfs.f_type == 0x01021994UL || (unsigned long) sfs.f_type == 0x858458f6UL)) {
        return nullptr;
    }
#endif
    int n_threads = 32;
    if (const char * e = getenv("LLAMA_PLE_PROMPT_READ_THREADS")) {
        n_threads = std::max(1, atoi(e));
    }
    return new llama_ple_prompt_reader(tensor, fd, file_offs, n_threads);
#endif
}
