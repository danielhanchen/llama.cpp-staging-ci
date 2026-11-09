#pragma once

#include "llama.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_tensor;

// Advisory prefetch of the rows of a lazily read, memory mapped table (the PLE table of gemma4 / qwen4exp).
// The caller computes the row ids of a future ubatch and submits them; persistent worker threads turn them into page
// runs and warm the page cache (madvise / populate / pread). Nothing here writes to the mapping, so the results of the
// model cannot change, only how many page faults the gather of the rows takes.
struct llama_row_prefetcher {
    struct stats {
        std::atomic<uint64_t> n_jobs    {0}; // jobs done
        std::atomic<uint64_t> n_dropped {0}; // jobs replaced by newer ones before they started
        std::atomic<uint64_t> n_rows    {0};
        std::atomic<uint64_t> n_runs    {0}; // merged page runs
        std::atomic<uint64_t> n_bytes   {0}; // bytes of the page runs
        std::atomic<uint64_t> n_errors  {0};
        std::atomic<uint64_t> t_busy_us {0}; // submit -> done, summed over jobs
        std::atomic<uint64_t> n_majflt  {0}; // major faults taken by the workers (populate)
    };

    // fd: a file descriptor of the file holding the table, used by PREAD (-1: PREAD falls back to POPULATE), not owned
    // file_offs: offset of tensor->data in that file
    llama_row_prefetcher(enum llama_ple_prefetch_mode mode, int n_threads, const ggml_tensor * tensor, int fd, size_t file_offs);
    ~llama_row_prefetcher();

    // rows: row ids of `tensor`, any order, duplicates allowed, negative ids are skipped
    void submit(std::vector<int32_t> rows);

    enum llama_ple_prefetch_mode get_mode() const { return mode; }
    int get_n_threads() const { return (int) workers.size(); }

    stats st;

private:
    struct run {
        uintptr_t begin;
        uintptr_t end;
    };

    struct job {
        std::vector<int32_t> rows;
        int64_t t_submit_us = 0;
    };

    void worker_main(int id);
    void build_runs(const std::vector<int32_t> & rows, std::vector<run> & out) const;
    void work_runs(int id);

    enum llama_ple_prefetch_mode mode;

    const ggml_tensor * tensor;
    const char *        data;
    size_t              row_bytes;
    size_t              row_stride;
    int64_t             n_rows_max;
    size_t              page;

    int    fd;
    size_t file_offs;

    std::atomic<bool> populate_ok {true}; // MADV_POPULATE_READ not refused yet

    std::mutex              mtx;
    std::condition_variable cv;      // workers wait here for a job / a run phase
    std::condition_variable cv_done; // the leader waits here for the helpers to finish a phase
    std::deque<job>         queue;   // pending jobs, at most max_pending
    bool                    stop = false;

    // the run phase of the current job, shared by all workers
    std::vector<run>      runs;
    std::atomic<size_t>   run_next {0};
    uint64_t              phase    = 0; // incremented for each run phase
    int                   n_active = 0; // helpers still in the current phase

    std::vector<std::thread> workers;

    static constexpr size_t max_pending = 2;
    static constexpr size_t step_bytes  = 128*1024; // madvise / pread granularity
};

// the mode chosen by the env (LLAMA_PLE_PREFETCH: 0|off|madvise|populate|pread) or the params, AUTO resolved
enum llama_ple_prefetch_mode llama_ple_prefetch_resolve(enum llama_ple_prefetch_mode requested);

// an independent, buffered, read-only descriptor of the file behind fd (POSIX_FADV_RANDOM over [offs, offs + size)),
// -1 where unsupported; the caller owns it (llama_prefetch_close_fd)
int  llama_prefetch_reopen_fd(int fd, size_t offs, size_t size);
void llama_prefetch_close_fd(int fd);

// k19-disktt: the pages a prompt ubatch gathers from a lazily mapped table on a disk (the Flash-Next PLE table, 16 rows of 90 bytes on
// 16 random pages per token), read before the gather by a pool of threads with buffered preads (the page cache then serves the gather's
// faults). The gather of a lazy MADV_RANDOM mapping faults one 4 KiB page at a time on its threads, and the WILLNEED hint is dropped when
// the kernel finds no free page cache for it, so a prompt read ~100k pages at the queue depth of the CPU threads (an 8k prompt on a
// model larger than RAM: 400 MB in ~35 s on a Colab A100 40 GB VM). Nothing is written to the mapping and the gather reads the same
// bytes, so the outputs cannot change.
struct llama_ple_prompt_reader {
    struct stats {
        std::atomic<uint64_t> n_calls {0};
        std::atomic<uint64_t> n_rows  {0};
        std::atomic<uint64_t> n_pages {0}; // pages read (runs of adjacent pages are read whole)
        std::atomic<uint64_t> n_reads {0}; // pread calls
        std::atomic<uint64_t> n_errors{0};
        std::atomic<uint64_t> t_us    {0}; // wall time of read(), summed
    };

    // fd: buffered descriptor of the file holding the tensor (not owned); file_offs: offset of tensor->data in that file
    llama_ple_prompt_reader(const ggml_tensor * tensor, int fd, size_t file_offs, int n_threads);
    ~llama_ple_prompt_reader();

    // reads the pages of the given rows (negative or out of range ids skipped), returns when all reads are done
    void read(const int32_t * rows, size_t n_rows);

    int get_n_threads() const { return (int) workers.size() + 1; }

    stats st;

private:
    void work();
    void worker_main();

    const char * data;
    size_t       row_bytes;
    size_t       row_stride;
    int64_t      n_rows_max;
    size_t       page = 4096;
    int          fd;
    size_t       file_offs;

    std::vector<std::pair<uint64_t, uint64_t>> runs; // [file offset, length) of the current call
    std::atomic<size_t> run_next {0};

    std::mutex              mtx;
    std::condition_variable cv;
    std::condition_variable cv_done;
    uint64_t                phase    = 0;
    int                     n_active = 0;
    bool                    stop     = false;
    std::mutex              call_mtx; // one read() at a time (contexts sharing the model)
    std::vector<std::thread> workers;
};

// the reader for a lazily mapped table on a real file system (not tmpfs / ramfs), Linux / macOS / BSD; nullptr where it does not apply or
// LLAMA_PLE_PROMPT_READ=0 (threads: LLAMA_PLE_PROMPT_READ_THREADS, default 32)
llama_ple_prompt_reader * llama_ple_prompt_reader_make(const ggml_tensor * tensor, int fd, size_t file_offs);
// rows at least this many (LLAMA_PLE_PROMPT_READ_MIN_ROWS, default 256) go through the reader, fewer keep the WILLNEED hint
size_t llama_ple_prompt_read_min_rows();
