// Ported from ikawrakow/ik_llama.cpp ggml/src/ggml-moe-prefetch.cpp at e34d163a (PR #2101, author dmaivel), MIT.
// Copyright (c) 2024-2025 The ik_llama.cpp authors. See LICENSE.
//
// k16-disk changes to the copy (the worker pool, the chunking, the mincore skip, the tickets and MADV_COLD are ik's):
// - public API (ggml/include/ggml-moe-prefetch.h, GGML_API); GGML_OP_MOE_FUSED_UP_GATE removed (not an op of this ggml)
// - the registry holds reference-counted ranges (llama registers the host expert tensors of file mappings, not whole mappings,
//   so dense weights, embeddings and the PLE / Engram tables are never read by the engine); optional MADV_RANDOM on them
// - a worker reads a chunk with an exact-range MADV_WILLNEED (async readahead of exactly the chunk) before MADV_POPULATE_READ, so
//   the populate waits for one batch of IO instead of faulting the chunk in read-around windows
// - node enqueues (kernel hook) do not depend on the scheduler epoch (CPU graphs computed outside the scheduler, e.g. the MoE
//   cache's host plans, never bump it); lookahead enqueues keep ik's per-epoch idempotence
// - lookahead byte budget (queued + in flight); the kernel hook is a no-op while the engine is off (no WILLNEED fallback), so the
//   kill switch leaves the process as without the engine
// - counters (ggml_moe_prefetch_stats), an atomic enabled flag for the per-MUL_MAT_ID hook
#include "ggml-moe-prefetch.h"

#if defined(__linux__)

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// chunk granularity; big enough to amortize syscall cost,
// small enough to spread one expert (~MBs) across several workers.
static constexpr size_t GGML_MOE_PREFETCH_CHUNK = 2u*1024u*1024u;
// lookahead jobs are dropped beyond this queue depth so a stalled consumer
// cannot accumulate unbounded work.
static constexpr size_t GGML_MOE_PREFETCH_MAX_QUEUE = 65536;

// experts are faulted in with MADV_POPULATE_READ, which brings pages into the
// page cache AND this mm's page tables, so consumers take no minor faults.
// Unsupported kernels leave the engine off
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

namespace {

struct mapping_entry {
    uintptr_t base;
    size_t    size;
    int       refs;
    bool      rand_read;
};

struct ticket {
    std::atomic<int> pending{0};
    uint64_t         epoch = 0;
    bool             track_reads = false;
    // chunks the workers actually read from storage this epoch (i.e. pages
    // that were cold before the sweep); guarded by prefetch_pool::mtx.
    // Used by ggml_moe_prefetch_cold() to deactivate streaming traffic.
    std::vector<std::pair<uintptr_t, uint32_t>> read_chunks;
};

struct job {
    uintptr_t addr;
    uint32_t  len;
    bool      lookahead;
    std::shared_ptr<ticket> tk;
};

struct prefetch_pool {
    std::mutex               mtx;
    std::condition_variable  cv_work;   // workers sleep here
    std::condition_variable  cv_done;   // waiters sleep here
    std::deque<job>          queue;
    std::vector<std::thread> workers;
    bool                     shutdown = false;

    std::unordered_map<const void *, std::shared_ptr<ticket>> tickets;

    // lookahead bytes queued or being read (budget)
    std::atomic<size_t> lookahead_bytes{0};

    ~prefetch_pool() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            shutdown = true;
            // complete outstanding tickets so no waiter blocks forever
            for (auto & j : queue) {
                if (j.tk) {
                    j.tk->pending.fetch_sub(1, std::memory_order_acq_rel);
                }
            }
            queue.clear();
            lookahead_bytes.store(0);
            cv_work.notify_all();
            cv_done.notify_all();
        }
        for (auto & w : workers) {
            if (w.joinable()) w.join();
        }
        workers.clear();
    }

    void start(int n_threads) {
        stop();
        std::lock_guard<std::mutex> lock(mtx);
        shutdown = false;
        workers.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([this] { run(); });
        }
    }

    // k16-disk: all_cold = no page of the chunk was resident (only such chunks are recorded for MADV_COLD: a partly warm chunk
    // holds working-set pages)
    static bool chunk_resident(uintptr_t addr, size_t len, bool & all_cold) {
        all_cold = false;
        const long page = sysconf(_SC_PAGESIZE);
        const uintptr_t astart = addr & ~(uintptr_t)(page - 1);
        const size_t    alen   = (addr + len) - astart;
        const size_t    npages = (alen + page - 1)/page;
        if (npages > GGML_MOE_PREFETCH_CHUNK/4096 + 2) {
            return false;
        }
        unsigned char vec[GGML_MOE_PREFETCH_CHUNK/4096 + 2];
        if (mincore((void *)astart, alen, vec) != 0) {
            return false;
        }
        bool all_resident = true;
        all_cold = true;
        for (size_t i = 0; i < npages; ++i) {
            const bool r = (vec[i] & 1) != 0;
            all_resident &= r;
            all_cold     &= !r;
        }
        return all_resident;
    }

    void run();
};

struct prefetch_state {
    std::shared_mutex          reg_mtx; // k16-disk: workers hold it shared while they touch a range, so an unregister waits for them
    std::vector<mapping_entry> mappings;

    std::atomic<uint64_t> epoch{1};
    std::atomic<bool>     on{false};
    std::atomic<size_t>   budget{0};

    std::mutex                     pool_mtx;
    std::shared_ptr<prefetch_pool> pool;

    // cumulative observability counters (GGML_MOE_PREFETCH_DEBUG, ggml_moe_prefetch_stats)
    std::atomic<uint64_t> n_jobs{0};
    std::atomic<uint64_t> n_skipped{0};
    std::atomic<uint64_t> bytes_populated{0};
    std::atomic<uint64_t> n_colded{0};
    std::atomic<uint64_t> n_node{0};
    std::atomic<uint64_t> n_tensor{0};
    std::atomic<uint64_t> n_dropped{0};

    ~prefetch_state() {
        {
            std::lock_guard<std::mutex> lock(pool_mtx);
            on.store(false);
            pool.reset(); // joins the workers before the counters go
        }
        if (getenv("GGML_MOE_PREFETCH_DEBUG") && n_jobs.load() > 0) {
            fprintf(stderr, "ggml_moe_prefetch: jobs=%llu skipped_resident=%llu bytes_populated=%.2f GiB colded=%llu node=%llu tensor=%llu dropped=%llu\n",
                    (unsigned long long) n_jobs.load(), (unsigned long long) n_skipped.load(),
                    (double) bytes_populated.load()/(1024.0*1024.0*1024.0), (unsigned long long) n_colded.load(),
                    (unsigned long long) n_node.load(), (unsigned long long) n_tensor.load(), (unsigned long long) n_dropped.load());
        }
    }
};

static prefetch_state & state() {
    static prefetch_state s;
    return s;
}

static bool is_mapped_locked(const void * p, size_t len);

void prefetch_pool::run() {
    const long page = sysconf(_SC_PAGESIZE);
    auto & s = state();
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_work.wait(lock, [this] { return shutdown || !queue.empty(); });
            if (shutdown) return;
            j = std::move(queue.front());
            queue.pop_front();
        }
        s.n_jobs.fetch_add(1, std::memory_order_relaxed);
        bool done_read = false;
        bool all_cold  = false;
        // k16-disk: the range must still be registered while the worker touches it (an unregister takes reg_mtx exclusively, so the
        // address can never belong to a later mapping here)
        std::shared_lock<std::shared_mutex> reg(s.reg_mtx);
        if (!is_mapped_locked((const void *) j.addr, j.len)) {
            // unregistered after the enqueue: skip
        } else if (chunk_resident(j.addr, j.len, all_cold)) {
            // skip when every page is already resident; keeps the drain rate high when the cache is warm
            s.n_skipped.fetch_add(1, std::memory_order_relaxed);
        } else {
            const uintptr_t astart = j.addr & ~(uintptr_t)(page - 1);
            const size_t    alen   = ((j.addr + j.len + page - 1) & ~(uintptr_t)(page - 1)) - astart;
            // k16-disk: exact-range async readahead first (one batch of IO for the chunk, also under MADV_RANDOM), then the populate
            // waits for it and maps the pages
            (void) madvise((void *)astart, alen, MADV_WILLNEED);
            if (madvise((void *)astart, alen, MADV_POPULATE_READ) == 0) {
                s.bytes_populated.fetch_add(alen, std::memory_order_relaxed);
                done_read = true;
            } // on failure the fault path takes over
        }
        reg.unlock();
        if (j.lookahead) {
            lookahead_bytes.fetch_sub(std::min<size_t>(j.len, lookahead_bytes.load()), std::memory_order_relaxed);
        }
        if (j.tk) {
            // k16-disk: the read is published before the ticket completes, so a waiter / MADV_COLD sees it
            std::lock_guard<std::mutex> lock(mtx);
            if (done_read && all_cold && j.tk->track_reads) {
                j.tk->read_chunks.emplace_back(j.addr, j.len);
            }
            if (j.tk->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                cv_done.notify_all();
            }
        }
    }
}

// true when [p, p+len) lies inside a registered range (caller holds reg_mtx)
static bool is_mapped_locked(const void * p, size_t len) {
    auto & s = state();
    const uintptr_t a = (uintptr_t)p;
    for (const auto & m : s.mappings) {
        if (a >= m.base && a + len <= m.base + m.size) return true;
    }
    return false;
}

static bool is_mapped(const void * p, size_t len) {
    std::shared_lock<std::shared_mutex> lock(state().reg_mtx);
    return is_mapped_locked(p, len);
}

// collect [offset, offset+len) ranges of the selected experts of weight tensor w
static void collect_selected_ranges(const ggml_tensor * w, const ggml_tensor * ids,
        std::vector<std::pair<size_t, size_t>> & ranges) {
    const int64_t n_as   = w->ne[2];
    const size_t  stride = w->nb[2];
    const size_t  wbytes = ggml_nbytes(w);

    if (n_as <= 1 || stride == 0) {
        ranges.emplace_back(0, wbytes);
        return;
    }

    std::vector<uint32_t> seen((n_as + 31)/32, 0);
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            const int32_t id = *(const int32_t *)((const char *)ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
            if (id < 0 || id >= n_as) continue; // ids may hold -1 sentinels
            seen[id >> 5] |= 1u << (id & 31);
        }
    }
    // coalesce consecutive experts into single ranges
    int64_t id = 0;
    while (id < n_as) {
        while (id < n_as && !(seen[id >> 5] & (1u << (id & 31)))) ++id;
        if (id >= n_as) break;
        int64_t first = id;
        while (id < n_as && (seen[id >> 5] & (1u << (id & 31)))) ++id;
        const size_t off = (size_t)first*stride;
        const size_t len = std::min<size_t>((size_t)(id - first)*stride, wbytes - off);
        if (off < wbytes && len > 0) {
            ranges.emplace_back(off, len);
        }
    }
}

static std::shared_ptr<prefetch_pool> get_pool() {
    auto & s = state();
    std::lock_guard<std::mutex> plock(s.pool_mtx);
    return s.pool;
}

// enqueue ranges of tensor w; returns false when the engine is off or w is not registered
static bool enqueue_ranges(const ggml_tensor * w, const std::vector<std::pair<size_t, size_t>> & ranges, bool urgent, bool track_reads) {
    auto & s = state();
    if (!s.on.load(std::memory_order_acquire)) return false;
    std::shared_ptr<prefetch_pool> pool_sp = get_pool();
    if (!pool_sp) return false;

    if (!is_mapped(w->data, ggml_nbytes(w))) return false;

    prefetch_pool & pool = *pool_sp;
    std::vector<job> jobs;
    size_t total = 0;
    for (const auto & r : ranges) {
        for (size_t o = r.first; o < r.first + r.second; o += GGML_MOE_PREFETCH_CHUNK) {
            const size_t len = std::min(GGML_MOE_PREFETCH_CHUNK, r.first + r.second - o);
            jobs.push_back({(uintptr_t)w->data + o, (uint32_t)len, !urgent, nullptr});
            total += len;
        }
    }
    if (jobs.empty()) return true;

    std::lock_guard<std::mutex> lock(pool.mtx);
    if (pool.shutdown) return false;

    if (urgent) {
        // the routed experts of a running product: no ticket (nobody waits for them), dropped only beyond the queue cap (a stalled
        // drive: the demand faults read them anyway)
        if (pool.queue.size() > GGML_MOE_PREFETCH_MAX_QUEUE) {
            s.n_dropped.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        s.n_node.fetch_add(1, std::memory_order_relaxed);
        pool.queue.insert(pool.queue.begin(), std::make_move_iterator(jobs.begin()), std::make_move_iterator(jobs.end()));
        pool.cv_work.notify_all();
        return true;
    }

    const uint64_t cur_epoch = s.epoch.load(std::memory_order_relaxed);
    auto & tk = pool.tickets[w];
    if (!tk) tk = std::make_shared<ticket>();
    if (tk->epoch == cur_epoch) {
        return true; // already enqueued for this scheduler pass
    }
    const size_t budget = s.budget.load(std::memory_order_relaxed);
    if (pool.queue.size() > GGML_MOE_PREFETCH_MAX_QUEUE || (budget > 0 && pool.lookahead_bytes.load() + total > budget)) {
        s.n_dropped.fetch_add(1, std::memory_order_relaxed);
        return true; // drop lookahead under backpressure; do not mark the epoch
    }
    s.n_tensor.fetch_add(1, std::memory_order_relaxed);
    tk->epoch = cur_epoch;
    tk->track_reads = track_reads;
    tk->read_chunks.clear();
    for (auto & j : jobs) {
        j.tk = tk;
    }
    pool.lookahead_bytes.fetch_add(total, std::memory_order_relaxed);
    tk->pending.fetch_add((int)jobs.size(), std::memory_order_acq_rel);
    pool.queue.insert(pool.queue.end(), std::make_move_iterator(jobs.begin()), std::make_move_iterator(jobs.end()));
    pool.cv_work.notify_all();
    return true;
}

static void advise_range(uintptr_t base, size_t size, int advice) {
    const uintptr_t page_mask = (uintptr_t)sysconf(_SC_PAGESIZE) - 1;
    const uintptr_t a = (base + page_mask) & ~page_mask; // whole pages inside the range only (neighbours keep their advice)
    const uintptr_t e = (base + size) & ~page_mask;
    if (e > a) {
        (void) madvise((void *)a, e - a, advice);
    }
}

} // namespace

void ggml_moe_prefetch_register_mapping(const void * base, size_t size, bool rand_read) {
    if (!base || size == 0) return;
    auto & s = state();
    std::unique_lock<std::shared_mutex> lock(s.reg_mtx);
    for (auto & m : s.mappings) {
        if (m.base == (uintptr_t)base) { // contexts may re-register the same model
            m.refs++;
            if (m.rand_read && !rand_read) { // a registrant that needs read-around wins (e.g. an MTP context without a weight store)
                advise_range(m.base, m.size, MADV_NORMAL);
                m.rand_read = false;
            }
            return;
        }
    }
    s.mappings.push_back({(uintptr_t)base, size, 1, rand_read});
    if (rand_read) {
        advise_range((uintptr_t)base, size, MADV_RANDOM);
    }
}

void ggml_moe_prefetch_unregister_mapping(const void * base) {
    auto & s = state();
    // waits for the workers touching a registered range; a job queued for this range is skipped after the erase
    std::unique_lock<std::shared_mutex> lock(s.reg_mtx);
    for (size_t i = 0; i < s.mappings.size(); ++i) {
        mapping_entry & m = s.mappings[i];
        if (m.base == (uintptr_t)base && --m.refs <= 0) {
            if (m.rand_read) {
                advise_range(m.base, m.size, MADV_NORMAL);
            }
            s.mappings.erase(s.mappings.begin() + i);
            return;
        }
    }
}

static bool populate_read_supported() {
    const long page = sysconf(_SC_PAGESIZE);
    void * p = mmap(nullptr, (size_t)page, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return false;
    const bool ok = madvise(p, (size_t)page, MADV_POPULATE_READ) == 0;
    munmap(p, (size_t)page);
    return ok;
}

void ggml_moe_prefetch_set_n_threads(int n_threads) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.pool_mtx);
    if (n_threads <= 0) {
        s.on.store(false, std::memory_order_release);
        s.pool.reset();
        return;
    }
    if (!populate_read_supported()) {
        fprintf(stderr, "%s: MADV_POPULATE_READ not supported; MoE prefetch disabled\n", __func__);
        s.on.store(false, std::memory_order_release);
        s.pool.reset();
        return;
    }
    if (s.pool && s.pool->workers.size() == (size_t)n_threads) {
        return;
    }
    s.on.store(false, std::memory_order_release);
    s.pool.reset();
    s.pool = std::make_shared<prefetch_pool>();
    s.pool->start(n_threads);
    s.on.store(true, std::memory_order_release);
}

bool ggml_moe_prefetch_enabled(void) {
    return state().on.load(std::memory_order_acquire);
}

void ggml_moe_prefetch_set_budget(size_t bytes) {
    state().budget.store(bytes, std::memory_order_relaxed);
}

void ggml_moe_prefetch_new_epoch(void) {
    state().epoch.fetch_add(1, std::memory_order_relaxed);
}

void ggml_moe_prefetch_node(const struct ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT_ID) return;
    const ggml_tensor * w0  = node->src[0];
    const ggml_tensor * ids = node->src[2];
    if (!w0 || !w0->data || !ids || !ids->data) return;

    std::vector<std::pair<size_t, size_t>> ranges;
    collect_selected_ranges(w0, ids, ranges);
    enqueue_ranges(w0, ranges, /*urgent =*/ true, /*track_reads =*/ false);
}

void ggml_moe_prefetch_tensor(const struct ggml_tensor * w) {
    if (!w || !w->data) return;
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.emplace_back(0, ggml_nbytes(w));
    enqueue_ranges(w, ranges, /*urgent =*/ false, /*track_reads =*/ true);
}

bool ggml_moe_prefetch_is_registered(const struct ggml_tensor * w) {
    return w && w->data && is_mapped(w->data, ggml_nbytes(w));
}

#ifndef MADV_COLD
#define MADV_COLD 20
#endif
void ggml_moe_prefetch_cold(const struct ggml_tensor * w) {
    auto & s = state();
    std::shared_ptr<prefetch_pool> pool = get_pool();
    if (!pool) return;
    std::vector<std::pair<uintptr_t, uint32_t>> chunks;
    {
        std::lock_guard<std::mutex> lock(pool->mtx);
        auto it = pool->tickets.find(w);
        if (it == pool->tickets.end() || !it->second->track_reads) return;
        chunks.swap(it->second->read_chunks);
    }
    // k16-disk: whole pages inside each chunk only (a page shared with a neighbouring tensor keeps its state), ranges still registered
    std::shared_lock<std::shared_mutex> reg(s.reg_mtx);
    for (const auto & c : chunks) {
        if (is_mapped_locked((const void *) c.first, c.second)) {
            advise_range(c.first, c.second, MADV_COLD);
        }
    }
    s.n_colded.fetch_add(chunks.size(), std::memory_order_relaxed);
}

void ggml_moe_prefetch_wait(const struct ggml_tensor * w) {
    std::shared_ptr<prefetch_pool> pool = get_pool();
    if (!pool) return;
    std::shared_ptr<ticket> tk;
    {
        std::lock_guard<std::mutex> lock(pool->mtx);
        auto it = pool->tickets.find(w);
        if (it == pool->tickets.end()) return;
        tk = it->second;
    }
    if (tk->pending.load(std::memory_order_acquire) <= 0) return;
    // the shared_ptr keeps the pool alive; shutdown wakes cv_done
    std::unique_lock<std::mutex> lock(pool->mtx);
    pool->cv_done.wait(lock, [&] {
        return pool->shutdown || tk->pending.load(std::memory_order_acquire) <= 0;
    });
}

void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor * node, int ith) {
    if (ith != 0 || !state().on.load(std::memory_order_relaxed)) return;
    // fire-and-forget enqueue of the routed experts; we deliberately do not wait: the compute threads' demand faults overlap with the
    // workers' populates, which run ahead in the same expert-index order.
    ggml_moe_prefetch_node(node);
}

void ggml_moe_prefetch_stats(struct ggml_moe_prefetch_counters * out) {
    auto & s = state();
    out->n_jobs          = s.n_jobs.load();
    out->n_skipped       = s.n_skipped.load();
    out->bytes_populated = s.bytes_populated.load();
    out->n_colded        = s.n_colded.load();
    out->n_node          = s.n_node.load();
    out->n_tensor        = s.n_tensor.load();
    out->n_dropped       = s.n_dropped.load();
    std::shared_lock<std::shared_mutex> lock(s.reg_mtx);
    out->n_ranges = s.mappings.size();
    uint64_t b = 0;
    for (const auto & m : s.mappings) {
        b += m.size;
    }
    out->bytes_registered = b;
}

#else // !__linux__

void ggml_moe_prefetch_register_mapping(const void *, size_t, bool) {}
void ggml_moe_prefetch_unregister_mapping(const void *) {}
void ggml_moe_prefetch_set_n_threads(int) {}
bool ggml_moe_prefetch_enabled(void) { return false; }
void ggml_moe_prefetch_set_budget(size_t) {}
void ggml_moe_prefetch_new_epoch(void) {}
void ggml_moe_prefetch_node(const struct ggml_tensor *) {}
void ggml_moe_prefetch_tensor(const struct ggml_tensor *) {}
bool ggml_moe_prefetch_is_registered(const struct ggml_tensor *) { return false; }
void ggml_moe_prefetch_wait(const struct ggml_tensor *) {}
void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor *, int) {}
void ggml_moe_prefetch_cold(const struct ggml_tensor *) {}
void ggml_moe_prefetch_stats(struct ggml_moe_prefetch_counters * out) { *out = {}; }

#endif
