// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "llama-moe-cache.h"
#include "llama-sysmem.h"

#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-weight-store.h"
#include "llama-moe-host-cache.h"
#include "llama-build-id.h"
#include "llama-moe-trial-persist.h"
#include "llama-autoprof.h"

#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-moe-mailbox.h"

// k19-q4xcache: see ggml.c (declared here so that ggml.h stays unchanged)
extern "C" { GGML_API void ggml_moe_cache_assign_lookup_counts(struct ggml_tensor * op); }
extern "C" { GGML_API void ggml_moe_cache_set_core(struct ggml_tensor * op, int32_t n_core); }
// k19-g4pref: see ggml.c
extern "C" { GGML_API struct ggml_tensor * ggml_moe_cache_prefetch(struct ggml_context * ctx, struct ggml_tensor * logits, struct ggml_tensor * map,
        struct ggml_tensor * state, struct ggml_tensor ** banks, const void ** host, int n_banks, int il, int n_top, int max_fill,
        int n_core, struct ggml_tensor * scores, struct ggml_tensor * dep); }

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cinttypes>
#include <climits>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#if defined(__linux__)
#include <sys/resource.h>
#endif
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#if defined(__linux__)
#include <sched.h>
#include <fcntl.h>
#endif

// k19-l4b: the logical CPUs this process may run on (the affinity mask on Linux, else the hardware threads)
static int llama_moe_n_cpus() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0) {
        return CPU_COUNT(&set);
    }
#endif
    return (int) std::max(1u, std::thread::hardware_concurrency());
}

// acquire load / release store / spin pause on the host mailbox words, which the device reads and writes through mapped pinned memory:
// the GCC / Clang builtins, on MSVC (no __atomic_* / __builtin_ia32_pause) the interlocked intrinsics (full barriers) and _mm_pause
static inline uint32_t llama_mb_load_acq(const uint32_t * p) {
#if defined(_MSC_VER) && !defined(__clang__)
    return (uint32_t) _InterlockedCompareExchange((volatile long *) p, 0, 0);
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

static inline void llama_mb_store_rel(uint32_t * p, uint32_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
    _InterlockedExchange((volatile long *) p, (long) v);
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

static inline void llama_mb_spin_pause() {
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && !defined(__clang__) && defined(_M_ARM64)
    __yield();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// k19-exactmtp: LLAMA_MTP_EXACT=1 (the drafted text equals the no-draft text) and LLAMA_MOE_EXACT_HOST unset or 0: the MoE cache never
// computes an expert on the host unless LLAMA_MOE_CACHE_HYBRID / LLAMA_MOE_CACHE_FETCH / LLAMA_MOE_CPU_MISS=1 asks for it
static bool llama_moe_exact_host_off() {
    static const bool value = [] {
        const char * x = getenv("LLAMA_MTP_EXACT");
        const char * k = getenv("LLAMA_MOE_EXACT_HOST");
        return x != nullptr && atoi(x) != 0 && (k == nullptr || atoi(k) == 0);
    }();
    return value;
}

bool llama_moe_cache_device_lookup_supported(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg != nullptr &&
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_map")   != nullptr &&
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_unmap") != nullptr;
}

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    uint64_t gen = 0; // incremented by each change of the slots, so the device copy of the state can be refreshed

    int32_t n_lent = 0; // the last n_lent slots are lent (see set_lent): they are not in the list and hold no expert
    // k19-ptres: the first n_core slots are a core (see set_core): they hold their experts for good, are not in the list (never a
    // victim) and keep their experts through clear and load
    int32_t n_core = 0;

    // LLAMA_MOE_CACHE_POLICY=decay: decayed use counters of all experts [n_layer*n_expert], also kept for the experts that are not
    // cached, shared with the device lookup (ggml_moe_cache_set_scores): growing weights, each distinct expert of a plan adds w, the
    // plans of first_layer (once per batch) first multiply w by grow_q16/65536 and halve every counter and w when w reaches w_max;
    // the misses evict the slots with the lowest counters (empty first), ties by LRU order
    static constexpr int32_t w_init    = 1 << 16;
    static constexpr int32_t w_max     = 1 << 22;
    static constexpr int32_t score_max = 1 << 30;
    int32_t grow_q16    = 0; // 0: LRU
    int32_t first_layer = -1;
    int32_t w           = w_init;
    std::vector<int32_t> scores;
    std::vector<uint32_t> pinned; // [n_slots] seen_gen of the plan whose hits use the slot
    std::vector<std::pair<int64_t, int32_t>> cand;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, int32_t grow_q16 = 0, int32_t first_layer = -1) {
        this->n_expert    = n_expert;
        this->n_slots     = n_slots;
        this->grow_q16    = grow_q16;
        this->first_layer = first_layer;
        w = w_init;
        scores.assign(grow_q16 > 0 ? (size_t) n_layer*n_expert : 0, 0);
        pinned.assign(grow_q16 > 0 ? n_slots : 0, 0);
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    // every slot empty, in the order of init
    void clear() {
        GGML_ASSERT(n_lent == 0);
        std::fill(key_of.begin() + n_core, key_of.end(), -1);
        std::fill(scores.begin(), scores.end(), 0);
        w = w_init;
        for (int32_t s = n_core; s < n_slots; ++s) {
            prev[s] = s > n_core ? s - 1 : -1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        // the core keeps its experts (callers may have emptied the slot maps)
        for (int32_t s = 0; s < n_core; ++s) {
            slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = s;
        }
        head = n_slots > n_core ? n_core : -1;
        tail = n_slots > n_core ? n_slots - 1 : -1;
        gen++;
    }

    // k19-ptres: slots [0, n) become the core: each must hold an expert; they leave the list for good (n = 0: the core slots come back
    // to the list as the most recently used)
    bool set_core(int32_t n) {
        GGML_ASSERT(n >= 0 && n + n_lent < n_slots);
        if (n > 0) {
            GGML_ASSERT(n_core == 0);
            for (int32_t s = 0; s < n; ++s) {
                if (key_of[s] < 0) {
                    return false;
                }
            }
            for (int32_t s = 0; s < n; ++s) {
                unlink(s);
            }
        } else {
            for (int32_t s = 0; s < n_core; ++s) {
                prev[s] = tail;
                next[s] = -1;
                if (tail >= 0) {
                    next[tail] = s;
                } else {
                    head = s;
                }
                tail = s;
            }
        }
        n_core = n;
        gen++;
        return true;
    }

    void unlink(int32_t s) {
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail = prev[s];
        }
        prev[s] = -1;
        next[s] = -1;
    }

    // empty slot s (it stays in the list, k13-small release); never a core slot
    void evict(int32_t s) {
        GGML_ASSERT(s >= n_core);
        if (key_of[s] >= 0) {
            slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            key_of[s] = -1;
            gen++;
        }
    }

    // lend the last n slots (their experts are evicted), the slots no longer lent come back empty as the least recently used
    void set_lent(int32_t n) {
        GGML_ASSERT(n >= 0 && n < n_slots && n_slots - n >= n_core + (n_core > 0 ? 1 : 0));
        for (int32_t s = n_slots - n; s < n_slots - n_lent; ++s) {
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
                key_of[s] = -1;
            }
            unlink(s);
        }
        for (int32_t s = n_slots - n_lent; s < n_slots - n; ++s) {
            prev[s] = -1;
            next[s] = head;
            if (head >= 0) {
                prev[head] = s;
            } else {
                tail = s;
            }
            head = s;
        }
        n_lent = n;
        gen++;
    }

    // k17-small (lazy KV step 2, see llama_moe_cache::shrink): the slots [n_new, n_slots) leave the cache for good (their experts are
    // evicted, they leave the list); no slot may be lent
    void retire(int32_t n_new) {
        GGML_ASSERT(n_lent == 0 && n_new > n_core && n_new <= n_slots); // (k19-ptres: never into the core)
        for (int32_t s = n_new; s < n_slots; ++s) {
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
                key_of[s] = -1;
            }
            unlink(s);
        }
        n_slots = n_new;
        key_of.resize(n_new);
        prev.resize(n_new);
        next.resize(n_new);
        if (!pinned.empty()) {
            pinned.resize(n_new);
        }
        gen++;
    }

    // k19-t4small (decode extension, see llama_moe_cache::ext_grow): the slots [n_slots, n_new) join the cache empty, as the least
    // recently used (the next victims); no slot may be lent
    void unretire(int32_t n_new) {
        GGML_ASSERT(n_lent == 0 && n_new >= n_slots);
        const int32_t n_old = n_slots;
        key_of.resize(n_new, -1);
        prev.resize(n_new, -1);
        next.resize(n_new, -1);
        if (!pinned.empty()) {
            pinned.resize(n_new, 0);
        }
        n_slots = n_new;
        for (int32_t s = n_new - 1; s >= n_old; --s) {
            prev[s] = -1;
            next[s] = head;
            if (head >= 0) {
                prev[head] = s;
            } else {
                tail = s;
            }
            head = s;
        }
        gen++;
    }

    // move slot s to the tail (most recently used); a core slot stays out of the list
    void touch(int32_t s) {
        if (s == tail || s < n_core) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    // move slot s to the head (the next to be evicted), k15-pred
    void to_head(int32_t s) {
        if (s == head || s < n_core) {
            return;
        }
        unlink(s);
        prev[s] = -1;
        next[s] = head;
        if (head >= 0) {
            prev[head] = s;
        } else {
            tail = s;
        }
        head = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    // evicted: the slots whose expert the misses replace and that expert's key (il*n_expert + e), if not nullptr
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, std::vector<std::pair<int32_t, int32_t>> * evicted = nullptr) {
        fills.clear();
        if (evicted != nullptr) {
            evicted->clear();
        }
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            std::fill(pinned.begin(), pinned.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) (n_slots - n_lent)) {
            return false;
        }
        if (n_core > 0) {
            // the misses and the hits outside the core take slots of the list
            size_t n_in_core = 0;
            for (int32_t e : uniq) {
                n_in_core += slot_map[il][e] >= 0 && slot_map[il][e] < n_core;
            }
            if (uniq.size() - n_in_core > (size_t) (n_slots - n_lent - n_core)) {
                return false;
            }
        }

        gen++;

        int32_t * slots = slot_map[il];

        if (grow_q16 > 0) {
            if (il == first_layer) {
                w = (int32_t) (((int64_t) w*grow_q16) >> 16);
                if (w >= w_max) {
                    for (int32_t & v : scores) {
                        v >>= 1;
                    }
                    w >>= 1;
                }
            }
            int32_t * sc = scores.data() + (size_t) il*n_expert;
            for (int32_t e : uniq) {
                sc[e] = std::min(sc[e] + w, score_max);
            }
        }

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                if (grow_q16 > 0) {
                    pinned[slots[e]] = seen_gen;
                }
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        if (grow_q16 > 0) {
            // the victims: the lowest counters of the slots not used by the hits (empty: -1), ties by LRU order (the head first);
            // given to the sorted misses in slot order, so the uploads can still be merged
            const size_t n_miss = uniq.size() - n_hit;
            if (n_miss == 0) {
                return true;
            }
            cand.clear();
            int64_t pos = 0;
            for (int32_t s = head; s >= 0; s = next[s], ++pos) {
                if (pinned[s] == seen_gen) {
                    continue;
                }
                const int64_t v = key_of[s] >= 0 ? scores[key_of[s]] : -1;
                cand.push_back({ (v + 1) << 32 | pos, s }); // v + 1 >= 0 and pos < 2^31
            }
            GGML_ASSERT(cand.size() >= n_miss);
            std::nth_element(cand.begin(), cand.begin() + (n_miss - 1), cand.end());
            cand.resize(n_miss);
            std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.second < b.second; });
            size_t j = 0;
            for (int32_t e : uniq) {
                if (slots[e] >= 0) {
                    continue;
                }
                const int32_t s = cand[j++].second;
                if (key_of[s] >= 0) {
                    slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
                    // the exclusive RAM tier copies these back before their slots are overwritten (without them it only drains)
                    if (evicted != nullptr) {
                        evicted->emplace_back(s, key_of[s]);
                    }
                }
                key_of[s] = il*n_expert + e;
                slots[e] = s;
                touch(s);
                fills.push_back({ e, s });
            }
            return true;
        }
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
                if (evicted != nullptr) {
                    evicted->emplace_back(s, key_of[s]);
                }
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }

    // replace the state with the one computed by the device lookup
    // the slots with the reference bit set were used more recently, they go to the tail
    void load(const std::vector<int32_t> & layers, const int32_t * dev_key_of, const int32_t * dev_ref, const int32_t * dev_scores = nullptr, int32_t dev_w = 0) {
        // the lent slots (a device seed of a large batch, see devseed_end) stay out of the list, the device never fills them
        if (dev_scores != nullptr) {
            std::copy(dev_scores, dev_scores + scores.size(), scores.begin());
            w = dev_w > 0 ? dev_w : w_init;
        }
        for (int32_t il : layers) {
            std::fill(slot_map[il], slot_map[il] + n_expert, -1);
        }
        head = -1;
        tail = -1;
        auto append = [&](int32_t s) {
            prev[s] = tail;
            next[s] = -1;
            if (tail >= 0) {
                next[tail] = s;
            } else {
                head = s;
            }
            tail = s;
        };
        for (int32_t s = n_slots - n_lent; s < n_slots; ++s) {
            GGML_ASSERT(dev_key_of[s] < 0);
            key_of[s] = -1;
            prev[s]   = -1;
            next[s]   = -1;
        }
        // the core (never evicted by the device) keeps its experts and stays out of the list
        for (int32_t s = 0; s < n_core; ++s) {
            GGML_ASSERT(dev_key_of[s] == key_of[s]);
            slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = s;
            prev[s] = -1;
            next[s] = -1;
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int32_t s = n_core; s < n_slots - n_lent; ++s) {
                const bool used = dev_key_of[s] >= 0 && dev_ref[s] != 0;
                if ((pass == 0) == used) {
                    continue;
                }
                key_of[s] = dev_key_of[s];
                if (key_of[s] >= 0) {
                    slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = s;
                }
                append(s);
            }
        }
        gen++;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

// per-expert scales and biases of the experts kept in host memory, a tiny fraction of the experts, so a copy is kept on the device
static std::vector<ggml_tensor *> llama_moe_cache_layer_small(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_exps_s, layer.ffn_up_exps_s, layer.ffn_down_exps_s,
                             layer.ffn_gate_exps_b, layer.ffn_up_exps_b, layer.ffn_down_exps_b, layer.ffn_gate_up_exps_b }) {
        if (t != nullptr && llama_moe_cache_is_host_weight(t)) {
            res.push_back(t);
        }
    }
    return res;
}


// the cache of a layer set holds layer il
static bool llama_moe_cache_in_set(const llama_model & model, llama_moe_cache::layer_set set, size_t il) {
    switch (set) {
        case llama_moe_cache::layer_set::trunk: return il <  model.hparams.n_layer();
        case llama_moe_cache::layer_set::nextn: return il >= model.hparams.n_layer();
        default:                                return true;
    }
}

// a device whose MUL_MAT_ID and ADD_ID implement the -1 id of the cache (a slot that is not cached or not computed: the row is
// zeroed / nothing is added, see ggml_mul_mat_id and ggml_add_id): ggml-cuda (CUDA, ROCm, MUSA). Metal and Vulkan read out of
// bounds or leave stale rows for it, and an RPC device's remote backend is unknown, so their layers are not cached
static bool llama_moe_cache_device_ok(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }
    const auto type = ggml_backend_dev_type(dev);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
    return name != nullptr && (strcmp(name, "CUDA") == 0 || strcmp(name, "ROCm") == 0 || strcmp(name, "MUSA") == 0);
}

// the experts of layer il that a cache can hold: all of them in host memory, the layer assigned to a GPU whose ops take -1 ids
static bool llama_moe_cache_layer_cacheable(const llama_model & model, size_t il, std::vector<ggml_tensor *> & experts) {
    experts = llama_moe_cache_layer_experts(model.layers[il]);
    if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
        return false;
    }
    return llama_moe_cache_device_ok(model.dev_layer((int) il));
}

// host bytes of the experts that the cache of a layer set can hold
static size_t llama_moe_cache_set_bytes(const llama_model & model, llama_moe_cache::layer_set set) {
    size_t res = 0;
    std::vector<ggml_tensor *> experts;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        if (llama_moe_cache_in_set(model, set, il) && llama_moe_cache_layer_cacheable(model, il, experts)) {
            for (const ggml_tensor * t : experts) {
                res += ggml_nbytes(t);
            }
        }
    }
    return res;
}
}

struct llama_moe_cache::impl {
    bool force_large = false; // k15-pp2: see llama_moe_cache::set_force_large
    llama_weight_store * wstore = nullptr;
    // k11: the graph allocation of the graph allocated next may be lent too (a merged layer-major group, see llama_moe_cache::set_graph_lend)
    bool graph_lend_scope = false;
    int64_t n_releases = 0; // release (k13-small)

    // k17-small (lazy KV step 2, LLAMA_KV_LAZY=1 with LLAMA_MOE_CACHE_SHRINK unset or != 0): the banks of a device are a lazy buffer of the
    // CUDA backend (address range reserved whole, backed at load), so the cache can give memory back to a KV cache that grows (shrink)
    bool shrink_on = false;
    ggml_backend_buffer_t (*lazy_alloc_fn)(ggml_backend_buffer_type_t, size_t) = nullptr;
    bool   (*lazy_commit_fn)(ggml_backend_buffer_t, size_t, size_t)            = nullptr;
    size_t (*lazy_decommit_fn)(ggml_backend_buffer_t, size_t, size_t)          = nullptr;
    bool   (*lazy_commit_run_fn)(ggml_backend_buffer_t, size_t, size_t)        = nullptr; // k19-t4small: one allocation per span
    int64_t n_shrinks     = 0; // shrink calls that gave memory back
    int64_t n_retired     = 0; // slots retired by them
    size_t  shrunk_bytes  = 0; // device memory they gave back
    bool    graphs_stale  = false; // a shrink changed the slot counts the graphs were built with (see take_graphs_stale)
    // k19-t4small (decode extension, see llama_moe_cache::ext_grow): the banks are a lazy buffer with room for group::n_slots_cap slots
    bool    ext_on        = false;
    int64_t n_ext_grows   = 0;
    int64_t n_ext_parks   = 0;
    size_t  ext_max_bytes = 0; // the most the extension held
    int64_t ext_us        = 0; // host time of ext_grow / ext_park
    int32_t ext_fail      = 0; // grows that could not back their slots
    ggml_context_ptr ctx_banks_op; // the views of group::banks_op
    std::vector<llama_weight_store::region> ws_regions;

    // RAM tier (see llama_moe_host_cache): experts of the groups streamed from the model files, behind the device slots
    // --moe-host-cache-mib (-1 auto), LLAMA_MOE_HOST_CACHE=0 off, LLAMA_MOE_HOST_CACHE_POLICY=exclusive|inclusive
    std::unique_ptr<llama_moe_host_cache> host;
    std::vector<bool> host_group; // [group] served by the RAM tier (host lookups, never mapped for the device)
    // exclusive with spare room (budget + device slots >= streamed experts): promoted experts keep their RAM copy as a duplicate (the
    // first to go) and RAM misses fill free slots; without spare room promotions free the RAM slot and misses go to the device only
    // (V16 / R29 and the T4 at R31: keeping duplicates there cost the prompts their RAM admissions, decode -53% / -32%)
    bool host_spare = false;
    // k19-copyback: a device eviction whose expert RAM still holds (a duplicate kept at promotion) needs no copy back: the weights are
    // read-only, so the RAM bytes equal the device bytes. LLAMA_MOE_COPYBACK_SKIP=0 copies them back over that RAM copy anyway (same
    // bytes, same slot, the slot waits for the copy as a real copy back does): the A/B arm for the skip, cache decisions unchanged
    bool copyback_skip = [] {
        const char * e = getenv("LLAMA_MOE_COPYBACK_SKIP");
        return e == nullptr || atoi(e) != 0;
    }();
    bool host_verify = false;     // LLAMA_WEIGHT_STREAM_VERIFY=1: every upload from the tier is compared with the mapping
    struct host_stats {
        uint64_t hits = 0, misses = 0;        // device misses of small batches (and refills) found / not found in RAM
        uint64_t read = 0, read_bytes = 0;    // RAM misses read into a slot (inclusive)
        uint64_t demoted = 0, demote_bytes = 0, demote_skip = 0; // device evictions copied back into RAM (exclusive)
        uint64_t demote_kept = 0;             // device evictions RAM still held a copy of (no copy back)
        uint64_t demote_kept_bytes = 0;       // the bytes of those copies back skipped
        uint64_t demote_forced = 0, demote_forced_bytes = 0; // LLAMA_MOE_COPYBACK_SKIP=0: copied back over the identical RAM copy anyway
        uint64_t promoted = 0;                // RAM experts moved to the device (exclusive: their slots are freed)
        uint64_t up_bytes = 0;                // bytes uploaded from RAM by small batches and refills
        uint64_t admitted = 0;                // experts admitted from the uploads of prompts
        uint64_t prefilled = 0;               // experts copied from the page cache at load
        uint64_t verify_regions = 0, verify_bytes = 0, verify_mismatch = 0;
        uint64_t avail_init = 0;
        size_t   registered = 0;              // host memory registered for the device lookup (whole expert buffers)
    } hst;
    std::vector<std::pair<int32_t, int32_t>> evicted_tmp;
    std::vector<llama_weight_store::host_read> host_reads;
    std::vector<std::pair<int32_t, moe_cache_lru::fill>> host_pending;

    // layer-ahead reads of the RAM tier (LLAMA_MOE_HOST_CACHE_AHEAD=N, 1..4, 0 = off): small batches, host lookups only. The router
    // input of layer p (read with the ids of p) is scored on the host with the routers of layers p+1..p+N, while the device computes
    // split p (before the next lookup synchronizes); the top experts that neither the device nor RAM hold are read into RAM slots by
    // the IO threads behind every demand read. A lookup that routes an expert still being read waits for that read only.
    // Exact: only which RAM slots hold (file) bytes changes, never what a lookup computes with
    struct ahead_router {
        std::vector<int8_t>   w;  // int8 [n_expert][n_embd] (rows of ffn_gate_inp, symmetric per row): integer dot products vectorize at
                                  // the baseline ISA (3x the bf16 loop, half its memory; a prediction only ranks experts)
        std::vector<float>    ws; // [n_expert] scale of each row
        std::vector<float>    b;  // ffn_gate_inp_b, empty if none
        std::vector<float>    pb; // ffn_exp_probs_b, empty if none
    };
    int     ahead_depth   = 0;
    int     ahead_topk    = 0;
    int     ahead_max_inflight = 0; // reads ahead in flight at most (experts), LLAMA_MOE_HOST_CACHE_AHEAD_INFLIGHT
    int     ahead_gating  = 0;
    int64_t ahead_n_embd  = 0;
    static constexpr int64_t ahead_max_tokens = 4; // small batches of more tokens are not predicted
    std::vector<ahead_router>        routers;  // [n_layer]
    std::vector<const ggml_tensor *> router_w; // [n_layer] ffn_gate_inp of the layer
    std::vector<float>    ahead_x;             // router input of layer ahead_il, [ahead_nt][n_embd]
    int64_t  ahead_nt    = 0;
    int32_t  ahead_il    = -1;
    uint64_t ahead_epoch   = 1; // ubatch being computed (see prepare): the lookups of one ubatch follow the layers in order
    uint64_t ahead_x_epoch = 0; // ubatch of the router input in ahead_x
    std::vector<uint64_t> ahead_ticket;        // [slot] weight store ticket of its read ahead, 0 = none
    std::vector<int32_t>  ahead_key;           // [slot] key read ahead and not used yet, -1 = none
    std::vector<int32_t>  ahead_inflight;      // slots with a ticket
    std::vector<std::vector<int32_t>> ahead_pred; // [n_layer] experts last predicted for the layer (recall)
    std::vector<float>    ahead_scores;
    std::vector<int16_t>  ahead_xq;   // router input of one token, int16 (symmetric)
    std::vector<std::pair<float, int32_t>> ahead_rank;
    std::vector<llama_weight_store::host_read> ahead_reads;
    size_t ahead_router_bytes = 0;
    struct ahead_stats {
        uint64_t layers = 0, predicted = 0, issued = 0, bytes = 0, used = 0, late = 0, no_slot = 0, waited_us = 0, routed = 0, recalled = 0,
                 held = 0, predict_us = 0, cancelled = 0, capped = 0;
    } ast;
    std::vector<moe_cache_lru::fill> fills_rest;

    // the tier the weight store sees: the expert tensors of the RAM tier layers, at expert granularity
    struct tier_adapter : llama_weight_tier {
        impl * m = nullptr;
        void runs(const ggml_tensor * src, size_t off, size_t size, bool lease, std::vector<run> & out) override { m->tier_runs(src, off, size, lease, out); }
        void new_plan() override { m->host->new_plan(); }
        uint64_t generation() const override { return m->host->generation(); }
        void offer(const ggml_tensor * src, size_t off, size_t n, const uint8_t * p) override { m->tier_offer(src, off, n, p); }
        void uploaded(ggml_backend_t backend) override { m->host->commit(backend); }
    } tier;

    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views (ctx_bank, buf_bank: freed while the cache is released, see release), the other device tensors
        ggml_context_ptr ctx_bank;
        ggml_backend_buffer_ptr buf_bank;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
        bool released = false; // k13-small: the banks are freed for a prompt (release / restore)

        int lend_auto = -1; // automatic lending (LLAMA_MOE_CACHE_LEND unset): -1 not decided yet, 0 off, 1 on
        size_t dry_need = 0; // the largest staging need a measurement (dry lend) asked for, see lend_need_max
        size_t dry_save = 0; // k15-fn: the staging copies a measurement found the device would lend (lend_saves), see lend_save_need
        bool lend_saving = false; // k15-fn: lending is on for the memory it frees (lend_saves), see unlend
        size_t mem_reserve = 0; // device memory kept free for a later model or context (set_mem_reserve), not free for lend_decide
        bool   bank_lazy    = false; // k17-small: buf_bank is a lazy buffer (shrink_on), see alloc_banks
        size_t shrink_floor = SIZE_MAX; // k17-small: shrink keeps at least this many bytes of slots on the device (set_shrink_floor)
    };

    // LLAMA_MOE_CACHE_SEED_CHECK=1: after each seeding ubatch, compare the seeded slots with the host experts (debug)
    struct seed_rec {
        const ggml_tensor * w;
        const ggml_tensor * bank;
        int32_t expert;
        int32_t slot;
    };

    // the layer being seeded from a large batch, per group: with several devices the copies of their splits can interleave
    struct seed_state {
        int32_t il = -1;
        std::vector<int32_t> old_slots; // slot map before the seed, the copies of the other expert tensors of the layer read it
        std::vector<moe_cache_lru::fill> fills;
        uint32_t done = 0; // expert tensors whose slots were filled
        std::vector<seed_rec> pending; // LLAMA_MOE_CACHE_SEED_CHECK: copies of the layer being seeded

        // LRU state before the plan of the seed, restored if the seed is dropped
        std::vector<int32_t> old_key_of;
        std::vector<int32_t> old_prev;
        std::vector<int32_t> old_next;
        std::vector<int32_t> old_scores; // LLAMA_MOE_CACHE_POLICY=decay: the counters and the weight
        int32_t  old_w    = 0;
        int32_t  old_head = -1;
        int32_t  old_tail = -1;
        uint64_t gen      = 0; // generation of the LRU after the plan, the snapshot is only valid while it is unchanged
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        int32_t n_slots_load = 0; // k17-small: n_slots at load (the banks keep that size, shrink retires the tail slots)
        int32_t n_slots_min  = 0; // k17-small: shrink keeps at least this many (set_shrink_floor; 0: not set, nothing is retired)
        int32_t n_slots_fit  = 0; // k19-t4small: the slots the budget gave (ext_park goes back to them)
        int32_t n_slots_cap  = 0; // k19-t4small: the slots the banks have room for (n_slots_fit + the decode extension, see ext_grow)
        // k17-small: once slots are retired, views of the banks over the slots left and the spare slot ([ne0, ne1, n_slots + 1], the
        // shape the lookup ops require), passed to the device lookup / fetch ops instead of the banks (see op_banks); empty before
        std::vector<ggml_tensor *> banks_op;
        moe_cache_lru lru;
        seed_state sd;
        // k15-pred r19: the key each slot held when the device lookups of the last small ubatches had used it (-1: not used); the
        // uploads ahead take none of these slots while they still hold that key (LLAMA_MOE_UPLOAD_AHEAD_KEEP_DECODE)
        std::vector<int32_t> ua_dec_key;

        // device lookup: slot maps of all layers [n_expert, n_layer] and eviction state, see ggml_moe_cache_lookup
        ggml_tensor * map   = nullptr;
        ggml_tensor * state = nullptr;
        uint64_t dev_gen   = UINT64_MAX; // generation of the host state last copied to the device
        bool     dev_newer = false;      // the device state changed since it was last copied to the host
        size_t   expert_bytes = 0;       // all the expert tensors of one expert
        bool     dev_lookup   = false;   // small batches are looked up on the device (per group, so mixed devices work)
        ggml_tensor * counts  = nullptr; // hybrid: use counters of the experts [n_expert, n_layer + 1], see ggml_moe_cache_assign
        ggml_tensor * scores  = nullptr; // LLAMA_MOE_CACHE_POLICY=decay: decayed use counters [n_expert, n_layer], see ggml_moe_cache_set_scores
        bool     devseed      = false;   // large prefetched batches seed the group on the device (see devseed_plan)
        bool     host_read    = false;   // the graph being allocated reads the banks through host lookups (see set_graph)
        int64_t  host_need    = 0;       // the most distinct experts one of its host lookups may need in a layer of the group
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * prompt_map = nullptr; // k15-lfm2: I32 [1, n_expert] in host memory, the slots a large batch reads in place (see prompt_copy)
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
        std::vector<const void *> host_dev; // device-accessible pointers to their data, for the device lookup
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
        size_t calls  = 0; // copy calls issued
    };

    size_t n_large_ubatches = 0; // ubatches of more than max_batch tokens

    static constexpr int64_t max_batch = 32;

    // large batches copy runs of cached experts of at least this size from the cache
    static constexpr size_t min_copy_bytes = 4u*1024*1024;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;   // experts copied from the cache for large batches
    stats stats_upload; // experts uploaded for large batches
    stats stats_seed;   // experts admitted from large batches
    size_t n_batches = 0; // batched copy calls (one per expert tensor) for large batches

    std::vector<int32_t> seed_ids;
    std::vector<std::vector<int32_t>> refill; // [n_layer] experts of the last seed waiting for the lent slots to come back
    bool  refill_enabled = true;
    // the routing of the end of a prompt that did not seed (see set_refill_counts): the slots that come back after it get its most used experts
    std::vector<std::vector<int32_t>> refill_counts; // [n_layer]
    bool   refill_routed   = false; // LLAMA_MOE_CACHE_REFILL_ROUTED=1 (opt-in): else the slots get back the experts they held before they were lent
    size_t n_refill_routed = 0;    // refills that used the routing of a prompt
    bool   refill_routed_once = false; // k13-small: the next restore refills by the routing (set_refill_routed_once)
    bool  verify_enabled = false; // LLAMA_MOE_CACHE_VERIFY=1 (debug): compare the cached experts with host memory
    stats stats_refill;
    std::vector<std::vector<int32_t>> seed_counts; // [n_layer] routing of an earlier ubatch of the prompt, see set_seed_counts
    std::vector<ggml_backend_copy_region> seed_regions;
    size_t seed_unstaged = 0; // seeds of a layer dropped because a planned expert was not staged in the copy of an expert tensor
    size_t seed_skipped  = 0; // prompt ends that did not seed because the graph of the seed did not fit (see seed_unavailable)
    int64_t devseed_min     = INT64_MAX; // graphs of ubatches of at least this many tokens get the seed plans
    size_t devseed_graphs   = 0; // graphs seeded on the device
    size_t devseed_disabled = 0; // seed plans not run: their fills are not all on the device (see devseed_prepare)
    size_t devseed_admitted = 0; // experts they admitted
    size_t devseed_bytes    = 0;
    size_t  devseed_replays      = 0; // device seeds replayed after a merged layer-major graph (devseed_replay)
    size_t  devseed_replay_bytes = 0; // bytes they uploaded from host memory
    int64_t devseed_replay_us    = 0; // their time (plans + uploads), synchronous
    int64_t devseed_replay_up_us = 0; // of which the uploads
    bool   prefill_deferred = false; // the default seeding skipped the prefill at load, it runs if prompts cannot seed
    bool   prefilled_dev    = false; // prefill() filled device slots
    bool   resident_mode    = false; // resident mode: device slots and RAM tier filled at load (see init_host_cache)

    bool seed_check = false;
    std::vector<seed_rec> seed_committed; // seeded layers, not yet checked
    size_t seed_checked    = 0;
    size_t seed_mismatches = 0;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer
    std::unordered_map<const ggml_tensor *, ggml_tensor *> resident; // small per-expert host tensors -> device copies

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // k15-lfm2: in-place prompts (LLAMA_MOE_PROMPT_INPLACE=1). A large batch reads the experts of its layer in place: the graph looks up
    // their slots in a prompt map (a host tensor, copied by prompt_copy as the scheduler copies a weight) and the product reads the bank of
    // the group; the missing experts are uploaded into lent slots of the bank, at lent base + expert, so nothing stages the cached experts
    // (Kaggle T4 LFM2.5-24B-A2B 16 GB, 8k prompt: 442 ms of copy_regions copying every cached expert into the staging copy per ubatch).
    // Only where every small batch runs on the device from the slots (no CPU-miss rule, no hybrid gate): the slots then decide nothing
    // about the text, the prompt reads the same expert bytes with the same product. The group with lent slots is the largest of the device;
    // a group that holds every expert of its layers admits its misses instead. A seed of a layer is issued after the layer's product, from
    // its lent slots, when the next layer is copied (prompt_flush)
    std::unordered_map<const ggml_tensor *, int32_t> prompt_of; // prompt map -> layer
    bool prompt_inplace_env = [] {
        const char * e = getenv("LLAMA_MOE_PROMPT_INPLACE");
        return e != nullptr && atoi(e) != 0;
    }();
    bool prompt_inplace_blocked = false; // the context runs layer-major prompts (their seeds read the staging copies)
    bool inplace_graph = false;          // the graph being allocated reads prompt maps (set_graph)
    struct prompt_seed_t {
        int32_t il = -1;
        int32_t base = 0;
        ggml_backend_t backend = nullptr;
        std::vector<int32_t> counts;
    } pseed;
    ggml_context_ptr ctx_pviews;
    size_t n_prompt_inplace = 0, n_prompt_fallback = 0, prompt_hits = 0, prompt_misses = 0, prompt_bytes = 0, prompt_seeds = 0, n_prompt_ring_waits = 0, n_prompt_seed_left = 0;
    bool prompt_log_done = false;
    mutable bool prompt_why_logged = false;

    // lending (see lend): LLAMA_MOE_CACHE_LEND=0 off, 1 on with one cache device, 2 also with several (not validated)
    int    lend_mode    = 1;     // LLAMA_MOE_CACHE_LEND: 0 off, 1 staging copies, 2 also the graph allocator (experimental)
    bool   lend_auto    = true;  // LLAMA_MOE_CACHE_LEND unset: lend only if the staging copies are a large share of the cache, see lend
    bool   seed_own     = false; // prompts seed from their own ids by default (LLAMA_MOE_CACHE_SEED unset or 2), see lend
    double lend_min_frac = 0.08; // LLAMA_MOE_CACHE_LEND_MIN_FRAC: that share
    bool   lend_save     = false; // k15-fn LLAMA_MOE_CACHE_LEND_SAVE=1 (opt-in): also lend where it frees lend_save_min bytes, see lend_saves
    size_t lend_save_min = 256ull*1024*1024; // LLAMA_MOE_CACHE_LEND_SAVE_MIB
    bool   lend_save_refill = true; // LLAMA_MOE_CACHE_LEND_SAVE_REFILL=0: slots lent for lend_saves come back empty (lazy), see unlend
    bool   lend_multi   = false; // LLAMA_MOE_CACHE_LEND_MULTI=1: also with the cache on several devices (not validated)
    bool   lend_off_log = false;
    size_t n_lends      = 0; // graphs whose staging copies used lent slots
    int32_t lend_max    = 0; // most slots lent at once

    // device lookup for small batches: the selected experts stay on the device and the misses are read from mapped host memory
    bool device_lookup = false; // requested, see LLAMA_MOE_CACHE_DEVICE; each group then checks its device

    // eviction policy (LLAMA_MOE_CACHE_POLICY): decay (default) = decayed use counters of every expert, the misses evict the lowest
    // ones (see moe_cache_lru and ggml_moe_cache_set_scores), for the groups that cache less than decay_max_share of their experts, the
    // others keep clock; clock = CLOCK on the device, LRU on the host for every group (measured decode vs clock: gemma-4 8 GB 1.084x
    // on a G4 and 1.051x on a T4, Qwen3.6 8 GB 1.04x on a G4, neutral at 38-44% cached, 0.986x with more misses at 41% on
    // Qwen3.8-Flash-Next, which the 30% gate keeps on clock; greedy outputs identical)
    bool   policy_decay    = true;
    double half_life       = 0.0; // LLAMA_MOE_CACHE_HALF_LIFE, in tokens; 0 = clamp(64 x cached share, 8, 64)
    double decay_max_share = 0.3; // LLAMA_MOE_CACHE_DECAY_MAX_SHARE
    std::vector<std::pair<void (*)(void *), void *>> mapped; // host buffers mapped by the cache, with the function that unmaps them

    // hybrid small batches (device lookups only): the host computes the missing experts that are not fetched
    bool  hybrid       = false; // LLAMA_MOE_CACHE_HYBRID / LLAMA_MOE_CACHE_FETCH (default off)
    bool  hybrid_probe = false; // the fetch share is measured before the first small batch (LLAMA_MOE_CACHE_FETCH not set)
    bool  hybrid_live  = false; // the graphs of small batches use the hybrid path
    float hybrid_share = 1.0f;  // share of the missing experts that are fetched and admitted, 1 = all (no hybrid path)
    bool  hybrid_fixed = false; // the share does not depend on timing: the state is reset for every request (see hybrid_reset)
    // deterministic share (LLAMA_MOE_CACHE_HYBRID unset or 2): set from the decode threads by hybrid_set_threads, not measured
    bool  hybrid_det   = false;
    // default gate (LLAMA_MOE_CACHE_HYBRID unset): on only while the cache holds at most hybrid_max_frac of the host experts' bytes;
    // measured on Qwen3.8-Flash-Next UD-IQ1_S: +27% decode at 3% of the experts, a tie at 14%, -13% at 24%, -20% at 68%
    bool   hybrid_gate     = false;
    double hybrid_max_frac = 0.10; // LLAMA_MOE_CACHE_HYBRID_MAX_FRAC
    double cache_frac      = -1.0; // cached share of the host experts' bytes, -1 before the slots are sized
    double host_uncached   = 0.0;  // bytes of the host experts the slots cannot hold (host experts minus slots, uncached groups count whole)
    double host_total      = 0.0;  // k19-widefix: bytes of the host experts (cache_frac's denominator)
    static constexpr double hybrid_k = 6.0; // see set_threads
    bool  overlap      = false; // LLAMA_MOE_CACHE_FETCH_OVERLAP: single token ubatches compute the hits while the missing experts are copied

    // overlap auto: the overlap saves the time of the experts computed beside the copies and costs a few nodes per layer, which is a loss
    // when few experts are missing on a fast device; trials of 4 blocks of ov_block single token ubatches: plain, overlap, overlap, plain
    // (a trend cancels out), the first ov_skip of a block not counted (the graph is built and captured again); kept only if the median
    // of the overlap is at least ov_margin faster; trials again after ov_gap single token ubatches (doubled each time, at most
    // ov_gap_max), as the misses change with the cache and the prompts
    static constexpr int    ov_block   = 12;
    static constexpr int    ov_skip    = 3;
    static constexpr int    ov_first   = 64; // the misses go down while the cache warms up
    static constexpr int    ov_gap_max = 8192;
    static constexpr double ov_margin  = 0.01;
    int     ov_mode   = 1;

    // default rule (LLAMA_MOE_CACHE_FETCH_OVERLAP unset or 1): on while the bytes of the missing experts per cached layer and single token
    // ubatch reach ov_min (ggml_backend_moe_fetch_overlap_min_bytes, LLAMA_MOE_CACHE_FETCH_OVERLAP_MIN_MB); ov_min = 0: always on
    // measured from the device lookup counters (the same for a given prompt and configuration, not a timing) over windows of single
    // token ubatches: the first ov_rule_win after ov_rule_skip, then ov_rule_win2; off again below ov_rule_hyst*ov_min
    static constexpr int    ov_rule_skip = 8;
    static constexpr int    ov_rule_win  = 32;
    static constexpr int    ov_rule_win2 = 128;
    static constexpr double ov_rule_hyst = 0.75;
    bool     ov_rule   = false;
    double   ov_min    = 0.0;
    int      ov_seen   = 0;     // single token ubatches seen
    int      ov_win    = ov_rule_win;
    int      ov_n1     = 0;     // single token ubatches of the window
    bool     ov_have0  = false; // the window has started
    uint64_t ov_m0     = 0;     // bytes of the missing experts at the start of the window
    int      ov_nwin   = 0;
    bool    ov_auto   = false;
    bool    ov_live   = false; // the graphs of single token ubatches use the overlap
    int     ov_done   = -1;    // single token ubatches of the trial, -1 if there is no trial
    int     ov_wait   = -1;    // single token ubatches until the next trial, -1 if there is none
    int     ov_gap    = 256;
    int     ov_n      = 0;     // trials done
    int     ov_arm    = 0;     // path of the previous ubatch (1 = overlap)
    int     ov_skipn  = 0;
    int64_t ov_t_prev = -1;    // time of the previous counted single token ubatch, -1 if none
    std::vector<int64_t> ov_t[2];

    // auto (LLAMA_MOE_CACHE_HYBRID=1): trials of 5 blocks of hybrid_block small batches: plain (warm-up,
    // not counted), plain, hybrid, hybrid, plain (a trend such as the cache warming up cancels out); the time from one small batch to the
    // next is credited to the path of the first, except for the first hybrid_skip batches after a change of path; the hybrid path is
    // kept only if its median is at least hybrid_margin faster, then checked again by a new trial after win_gap small batches (doubled
    // every time, the misses go down as the cache warms up); once the plain cache is faster, the hybrid path stays off; a hybrid path
    // much slower than the plain cache ends a trial early; the plain blocks keep the use counters (see hybrid_assign)
    static constexpr int    hybrid_block  = 8;
    static constexpr double hybrid_margin = 0.03;
    static constexpr int    hybrid_gap_max = 1024;
    bool    hybrid_auto = false;
    int     win_done    = -1; // small batches of the trial, -1 if there is no trial
    int     win_wait    = -1; // small batches until the next trial, -1 if there is none
    int     win_gap     = 64;
    int     win_n       = 0;  // trials done
    int64_t win_t_prev  = -1; // time of the previous small batch, -1 if it is not counted
    int     win_arm     = 0;  // path of the previous small batch (1 = hybrid)
    int     win_skip    = 0;  // small batches not counted after a change of path
    static constexpr int hybrid_skip = 3;
    std::vector<int64_t> win_t[2];

    // CPU computes the misses [LLAMA_MOE_CPU_MISS] (see cm_calibrate and cm_rule): per batch size n (1..cm_max tokens), the host computes
    // the missing experts that are not fetched when a measured cost model says it is faster than fetching them all
    //   t_cpu(n) = cm_t1 + (tbar(n) - 1)*cm_dt   host time per missing expert, tbar(n) = tokens per distinct expert of a batch of n
    //   share(n) = t_cpu/(t_cpu + cm_tup), on log levels 1/8, 1/4, 1/2, 3/4 (the device fetches that share, the host computes the rest)
    //   on(n): m(n)*(cm_tup - max(share*cm_tup, (1 - share)*t_cpu)) > cm_margin*cm_split, m(n) = missing experts per layer of a batch of n,
    //          from the measured miss rate of the distinct experts looked up (device counters: the same inputs give the same decisions)
    // the cache state carries over between requests (no reset as with a fixed share), so layer-major prompts keep running
    static constexpr int64_t cm_max_limit = 255; // below the default prefetch threshold (256): prefetched prompts keep the k9 path
    bool    cm      = false;
    bool    cm_dbg  = false;  // LLAMA_MOE_CPU_MISS_LOG=1: every rule window
    bool    cm_force = false; // LLAMA_MOE_CPU_MISS=2: on for every batch size, no rule
    // k16 route tail skip (LLAMA_MOE_ROUTE_TAIL_SKIP, see tail_skip): the rank threshold (0 = off), whether it is the default (the
    // backend may decline a default, see ggml_moe_cache_set_tail_skip), and the largest window it applies to
    int32_t tail_rank       = 0;
    bool    tail_dflt       = false;
    int64_t tail_max_tokens = 8;
    // k19-hostrow: wide hybrid windows (LLAMA_MOE_HYBRID_WIDE, see use_device): a hybrid small batch of up to wide_max tokens takes the
    // assign path even when its routed experts may outnumber the slots (the assign admits at most the unpinned slots, the host computes
    // the rest), instead of every expert of the window on the CPU
    bool    hybrid_wide     = false;
    int64_t wide_max        = 8;
    // k19-place8 (LLAMA_MOE_PLACE8_UB set by common's fit, see common_fit_place8): a hybrid group's small batch of 2+ tokens takes the
    // CPU path of a batch whose experts do not fit the slots (every expert of the batch on the host), as with the few dozen slots of the
    // placement before, not the hybrid assign. The decode-sized cache holds the experts of tens of tokens, so the prompt's short tail
    // batches would take the multi-token hybrid assign, which writes non-finite values at fn4 8 GB (plans/k19_lookup8.md section 4,
    // fixed on k19-nanfix); LLAMA_MOE_PLACE8_SINGLE=0 lets them take it
    bool    place8_single   = false;
    bool    restore_soft    = false; // k19-place8: restore leaves the banks released when they do not allocate (set_restore_soft)
    int     n_restore_fail  = 0;
    // k11 LLAMA_MOE_CPU_MISS_PROMPT=N (opt-in, default 0 = off): prompt ubatches of max_batch + 1 .. N tokens always take the host path
    // once calibrated, at the rule's fetch share but at most 1/4 (the host computes at least three quarters of the misses, the experts
    // with the fewest uses), without waiting for a trial on the small batches. G4, LFM2.5-24B-A2B 8 GB, a 38-token prompt: 6.1 GB of uploads
    // (PCIe-bound) -> 0.4 GB. The host's expert products are not the device's bit for bit: other text, KLD-gated
    int64_t cm_prompt_force = 0;
    // LLAMA_MOE_CPU_MISS_PROMPT_SHARE=f: the fetch share of those ubatches instead of the rule's (0 < f <= 1; the host computes the rest)
    float   cm_prompt_share = -1.0f;
    // k10: with LLAMA_MOE_CACHE_HYBRID unset, a cache that holds at most hybrid_max_frac of the host experts keeps k9's default hybrid gate
    // (deterministic share) instead of the CPU-miss rule [LLAMA_MOE_CPU_MISS_GATE=0: the rule there too]
    bool    cm_gate  = false;
    size_t  cm_n_prompt = 0;  // batches above max_batch served by the host path
    size_t  cm_n_small  = 0;  // batches of at most max_batch tokens with the host path on
    bool    cm_cal  = false;  // calibrated
    int64_t cm_max  = 128;    // LLAMA_MOE_CPU_MISS_MAX_TOKENS: largest batch the host path may serve (> 32: short prompts)
    double  cm_t1    = 0.0;   // us, host: one missing expert of one token
    double  cm_dt    = 0.0;   // us, host: each further token of that expert
    double  cm_tup   = 0.0;   // us, upload of one expert
    double  cm_split = 0.0;   // us per layer: the device-host-device round trip of a host split
    static constexpr double cm_margin = 1.5;
    static constexpr double cm_hyst   = 0.5; // a size that is on stays on until the gain falls below cm_hyst*the threshold
    std::vector<float>   cm_share; // [cm_max + 1]
    std::vector<uint8_t> cm_on;    // [cm_max + 1]
    uint32_t cm_gen  = 0;          // changes with every decision (graph key)
    // rule windows over small batches: skip cm_skip, then windows of cm_win1, then cm_win2 batches
    static constexpr int cm_skip = 2;
    static constexpr int cm_win1 = 8;
    static constexpr int cm_win2 = 128;
    int      cm_seen  = 0;
    // k13-fn: one re-calibration of the host timings after cm_recal_after small batches (LLAMA_MOE_CPU_MISS_RECAL=N, 0 = off), kept
    // only if it disagrees with the start-up one by more than cm_recal_tol (then the faster host time wins)
    int      cm_recal_after = 64;
    int      cm_recal_wait  = -1;
    static constexpr double cm_recal_tol = 0.20;
    // k15-lfm2 LLAMA_MOE_CPU_MISS_STABLE (default 1; =0 = the k15 rule; plans/k15_lfm2.md sections 13 and 17: on the v2 prompts all 8
    // stable launches gave the k15 rule's text, KLD 0 between them): one decision per launch, not one of two modes drawn from a noisy timing
    // sample. The fetch share of a small batch sat on the 1/4 - 1/2 level boundary (raw share 0.354, the geometric midpoint) in the
    // cells whose calibrated raw share is 0.33-0.43 (G4: Qwen3.8-Flash-Next UD-IQ4_XS 32 GB host 22.6-34.5 us per expert against an
    // upload of 46 us, Qwen3.6 UD-Q3_K_M 8 GB 14.5-18.6 against 29): launches that drew a fast start-up sample fetched 1/4 and computed
    // twice the host rows (45435 vs 22865 per run, fn4) at 28.6-30.5 us per row in the decode (not the 23 us of the sample), and decoded
    // 7-9% faster than the launches at 1/2. The balance the share comes from treats a fetch as running beside the device work, while
    // the fetch reads host memory in the device stream and the host experts run beside it: small batches take 1/4 up to a raw share of
    // cm_small_half (LLAMA_MOE_CPU_MISS_SMALL_HALF, default 0.47; 0.354 = k15), which holds a host up to 0.89x the upload time per
    // expert at 1/4. Of 3500 start-up calibrations in this workspace's G4 / L4 / T4 logs, the ones in 0.354-0.472 (Flash-Next, q3m,
    // a few gemma-4 / q4x launches) are the minority mode of their cell; the other levels are unchanged. Also an agreeing re-calibration
    // keeps the faster host time and decides again (see cm_calibrate)
    const bool cm_stable = [] {
        const char * e = getenv("LLAMA_MOE_CPU_MISS_STABLE");
        return e == nullptr || atoi(e) != 0;
    }();
    const double cm_small_half = [this] {
        const char * e = getenv("LLAMA_MOE_CPU_MISS_SMALL_HALF");
        return !cm_stable ? 0.354 : e != nullptr ? std::min(0.612, std::max(0.177, atof(e))) : 0.47;
    }();
    // k19-l4poly (LLAMA_MOE_CPU_MISS_SMALL_HALF_TEAM, default 0.6): on a host whose host experts run on the host team (single_team_host,
    // a small host), small batches take 1/4 up to a raw share of 0.6 (a host up to 1.5x the upload time per expert). Colab L4 Flint 16 GB
    // sits on the 0.47 boundary: start-up / persisted calibrations of 116-121 us per expert (raw 0.45) fetched 1/4, those of 158-160 us
    // (raw 0.53, a cold first launch, then persisted for the bin) fetched 1/2 and lost their trials (chat 54.7 vs 57.5 tok/s, same head);
    // a fixed 1/4 decoded +13% (k19-flint), 1/8 +10%, every miss on the host -12% at 2k. Unchanged where LLAMA_MOE_CPU_MISS_SMALL_HALF
    // is set, without cm_stable, and on hosts without the team (G4, 4-vCPU T4)
    double cm_small_half_eff() const {
        static const double team_half = getenv("LLAMA_MOE_CPU_MISS_SMALL_HALF_TEAM") != nullptr ?
            std::min(0.80, std::max(0.177, atof(getenv("LLAMA_MOE_CPU_MISS_SMALL_HALF_TEAM")))) : 0.6;
        static const bool user_half = getenv("LLAMA_MOE_CPU_MISS_SMALL_HALF") != nullptr;
        return cm_stable && !user_half && single_team_host() ? std::max(cm_small_half, team_half) : cm_small_half;
    }
    bool     cm_recal_pending = false; // (timed only between requests: 2 x ~40 ms inside a reply cost chat decode 2% on G4)
    int      cm_nb    = 0;     // batches in the window
    int      cm_win   = cm_win1;
    bool     cm_have0 = false;
    // k19-tokcost: the server's verify-graph prewarm (dummy windows of one repeated token) is not a decode: while paused, its batches
    // count in no miss-rate window (one such window measured 0.7% and turned the host path off for the next request: fn4 24 GB)
    bool     cm_paused = false;
    uint64_t cm_c0[3] = { 0, 0, 0 }; // hits, fetched, left to the host at the start of the window
    int      cm_nwin  = 0;
    double   cm_rate  = -1.0;  // miss rate of the distinct experts looked up in the last window
    bool     cm_measured = false; // the decisions come from a measured miss rate (before: the prior, see cm_calibrate)
    // small batches (at most max_batch tokens): the cost model gives the candidate sizes and the share, a timing trial keeps the host path only
    // where it is faster: the time of a split boundary (device sync, host threads, copies, the CUDA graph cut in two) is not something the
    // calibration sees (G4: modelled 25 us per layer, measured 60-70 us: Qwen3.6 UD-Q4_K_XL 12 GB without a draft -20% decode with the model
    // alone). Trial = 5 blocks of cmt_block small batches: host path off (warm-up, not counted), off, on, on, off; each counted small batch is
    // timed from before its graph to its end (synchronized), the first cmt_skip of a block are not counted (graphs rebuilt); on if the median is at least
    // cmt_margin faster; trials again after cmt_gap small batches (doubled each time, at most cmt_gap_max) since the misses change with the
    // cache and the prompts. LLAMA_MOE_CPU_MISS_TRIAL=0: the model alone decides
    static constexpr int    cmt_block   = 8;
    static constexpr int    cmt_skip    = 3;
    static constexpr double cmt_margin  = 0.03;
    static constexpr int    cmt_gap_max = 4096;
    bool     cm_trial      = true;
    // k13: frozen small batches (cm_freeze): batches of at most cm_frozen_max tokens fetch nothing, the host computes every miss, no
    // timing trial or miss-rate window changes that; which expert of a token the host computes then depends only on its own ids and
    // the cache the prompts left, so a verify window splits each token as its single token decode does
    static constexpr int64_t cm_frozen_max = 8;
    bool     cm_frozen     = false;
    // LLAMA_MOE_CPU_MISS_FROZEN_FETCH=1: the freeze pins only the decisions (host path on for the frozen sizes, no trials or windows);
    // small batches still fetch their share and the assign still admits, so the cache follows the decode (not exact under drafts)
    bool     cm_frozen_fetch = false;
    // k19-flint LLAMA_MOE_CPU_MISS_DECODE (default 0 = off; 1 = on): decode batches of 1..LLAMA_MOE_CPU_MISS_DECODE_MAX tokens (default
    // and at most the mailbox's single-graph sizes) take the host path at the fixed fetch share LLAMA_MOE_CPU_MISS_DECODE_SHARE (default
    // 1/8) once calibrated, with no cost-model or timing-trial verdict: the device fetches the misses with the highest use counters
    // (staged, they become hits) and the host team computes the rest through the mailbox while the device computes the hits. Where the
    // cache holds a large share of the experts (Colab L4 Flint 16 GB, 55.8%) the timing trials only count batches of the sizes the rule
    // turned on (3 and more there), n-gram verify windows are too rare to fill one, so the warm phase's "host path off" stays for the
    // whole launch (0 host experts) and every single-token miss is read over PCIe (~44 experts, 80 MiB per pass). Only with the mailbox
    // (no per-layer graph split) and not with frozen small batches. The host's expert products are not the device's bit for bit (int8
    // activations on the host): other text, KLD-gated; the share is fixed, so one prompt gives one text in every launch
    int      cm_dec_on    = -1;    // -1 not read yet
    int64_t  cm_dec_max   = 0;
    float    cm_dec_share = 0.125f;
    bool     cm_small_ok   = true;  // the last trial kept the host path for small batches
    bool     cm_small_live = true;  // the graphs of small batches use it now
    int      cmt_done   = -1;       // small batches of the running trial, -1 if none
    int      cmt_wait   = -1;       // small batches until the next trial
    int      cmt_gap    = 64;
    int      cmt_n      = 0;
    int      cmt_arm    = 0;
    int      cmt_skipn  = 0;
    int64_t  cmt_t_prev = -1;
    std::vector<int64_t> cmt_t[2];
    // the host is not as fast as calibrated (a busy host: the last trial found the host path cmt_host_bad times slower or worse): the
    // short prompts do not take it either
    static constexpr double cmt_host_bad = 1.25;
    bool     cmt_host_ok = true;
    // k19-cmtpersist (LLAMA_MOE_CPU_MISS_PERSIST, default 1; =0 off): the trials' verdicts in the calibration file, replayed by later
    // launches with the same file and tag (see src/llama-moe-trial-persist.h); hysteresis LLAMA_MOE_CPU_MISS_TRIAL_HYST (default 0.03)
    // on the trials a launch times after its first verdict
    int      cmt_ps_state = -1;           // -1 not set up yet, 0 off, 1 on
    std::vector<llama_moe_tp::rec> cmt_rec;   // replayed and appended verdicts, in trial order
    size_t   cmt_rec_file = 0;            // how many of them came from the file
    bool     cmt_rec_dirty = false;       // timed verdicts not in the file yet
    std::string cmt_ps_path, cmt_ps_key, cmt_ps_tag;
    int      cmt_len_now = 0;             // counted small batches of the running trial (cmt_done before an early end)
    // k15-pred r22b, short prompts (LLAMA_MOE_CPU_MISS_PROMPT_AUTO=1, opt-in; 2 = every size a candidate, diagnostic): a prompt ubatch of max_batch + 1 .. cm_max
    // tokens that the cost model puts on the host path (cm_on, from the calibration and the measured miss rate) takes it without the
    // small-batch trials, but only while timing says it pays, as Strata's STRATA_PREFILL_CPU_SHARE=auto does: the first such ubatches
    // alternate with and without it (pt_need of each), timed to their end per token, and the host path is kept only if its median is at
    // least cmt_margin faster; then one in pt_every runs the other way and the last pt_keep times of each decide again. The host's expert
    // products are not the device's bit for bit (KLD-gated, like Strata's)
    static constexpr int pt_need  = 3;
    static constexpr int pt_keep  = 5;
    static constexpr int pt_every = 29;
    bool     pt_auto    = false;  // r22b G4 t17p: opt-in (decided off on 6 of 7 cells, RSS +20..81 MiB, see plans/k15_models.md round 22b)
    bool     pt_any     = false;  // LLAMA_MOE_CPU_MISS_PROMPT_AUTO=2 (diagnostic): every prompt size in range is a candidate, the timing still decides
    int      pt_state   = -1;     // -1 trial, 0 off, 1 on
    int      pt_arm     = 1;      // the path of the current / next counted prompt ubatch
    int      pt_count   = 0;      // counted prompt ubatches since the decision
    int64_t  pt_t_prev  = -1;
    int64_t  pt_n_prev  = 0;
    int      pt_n_dec   = 0;
    std::vector<double> pt_t[2];  // us per token
    // warm phase: a trial in a cache that is still filling compares the paths at a miss rate that is about to fall, and the host path
    // admits fewer experts, so it fills slower (G4 Qwen3.6 UD-Q4_K_XL 12 GB MTP: trial 1 on at a 41.6% miss rate, then off at 15.2%,
    // chat decode -10%): small batches fetch every miss and windows of cm_win1 batches measure the miss rate until it stops falling (a
    // window at least cmw_flat of the one before) or cmw_max batches, then the first trial starts
    static constexpr double cmw_flat = 0.85;
    static constexpr int    cmw_max  = 160;
    bool     cmw_on   = false;
    double   cmw_prev = -1.0;
    int      cmw_n    = 0;
    // short prompts (above max_batch): the plain cache pipelines the copies of their spilled experts with the compute of the layers
    // before (prefetch), which the cost model of the per layer fetch does not see, and each split moves the activations of the batch
    // both ways (G4 gpt-oss-20b Q2_K 8 GB, 95-token prompt: -19% with the margin of the small batches): the gain must beat
    // cm_margin_p*(cm_split + n*cm_act)
    static constexpr double cm_margin_p = 4.0;
    // small batches: the split boundary costs more than the calibration sees (G4 trials: modelled 25-50 us per layer, measured 60-70 us),
    // and a trial is not free (2 of its 5 blocks run the slower path, every counted batch is synchronized): the model makes a size a
    // candidate (and so a trial possible) only if its gain beats cm_split_scale times the calibrated split (G4 Qwen3.6 UD-Q3_K_M 12 GB
    // MTP: 14 host batches per launch, every trial off, -7% decode from the trials alone). LLAMA_MOE_CPU_MISS_SPLIT_SCALE (3)
    double   cm_split_scale = 3.0;
    bool     cm_split_scale_env = false; // LLAMA_MOE_CPU_MISS_SPLIT_SCALE set: it applies to every size (k19-l4poly)
    static double cm_split_scale_team() {
        static const double s = getenv("LLAMA_MOE_CPU_MISS_SPLIT_SCALE_TEAM") != nullptr ?
            std::max(0.0, atof(getenv("LLAMA_MOE_CPU_MISS_SPLIT_SCALE_TEAM"))) : 0.5;
        return s;
    }
    double   cm_act   = 0.0;   // us per token: its activations to the host and its expert output back

    // " (mtp)" for the cache of an MTP context, in its log lines
    const char * tag = "";

    // warm start from a routing profile (k11, opt-in): LLAMA_MOE_CACHE_PROFILE=<path> fills the device slots at load from the ranked
    // (layer, expert) list a run saved with LLAMA_MOE_CACHE_PROFILE_SAVE=<path> (its resident experts by decayed use, then the other used
    // ones), the most routed first over the whole group instead of an equal share per layer (Strata v0.2.2 --expert-profile). The file is
    // <path>.<n_layer>.<n_expert>[.mtp], so a draft model or an MTP context keeps its own. Only where experts sit changes: a cached and a
    // fetched expert compute the same, so the output is unchanged unless experts are computed on the host (CPU misses), whose split follows
    // the residency. LLAMA_MOE_CACHE_PROFILE_SEED (1): the share of each layer a prompt may still seed with its routing when a profile
    // was loaded (0: prompts do not seed, the profile set stays until decode misses replace it)
    using prof_entry = llama_moe_profile_entry;
    std::string profile_file;      // load
    std::string profile_save_file; // save at exit
    std::vector<prof_entry> profile;
    bool   profile_loaded = false;
    double profile_seed   = 1.0;
    size_t profile_filled = 0;
    // k19-ramprof: LLAMA_MOE_HOST_PROFILE=<path> (a file of the format above, <path>.<n_layer>.<n_expert>[.mtp]) fills the RAM tier at load
    // in its order (llama_moe_profile_ram_order) before the page-cache fill (prefill_resident); unset or 0 = off (the page-cache fill alone)
    std::string host_profile_file;
    // k19-ramprof2 (see prepare): the profile is read at load (hprof_order), its experts in the page cache are copied at the first batch
    // (the warmup run, after the device prefill: the load pins no more than the page-cache fill it replaces, the device prefill reads as
    // before), the others are read by the weight store in the background (read_host_async: behind every demand read) from the first small
    // batch after a prompt on (not during the load), into free slots in heat order. A large batch cancels the reads that did not start
    // (back to hprof_pending, issued again at the next small batch into the slots its prompt left free) and, at the start of a request,
    // releases unused guesses until LLAMA_MOE_HOST_PROFILE_PROMPT_FREE (0.15) of the slots are free (its plan leases every held slot, so
    // only free slots keep what it reads). A lookup that routes an expert being read waits for it (its read moves in front). The slots
    // are guesses (llama_moe_host_cache::set_spec): any reservation takes them while unused.
    std::vector<prof_entry> hprof_order;   // heat order, until the copy
    bool   hprof_copied    = false;
    bool   hprof_prompted  = false;        // a large batch ran since the copy (the background reads may start)
    bool   hprof_prev_large = false;
    double hprof_prompt_free = 0.15;
    std::deque<prof_entry>  hprof_pending; // heat order
    bool   hprof_pinned    = false;        // the slots of the background reads were allocated (one pass)
    int    hprof_inflight_max = 1024;      // background reads in flight at most (experts), topped up at each small batch
    size_t hprof_load_read = (size_t) 6144 << 20; // LLAMA_MOE_HOST_PROFILE_LOAD_READ_MIB: background reads issued at the first batch (during the load)
    // k19-ramprof2 (inclusive tier): the share of each layout's slots the profile fill leaves free (LLAMA_MOE_HOST_PROFILE_ROOM, default 0.15
    // there, 0 on the exclusive tier, whose promotions free slots): the misses of the first requests are read into RAM by the CLOCK once no
    // slot is free, and a slot read once is as likely a victim as a guess (12 GB: the repeated chat re-read the first chat's experts)
    double hprof_room = 0.0;
    std::vector<uint64_t>   hprof_ticket;  // [slot] ticket of its background read, 0 = none
    std::deque<int32_t>     hprof_q;       // slots with a ticket (or collected since), issue order
    size_t hprof_read_cap  = SIZE_MAX;     // LLAMA_MOE_HOST_PROFILE_READ_MIB
    size_t hprof_read_left = SIZE_MAX;
    struct hprof_stats {
        uint64_t issued = 0, issued_bytes = 0, done = 0, waited = 0, waited_us = 0, cancelled = 0, rounds = 0, no_slot = 0, failed = 0, dup = 0;
        uint64_t load_copied = 0, load_copied_bytes = 0, load_skipped_uncached = 0, dropped = 0, requests = 0;
    } hps;
    // per request (k19-ramprof2 diagnostics): the counters at the start of the last request
    struct hprof_snap {
        uint64_t hits = 0, misses = 0, read = 0, admitted = 0, evicted = 0, no_slot = 0, spec_taken = 0, spec_used = 0, spec_dropped = 0, done = 0;
    } hprof_last;

    // k19-l4fill: the exclusive RAM tier refills the slots it freed with the experts the last prompts routed most that neither tier holds.
    // On the exclusive tier a device admission from RAM (a promotion) frees its RAM slot, and only a device eviction copied back (a demotion)
    // or a prompt's uploads (free slots only) fill one again: the slots the device takes back empty after a prompt (the lent slots) and
    // the evictions RAM still held a copy of are filled by promotions with no demotion beside them. Colab L4 full card, Qwen3.8-Flash-Next
    // UD-IQ4_XS from the VM disk: the 640-token request fills the tier (16692 slots), the chat after it starts with 1.3-1.4k free and
    // 13.1-13.5% of the experts held nowhere instead of ~8%, and misses that share of its routing. The routing of each prompt's last large
    // ubatch is counted (llama_context reads its recorded ids at the end of the prompt, fill_counts; older prompts halved); when the server
    // goes idle (llama_moe_host_fill_idle), the experts with a count that are neither on the device nor in RAM are read by the weight store
    // into free slots (or duplicates) whose memory is already allocated (never evicting, never a new pinned chunk), most routed first, as
    // speculative reads (behind every demand read and ring read, at most LLAMA_MOE_HOST_FILL_PARALLEL (2) at once); the next batch
    // publishes the done ones and cancels the reads that did not start. A filled slot is a guess (set_spec): until it is used any
    // reservation takes it like a free slot, so the next prompt's reads still find room. Placement only: a slot holds the file's bytes.
    // Not during the decode: refilled at every small batch (LLAMA_MOE_HOST_FILL=2, the first version) the reads shared the disk with the
    // decode's misses and the demotions took the guesses again (L4 full card: 9046 reads, 20 GB per launch; chat decode 18.1-18.2 s ->
    // 25.5-29.8 s for 256 tokens), while the chat after a 30 s idle gap found the tier full either way.
    // LLAMA_MOE_HOST_FILL=0 off (also off with a routing profile, LLAMA_MOE_HOST_PROFILE, or the reads ahead)
    int    fill_mode = [] { const char * e = getenv("LLAMA_MOE_HOST_FILL"); return e == nullptr ? 1 : atoi(e); }();
    int    fill_parallel = [] { const char * e = getenv("LLAMA_MOE_HOST_FILL_PARALLEL"); return e == nullptr ? 2 : std::max(0, atoi(e)); }();
    bool   fill_log = [] {
        const char * a = getenv("LLAMA_MOE_HOST_FILL_LOG");
        const char * b = getenv("LLAMA_MOE_PROMPT_DIAG");
        return (a != nullptr && atoi(a) != 0) || (b != nullptr && atoi(b) != 0);
    }();
    std::vector<std::vector<float>> fill_cnt;   // [n_layer][n_expert] routing counts of the last prompts (halved per prompt)
    std::vector<uint32_t> fill_layer_epoch;      // [n_layer] epoch whose counts the layer took (its expert tensors share one routing)
    uint32_t fill_epoch = 0;
    bool     fill_have = false;                  // counts were recorded
    bool     fill_order_dirty = false;
    std::vector<int32_t> hfill_order;            // keys (il*n_expert + e) with a count, most routed first
    std::vector<uint64_t> fill_ticket;           // [slot] ticket of its read, 0 = none
    std::deque<int32_t>   fill_q;                // slots with a ticket (or collected since), issue order
    bool     fill_prev_large = false;
    bool     fill_store_set = false;
    struct fill_stats {
        uint64_t issued = 0, issued_bytes = 0, done = 0, dup = 0, cancelled = 0, waited = 0, waited_us = 0, rounds = 0, failed = 0, prompts = 0, idle_rounds = 0, dup_marked = 0;
    } fls;
    // k19-l4fill diagnostics: where the free slots come from (exclusive tier)
    struct slot_src_stats {
        uint64_t prom_ev = 0;      // promotions by uploads with device evictions beside them (small batches)
        uint64_t prom_noev = 0;    // ... without (the refill of slots that came back, uploads into empty slots)
        uint64_t prom_seed = 0;    // promotions by a prompt's seed
        uint64_t dem_free = 0;     // demotions into a free slot or a duplicate
        uint64_t dem_evict = 0;    // demotions that evicted a RAM expert (or an unused guess)
        uint64_t dem_kept = 0;     // device evictions RAM still held (no slot taken)
        uint64_t dem_none = 0;     // demotions without a slot
        uint64_t empty_fills = 0;  // experts uploaded into device slots that held nothing (no eviction)
    } sss;

    // k19-ptres (LLAMA_MOE_CORE=1, opt-in): a never-evicted core of the device cache whose host copies are given back to the OS. With a
    // routing profile (LLAMA_MOE_CACHE_PROFILE, or kept by the cache itself in the calibration directory), the hottest experts of each
    // group are placed in its first slots at load and no lookup, assign, seed or lend ever evicts them (moe_cache_lru::set_core,
    // ggml_moe_cache_set_core); their pages of the model file are left out of the device registration and dropped from the page cache,
    // so the RAM they held goes to what the host needs (llama_context: the PLE table's part, see core_dropped_bytes). Every other expert
    // keeps today's paths (CPU misses and device copies from the registered mapping). Size: LLAMA_MOE_CORE_MIB, else what the PLE table
    // needs (table + other host weights + 6 GiB - MemAvailable); at most LLAMA_MOE_CORE_MAX_FRAC (0.4) of each group's slots, only
    // where the cache holds at least 25% of the host experts (smaller caches release their banks for prompts), one device, no RAM tier
    int    core_mode     = 0;
    bool   core_on       = false;   // the core was placed (groups with n_core > 0)
    size_t core_want     = 0;       // bytes asked for
    size_t core_bytes    = 0;       // bytes of the experts placed in the core
    size_t core_dropped  = 0;       // bytes of their pages not resident after the drop (mincore)
    size_t core_holes_b  = 0;       // bytes of whole pages inside core experts (left out of the registration)
    bool   core_profile_auto = false;
    uint64_t core_refused = 0;      // bank releases refused (they would empty the core)
    uint64_t core_saves   = 0;
    std::vector<std::pair<uintptr_t, uintptr_t>> core_holes; // page ranges of the core experts' host copies, sorted

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            layer_set set, llama_weight_store * ws, int64_t host_mib) :
            wstore(ws), n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        tag = set == layer_set::nextn ? " (mtp)" : "";
        single_no_alloc = model.hparams.no_alloc;
        {
            const std::string sfx = "." + std::to_string(model.layers.size()) + "." + std::to_string(model.hparams.n_expert) +
                (set == layer_set::nextn ? ".mtp" : "");
            if (const char * e = getenv("LLAMA_MOE_CACHE_PROFILE"); e != nullptr && *e) {
                profile_file = e + sfx;
            }
            if (const char * e = getenv("LLAMA_MOE_CACHE_PROFILE_SAVE"); e != nullptr && *e && !model.hparams.no_alloc) {
                profile_save_file = e + sfx;
            }
            if (const char * e = getenv("LLAMA_MOE_CACHE_PROFILE_SEED")) {
                profile_seed = std::min(1.0, std::max(0.0, atof(e)));
            }
            // k19-ptres: the core needs a profile; without one given, the cache keeps its own (read at load, written at each new request)
            if (const char * e = getenv("LLAMA_MOE_CORE"); e != nullptr && atoi(e) != 0 && set != layer_set::nextn && !model.hparams.no_alloc) {
                core_mode = atoi(e);
                if (profile_file.empty() && profile_save_file.empty()) {
                    const std::string dir = cm_cal_dir();
                    if (!dir.empty()) {
                        std::error_code ec;
                        std::filesystem::create_directories(dir, ec);
                        uint64_t h = 1469598103934665603ull;
                        for (unsigned char c : model.desc() + "|" + std::to_string(model.n_elements()) + "|" + std::to_string(model.size())) {
                            h ^= c;
                            h *= 1099511628211ull;
                        }
                        char name[64];
                        snprintf(name, sizeof(name), "/moe_core_profile_%016llx", (unsigned long long) h);
                        profile_file      = dir + name + sfx;
                        profile_save_file = profile_file;
                        core_profile_auto = true;
                    }
                }
            }
            if (const char * e = getenv("LLAMA_MOE_HOST_PROFILE"); e != nullptr && *e && strcmp(e, "0") != 0 && !model.hparams.no_alloc) {
                host_profile_file = e + sfx;
            }
            // k19-autoprof: the self-learned routing profile (main cache only; an explicit LLAMA_MOE_HOST_PROFILE wins)
            if (set != layer_set::nextn && !model.hparams.no_alloc && host_profile_file.empty()) {
                ap_mode = llama_autoprof::mode();
                if (ap_mode != 0) {
                    ap_file = llama_autoprof::path(llama_autoprof::dir(), "moe",
                        llama_autoprof::model_hash(model.desc(), model.n_elements(), model.size()), sfx);
                }
            }
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            if (llama_moe_cache_device_ok(ggml_backend_get_device(backends[i]))) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a CUDA / ROCm / MUSA GPU backend");
        }
        if (single_auto && devices.size() != 1) {
            single_req = false; // the default single-graph host misses: one GPU only (multi-GPU is untested)
        }
        // and at least LLAMA_MOE_SINGLE_MIN_CPUS (12) usable CPUs: the host thread waits beside a team of the decode threads - 1, which
        // on a small VM computes slower than the split's full team (Colab T4, 8 vCPUs, 4 decode threads: -2..-5% decode against the
        // split, the same host experts; Colab L4, 12 vCPUs: +3..+5%)
        if (single_auto && single_req) {
            const char * e = getenv("LLAMA_MOE_SINGLE_MIN_CPUS");
            const uint32_t min_cpus = e != nullptr ? (uint32_t) std::max(0, atoi(e)) : 12u;
            if (llama_sysmem::usable_cpus() < min_cpus) {
                single_req = false;
            }
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // LLAMA_MOE_CACHE_DEVICE=0 disables the device lookup, the host then plans the copies of each layer
        {
            const char * env = getenv("LLAMA_MOE_CACHE_DEVICE");
            device_lookup = env == nullptr || atoi(env) != 0;
            env = getenv("LLAMA_MOE_PIN_RANGES");
            pin_mode = env == nullptr ? 1 : std::max(0, std::min(2, atoi(env)));
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_POLICY");
            if (env != nullptr && strcmp(env, "clock") == 0) {
                policy_decay = false;
            } else if (env != nullptr && strcmp(env, "decay") != 0 && env[0] != '\0') {
                LLAMA_LOG_WARN("%s: unknown LLAMA_MOE_CACHE_POLICY=%s (clock or decay), using decay\n", __func__, env);
            }
            env = getenv("LLAMA_MOE_CACHE_HALF_LIFE");
            if (env != nullptr) {
                half_life = std::max(0.0, atof(env));
            }
            env = getenv("LLAMA_MOE_CACHE_DECAY_MAX_SHARE");
            if (env != nullptr) {
                decay_max_share = atof(env);
            }
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_SEED_CHECK");
            seed_check = env != nullptr && atoi(env) != 0;
        }
        // LLAMA_MOE_CACHE_HYBRID: unset = on while the cache holds a small share of the experts (see hybrid_gate), 0 = off,
        // 2 = on; both with a share from the decode threads (see hybrid_set_threads), so the same request gives the same output in every run;
        // 1 = auto (timing trials, kept only if faster) and 3 = on with a share measured at the first small batch: both depend on timing,
        // two runs can give different results;
        // LLAMA_MOE_CACHE_FETCH=0..1 sets the share (the same in every run, see hybrid_reset), unless LLAMA_MOE_CACHE_HYBRID=0
        {
            const char * env   = getenv("LLAMA_MOE_CACHE_HYBRID");
            const char * env_f = getenv("LLAMA_MOE_CACHE_FETCH");
            const char * env_x = getenv("LLAMA_MOE_CACHE_HYBRID_MAX_FRAC");
            const int    mode  = env != nullptr ? atoi(env) : -1;
            if (env_x != nullptr) {
                hybrid_max_frac = std::max(0.0, atof(env_x));
            }
            // LLAMA_MOE_CACHE_HYBRID=0 turns it off even with LLAMA_MOE_CACHE_FETCH set
            hybrid = device_lookup && mode != 0 && model.hparams.n_expert >= 4;
            // k19-exactmtp: with LLAMA_MTP_EXACT=1 the host never computes an expert by default (no CPU-miss rule, no hybrid gate), as
            // LLAMA_MOE_CPU_MISS=0 LLAMA_MOE_CACHE_HYBRID=0 would. A missing expert computed on the host runs CPU kernels where the device
            // runs MMVQ, and which experts miss follows the residency (drafts touch the experts of rejected tokens) and the timed rule, so
            // the drafted text left the no-draft text (L4 full card, q4x + MTP chat: 82 experts on the host in the drafted launch, none in
            // the no-draft one, other text after token 64). LLAMA_MOE_CACHE_HYBRID / _FETCH / LLAMA_MOE_CPU_MISS set by hand keep their
            // meaning; LLAMA_MOE_EXACT_HOST=1 keeps the default rules under LLAMA_MTP_EXACT=1 (the old behaviour)
            const char * env_cm0 = getenv("LLAMA_MOE_CPU_MISS");
            if (hybrid && env == nullptr && env_f == nullptr && (env_cm0 == nullptr || atoi(env_cm0) == 0) && llama_moe_exact_host_off()) {
                hybrid = false;
                if (!model.hparams.no_alloc) {
                    LLAMA_LOG_INFO("%s: MoE cache: LLAMA_MTP_EXACT=1: missing experts are fetched, never computed on the host (no CPU misses, "
                        "no hybrid gate; LLAMA_MOE_EXACT_HOST=1 keeps them)\n", __func__);
                }
            }
            // LLAMA_MOE_CPU_MISS (unset or 1 = on, 0 = off): with LLAMA_MOE_CACHE_HYBRID and LLAMA_MOE_CACHE_FETCH unset, the host computes
            // the missing experts per batch size where a calibration at the first small batch says it is faster (see cm_calibrate, cm_rule)
            {
                const char * env_cm = getenv("LLAMA_MOE_CPU_MISS");
                // k19-uma: unset = off where every cache device is an integrated GPU (unified memory, e.g. Strix Halo gfx1151): a fetch there
                // is a copy within the same DRAM, not over PCIe, so the rule's calibration prices the host path against a transfer that does
                // not exist (gfx1151 Qwen3.6-35B-A3B at an 8 GiB budget and LFM2.5-24B-A2B at 4 GiB: LLAMA_MOE_CPU_MISS=0 decodes 3-8% faster,
                // Linux and Windows). Discrete GPUs (GGML_BACKEND_DEVICE_TYPE_GPU) are unchanged; LLAMA_MOE_CPU_MISS=1 turns it on there too
                bool uma = !devices.empty();
                for (const auto & d : devices) {
                    uma = uma && ggml_backend_dev_type(ggml_backend_get_device(d.backend)) == GGML_BACKEND_DEVICE_TYPE_IGPU;
                }
                cm = hybrid && env == nullptr && env_f == nullptr && (env_cm == nullptr ? !uma : atoi(env_cm) != 0);
                if (uma && env_cm == nullptr && hybrid && env == nullptr && env_f == nullptr && !model.hparams.no_alloc) {
                    LLAMA_LOG_INFO("%s: MoE cache on an integrated GPU (unified memory): CPU misses off (LLAMA_MOE_CPU_MISS=1 on)\n", __func__);
                }
                // =2 (diagnostic, KLD gates): on for every batch size once calibrated, without the rule (the share is at most 3/4)
                cm_force = cm && env_cm != nullptr && atoi(env_cm) == 2;
                const char * env_t = getenv("LLAMA_MOE_CPU_MISS_TRIAL");
                cm_trial = !cm_force && (env_t == nullptr || atoi(env_t) != 0);
                const char * env_p = getenv("LLAMA_MOE_CPU_MISS_MAX_TOKENS");
                cm_max = std::max<int64_t>(1, std::min<int64_t>(cm_max_limit, env_p != nullptr ? atoll(env_p) : cm_max));
                if (const char * env_pf = getenv("LLAMA_MOE_CPU_MISS_PROMPT"); cm && env_pf != nullptr && atoll(env_pf) > max_batch) {
                    cm_prompt_force = std::min<int64_t>(cm_max_limit, atoll(env_pf));
                    cm_max = std::max(cm_max, cm_prompt_force);
                    if (const char * env_ps = getenv("LLAMA_MOE_CPU_MISS_PROMPT_SHARE"); env_ps != nullptr && atof(env_ps) > 0.0) {
                        cm_prompt_share = (float) std::min(1.0, atof(env_ps));
                    }
                }
                cm_dbg = getenv("LLAMA_MOE_CPU_MISS_LOG") != nullptr && atoi(getenv("LLAMA_MOE_CPU_MISS_LOG")) != 0;
                if (const char * env_pa = getenv("LLAMA_MOE_CPU_MISS_PROMPT_AUTO")) {
                    pt_auto = atoi(env_pa) != 0;
                    pt_any  = atoi(env_pa) == 2;
                }
                if (const char * env_s = getenv("LLAMA_MOE_CPU_MISS_SPLIT_SCALE")) {
                    cm_split_scale = std::max(0.0, atof(env_s));
                    cm_split_scale_env = true;
                }
                if (const char * env_r = getenv("LLAMA_MOE_CPU_MISS_RECAL")) {
                    cm_recal_after = std::max(0, atoi(env_r));
                }
            }
            // k10: k9's default gate (below) keeps its own domain, a cache of at most hybrid_max_frac of the host experts (decided once the
            // slots are sized): G4, Qwen3.8-Flash-Next UD-IQ1_S at 8 GB (2.2% cached), the rule kept batch size 1 off in some launches and
            // fetched every miss, decode 65.7 -> 49.6 tok/s against the gate's fixed share
            cm_gate = cm && !cm_force && mode < 0 && (getenv("LLAMA_MOE_CPU_MISS_GATE") == nullptr || atoi(getenv("LLAMA_MOE_CPU_MISS_GATE")) != 0);
            if (cm) {
                // no fixed share and no reset: the host computes misses only where the rule turns it on
                hybrid_probe = true;
                hybrid_auto  = false;
                hybrid_live  = false;
            } else if (hybrid && env_f != nullptr) {
                hybrid_share = std::min(1.0f, std::max(0.0f, (float) atof(env_f)));
                hybrid       = hybrid_share < 1.0f;
                hybrid_live  = hybrid;
                hybrid_fixed = hybrid;
            } else if (hybrid && (mode == 1 || mode == 3)) {
                hybrid_probe = true;
                hybrid_auto  = mode == 1;
            } else if (hybrid) {
                // unset, 2 or another value: the share is set by hybrid_set_threads
                hybrid_det   = true;
                hybrid_fixed = true;
                hybrid_gate  = mode < 0;
            }
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_LEND");
            lend_mode = env == nullptr ? 1 : atoi(env);
            lend_auto = env == nullptr;
            seed_own = seed_mode() == 2;
            env = getenv("LLAMA_MOE_CACHE_LEND_MIN_FRAC");
            if (env != nullptr) {
                lend_min_frac = std::max(0.0, atof(env));
            }
            env = getenv("LLAMA_MOE_CACHE_LEND_MULTI");
            lend_multi = env != nullptr && atoi(env) != 0;
            // opt-in: G4 k15gm_g4, Qwen3.8-Flash-Next UD-IQ4_XS + MTP at 32 GB: cache 18719 -> 19988 MiB with the same device memory, pinned
            // decode +5.1 / +1.9 / +3.1% (chat / 2k / 8k, same text), but prompts pay the refill of the lent slots: 2k 2441 -> 2313 tok/s
            // (TTFT 935 -> 986 ms), 8k -2..-3.5%; Qwen3.6 UD-Q4_K_XL 12 / 16 GB, Flash-Next 16 GB and LFM2.5 16 GB unchanged
            env = getenv("LLAMA_MOE_CACHE_LEND_SAVE");
            lend_save = env != nullptr && atoi(env) != 0;
            env = getenv("LLAMA_MOE_CACHE_LEND_SAVE_REFILL");
            lend_save_refill = env == nullptr || atoi(env) != 0;
            env = getenv("LLAMA_MOE_CACHE_LEND_SAVE_MIB");
            if (env != nullptr) {
                lend_save_min = (size_t) std::max(0ll, atoll(env))*1024*1024;
            }
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_REFILL");
            refill_enabled = env == nullptr || atoi(env) != 0;
            // opt-in: measured +0.2-0.4% decode and about -0.5% on 2k prompts (Qwen3.8-Flash-Next IQ1_S), not a clear gain
            env = getenv("LLAMA_MOE_CACHE_REFILL_ROUTED");
            refill_routed = env != nullptr && atoi(env) != 0;
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_VERIFY");
            verify_enabled = env != nullptr && atoi(env) != 0;
        }

        // LLAMA_MOE_CACHE_FETCH_OVERLAP (or the older LLAMA_MOE_CACHE_OVERLAP): 0 = off, single token ubatches copy the missing experts
        // before the expert FFN of the layer; unset or 1 = on while enough bytes of missing experts are copied per layer and token (see
        // ov_rule), 2 = on, 3 = auto, kept while timing trials find it faster (see ov_tick), 4 = on where the backend expects a gain
        // (ggml_backend_moe_fetch_overlap_default)
        // the results are the same either way (each slot is computed once, see build_moe_ffn), only the speed depends on the choice
        {
            const char * env = getenv("LLAMA_MOE_CACHE_FETCH_OVERLAP");
            if (env == nullptr) {
                env = getenv("LLAMA_MOE_CACHE_OVERLAP");
            }
            ov_mode = env != nullptr ? atoi(env) : 1;
            overlap = device_lookup && ov_mode != 0 && model.hparams.n_expert >= 4 && ov_mode >= 1 && ov_mode <= 4;
            ov_auto = overlap && ov_mode == 3;
            ov_live = overlap && !ov_auto;
            ov_wait = ov_auto ? ov_first : -1;
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        // and only the layers that the graphs of the context run (see layer_set)
        for (size_t il = 0; il < model.layers.size(); ++il) {
            if (!llama_moe_cache_in_set(model, set, il)) {
                continue;
            }
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;

        // the budget also holds the device copies of the small per-expert tensors and the state of the device lookup
        std::vector<size_t> fixed_of(groups.size(), 0);
        std::vector<size_t> budget_of(groups.size(), 0);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group  & g = groups[ig];
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  alignment = ggml_backend_buft_get_alignment(d.buft);
            for (int32_t il : g.layers) {
                for (const ggml_tensor * t : llama_moe_cache_layer_small(model.layers[il])) {
                    fixed_of[ig] += GGML_PAD(ggml_nbytes(t), alignment);
                }
            }
            if (device_lookup) {
                fixed_of[ig] += GGML_PAD((size_t) n_expert*model.layers.size()*sizeof(int32_t), alignment);
            }
            budget_of[ig] = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
        }
        // the device lookup state is padded like the state tensor allocated below
        auto group_size = [&](size_t ig, int32_t n_slots) {
            return alloc_size(groups[ig], n_slots) + fixed_of[ig] + (device_lookup ? GGML_PAD((8 + 3*(size_t) n_slots)*sizeof(int32_t), ggml_backend_buft_get_alignment(devices[groups[ig].id].buft)) : 0);
        };

        // a small batch (at most max_batch tokens) is read from the cache only if every group holds the experts of all of its tokens in
        // one layer (see get_slot_map), else its layers in that group compute from host memory, with other results than the device: a
        // group of few layers (e.g. the one layer of another type) gets few slots from the split by size, and a small cache then gives
        // other outputs than a large one for the same batch (the server runs the last 4 tokens of a prompt as their own batch). Each
        // group gets at least the slots of n tokens (LLAMA_MOE_CACHE_GROUP_MIN = n, at most max_batch, 0 = off), taken from the other
        // groups of the device; n is halved down to 1 while that takes more than LLAMA_MOE_CACHE_GROUP_MIN_FRAC (0.10) of the device
        // budget, or more than the other groups have above their own minimum. The default n = 4 covers the decode, the checkpoint tail
        // of the server and an MTP draft of 3; n = 32 (every small batch) cost gemma-4 26B-A4B 1-4% decode at an 8-12 GB tier (L4)
        {
            int64_t n_min = 4;
            if (const char * env = getenv("LLAMA_MOE_CACHE_GROUP_MIN")) {
                n_min = std::max<int64_t>(0, std::min<int64_t>(max_batch, atoll(env)));
            }
            double min_frac = 0.10;
            if (const char * env = getenv("LLAMA_MOE_CACHE_GROUP_MIN_FRAC")) {
                min_frac = std::max(0.0, atof(env));
            }
            for (size_t id = 0; id < devices.size() && n_min > 0; ++id) {
                size_t dev_budget = 0;
                for (size_t ig = 0; ig < groups.size(); ++ig) {
                    dev_budget += groups[ig].id == (int) id ? budget_of[ig] : 0;
                }
                if (dev_budget == 0) {
                    continue;
                }
                for (int64_t n = n_min; n >= 1; n /= 2) {
                    std::vector<size_t> need(groups.size(), 0);
                    size_t extra   = 0; // what the groups below their minimum lack
                    size_t surplus = 0; // what the others have above it
                    for (size_t ig = 0; ig < groups.size(); ++ig) {
                        const group & g = groups[ig];
                        if (g.id != (int) id) {
                            continue;
                        }
                        const int32_t n_expert = g.ref[0]->ne[2];
                        const int32_t n_slots  = (int32_t) std::min<int64_t>((int64_t) g.layers.size()*n_expert, std::min<int64_t>(n_expert, n*n_expert_used));
                        need[ig] = group_size(ig, n_slots);
                        if (budget_of[ig] < need[ig]) {
                            extra += need[ig] - budget_of[ig];
                        } else {
                            surplus += budget_of[ig] - need[ig];
                        }
                    }
                    if (extra == 0) {
                        break;
                    }
                    if ((double) extra > min_frac*(double) dev_budget || extra > surplus) {
                        continue;
                    }
                    // the others give up the extra in proportion to what they have above their minimum
                    size_t taken = 0;
                    for (size_t ig = 0; ig < groups.size(); ++ig) {
                        if (groups[ig].id != (int) id || budget_of[ig] < need[ig]) {
                            continue;
                        }
                        const size_t t = (size_t) std::ceil((double) extra*(double) (budget_of[ig] - need[ig])/(double) surplus);
                        const size_t d = std::min(t, budget_of[ig] - need[ig]);
                        budget_of[ig] -= d;
                        taken         += d;
                    }
                    for (size_t ig = 0; ig < groups.size(); ++ig) {
                        if (groups[ig].id == (int) id && budget_of[ig] < need[ig]) {
                            LLAMA_LOG_INFO("%s: %s: MoE cache group of %zu layers (%s): %.2f -> %.2f MiB, the experts of %" PRId64 " tokens per layer "
                                "(batches of at most %" PRId64 " tokens are read from the cache, LLAMA_MOE_CACHE_GROUP_MIN)\n", __func__,
                                ggml_backend_name(devices[id].backend), groups[ig].layers.size(), ggml_type_name(groups[ig].ref.back()->type),
                                budget_of[ig]/1024.0/1024.0, need[ig]/1024.0/1024.0, n, n);
                            budget_of[ig] = need[ig];
                        }
                    }
                    GGML_UNUSED(taken);
                    break;
                }
            }
        }

        // k19-t4small (decode extension, LLAMA_MOE_DECODE_EXT=1: on where it can be; unset or 0 off, opt-in until it is measured beyond
        // the T4 12 / 8 GB caps and its long48 max outliers are decided): the banks of the cache of the main
        // model (not an MTP context's) on a single device with CUDA VMM get room for LLAMA_MOE_DECODE_EXT_MAX_FRAC (0.6) more slots per
        // group, unbacked at load; the context grows the cache into the device memory the decode leaves free (llama_context::ext_unpark).
        // Not with the lazy KV cache's shrink (LLAMA_KV_LAZY), which retires slots for good
        double ext_frac = 0.0;
        {
            const char * e  = getenv("LLAMA_MOE_DECODE_EXT");
            const char * ef = getenv("LLAMA_MOE_DECODE_EXT_MAX_FRAC");
            ext_frac = ef != nullptr ? std::max(0.0, atof(ef)) : 0.6;
            const bool want = (e != nullptr && atoi(e) != 0) && ext_frac > 0.0 && set != layer_set::nextn && !model.hparams.no_alloc &&
                devices.size() == 1 && !llama_kv_lazy_alloc_on();
            ext_on = want && ext_probe(devices[0]);
            if (!ext_on) {
                ext_frac = 0.0;
            }
        }
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = budget_of[ig];
            const int32_t max_slots = g.layers.size()*n_expert;
            auto total_size = [&](int32_t n_slots) {
                return group_size(ig, n_slots);
            };
            while (g.n_slots < max_slots && total_size(g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            int32_t grow_q16 = 0;
            if (policy_decay && g.n_slots < decay_max_share*max_slots) {
                // half-life in tokens: a small cache turns over faster, so it forgets faster (fit on decode traces)
                const double hl = half_life > 0.0 ? half_life : std::min(64.0, std::max(8.0, 64.0*g.n_slots/max_slots));
                grow_q16 = (int32_t) std::min(2.0*65536.0, std::max(65537.0, std::round(65536.0*std::exp2(1.0/hl))));
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots, grow_q16, g.layers[0]);
            g.n_slots_load = g.n_slots;
            g.n_slots_fit  = g.n_slots;
            g.n_slots_cap  = ext_on ? std::min<int32_t>(max_slots, g.n_slots + (int32_t) std::ceil(ext_frac*g.n_slots)) : g.n_slots;
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size()) + (device_lookup ? 2 : 0) + (hybrid || overlap ? 1 : 0) +
                (device_lookup && grow_q16 > 0 ? 1 : 0);
            for (int32_t il : g.layers) {
                n_tensors[g.id] += llama_moe_cache_layer_small(model.layers[il]).size();
            }
            n_tensors_host  += 2*g.layers.size(); // slot map and prompt map
        }
        if (n_tensors_host == 0) {
            throw llama_moe_cache_too_small();
        }

        // the share of the host experts' bytes the cache holds (the uncached groups count as host experts), it decides the default hybrid gate
        {
            double cached = 0.0;
            double host   = 0.0;
            for (const group & g : groups) {
                size_t eb = 0;
                for (const ggml_tensor * t : g.ref) {
                    eb += t->nb[2];
                }
                cached += (double) g.n_slots*eb;
                host   += (double) g.host_bytes;
            }
            cache_frac    = host > 0.0 ? std::min(1.0, cached/host) : 1.0;
            host_uncached = std::max(0.0, host - cached);
            host_total    = host;
        }
        tail_skip_init(model, set);
        // k13-small: the default single-graph host misses need a cache that holds a real share of the experts. With almost every expert
        // on the host (G4 at 4 GB, Qwen3.6 Q3_K_M + MTP, 0.7% cached) the split decodes faster: 155 / 131 tok/s chat / 2k against 129 /
        // 120 with single-graph, same text (LLAMA_MOE_SINGLE=0 vs default at -fitt 256). The 8 GB tiers it was built for cache 17-38%.
        // LLAMA_MOE_SINGLE_MIN_FRAC (0.015) is that share; an explicit LLAMA_MOE_SINGLE keeps its meaning. 5% (the first value) lost
        // where more is cached: G4, single-graph on vs off, same text: Qwen3.6 Q3_K_M no draft 4 GB (4.7% cached) 137.1 / 133.0 / 132.6 vs
        // 118.7 / 115.1 / 114.7 tok/s, LFM2.5 4 GB (4.0%) 118.0 / 119.8 / 119.0 vs 103.1 / 104.3 / 102.9, Flash-Next UD-IQ1_S 8 GB (2.2%,
        // k14 re-bench) 78.1 / 78.2 / 76.8 vs 65.5 / 65.7 / 65.0, gpt-oss 4 GB (4.4%) equal; at 0.7% (Q3_K_M + MTP 4 GB) the split still
        // wins, 154.6 / 136.7 / 149.4 vs 138.9 / 112.7 / 131.0 (chat / 2k / 8k, G4 run k13r7a)
        if (single_auto && single_req) {
            const char * e = getenv("LLAMA_MOE_SINGLE_MIN_FRAC");
            const double min_frac = e != nullptr ? std::max(0.0, atof(e)) : 0.015;
            if (cache_frac >= 0.0 && cache_frac < min_frac) {
                single_req = false;
                if (!model.hparams.no_alloc) {
                    LLAMA_LOG_INFO("%s: MoE cache holds %.1f%% of the host experts: single-graph host misses off, the split computes them (below "
                        "%.1f%%, LLAMA_MOE_SINGLE_MIN_FRAC)\n", __func__, 100.0*cache_frac, 100.0*min_frac);
                }
            }
        }
        if (hybrid && cm && cm_gate && cache_frac <= hybrid_max_frac) {
            // the cache holds a small share of the experts: k9's default hybrid gate, deterministic share from the decode threads
            cm           = false;
            hybrid_probe = false;
            hybrid_auto  = false;
            hybrid_det   = true;
            hybrid_fixed = true;
            hybrid_gate  = true;
            if (!model.hparams.no_alloc) {
                LLAMA_LOG_INFO("%s: MoE cache: CPU misses off at %.1f%% of the host experts cached: the default hybrid gate computes the misses "
                    "(LLAMA_MOE_CPU_MISS_GATE=0 keeps the CPU-miss rule)\n", __func__, 100.0*cache_frac);
            }
        }
        if (hybrid && hybrid_gate && cache_frac > hybrid_max_frac) {
            // a large cache misses few experts: the host path then costs more (graph splits, slower warm-up) than the copies it saves
            hybrid       = false;
            hybrid_det   = false;
            hybrid_fixed = false;
        }
        if (hybrid_gate && !model.hparams.no_alloc) {
            LLAMA_LOG_INFO("%s: MoE cache holds %.1f%% of the host experts -> hybrid MoE cache %s by default (at most %.1f%%, LLAMA_MOE_CACHE_HYBRID_MAX_FRAC; LLAMA_MOE_CACHE_HYBRID=0 off, 2 on)\n",
                __func__, 100.0*cache_frac, hybrid ? "on" : "off", 100.0*hybrid_max_frac);
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx      = init_ctx(n_tensors[id]);
                devices[id].ctx_bank = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);
        ctx_pviews = init_ctx(8);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            ggml_context * ctx_bank = devices[g.id].ctx_bank.get();
            for (const ggml_tensor * t : g.ref) {
                // (k19-t4small: room for n_slots_cap slots, = n_slots without the decode extension)
                ggml_tensor * bank = ggml_new_tensor_3d(ctx_bank, t->type, t->ne[0], t->ne[1], g.n_slots_cap + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx_bank, bank, bank->ne[0], bank->ne[1], g.n_slots_cap, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                for (const ggml_tensor * t : llama_moe_cache_layer_small(model.layers[il])) {
                    const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
                    ggml_tensor * copy = ggml_dup_tensor(ctx, t);
                    ggml_format_name(copy, "moe_cache.%s", t->name);
                    resident[t] = copy;
                    devices[g.id].buf_size += GGML_PAD(ggml_nbytes(copy), alignment);
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                l.prompt_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.prompt_map, "moe_cache.prompt_map-%d", il);
                prompt_of[l.prompt_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.prompt_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);

            for (const ggml_tensor * t : g.ref) {
                g.expert_bytes += t->nb[2];
            }
            if (device_lookup) {
                const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
                g.map   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, g.ref[0]->ne[2], model.layers.size());
                g.state = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8 + 3*(int64_t) g.n_slots_cap);
                ggml_format_name(g.map,   "moe_cache.%zu.map",   ig);
                ggml_format_name(g.state, "moe_cache.%zu.state", ig);
                devices[g.id].buf_size += GGML_PAD(ggml_nbytes(g.map), alignment) + GGML_PAD(ggml_nbytes(g.state), alignment);
            }
            if (hybrid || overlap) {
                // outside of the budget, so the number of slots does not depend on the hybrid path
                const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
                g.counts = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, g.ref[0]->ne[2], model.layers.size() + 1);
                ggml_format_name(g.counts, "moe_cache.%zu.counts", ig);
                devices[g.id].buf_size += GGML_PAD(ggml_nbytes(g.counts), alignment);
            }
            if (device_lookup && g.lru.grow_q16 > 0) {
                // outside of the budget too: the same number of slots with either policy
                const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
                g.scores = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, g.ref[0]->ne[2], model.layers.size());
                ggml_format_name(g.scores, "moe_cache.%zu.scores", ig);
                devices[g.id].buf_size += GGML_PAD(ggml_nbytes(g.scores), alignment);
            }
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                d.buf_bank.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx_bank.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx_bank.get(), t)) {
                    t->buffer = d.buf_bank.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            // k17-small (lazy KV step 2): with LLAMA_KV_LAZY=1 the banks can give memory back to the KV cache as it grows (shrink);
            // LLAMA_MOE_CACHE_SHRINK=0 keeps them plain buffers (step 1: the KV cache only backs its cells lazily)
            {
                // (k19-vramlend: the banks are lazy where a KV cache allocated now would be, llama_kv_lazy_alloc_on: LLAMA_KV_LAZY=1, or
                // unset with the fit's grant; a drafter kept by --fit: plain banks, as without LLAMA_KV_LAZY)
                const char * e_sh = getenv("LLAMA_MOE_CACHE_SHRINK");
                shrink_on = llama_kv_lazy_alloc_on() && (e_sh == nullptr || atoi(e_sh) != 0);
            }
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf_bank.reset(alloc_banks(d));
                if (!d.buf_bank) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf_bank.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf_bank.get());
                if (ggml_get_first_tensor(d.ctx.get()) != nullptr) {
                    d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                    if (!d.buf) {
                        throw std::runtime_error("failed to allocate the MoE cache buffers");
                    }
                    ggml_backend_buffer_clear(d.buf.get(), 0);
                    d.buf_size += ggml_backend_buffer_get_size(d.buf.get());
                }
            }
            // k19-t4small: the banks have room for n_slots_cap slots, the cache starts with the fitted ones (as after a shrink to them)
            for (group & g : groups) {
                if (g.n_slots > 0 && g.n_slots_cap > g.n_slots) {
                    set_slot_count(g, g.n_slots);
                }
            }
            if (ext_on) {
                size_t fit = 0, cap = 0;
                for (const group & g : groups) {
                    fit += (size_t) g.n_slots_fit*g.expert_bytes;
                    cap += (size_t) g.n_slots_cap*g.expert_bytes;
                }
                LLAMA_LOG_INFO("%s: MoE cache: k19-t4small decode extension: room for %.2f MiB of slots past the %.2f MiB fitted (unbacked "
                    "until the decode grows into free device memory; LLAMA_MOE_DECODE_EXT=0 off)\n", __func__, (cap - fit)/1048576.0, fit/1048576.0);
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1

            for (const auto & [t, copy] : resident) {
                ggml_backend_tensor_set(copy, t->data, 0, ggml_nbytes(t));
            }
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // LLAMA_MOE_CACHE_PREFILL=1: fill the cache at load (default only with LLAMA_MOE_CACHE_SEED=1, where prompts do not seed it);
        // with the default seeding a prefilled cache made the first prompts slower, so it starts empty there
        if (!model.hparams.no_alloc) {
            const char * env  = getenv("LLAMA_MOE_CACHE_PREFILL");
            const int    seed = seed_mode();
            // k19-ptres: the core (LLAMA_MOE_CORE=1) takes the first slots before the rest of the profile fills the others
            // k19-core2: the profile LLAMA_MOE_CORE keeps by itself warm-starts the cache only where a core was placed
            // (LLAMA_MOE_CORE_WARM: unset = so, 1 = also without a core, 0 = never); without a core the warm start alone changed the text
            // and cost decode on Colab L4 fn4nd full card (8k decode 15.5 -> 13.9 tok/s) and A100 16 GB (8k 50.1 -> 47.5 tok/s), so
            // those tiers keep the cold start of LLAMA_MOE_CORE unset. An explicit LLAMA_MOE_CACHE_PROFILE always warm-starts.
            if (!profile_file.empty() && profile_read() && (core_place(model) || true) && core_warm_ok() && prefill_profile()) {
                profile_loaded = true;
            } else if (env != nullptr ? atoi(env) != 0 : seed == 1) {
                prefill();
            }
            // the default seeding may not fit in the compute buffer, the cache is then prefilled (see seed_unavailable)
            prefill_deferred = env == nullptr && seed == 2 && !profile_loaded;
        }

        if (!model.hparams.no_alloc) {
            init_host_cache(host_mib);
            init_ahead(model);
            init_upload_ahead(model);
        }

        if (device_lookup && !model.hparams.no_alloc) {
            init_device_lookup();
        } else if (device_lookup) {
            // memory measurement: build the graphs of the device lookup for the groups whose device can run it (mapping is only known after loading)
            for (group & g : groups) {
                g.dev_lookup = g.n_slots > 0 && lookup_capable(g);
            }
            for (layer & l : layers) {
                l.host_dev.assign(l.experts.size(), nullptr);
            }
        }

        // the hybrid path needs the device lookup, and one device: the host computes the misses of one device at a time
        if (hybrid || overlap) {
            std::vector<int32_t> devs;
            for (const group & g : groups) {
                if (g.dev_lookup && g.n_slots > 0 && g.counts != nullptr && std::find(devs.begin(), devs.end(), g.id) == devs.end()) {
                    devs.push_back(g.id);
                }
            }
            if (devs.size() > 1 && !model.hparams.no_alloc) {
                LLAMA_LOG_INFO("%s: the experts are cached on %zu devices, the hybrid MoE cache path is disabled (single device only)\n", __func__, devs.size());
            }
            if (devs.size() != 1) {
                overlap      = false;
                ov_auto      = false;
                ov_live      = false;
                ov_wait      = -1;
                hybrid       = false;
                hybrid_fixed = false;
                hybrid_det   = false;
                hybrid_probe = false;
                hybrid_live  = false;
                hybrid_share = 1.0f; // no hybrid line in the stats
                cm           = false;
            } else {
                ggml_backend_dev_t dev = ggml_backend_get_device(devices[devs[0]].backend);
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                if (overlap && ov_mode == 4) {
                    // the device hint alone: on where the backend expects a gain
                    using overlap_default_t = bool (*)(ggml_backend_dev_t);
                    auto fn = reg != nullptr ? (overlap_default_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_fetch_overlap_default") : nullptr;
                    if (fn == nullptr || !fn(dev)) {
                        overlap = false;
                        ov_live = false;
                        if (!model.hparams.no_alloc) {
                            LLAMA_LOG_INFO("%s: MoE cache: no fetch overlap on %s (LLAMA_MOE_CACHE_FETCH_OVERLAP=4)\n", __func__, ggml_backend_dev_name(dev));
                        }
                    }
                }
                if (overlap && ov_mode == 1) {
                    // the default rule: the missing expert bytes per layer and token the device needs to gain (see ov_rule)
                    using overlap_min_t = size_t (*)(ggml_backend_dev_t);
                    auto fn = reg != nullptr ? (overlap_min_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_fetch_overlap_min_bytes") : nullptr;
                    const size_t min_b = fn != nullptr ? fn(dev) : SIZE_MAX;
                    const char * env_mb = getenv("LLAMA_MOE_CACHE_FETCH_OVERLAP_MIN_MB");
                    if (env_mb != nullptr) {
                        ov_min = std::max(0.0, atof(env_mb))*1024.0*1024.0;
                    } else {
                        ov_min = min_b == SIZE_MAX ? -1.0 : (double) min_b;
                    }
                    if (ov_min < 0.0) {
                        overlap = false;
                        ov_live = false;
                        if (!model.hparams.no_alloc) {
                            LLAMA_LOG_INFO("%s: MoE cache: no fetch overlap by default on %s (LLAMA_MOE_CACHE_FETCH_OVERLAP=2 on)\n", __func__, ggml_backend_dev_name(dev));
                        }
                    } else {
                        ov_rule = ov_min > 0.0;
                        ov_live = !ov_rule;
                    }
                }
            }
            if (devs.size() == 1 && !model.hparams.no_alloc) {
                if (overlap) {
                    char buf[160];
                    snprintf(buf, sizeof(buf), "while at least %.2f MiB of missing experts per layer and token are copied", ov_min/1024.0/1024.0);
                    LLAMA_LOG_INFO("%s: MoE cache: single token ubatches compute the cached experts while the missing ones are copied, %s (LLAMA_MOE_CACHE_FETCH_OVERLAP=0 off, 2 on, 3 auto, 4 device hint)\n", __func__,
                        ov_auto ? "auto: kept while trials find it faster" : ov_rule ? buf : "on");
                }
                if (cm) {
                    LLAMA_LOG_INFO("%s: MoE cache: the host computes missing experts of batches of up to %" PRId64 " tokens where a calibration at the "
                        "first small batch and the measured miss rate say it is faster (LLAMA_MOE_CPU_MISS=0 off, LLAMA_MOE_CPU_MISS_MAX_TOKENS)\n", __func__, cm_max);
                } else if (hybrid && hybrid_probe) {
                    LLAMA_LOG_INFO("%s: hybrid MoE cache: the host computes the missing experts that are not fetched, fetch share measured at the first small batch, %s (LLAMA_MOE_CACHE_FETCH=0..1 to set the share)\n",
                        __func__, hybrid_auto ? "auto: kept only if trials of small batches are faster (LLAMA_MOE_CACHE_HYBRID=1)" : "on (LLAMA_MOE_CACHE_HYBRID=3)");
                } else if (hybrid && !hybrid_det) {
                    LLAMA_LOG_INFO("%s: hybrid MoE cache: the host computes the missing experts that are not fetched, fetch share = %.3f (LLAMA_MOE_CACHE_FETCH)\n", __func__, hybrid_share);
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            if (d.buf_bank) {
                ggml_backend_buffer_set_usage(d.buf_bank.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__, tag,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__, tag,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                    if (g.lru.grow_q16 > 0) {
                        LLAMA_LOG_INFO("%s: MoE cache policy decay: half-life %.1f tokens (weight x %d/65536 per token), victim = lowest decayed counter (LLAMA_MOE_CACHE_POLICY=clock for CLOCK/LRU)\n",
                            __func__, 1.0/std::log2(g.lru.grow_q16/65536.0), g.lru.grow_q16);
                    } else if (policy_decay) {
                        LLAMA_LOG_INFO("%s: MoE cache policy clock for these layers: they cache at least %.0f%% of their experts (LLAMA_MOE_CACHE_DECAY_MAX_SHARE)\n",
                            __func__, 100.0*decay_max_share);
                    }
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // the experts of layer il, most routed first: by the routing a prompt recorded (refill counts, see set_refill_counts, else the
    // seed counts), ties and layers without a record by id. A fresh load has no record, so the load-time fills go by id there.
    std::vector<int32_t> fill_order(int32_t il, int32_t n_expert) const {
        std::vector<int32_t> ids(n_expert);
        for (int32_t e = 0; e < n_expert; ++e) {
            ids[e] = e;
        }
        const std::vector<int32_t> * c = nullptr;
        if ((size_t) il < refill_counts.size() && (int32_t) refill_counts[il].size() == n_expert) {
            c = &refill_counts[il];
        } else if ((size_t) il < seed_counts.size() && (int32_t) seed_counts[il].size() == n_expert) {
            c = &seed_counts[il];
        }
        if (c != nullptr) {
            std::stable_sort(ids.begin(), ids.end(), [c](int32_t a, int32_t b) { return (*c)[a] > (*c)[b]; });
        }
        return ids;
    }

    // fill every slot at load, an equal share of the experts of each layer, so the first prompt reads them from the cache
    // like the experts a split keeps on the device (an empty cache uploads every expert of a large batch); most routed first where
    // a prompt recorded the routing (a late prefill, see seed_unavailable), else the lowest ids
    void prefill() {
        size_t n_fill  = 0;
        size_t n_bytes = 0;
        std::vector<int32_t> fill_ids;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            // a late prefill (see seed_unavailable) after device lookups plans with their slots, the host state is stale
            if (g.dev_newer) {
                download(g);
            }
            const int32_t n_expert = g.lru.n_expert;
            const size_t  n_layer  = g.layers.size();
            // a late prefill (see seed_unavailable) leaves the lent slots to the staging copies
            const int32_t n_slots  = g.n_slots - g.lru.n_lent - g.lru.n_core; // (k19-ptres: not into the core)
            for (size_t i = 0; i < n_layer; ++i) {
                const int32_t il = g.layers[i];
                const int32_t n  = std::min<int32_t>(n_expert, n_slots/n_layer + (i < n_slots % n_layer ? 1 : 0));
                fill_ids = fill_order(il, n_expert);
                fill_ids.resize(n);
                // the plan copies runs of consecutive ids and slots: keep the chosen experts in id order
                std::sort(fill_ids.begin(), fill_ids.end());
                size_t n_hit = 0;
                if (n == 0 || !g.lru.plan(il, fill_ids.data(), fill_ids.size(), fills, n_hit)) {
                    continue;
                }
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t expert_size = w->nb[2];
                    for (size_t j = 0; j < fills.size();) {
                        size_t m = 1;
                        while (j + m < fills.size() && fills[j + m].expert == fills[j].expert + (int32_t) m && fills[j + m].slot == fills[j].slot + (int32_t) m) {
                            m++;
                        }
                        ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) w->data + fills[j].expert*expert_size, fills[j].slot*expert_size, m*expert_size);
                        n_bytes += m*expert_size;
                        j += m;
                    }
                }
                n_fill += fills.size();
                // k19-l4held: on an exclusive tier the RAM copies of the experts the device now holds (the page-cache fill at load runs
                // first) are duplicates, the first slots a reservation takes (the layer quota counts them; LLAMA_MOE_HOST_LAYER_QUOTA=0 off)
                // k19-l4fill: also with the refill of freed slots (LLAMA_MOE_HOST_FILL, default on): where the bench's warm read left more
                // of the model in the page cache, 761-1254 such copies stayed in the tier to the chat after the 640-token request
                if (host && host->exclusive() && (host_quota_mode != 0 || fill_mode != 0) && host_layer(il)) {
                    for (const auto & f : fills) {
                        const int32_t h = host->held(il, f.expert);
                        if (h >= 0 && !host->is_dup(h)) {
                            host->set_dup(h, true);
                            n_prefill_dup++;
                        }
                    }
                }
            }
        }
        if (n_fill > 0) {
            prefilled_dev = true;
            LLAMA_LOG_INFO("%s: MoE cache prefilled with %zu experts, %.2f MiB (LLAMA_MOE_CACHE_PREFILL=0 to disable)%s\n", __func__, n_fill, n_bytes/1024.0/1024.0,
                n_prefill_dup > 0 ? (std::string("; k19-l4held: ") + std::to_string(n_prefill_dup) + " RAM copies of them marked as duplicates").c_str() : "");
        }
    }

    // the ranked (layer, expert) list of profile_file, see prof_entry; false (and a warning) if it is missing or does not match
    bool profile_read() {
        if (!llama_moe_profile_load(profile_file, profile_n_expert(), profile)) {
            LLAMA_LOG_WARN("llama_moe_cache%s: LLAMA_MOE_CACHE_PROFILE: %s is missing or does not match this model's cache, the cache starts empty\n",
                    tag, profile_file.c_str());
            return false;
        }
        return true;
    }

    // k19-ptres: the bytes the core should hold (0: no core), see core_mode
    size_t core_target(const llama_model & model) const {
        if (const char * e = getenv("LLAMA_MOE_CORE_MIB")) {
            return (size_t) std::max<int64_t>(0, atoll(e)) << 20;
        }
        const ggml_tensor * t = model.per_layer_tok_embd;
        if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
            return 0; // the freed RAM goes to a lazy PLE table: none here
        }
        uint64_t host = 0;
        for (const auto & [name, w] : model.tensors_by_name) {
            if (w != t && w->buffer != nullptr && ggml_backend_buffer_is_host(w->buffer)) {
                host += ggml_nbytes(w);
            }
        }
        const uint64_t avail = llama_sysmem::available();
        // the PLE rule's reserve (4 GiB) plus 2 GiB for the other file pages the page cache keeps beside the table
        const uint64_t need  = (uint64_t) ggml_nbytes(t) + host + (6ull << 30);
        return avail > 0 && need > avail ? (size_t) (need - avail) : 0;
    }

    // k19-core2: whether the auto profile of LLAMA_MOE_CORE warm-starts the cache (see the load)
    bool core_warm_ok() const {
        if (!core_profile_auto) {
            return true;
        }
        static const int w = getenv("LLAMA_MOE_CORE_WARM") ? atoi(getenv("LLAMA_MOE_CORE_WARM")) : -1;
        const bool ok = w > 0 || (w < 0 && core_on);
        if (!ok) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-core2: no warm start from the core's profile %s: %s (LLAMA_MOE_CORE_WARM=1 warm-starts anyway)\n",
                tag, profile_file.c_str(), w == 0 ? "LLAMA_MOE_CORE_WARM=0" : "no core was placed");
        }
        return ok;
    }

    // k19-core2: the smallest share of the host experts the device cache must hold for a core (LLAMA_MOE_CORE_MIN_FRAC, default 0.25):
    // smaller caches release their banks for prompts, which a core refuses
    static double core_min_frac() {
        static const double f = getenv("LLAMA_MOE_CORE_MIN_FRAC") ? std::max(0.0, atof(getenv("LLAMA_MOE_CORE_MIN_FRAC"))) : 0.25;
        return f;
    }

    // k19-ptres: places the core of each group (the first profile entries of its layers) in its first slots; false if no core
    bool core_place(const llama_model & model) {
        if (core_mode == 0 || profile.empty()) {
            return false;
        }
        size_t host_b = 0;
        size_t cache_b = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig >= 0) {
                for (const ggml_tensor * w : layers[il].experts) {
                    host_b += ggml_nbytes(w);
                }
            }
        }
        for (const group & g : groups) {
            cache_b += (size_t) g.n_slots*g.expert_bytes;
        }
        const char * why = nullptr;
        if (devices.size() != 1) {
            why = "the cache spans several devices";
        } else if (wstore != nullptr) {
            why = "the experts are streamed (weight store)";
        } else if (host_b == 0 || (double) cache_b < core_min_frac()*(double) host_b) {
            why = "the cache holds less than LLAMA_MOE_CORE_MIN_FRAC (default 25%) of the host experts";
        }
        core_want = why == nullptr ? core_target(model) : 0;
        if (why == nullptr && core_want == 0) {
            why = "nothing to make room for (no lazy PLE table short of RAM; LLAMA_MOE_CORE_MIB sets a size)";
        }
        if (why != nullptr) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-ptres core: off: %s\n", tag, why);
            return false;
        }
        static const double max_frac = getenv("LLAMA_MOE_CORE_MAX_FRAC") ? std::min(0.9, std::max(0.0, atof(getenv("LLAMA_MOE_CORE_MAX_FRAC")))) : 0.4;
        const double share = std::min(max_frac, (double) core_want/(double) cache_b);
        size_t n_core_all = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0 || g.lru.n_lent > 0) {
                continue;
            }
            // above the core the group keeps the slots of one whole layer, so every batch whose experts fit the slots without a core
            // still fits them (slots_fit), and large batches and their seeds keep a layer's worth of slots
            const int32_t n_want = std::min<int32_t>((int32_t) std::floor(share*g.n_slots), std::max<int32_t>(0, g.n_slots - g.lru.n_expert));
            std::vector<std::vector<int32_t>> core_ids(layers.size());
            int32_t n_take = 0;
            for (const prof_entry & p : profile) {
                if (layers[p.il].ig == (int32_t) ig && n_take < n_want) {
                    core_ids[p.il].push_back(p.e);
                    n_take++;
                }
            }
            if (n_take == 0) {
                continue;
            }
            // the list is in slot order (init): the misses of each plan take the next slots from 0 up
            bool ok = true;
            for (int32_t il : g.layers) {
                std::vector<int32_t> & ids = core_ids[il];
                if (ids.empty()) {
                    continue;
                }
                std::sort(ids.begin(), ids.end());
                size_t n_hit = 0;
                if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit) || n_hit != 0) {
                    ok = false;
                    break;
                }
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t expert_size = w->nb[2];
                    for (const auto & f : fills) {
                        ok = ok && f.slot < n_take;
                        ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) w->data + f.expert*expert_size, f.slot*expert_size, expert_size);
                    }
                }
            }
            if (!ok || !g.lru.set_core(n_take)) {
                LLAMA_LOG_WARN("llama_moe_cache%s: k19-ptres core: group %zu could not be placed in its first %d slots: no core there\n", tag, ig, n_take);
                continue;
            }
            core_bytes += (size_t) n_take*g.expert_bytes;
            n_core_all += n_take;
            // the page ranges inside the core experts' host copies (memory-mapped model files only)
            for (int32_t s = 0; s < n_take; ++s) {
                const int32_t il = g.lru.key_of[s] / g.lru.n_expert;
                const int32_t e  = g.lru.key_of[s] % g.lru.n_expert;
                for (const ggml_tensor * w : layers[il].experts) {
                    if (strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(w->buffer)), "CPU_Mapped") != 0) {
                        continue;
                    }
                    const uintptr_t page = 4096;
                    // (the first 512 bytes stay registered: a copy of the experts before it reads them as padding)
                    const uintptr_t a = ((uintptr_t) w->data + (uintptr_t) e*w->nb[2] + 512 + page - 1) & ~(page - 1);
                    const uintptr_t b = ((uintptr_t) w->data + (uintptr_t) (e + 1)*w->nb[2]) & ~(page - 1);
                    if (b > a) {
                        core_holes.emplace_back(a, b);
                    }
                }
            }
        }
        std::sort(core_holes.begin(), core_holes.end());
        std::vector<std::pair<uintptr_t, uintptr_t>> merged;
        for (const auto & h : core_holes) {
            if (!merged.empty() && h.first <= merged.back().second) {
                merged.back().second = std::max(merged.back().second, h.second);
            } else {
                merged.push_back(h);
            }
        }
        core_holes = std::move(merged);
        for (const auto & h : core_holes) {
            core_holes_b += h.second - h.first;
        }
        core_on = n_core_all > 0;
        LLAMA_LOG_INFO("llama_moe_cache%s: k19-ptres core: %zu experts (%.1f MiB of %.1f MiB asked, at most %.0f%% of each group's slots) placed for good from %s, "
            "%.1f MiB of their host pages to leave out of the registration (LLAMA_MOE_CORE=0 off)\n", tag, n_core_all, core_bytes/1048576.0,
            core_want/1048576.0, 100.0*max_frac, profile_file.c_str(), core_holes_b/1048576.0);
        return core_on;
    }

    // k19-ptres: the core holes inside [base, base + size)
    bool core_holes_in(uintptr_t base, size_t size) const {
        auto it = std::lower_bound(core_holes.begin(), core_holes.end(), std::make_pair(base, (uintptr_t) 0));
        if (it != core_holes.begin() && std::prev(it)->second > base) {
            return true;
        }
        return it != core_holes.end() && it->first < base + size;
    }

    // k19-ptres: registers [base, base + size) except the core holes, one mapping per run between them; the device reads registered
    // memory at its host address, so the runs together stand for the buffer (nullptr if a run fails or the device maps it elsewhere)
    void * core_map_buffer(ggml_backend_dev_t dev, void * (*host_map)(ggml_backend_dev_t, void *, size_t, bool *), void (*host_unmap)(void *),
            void * base, size_t size) {
        const size_t n_mapped = mapped.size();
        size_t bytes = 0;
        uintptr_t b = (uintptr_t) base;
        const uintptr_t end = (uintptr_t) base + size;
        auto it = std::lower_bound(core_holes.begin(), core_holes.end(), std::make_pair(b, (uintptr_t) 0));
        if (it != core_holes.begin() && std::prev(it)->second > b) {
            --it;
        }
        bool ok = true;
        while (b < end && ok) {
            uintptr_t e = end;
            uintptr_t next = end;
            if (it != core_holes.end() && it->first < end) {
                e    = std::max(b, it->first);
                next = std::min(end, it->second);
                ++it;
            }
            if (e > b) {
                bool registered = false;
                void * dp = host_map(dev, (void *) b, e - b, &registered);
                ok = dp == (void *) b;
                if (registered) {
                    mapped.emplace_back(host_unmap, (void *) b);
                    bytes += e - b;
                }
            }
            b = std::max(b, next);
        }
        if (!ok) {
            for (size_t i = n_mapped; i < mapped.size(); ++i) {
                mapped[i].first(mapped[i].second);
            }
            mapped.resize(n_mapped);
            return nullptr;
        }
        hst.registered += bytes;
        file_pin_add(bytes); // core runs: CPU_Mapped buffers only
        return base;
    }

    // k19-ptres: drops the pages of the core holes from this process and the page cache (they are only read again if something reads
    // the core experts on the host); core_dropped = the bytes found not resident afterwards
    void core_drop() {
#if defined(__linux__)
        struct map_ent { uintptr_t lo, hi; uint64_t off; std::string path; };
        std::vector<map_ent> maps;
        if (FILE * f = fopen("/proc/self/maps", "r")) {
            char line[4096];
            while (fgets(line, sizeof(line), f) != nullptr) {
                unsigned long lo = 0, hi = 0, off = 0;
                char perms[8] = {}, devs[32] = {};
                unsigned long inode = 0;
                int n_used = 0;
                if (sscanf(line, "%lx-%lx %7s %lx %31s %lu %n", &lo, &hi, perms, &off, devs, &inode, &n_used) >= 6 && inode != 0 && n_used > 0) {
                    std::string path = line + n_used;
                    while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) {
                        path.pop_back();
                    }
                    if (!path.empty() && path[0] == '/') {
                        maps.push_back({ lo, hi, off, path });
                    }
                }
            }
            fclose(f);
        }
        std::map<std::string, int> fds;
        const int64_t t0 = ggml_time_us();
        size_t n_fadv = 0;
        for (const auto & h : core_holes) {
            madvise((void *) h.first, h.second - h.first, MADV_DONTNEED);
            for (const map_ent & m : maps) {
                const uintptr_t a = std::max<uintptr_t>(h.first, m.lo);
                const uintptr_t b = std::min<uintptr_t>(h.second, m.hi);
                if (a >= b) {
                    continue;
                }
                auto it = fds.find(m.path);
                if (it == fds.end()) {
                    it = fds.emplace(m.path, open(m.path.c_str(), O_RDONLY)).first;
                }
                if (it->second >= 0 && posix_fadvise(it->second, (off_t) (m.off + (a - m.lo)), (off_t) (b - a), POSIX_FADV_DONTNEED) == 0) {
                    n_fadv++;
                }
            }
        }
        for (auto & [path, fd] : fds) {
            if (fd >= 0) {
                close(fd);
            }
        }
        size_t resident = 0;
        std::vector<unsigned char> vec;
        for (const auto & h : core_holes) {
            const size_t n_pages = (h.second - h.first)/4096;
            vec.assign(n_pages, 0);
            if (mincore((void *) h.first, h.second - h.first, vec.data()) == 0) {
                for (unsigned char c : vec) {
                    resident += (c & 1) ? 4096 : 0;
                }
            } else {
                resident += h.second - h.first;
            }
        }
        core_dropped = core_holes_b - std::min(core_holes_b, resident);
        LLAMA_LOG_INFO("llama_moe_cache%s: k19-ptres core: host copies of the core dropped: %.1f of %.1f MiB no longer resident (%zu ranges, %zu fadvise calls "
            "over %zu files, %.1f ms)\n", tag, core_dropped/1048576.0, core_holes_b/1048576.0, core_holes.size(), n_fadv, fds.size(), (ggml_time_us() - t0)/1000.0);
#endif
    }

    // experts per layer of the cache (0: a layer it does not cache), the shape a profile must match
    std::vector<int32_t> profile_n_expert() const {
        std::vector<int32_t> n(layers.size(), 0);
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig >= 0) {
                n[il] = groups[layers[il].ig].lru.n_expert;
            }
        }
        return n;
    }

    // fill the device slots of each group at load with the first profile entries of its layers (as many as it has slots)
    bool prefill_profile() {
        size_t n_fill  = 0;
        size_t n_bytes = 0;
        std::vector<std::vector<int32_t>> by_layer(layers.size());
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t n_avail = g.n_slots - g.lru.n_lent;
            int32_t n_take = 0;
            for (const prof_entry & p : profile) {
                if (layers[p.il].ig == (int32_t) ig && n_take < n_avail) {
                    by_layer[p.il].push_back(p.e);
                    n_take++;
                }
            }
            for (int32_t il : g.layers) {
                std::vector<int32_t> & ids = by_layer[il];
                std::sort(ids.begin(), ids.end());
                size_t n_hit = 0;
                if (ids.empty() || !g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
                    continue;
                }
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t expert_size = w->nb[2];
                    for (size_t j = 0; j < fills.size();) {
                        size_t m = 1;
                        while (j + m < fills.size() && fills[j + m].expert == fills[j].expert + (int32_t) m && fills[j + m].slot == fills[j].slot + (int32_t) m) {
                            m++;
                        }
                        ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) w->data + fills[j].expert*expert_size, fills[j].slot*expert_size, m*expert_size);
                        n_bytes += m*expert_size;
                        j += m;
                    }
                }
                n_fill += fills.size();
            }
            // the decay counters start from the profile's heat (relative to the weight of one use), so the first misses evict the coldest
            if (g.lru.grow_q16 > 0) {
                g.lru.w = moe_cache_lru::w_init;
                std::fill(g.lru.scores.begin(), g.lru.scores.end(), 0);
                for (const prof_entry & p : profile) {
                    if (layers[p.il].ig == (int32_t) ig) {
                        const double v = std::min<double>(moe_cache_lru::score_max, std::max(0.0, (double) p.heat)*moe_cache_lru::w_init);
                        g.lru.scores[(size_t) p.il*g.lru.n_expert + p.e] = (int32_t) v;
                    }
                }
                g.lru.gen++;
            }
        }
        profile_filled = n_fill;
        if (n_fill > 0) {
            prefilled_dev = true;
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: MoE cache warm start from %s: %zu of %zu profile experts in the device slots, %.2f MiB, prompts seed %.0f%% "
                "of their share (LLAMA_MOE_CACHE_PROFILE_SEED; unset LLAMA_MOE_CACHE_PROFILE to disable)\n", tag, profile_file.c_str(), n_fill,
                profile.size(), n_bytes/1048576.0, 100.0*profile_seed);
        return n_fill > 0;
    }

    // at exit (LLAMA_MOE_CACHE_PROFILE_SAVE): the resident experts of every group by decayed use (the device state when the device
    // lookup planned last), then the used experts that are not resident; heat = counter / weight of one use (resident without the
    // decay policy: 1)
    void profile_save(bool quiet = false) {
        if (profile_save_file.empty()) {
            return;
        }
        std::vector<prof_entry> out_res;
        std::vector<prof_entry> out_rest;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t n_expert = g.lru.n_expert;
            std::vector<int32_t> key_of = g.lru.key_of;
            std::vector<int32_t> sc     = g.lru.scores;
            int32_t w = g.lru.w;
            if (g.dev_newer && g.state != nullptr && g.state->data != nullptr) {
                ggml_backend_tensor_get(g.state, key_of.data(), n_hdr*sizeof(int32_t), key_of.size()*sizeof(int32_t));
                if (g.scores != nullptr && g.scores->data != nullptr) {
                    ggml_backend_tensor_get(g.scores, sc.data(), 0, sc.size()*sizeof(int32_t));
                    ggml_backend_tensor_get(g.state, &w, 7*sizeof(int32_t), sizeof(int32_t));
                }
            }
            const bool decay = !sc.empty() && w > 0;
            // k19-ptres (LLAMA_MOE_CORE): without decayed counts the use counters of the assign rank the experts (resident or not, one list):
            // the core takes the first entries of each group
            if (core_mode != 0 && !decay && g.counts != nullptr && g.counts->data != nullptr) {
                std::vector<int32_t> cnt((size_t) g.counts->ne[0]*g.counts->ne[1]);
                ggml_backend_tensor_get(g.counts, cnt.data(), 0, cnt.size()*sizeof(int32_t));
                for (int32_t il : g.layers) {
                    for (int32_t e = 0; e < n_expert; ++e) {
                        const int32_t c = cnt[(size_t) il*g.counts->ne[0] + e];
                        if (c > 0) {
                            out_res.push_back({ il, e, (float) c });
                        }
                    }
                }
                continue;
            }
            std::vector<bool> res((size_t) layers.size()*n_expert, false);
            for (int32_t k : key_of) {
                if (k >= 0 && k < (int32_t) res.size()) {
                    res[k] = true;
                    out_res.push_back({ k / n_expert, k % n_expert, decay ? (float) sc[k]/w : 1.0f });
                }
            }
            if (decay) {
                for (int32_t il : g.layers) {
                    for (int32_t e = 0; e < n_expert; ++e) {
                        const size_t k = (size_t) il*n_expert + e;
                        if (!res[k] && sc[k] > 0) {
                            out_rest.push_back({ il, e, (float) sc[k]/w });
                        }
                    }
                }
            }
        }
        auto by_heat = [](const prof_entry & a, const prof_entry & b) {
            return a.heat != b.heat ? a.heat > b.heat : a.il != b.il ? a.il < b.il : a.e < b.e;
        };
        std::sort(out_res.begin(), out_res.end(), by_heat);
        std::sort(out_rest.begin(), out_rest.end(), by_heat);
        // k19-ptres: a request boundary early in a launch (few experts used yet) keeps the profile of the earlier launch
        if (quiet && out_res.size() + out_rest.size() < (size_t) std::max<int64_t>(1, (int64_t) profile.size()/2)) {
            return;
        }
        const size_t n_rest = out_rest.size();
        out_res.insert(out_res.end(), out_rest.begin(), out_rest.end());
        const bool ok = llama_moe_profile_write(profile_save_file, layers.size(), out_res);
        if (core_saves++ < 2 || !quiet) {
            LLAMA_LOG_INFO("llama_moe_cache%s: MoE cache profile %s %s: %zu resident + %zu other used experts\n", tag, ok ? "saved to" : "NOT saved to",
                    profile_save_file.c_str(), out_res.size() - n_rest, n_rest);
        }
    }

    // k19-ptres: at each new request (llama_context, before its first batch): the core still in its slots on the device (a slot that lost
    // its expert would make the device read a host copy that is no longer registered), and the profile written for the next load
    uint64_t core_checks = 0;
    uint64_t core_bad    = 0;
    void core_request() {
        if (core_mode == 0) {
            return;
        }
        if (core_on) {
            for (group & g : groups) {
                if (g.lru.n_core == 0) {
                    continue;
                }
                std::vector<int32_t> key(g.lru.n_core);
                // the device state only where it is current (newer than the host's, or uploaded since the last host change): before the
                // first upload it is empty, and an upload replaces it with the host state anyway
                const bool dev_cur = g.dev_newer || g.dev_gen == g.lru.gen;
                if (dev_cur && g.dev_lookup && g.state != nullptr && g.state->data != nullptr && !devices[g.id].released) {
                    ggml_backend_synchronize(devices[g.id].backend);
                    ggml_backend_tensor_get(g.state, key.data(), n_hdr*sizeof(int32_t), key.size()*sizeof(int32_t));
                } else {
                    std::copy(g.lru.key_of.begin(), g.lru.key_of.begin() + g.lru.n_core, key.begin());
                }
                for (int32_t s = 0; s < g.lru.n_core; ++s) {
                    const int32_t k = g.lru.key_of[s];
                    const bool ok = key[s] == k && k >= 0 && g.lru.slot_map[k / g.lru.n_expert][k % g.lru.n_expert] == s;
                    core_bad += ok ? 0 : 1;
                }
                core_checks += g.lru.n_core;
            }
            if (core_bad > 0) {
                GGML_ABORT("k19-ptres core: %llu core slots lost their expert (a host copy the device may read is no longer registered)", (unsigned long long) core_bad);
            }
        }
        profile_save(true);
    }

    // single-graph host misses (k11, LLAMA_MOE_SINGLE=1, opt-in): batches of up to GGML_MOE_MAILBOX_MAX_TOKENS tokens publish the host ids
    // and activations of each layer to a mailbox in mapped host memory (ggml_moe_mailbox_publish) instead of a graph split; a host
    // thread (single_service) computes those experts with the ops of the split path (mul_mat_id up / gate, swiglu, mul_mat_id down), on
    // a CPU backend of single_threads threads, so the results are the ones of the split; the merge waits for it on the device. A merge
    // never gives up (see single_timeout_ms).
    struct single_plan {
        ggml_context_ptr ctx;
        ggml_cgraph    * gf     = nullptr;
        ggml_gallocr_t   galloc = nullptr;
    };
    struct mailbox {
        ggml_backend_dev_t dev  = nullptr;
        void * host   = nullptr;
        void * mdev   = nullptr;
        void * ctl    = nullptr;
        int    n_used = 0;
        int    n_embd = 0;
        ggml_tensor * w_up   = nullptr;
        ggml_tensor * w_gate = nullptr;
        ggml_tensor * w_down = nullptr;
        ggml_backend_buffer_ptr buf; // the mailbox as a CPU buffer (the inputs and output of the plans)
        uint32_t done = 0;
        int64_t  h_us = 0; // k19-overlap GGML_MOE_MAILBOX_TIMING: host time of this mailbox's batches
        int64_t  h_n  = 0;
        int64_t  h_pick_ns  = 0; // sum of (host CLOCK_REALTIME at the pickup - the publish's globaltimer stamp), signed
        int64_t  h_pick_n   = 0;
        int64_t  h_pick_min = INT64_MAX;
        int64_t  h_pre_ns   = 0; // pickup -> start of the host compute, end of it -> completed (host steady clock)
        int64_t  h_post_ns  = 0;
        single_plan plans[GGML_MOE_MAILBOX_MAX_TOKENS + 1];
    };
    // LLAMA_MOE_SINGLE unset: on where it was tested (k11 round 16): CUDA (the backend has mapped mailboxes: not HIP / MUSA), one GPU
    // in the context, not Windows, and only the expert form the host plans compute (separate gate / up, SiLU: see build_moe_ffn);
    // everything else keeps the split. LLAMA_MOE_SINGLE=0 off, =1 forces the request on any device with mailboxes
    const bool single_auto = getenv("LLAMA_MOE_SINGLE") == nullptr;
    bool single_req = [] {
        const char * e = getenv("LLAMA_MOE_SINGLE");
#if defined(_WIN32)
        return e != nullptr && atoi(e) != 0;
#else
        return e == nullptr || atoi(e) != 0;
#endif
    }();
    // LLAMA_MOE_SINGLE_TIMEOUT_MS (default 1000): only a warning. The merges never give up: they wait for the host thread as long as it
    // computes, like the split path waits for its host graph, so the result is always the split's (k12 report: a 1000 ms timeout on a
    // host at load 500-1100 wrote NaN rows into the ubatch it gave up on, and into the KV cells later requests read). The host thread
    // logs a batch that took longer than this; a host graph that fails is computed again, and a second failure aborts
    const int single_timeout_ms = [] {
        const char * e = getenv("LLAMA_MOE_SINGLE_TIMEOUT_MS");
        return e != nullptr ? std::max(0, atoi(e)) : 1000;
    }();
    // the split the rule weighs for these batches: a publish and a merge instead of a graph split (LLAMA_MOE_SINGLE_SPLIT_FRAC), only
    // once this cache has a mailbox (a form the host thread computes: G4 gpt-oss-20b Q2_K, with biases, kept the split and lost 18-20%
    // decode when the rule weighed it as a mailbox)
    const double single_split_frac = [] {
        const char * e = getenv("LLAMA_MOE_SINGLE_SPLIT_FRAC");
        return e != nullptr ? std::max(0.0, atof(e)) : 0.2;
    }();
    bool single_off     = false;  // a mailbox failed: back to the split for good
    int  single_warned  = 0;      // slow host batches logged
    bool single_no_alloc = false; // a context that only measures memory: no mailboxes
    int  single_threads = 0;
    bool single_reg     = false;  // registered with the hub
    std::vector<std::unique_ptr<mailbox>> mboxes; // by layer, owned (created by the graph builder)
    std::unique_ptr<std::atomic<mailbox *>[]> mbox_live; // by layer, read by the host thread
    size_t mbox_n = 0;
    std::atomic<bool> single_failed{false};
    uint64_t single_batches = 0;
    uint64_t single_rows    = 0; // host expert rows computed
    int64_t  single_us      = 0; // time in the host graphs
    int64_t  single_max_us  = 0;
    uint64_t single_slow    = 0; // host graphs above 1 ms

    // one host thread and one CPU backend for every cache of the process (the target and its MTP draft context alternate, two threads
    // with a CPU team each would compete for the same cores): the thread serves the mailboxes of the registered caches
    struct single_hub_t {
        // the lifecycle (registration, and the last owner's stop / join / CPU backend reset) of caches created and freed on several
        // threads: without it a cache registering while the last owner stopped saw the thread still joinable and started none, then
        // the stop joined it and reset the CPU backend, and the new cache's host-miss merges waited forever. Not taken by the thread
        std::mutex           life_mtx;
        std::mutex           reg_mtx; // the owners; held by the thread while it serves them
        std::vector<impl *>  owners;
        std::thread          thr;
        std::atomic<bool>    stop{false};
        ggml_backend_ptr     cpu;
        int                  threads = 1;
        void               (*team_stop)(void) = nullptr; // k19-l4b: joins the CPU backend's host team (the last owner's stop)
        bool                 team = false;                // k19-l4b: the batches run on that team (LLAMA_MOE_SINGLE_THREADS_WIDE)
        bool                 blocking = false;            // k19-l4poly: the CUDA device wait blocks while this team runs
        // k18 (LLAMA_MOE_SINGLE_FUSED, default on; =0 the graph of each batch as before): the CPU backend's one-region form of the batch's
        // host experts (ggml_cpu_moe_host_ffn: one barrier instead of the graph's seven, no plan; bit-identical to the graph), with
        // its scratch; null when the CPU backend has none (the graph then)
        bool (*fused)(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const float *, const int32_t *, float *, int, int, int,
                void *, size_t) = nullptr;
        size_t (*fused_wsize)(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, int, int, int) = nullptr;
        std::vector<uint8_t> fused_buf;
        uint64_t             fused_batches = 0;
        std::atomic<int64_t> kick_us{0};
        std::mutex              mtx;
        std::condition_variable cv;
    };
    static single_hub_t & single_hub() {
        static single_hub_t h;
        return h;
    }
    static int64_t single_steady_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static int64_t single_rt_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    static int64_t single_now_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // the rule weighs the cheaper split once this cache has a mailbox, or before the first one when every MoE layer the graphs built
    // has the form the host plans compute and the device can map mailboxes: a decode of one token at a time (no draft) only gets a
    // mailbox once the rule turns the host path on for it, which it did not while it weighed a full split (G4 LFM2.5 8 GB: 0 host
    // experts, while a fixed share with the mailbox gave +46..+69% decode)
    int  single_form_state = 0; // 0: no MoE layer seen, 1: every one has the form, -1: one does not (or no mailboxes on its device)
    void single_note_form(int32_t il, bool ok) {
        if (!ok) {
            cm_gen += single_form_state != -1 && cm_dec_on > 0 ? 1 : 0; // k19-flint: the decode host path is off from now on
            single_form_state = -1;
        } else if (single_form_state == 0) {
            ggml_backend_dev_t dev = il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0 ? ggml_backend_get_device(backend(il)) : nullptr;
            ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
            const bool alloc = reg != nullptr && ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_mailbox_alloc") != nullptr &&
                (!single_auto || strcmp(ggml_backend_reg_name(reg), "CUDA") == 0);
            single_form_state = alloc ? 1 : -1;
            cm_gen += !alloc && cm_dec_on > 0 ? 1 : 0;
            if (single_form_state == 1) {
                cm_skip_trials(true);
            }
        }
    }
    // LLAMA_MOE_SINGLE_KICK_GATE=0: kick the host thread for every small graph (k11 round 16-19 behaviour)
    const bool single_kick_gate = [] {
        const char * e = getenv("LLAMA_MOE_SINGLE_KICK_GATE");
        return e == nullptr || atoi(e) != 0;
    }();
    bool single_weigh() const {
        return single_reg || (single_weigh_early && single_form_state == 1 && single_req && !single_off && !single_no_alloc);
    }
    // LLAMA_MOE_SINGLE_WEIGH_EARLY=0: weigh the cheaper split only once a mailbox exists (k11 round 16 behaviour)
    const bool single_weigh_early = [] {
        const char * e = getenv("LLAMA_MOE_SINGLE_WEIGH_EARLY");
        return e == nullptr || atoi(e) != 0;
    }();

    // k17-models LLAMA_MOE_SINGLE_MAX_TOKENS (1..GGML_MOE_MAILBOX_MAX_TOKENS, default 4 = k15; 8 = opt-in): the largest batch that
    // publishes to the mailbox. Above it a batch takes the graph split per MoE layer (device sync, host threads, the CUDA graph cut). The
    // host plans are the split's ops on the same rows, so 5-8 row batches on the mailbox are exact (cloud gate, plans/k17_models.md 4a:
    // md5 equal, 4.7x the host batches). Opt-in because it does not pay: Colab L4 (6 host threads) LFM2.5 8 GB with the same drafts in
    // both arms, 2k 84.9 vs 90.2 tok/s, 8k (LONG_CM=3) 74.5 vs 78.1; G4 Flint long48 within 1% (plans/k17_models.md 4c / 4d)
    // k19-miss48: on a many-core host (LLAMA_MOE_SINGLE_WIDE_MIN_CPUS, default 32 logical CPUs; LLAMA_MOE_SINGLE_WIDE=0 off) the default
    // is GGML_MOE_MAILBOX_MAX_TOKENS (8), so a DFlash2 7-row verify publishes its host experts to the mailbox instead of taking the
    // eager graph split of each MoE layer (device sync, host graph, CUDA graph cut). Colab G4 (48 vCPU, 23 decode threads), Flash-Next
    // UD-IQ4_XS DFlash2 n6, same text (md5 equal): 32 GB chat 193.8 / 191.3 -> 205.3 / 205.8 tok/s, 8k 127.2 / 126.9 -> 135.1 / 135.9
    // (plans/k19_miss48.md). Only where the cache holds at least LLAMA_MOE_SINGLE_WIDE_MIN_FRAC of the host experts (see single_max).
    // The 12-vCPU hosts of the k17 / k19-ampere measurements (Colab L4 / A100: 8 rows did not pay) keep 4. Above the k15-np share (0.50,
    // LLAMA_MOE_SINGLE_NOTE_PLAIN_MAX_FRAC) the rule still weighs the full split, so a 48 GB tier (70.8% cached) keeps fetching its
    // misses: there the host path was level per pass and cost MTP 6% (round 1)
    static bool single_wide_host() {
        static const bool v = [] {
            const char * e  = getenv("LLAMA_MOE_SINGLE_WIDE");
            const char * em = getenv("LLAMA_MOE_SINGLE_WIDE_MIN_CPUS");
            const int min_cpus = em != nullptr ? atoi(em) : 32;
            return !(e != nullptr && atoi(e) == 0) && llama_moe_n_cpus() >= min_cpus;
        }();
        return v;
    }
    const bool single_max_env = getenv("LLAMA_MOE_SINGLE_MAX_TOKENS") != nullptr;
    const int single_max_tok = [] {
        const char * e = getenv("LLAMA_MOE_SINGLE_MAX_TOKENS");
        return std::max(1, std::min<int>(GGML_MOE_MAILBOX_MAX_TOKENS, e != nullptr ? atoi(e) : (single_wide_host() ? GGML_MOE_MAILBOX_MAX_TOKENS : 4)));
    }();
    // the largest batch on the mailbox now: the wide default only where the cache holds at least LLAMA_MOE_SINGLE_WIDE_MIN_FRAC of the
    // host experts (cache_frac is known after the slots are sized; before that the wide default counts)
    int single_max() const {
        // LLAMA_MOE_SINGLE_WIDE_MIN_FRAC (default 0.20): below it the host expert compute of a 5..8-row batch is the critical path and
        // the eager split (which overlaps a little device work per layer) is level or better: G4 fn4 16 GB (14.8% cached) chat -0.6%, 8k
        // -4.5% (same text), 12 GB DFlash2 admitted -4% (prof_1105_small 2d, w7 37-41 -> 47.8 ms); 24 GB (27.5%) +6 / +3.5%, 32 GB (42%)
        // +6.7%
        // k19-g4pass: default 0.12 (LLAMA_MOE_G4PASS=0: 0.20). Since k19-overlap (one activation quantization, no id -1 rows) the mailbox
        // wins at 16 GB too: Colab G4 Flash-Next UD-IQ4_XS DFlash2 n6, 16 GB (15.5% cached), same text, w7 27.1 -> 24.9 ms, chat 161.9 ->
        // 179.1 tok/s, 8k 120.9 -> 126.9 (gp_p1, 2 launches). nsys (node trace, ms per w7 pass): eager split 33.5 wall = 25.0 busy + 7.7 idle
        // inside the verify (111 launches, the device waiting on each layer's host split) vs the mailbox 29.8 = 27.3 + 1.6 (29 launches,
        // host-expert wait + merge 3.6). 12 GB (about 9% cached, DFlash2 not admitted by default) keeps the split
        static const double min_frac = getenv("LLAMA_MOE_SINGLE_WIDE_MIN_FRAC") != nullptr ? atof(getenv("LLAMA_MOE_SINGLE_WIDE_MIN_FRAC")) :
            (g4pass_on() ? 0.12 : 0.20);
        return !single_max_env && single_max_tok > 4 && cache_frac >= 0.0 && cache_frac < min_frac ? 4 : single_max_tok;
    }

    // k19-l4poly: the CPU backend's host team serves (or, before the first mailbox, would serve) this cache's host batches: a small
    // host whose expert compute is the critical path (the rule of single_register: decode threads + a third, more than the old team)
    bool single_team_host() const {
        if (single_reg) {
            return single_hub().team;
        }
        const char * e  = getenv("LLAMA_MOE_SINGLE_THREADS");
        const char * ew = getenv("LLAMA_MOE_SINGLE_THREADS_WIDE");
        const char * et = getenv("GGML_MOE_HOST_TEAM");
        const char * em = getenv("LLAMA_MOE_SINGLE_THREADS_MAX");
        const int wide_mode = ew != nullptr ? atoi(ew) : 1;
        if ((e != nullptr && atoi(e) > 0) || wide_mode == 0 || !(wide_mode == 2 || hybrid) || (et != nullptr && atoi(et) == 0) || single_threads <= 0) {
            return false;
        }
        const int wmax = em != nullptr && atoi(em) > 0 ? atoi(em) : 16;
        return single_team_size(wmax) > std::max(1, single_threads - 1);
    }
    // the host team's size: the decode threads + a third, at most the CPUs - 4 and wmax; k19-l4poly LLAMA_MOE_SINGLE_THREADS_TEAM=N (test
    // knob): N threads, at most the CPUs - 1
    int single_team_size(int wmax) const {
        static const int team_env = getenv("LLAMA_MOE_SINGLE_THREADS_TEAM") != nullptr ? atoi(getenv("LLAMA_MOE_SINGLE_THREADS_TEAM")) : 0;
        if (team_env > 0) {
            return std::max(1, std::min(team_env, llama_moe_n_cpus() - 1));
        }
        if (single_wait_blocking_here()) {
            return std::min({ single_threads + std::max(1, 2*single_threads/3), llama_moe_n_cpus() - 2, wmax });
        }
        return std::min({ single_threads + std::max(1, single_threads/3), llama_moe_n_cpus() - 4, wmax });
    }
    // k19-l4poly (LLAMA_MOE_SINGLE_WAIT_BLOCKING, default 1; 0 = off): the CUDA backend's blocking device wait. With it the thread that waits
    // for the graph sleeps instead of spinning, so the host team takes the decode threads + two thirds (at most the CPUs - 2) instead
    // of + a third (CPUs - 4). Colab L4 (6 decode threads, 12 vCPUs) LFM2.5 4 GB chat: 10 threads blocking 45.2 tok/s, 10 spinning
    // 41.9, 11 blocking 43.0, 11 yielding 33.5 (bimodal), 8 spinning (k19-l4b) 42.0; PolyStrata 46.4 (l4p_g5a)
    // only where the cache holds at most LLAMA_MOE_SINGLE_WAIT_BLOCKING_MAX_FRAC (0.25) of the host experts: Colab L4 Flint 16 GB
    // (55.8% cached) with a team of 10 waiting blocked: chat 57.6 [50.0-57.6] vs 58.3, 2k 66.1 vs 73.3; LFM2.5 8 GB (40%) on the base's
    // own texts chat +4%, 2k -3%, 8k -4% (l4p_f1; l4p_g5a +3.0 / -5.8 / -1.9%, l4p_a7 +8 / +9 / +13% on a slower VM)
    // k19-l4c: also where the deferred admission takes the cache (fetch_defer_frac: more than a quarter of the host experts cached): it
    // moves the admitted experts to the host, so the host team is the critical path there too. Colab L4 LFM2.5 8 GB (40% cached), ms per
    // pass at chat: base 18.7, deferral 18.2, deferral + team of 10 waiting blocked 17.3, PolyStrata 17.0; at 2k / 8k the same text
    // costs the same per pass with either team (39.7 vs 39.8 / 37.9 ms at 3.61 tokens per pass, 28.1 vs 27.9 / 27.7 at 2.39) (l4c_a3).
    // An explicit LLAMA_MOE_SINGLE_WAIT_BLOCKING_MAX_FRAC keeps its meaning
    bool single_wait_blocking_here() const {
        static const bool user_frac = getenv("LLAMA_MOE_SINGLE_WAIT_BLOCKING_MAX_FRAC") != nullptr;
        static const double max_frac = user_frac ? std::max(0.0, atof(getenv("LLAMA_MOE_SINGLE_WAIT_BLOCKING_MAX_FRAC"))) : 0.25;
        return single_wait_blocking() != nullptr && cache_frac >= 0.0 && (cache_frac <= max_frac || (!user_frac && fetch_defer_frac()));
    }
    // k19-l4c: the deferred admission's own gate without the host-team test (LLAMA_MOE_FETCH_DEFER on, more than
    // LLAMA_MOE_FETCH_DEFER_MIN_FRAC of the host experts cached); fetch_defer adds the team, the batch size and the mailbox
    static int fetch_defer_mode() {
        static const int mode = [] {
            const char * e = getenv("LLAMA_MOE_FETCH_DEFER");
            return e != nullptr ? atoi(e) : 1;
        }();
        return mode;
    }
    // and at most LLAMA_MOE_FETCH_DEFER_MAX_FRAC (0.5): Colab L4 Flint 16 GB (56% cached) gained only on chat (59.9 vs 59.1 tok/s)
    // while its 2k cell lost 19% (59.6 vs 73.4: the cheaper single token steps price the n-gram windows out, 27 vs 90 accepted
    // drafts per 256, the same text in 3 of 3 launches), and with its host path off (the stack's k19-l4poly levers off) the blocking
    // wait alone cost 2.4% (52.7 vs 54.0, same text) (l4c_f4)
    bool fetch_defer_frac() const {
        static const double min_frac = getenv("LLAMA_MOE_FETCH_DEFER_MIN_FRAC") != nullptr ?
            std::max(0.0, atof(getenv("LLAMA_MOE_FETCH_DEFER_MIN_FRAC"))) : 0.25;
        static const double max_frac = getenv("LLAMA_MOE_FETCH_DEFER_MAX_FRAC") != nullptr ?
            std::max(0.0, atof(getenv("LLAMA_MOE_FETCH_DEFER_MAX_FRAC"))) : 0.50;
        return fetch_defer_mode() != 0 && cache_frac > min_frac && cache_frac <= max_frac;
    }
    static void (*single_wait_blocking())(bool) {
        static void (*fn)(bool) = [] {
            const char * e = getenv("LLAMA_MOE_SINGLE_WAIT_BLOCKING");
            if (e != nullptr && atoi(e) == 0) {
                return (void (*)(bool)) nullptr;
            }
            ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
            return reg != nullptr ? (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_wait_blocking") : nullptr;
        }();
        return fn;
    }

    bool single_usable(int64_t n_tokens) const {
        return single_req && !single_off && !single_no_alloc && n_tokens >= 1 && n_tokens <= single_max();
    }

    bool single_register() {
        if (single_reg) {
            return true;
        }
        single_hub_t & h = single_hub();
        std::lock_guard<std::mutex> life(h.life_mtx);
        std::lock_guard<std::mutex> lock(h.reg_mtx);
        if (!h.cpu) {
            h.cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
            if (!h.cpu) {
                return false;
            }
            ggml_backend_reg_t creg = ggml_backend_dev_backend_reg(ggml_backend_get_device(h.cpu.get()));
            auto set_n = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_set_n_threads");
            // LLAMA_MOE_SINGLE_THREADS (default: the decode threads - 1): the caller of the graph waits on the device beside this team,
            // so a team as large as the decode threads puts two threads on one core
            const char * e = getenv("LLAMA_MOE_SINGLE_THREADS");
            h.threads = e != nullptr && atoi(e) > 0 ? atoi(e) : std::max(1, single_threads - 1);
            // k19-l4b LLAMA_MOE_SINGLE_THREADS_WIDE (default 1; 0 = off, 2 = on every placement): where the host computes most of the
            // misses (the hybrid gate: a cache of at most LLAMA_MOE_CACHE_HYBRID_MAX_FRAC of the host experts), the host experts are the
            // critical path of a decode step, so the batch runs on the CPU backend's own team (ggml_cpu_moe_host_ffn_team,
            // GGML_MOE_HOST_TEAM=0 off) of the decode threads + max(1, the decode threads / 3) threads, at most the logical CPUs of the
            // process - 4 (the caller of the graph waits on the device, the server and the drafter keep a few) and
            // LLAMA_MOE_SINGLE_THREADS_MAX (16); only where that is more than the decode threads - 1. Elsewhere the OpenMP region and
            // the team size of before.
            // Colab L4 (6 decode threads, 12 vCPUs, LFM2.5 4 GB, k19l4b_l4): 5 -> 8 threads, chat decode +8.9% (36.3-36.9 vs 33.6-35.4
            // tok/s), 2k within the spread; a team of 11 (every vCPU but the caller's) was bimodal (27.6-38.5) and slower on median.
            // Colab G4 q4x 12 GB (CPU-miss rule, not hybrid): the team on every placement cost 0.8% (p1103_g4l4b), so it stays off there.
            // The team and its size do not change a result (each row is computed once with the same kernel) and the hybrid fetch share
            // is set from the decode threads, so the text is the same (md5 equal in every launch).
            h.team_stop = (void (*)(void)) ggml_backend_reg_get_proc_address(creg, "ggml_cpu_moe_host_team_stop");
            auto team_on = (bool (*)(void)) ggml_backend_reg_get_proc_address(creg, "ggml_cpu_moe_host_team_on");
            const char * ew = getenv("LLAMA_MOE_SINGLE_THREADS_WIDE");
            const int wide_mode = ew != nullptr ? atoi(ew) : 1;
            if ((e == nullptr || atoi(e) <= 0) && wide_mode != 0 && (wide_mode == 2 || hybrid) && team_on != nullptr && team_on()) {
                const char * em = getenv("LLAMA_MOE_SINGLE_THREADS_MAX");
                const int wmax = em != nullptr && atoi(em) > 0 ? atoi(em) : 16;
                const int wide = single_team_size(wmax);
                if (wide > h.threads) {
                    h.threads = wide;
                    h.team    = true;
                    if (single_wait_blocking_here()) {
                        single_wait_blocking()(true);
                        h.blocking = true;
                    }
                }
            }
            if (set_n != nullptr) {
                set_n(h.cpu.get(), h.threads);
            }
            const char * ef = getenv("LLAMA_MOE_SINGLE_FUSED");
            if (ef == nullptr || atoi(ef) != 0) {
                h.fused       = (decltype(h.fused))       ggml_backend_reg_get_proc_address(creg, h.team ? "ggml_cpu_moe_host_ffn_team" : "ggml_cpu_moe_host_ffn");
                h.fused_wsize = (decltype(h.fused_wsize)) ggml_backend_reg_get_proc_address(creg, "ggml_cpu_moe_host_ffn_wsize");
                if (h.fused == nullptr || h.fused_wsize == nullptr) {
                    h.fused = nullptr;
                    h.fused_wsize = nullptr;
                }
            }
        }
        h.owners.push_back(this);
        single_reg = true;
        if (!h.thr.joinable()) {
            h.stop.store(false, std::memory_order_relaxed);
            h.thr = std::thread(single_service);
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: single-graph host misses on (LLAMA_MOE_SINGLE): batches of up to %d tokens, %d host threads "
            "(one host thread for every cache of the process%s), slow-batch warning above %d ms (merges never time out), %s\n", tag, single_max(), h.threads,
            h.team ? (h.blocking ? "; the CPU backend's host team, the host experts are the critical path: LLAMA_MOE_SINGLE_THREADS_WIDE=0 off; "
                "the device wait blocks (LLAMA_MOE_SINGLE_WAIT_BLOCKING=0 off)" :
                "; the CPU backend's host team, the host experts are the critical path: LLAMA_MOE_SINGLE_THREADS_WIDE=0 off") : "",
            single_timeout_ms, h.fused != nullptr ? "one-region host experts (LLAMA_MOE_SINGLE_FUSED=0: the graph)" : "host experts by graph");
        return true;
    }

    mailbox * single_mailbox(int32_t il, int n_used, int n_embd, ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down) {
        if (il < 0 || il >= (int32_t) layers.size()) {
            return nullptr;
        }
        if (!mbox_live) {
            mboxes.resize(layers.size());
            mbox_live.reset(new std::atomic<mailbox *>[layers.size()]);
            for (size_t i = 0; i < layers.size(); ++i) {
                mbox_live[i].store(nullptr, std::memory_order_relaxed);
            }
            mbox_n = layers.size();
        }
        if (mboxes[il]) {
            mailbox & mb = *mboxes[il];
            return mb.n_used == n_used && mb.n_embd == n_embd && mb.w_up == up && mb.w_gate == gate && mb.w_down == down ? &mb : nullptr;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend(il));
        ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto alloc = reg != nullptr ? (ggml_backend_moe_mailbox_alloc_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_mailbox_alloc") : nullptr;
        if (alloc == nullptr || (single_auto && strcmp(ggml_backend_reg_name(reg), "CUDA") != 0)) {
            if (!single_auto) {
                LLAMA_LOG_WARN("llama_moe_cache%s: LLAMA_MOE_SINGLE: the device backend has no mapped mailboxes, the split path stays\n", tag);
            }
            single_off = true;
            return nullptr;
        }
        auto mb = std::make_unique<mailbox>();
        const size_t size = ggml_moe_mailbox_size(n_used, n_embd);
        mb->host = alloc(dev, size, &mb->mdev, &mb->ctl);
        if (mb->host == nullptr) {
            LLAMA_LOG_WARN("llama_moe_cache%s: LLAMA_MOE_SINGLE: no mapped mailbox (%zu bytes), the split path stays\n", tag, size);
            single_off = true;
            return nullptr;
        }
        mb->dev    = dev;
        mb->n_used = n_used;
        mb->n_embd = n_embd;
        mb->w_up   = up;
        mb->w_gate = gate;
        mb->w_down = down;
        mb->buf.reset(ggml_backend_cpu_buffer_from_ptr(mb->host, size));
        // the plans of every batch size now, not in the first batch the device waits for
        for (int n = 1; n <= single_max_tok; ++n) {
            single_plan * p = nullptr;
            if (!single_plan_get(*mb, n, p)) {
                LLAMA_LOG_WARN("llama_moe_cache%s: LLAMA_MOE_SINGLE: no host plan for layer %d, the split path stays\n", tag, il);
                single_off = true;
                return nullptr;
            }
        }
        mboxes[il] = std::move(mb);
        mbox_live[il].store(mboxes[il].get(), std::memory_order_release);
        if (!single_register()) {
            single_off = true;
            return nullptr;
        }
        return mboxes[il].get();
    }

    // k19-overlap GGML_MOE_MAILBOX_TIMING=1 (diagnostic): per layer, the device work between the publish and the merge, the merge's wait
    // for the host (device clock, see moe_mailbox_merge) and the host time of the batches (host clock), totals and by bands of layers
    void single_timing_report() {
        const char * e = getenv("GGML_MOE_MAILBOX_TIMING");
        if (e == nullptr || atoi(e) == 0) {
            return;
        }
        double t[6] = { 0, 0, 0, 0, 0, 0 };
        int64_t hn = 0;
        int64_t hus = 0;
        double  pick = 0, pre = 0, post = 0;
        int64_t pick_n = 0;
        int64_t min_d1 = INT64_MAX, min_d2 = INT64_MAX;
        std::string bands;
        double b[5] = { 0, 0, 0, 0, 0 };
        int64_t bhn = 0, bhus = 0;
        int b0 = 0;
        for (size_t i = 0; i < mboxes.size(); ++i) {
            if (mboxes[i]) {
                const volatile uint64_t * tm = ((ggml_moe_mailbox_hdr *) mboxes[i]->host)->tm;
                for (int k = 0; k < 6; ++k) {
                    t[k] += k == 5 ? (double) (int64_t) tm[k] : (double) tm[k]; // [5]: a signed sum (two clocks)
                }
                for (int k = 0; k < 5; ++k) {
                    b[k] += (double) tm[k];
                }
                pick += (double) mboxes[i]->h_pick_ns; pick_n += mboxes[i]->h_pick_n;
                pre  += (double) mboxes[i]->h_pre_ns;  post += (double) mboxes[i]->h_post_ns;
                min_d1 = std::min(min_d1, mboxes[i]->h_pick_min);
                if (tm[0] > 0) {
                    min_d2 = std::min(min_d2, (int64_t) tm[8]);
                }
                hn  += mboxes[i]->h_n;  bhn  += mboxes[i]->h_n;
                hus += mboxes[i]->h_us; bhus += mboxes[i]->h_us;
            }
            if ((i + 1) % 8 == 0 || i + 1 == mboxes.size()) {
                if (b[0] > 0) {
                    bands += " L" + std::to_string(b0) + "-" + std::to_string(i) + " " + std::to_string((int) (b[1]/b[0]/1e3 + 0.5)) + "/" +
                        std::to_string((int) (b[2]/b[0]/1e3 + 0.5)) + "/" + std::to_string(bhn > 0 ? (int) ((double) bhus/bhn + 0.5) : 0);
                }
                for (double & x : b) {
                    x = 0;
                }
                bhn = bhus = 0;
                b0 = (int) i + 1;
            }
        }
        if (t[0] <= 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: GGML_MOE_MAILBOX_TIMING: no timed batch\n", tag);
            return;
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: GGML_MOE_MAILBOX_TIMING: %.0f batches: device beside the host %.1f us, merge wait %.1f us "
            "(%.1f%% waited over 2 us; %.1f ms in all), publish to merge end %.1f us, host %.1f us per batch (%" PRId64 " host batches); "
            "by layers device/wait/host us:%s\n", tag, t[0], t[1]/t[0]/1e3, t[2]/t[0]/1e3, 100.0*t[3]/t[0], t[2]/1e6, t[4]/t[0]/1e3,
            hn > 0 ? (double) hus/hn : 0.0, hn, bands.c_str());
        // the two clocks (host CLOCK_REALTIME, device globaltimer) are apart by an offset: taken as half the difference of the smallest
        // one-way samples each way (equal smallest latencies assumed)
        const double d1 = pick_n > 0 ? pick/pick_n : 0.0;
        const double d2 = t[5]/t[0];
        const double off = pick_n > 0 && min_d2 != INT64_MAX ? 0.5*((double) min_d1 - (double) min_d2) : 0.0;
        LLAMA_LOG_INFO("llama_moe_cache%s: GGML_MOE_MAILBOX_TIMING: publish to host pickup %.1f us, host completion to the end of the merge's "
            "wait %.1f us (clock offset %.1f us from the smallest samples %.1f / %.1f us); host pickup to compute %.2f us, compute to completed "
            "%.2f us\n", tag, (d1 - off)/1e3, (d2 + off)/1e3, off/1e3, min_d1/1e3, min_d2/1e3, pick_n > 0 ? pre/pick_n/1e3 : 0.0,
            pick_n > 0 ? post/pick_n/1e3 : 0.0);
    }

    // the plan of n_tokens tokens of a mailbox: the experts of the host ids, reading and writing the mailbox
    static bool single_plan_get(mailbox & mb, int n_tokens, single_plan *& out) {
        single_plan & p = mb.plans[n_tokens];
        if (p.gf == nullptr) {
            ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false), nullptr, true };
            p.ctx.reset(ggml_init(ip));
            ggml_context * c = p.ctx.get();
            ggml_tensor * x   = ggml_new_tensor_3d(c, GGML_TYPE_F32, mb.n_embd, 1, n_tokens);
            ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, mb.n_used, n_tokens);
            ggml_tensor * up   = ggml_mul_mat_id(c, mb.w_up,   x, ids);
            ggml_tensor * gate = ggml_mul_mat_id(c, mb.w_gate, x, ids);
            ggml_tensor * act  = ggml_swiglu_split(c, gate, up);
            ggml_tensor * dn   = ggml_mul_mat_id(c, mb.w_down, act, ids);
            if (dn->ne[0] != mb.n_embd || dn->ne[1] != mb.n_used || dn->ne[2] != n_tokens || !ggml_is_contiguous(dn)) {
                return false;
            }
            char * base = (char *) mb.host;
            for (auto [t, off] : { std::pair<ggml_tensor *, size_t>{ x, ggml_moe_mailbox_off_x(mb.n_used) },
                                   { ids, ggml_moe_mailbox_off_ids() }, { dn, ggml_moe_mailbox_off_out(mb.n_used, mb.n_embd) } }) {
                if (ggml_backend_tensor_alloc(mb.buf.get(), t, base + off) != GGML_STATUS_SUCCESS) {
                    return false;
                }
            }
            p.gf = ggml_new_graph_custom(c, 16, false);
            ggml_build_forward_expand(p.gf, dn);
            p.galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            if (p.galloc == nullptr || !ggml_gallocr_alloc_graph(p.galloc, p.gf)) {
                return false;
            }
        }
        out = &p;
        return true;
    }

    // one pass over the mailboxes of this cache, true if one had work
    bool single_poll(ggml_backend_t cpu) {
        // LLAMA_MOE_SINGLE_DEBUG_STALL=N (test of the timeout): the thread stops answering after N batches, as if it died
        static const int64_t stall_after = [] {
            const char * e = getenv("LLAMA_MOE_SINGLE_DEBUG_STALL");
            return e != nullptr ? atoll(e) : -1;
        }();
        bool work = false;
        for (size_t i = 0; i < mbox_n; ++i) {
            mailbox * mbp = mbox_live[i].load(std::memory_order_acquire);
            if (mbp == nullptr) {
                continue;
            }
            mailbox & mb = *mbp;
            auto * hdr = (ggml_moe_mailbox_hdr *) mb.host;
            const uint32_t req = llama_mb_load_acq(&hdr->request);
            if (req == mb.done || (stall_after >= 0 && (int64_t) single_batches >= stall_after)) {
                continue;
            }
            // LLAMA_MOE_SINGLE_DEBUG_DELAY_PHASE=1 (test): the delay before the pickup, as if the host thread were not scheduled
            single_debug_delay(false);
            work = true;
            bool ok = false;
            const int n_tokens = hdr->n_tokens;
            single_plan * p = nullptr;
            // k19-overlap GGML_MOE_MAILBOX_TIMING: the pickup against the publish's device stamp (both clocks in ns since the epoch)
            static const bool tm_on = [] { const char * e = getenv("GGML_MOE_MAILBOX_TIMING"); return e != nullptr && atoi(e) != 0; }();
            const int64_t t_pick = tm_on ? single_steady_ns() : 0;
            int64_t t_end = 0;
            if (tm_on) {
                const int64_t d = single_rt_ns() - (int64_t) ((volatile uint64_t *) hdr->tm)[6];
                mb.h_pick_ns += d;
                mb.h_pick_n++;
                mb.h_pick_min = std::min(mb.h_pick_min, d);
            }
            // picked up: the merge on the device waits for this batch without a timeout from now on (see moe_mailbox_merge)
            llama_mb_store_rel(&hdr->started, req);
            if (!single_failed.load(std::memory_order_relaxed) && n_tokens >= 1 && n_tokens <= GGML_MOE_MAILBOX_MAX_TOKENS &&
                    single_plan_get(mb, n_tokens, p)) {
                const int64_t t0 = single_now_us();
                if (tm_on) {
                    mb.h_pre_ns += single_steady_ns() - t_pick;
                }
                single_hub_busy(true);
                single_debug_delay(true);
                single_hub_t & hub = single_hub();
                if (hub.fused != nullptr) {
                    const size_t ws = hub.fused_wsize(mb.w_up, mb.w_gate, mb.w_down, mb.n_used, n_tokens, hub.threads);
                    if (ws > 0 && hub.fused_buf.size() < ws) {
                        hub.fused_buf.resize(ws);
                    }
                    char * hb = (char *) mb.host;
                    ok = ws > 0 && hub.fused(mb.w_up, mb.w_gate, mb.w_down, (const float *) (hb + ggml_moe_mailbox_off_x(mb.n_used)),
                        (const int32_t *) (hb + ggml_moe_mailbox_off_ids()), (float *) (hb + ggml_moe_mailbox_off_out(mb.n_used, mb.n_embd)),
                        mb.n_used, n_tokens, hub.threads, hub.fused_buf.data(), hub.fused_buf.size());
                    hub.fused_batches += ok;
                }
                if (!ok) {
                    ok = ggml_backend_graph_compute(cpu, p->gf) == GGML_STATUS_SUCCESS;
                }
                if (!ok) {
                    // the device waits for this batch: computed again, a second failure is fatal (never a wrong row)
                    ok = ggml_backend_graph_compute(cpu, p->gf) == GGML_STATUS_SUCCESS;
                    GGML_ASSERT(ok && "single-graph host misses: the host graph failed twice");
                }
                single_hub_busy(false);
                const int64_t dt = single_now_us() - t0;
                t_end = tm_on ? single_steady_ns() : 0;
                single_us += dt;
                mb.h_us += dt;
                mb.h_n++;
                single_slow += dt > 1000;
                single_max_us = std::max(single_max_us, dt);
                if (single_timeout_ms > 0 && dt > 1000*(int64_t) single_timeout_ms && single_warned++ < 8) {
                    LLAMA_LOG_WARN("llama_moe_cache%s: single-graph host misses: a host batch took %.1f ms (above %d ms, "
                        "LLAMA_MOE_SINGLE_TIMEOUT_MS); the device waited for it\n", tag, dt/1000.0, single_timeout_ms);
                }
                const int32_t * ids = (const int32_t *) ((const char *) mb.host + ggml_moe_mailbox_off_ids());
                for (int j = 0; j < mb.n_used*n_tokens; ++j) {
                    single_rows += ids[j] >= 0;
                }
            }
            if (!ok) {
                single_failed.store(true, std::memory_order_relaxed);
                llama_mb_store_rel(&hdr->error, 2u);
            }
            single_batches++;
            mb.done = req;
            if (tm_on) {
                if (t_end != 0) {
                    mb.h_post_ns += single_steady_ns() - t_end;
                }
                ((volatile uint64_t *) hdr->tm)[7] = (uint64_t) single_rt_ns();
            }
            llama_mb_store_rel(&hdr->completed, req);
        }
        return work;
    }

    // nonzero busy in every mailbox of the process while the host thread computes one batch: merges waiting on another mailbox
    // (an MTP context) do not time out behind it. Called by the host thread with reg_mtx held
    static void single_hub_busy(bool on) {
        // k18 (GGML_MOE_MAILBOX_FAST, default on): the merges of this process never time out (single_merge passes 0), and only a merge
        // with a timeout reads busy, so the stores (two per mailbox of every cache per host batch, on lines the device polls) are skipped
        static const bool fast = [] {
            const char * e = getenv("GGML_MOE_MAILBOX_FAST");
            return e == nullptr || atoi(e) != 0;
        }();
        if (fast) {
            return;
        }
        single_hub_t & h = single_hub();
        for (impl * o : h.owners) {
            for (size_t i = 0; i < o->mbox_n; ++i) {
                mailbox * mbp = o->mbox_live[i].load(std::memory_order_acquire);
                if (mbp != nullptr) {
                    llama_mb_store_rel(&((ggml_moe_mailbox_hdr *) mbp->host)->busy, on ? 1u : 0u);
                }
            }
        }
    }

    // test only: LLAMA_MOE_SINGLE_DEBUG_DELAY_MS=D sleeps D ms in every LLAMA_MOE_SINGLE_DEBUG_DELAY_EVERY-th (25) host batch, after
    // the pickup (a slow host batch, the default: the merge must keep waiting) or before it (LLAMA_MOE_SINGLE_DEBUG_DELAY_PHASE=1:
    // a host thread that is not scheduled; a delay above LLAMA_MOE_SINGLE_TIMEOUT_MS then times the merge out)
    void single_debug_delay(bool picked) {
        static const int delay_ms = [] { const char * e = getenv("LLAMA_MOE_SINGLE_DEBUG_DELAY_MS"); return e != nullptr ? atoi(e) : 0; }();
        static const int every    = [] { const char * e = getenv("LLAMA_MOE_SINGLE_DEBUG_DELAY_EVERY"); return e != nullptr ? std::max(1, atoi(e)) : 25; }();
        static const int phase    = [] { const char * e = getenv("LLAMA_MOE_SINGLE_DEBUG_DELAY_PHASE"); return e != nullptr ? atoi(e) : 0; }();
        if (delay_ms > 0 && (phase == 1) != picked && (single_batches + 1) % every == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }


    static void single_service() {
        single_hub_t & h = single_hub();
        int64_t last_work = 0;
        while (!h.stop.load(std::memory_order_acquire)) {
            bool work = false;
            {
                std::lock_guard<std::mutex> lock(h.reg_mtx);
                for (impl * o : h.owners) {
                    work = o->single_poll(h.cpu.get()) || work;
                }
            }
            const int64_t now = single_now_us();
            if (work) {
                last_work = now;
            } else if (now - std::max(last_work, h.kick_us.load(std::memory_order_relaxed)) > 20000) {
                // no decode graph started for 20 ms: sleep until the next one (single_kick), at most 1 ms at a time, since a graph
                // that publishes without a kick (layer-major, a draft context's first graph) would wait for this sleep (G4 nsys: the
                // first decode forward after an 8k prompt waited 100 ms in the merges with a 100 ms sleep, -8.6% prompt)
                std::unique_lock<std::mutex> lock(h.mtx);
                h.cv.wait_for(lock, std::chrono::milliseconds(1));
            } else {
                llama_mb_spin_pause();
            }
        }
    }

    // before the graph of a ubatch of n_tokens: wakes the host thread for the batches that may publish; a failed mailbox turns the path
    // off (the graphs are rebuilt)
    void single_kick(int64_t n_tokens) {
        if (!single_reg || single_off) {
            return;
        }
        bool failed = single_failed.load(std::memory_order_relaxed);
        for (auto & mb : mboxes) {
            failed = failed || (mb && llama_mb_load_acq(&((ggml_moe_mailbox_hdr *) mb->host)->error) != 0);
        }
        if (failed) {
            LLAMA_LOG_ERROR("llama_moe_cache%s: single-graph host misses failed (a merge waited %d ms or the host thread failed): "
                "the split path from now on\n", tag, single_timeout_ms);
            single_off = true;
            cm_gen++;
            return;
        }
        if (n_tokens < 1 || n_tokens > single_max()) {
            return;
        }
        // no publish while the CPU-miss rule has the host path off for this batch size: the host thread sleeps instead of polling
        // (G4 Flash-Next UD-IQ4_XS 32 GB, trials turned small batches off: 8k decode -3.5% against LLAMA_MOE_SINGLE=0 in 6 of 6 launches)
        if (cm && single_kick_gate && cm_share_of(n_tokens) >= 1.0f) {
            return;
        }
        single_hub_t & h = single_hub();
        const int64_t now  = single_now_us();
        const int64_t prev = h.kick_us.exchange(now, std::memory_order_relaxed);
        if (now - prev > 20000) {
            h.cv.notify_one();
        }
    }

    void single_stop_all() {
        // the context synchronized its backends before it frees the cache (they may be gone here)
        if (single_reg) {
            single_hub_t & h = single_hub();
            // held across the erase and the stop / join of the last owner (reg_mtx is released for the join: the thread takes it)
            std::lock_guard<std::mutex> life(h.life_mtx);
            bool last = false;
            {
                std::lock_guard<std::mutex> lock(h.reg_mtx);
                h.owners.erase(std::remove(h.owners.begin(), h.owners.end(), this), h.owners.end());
                last = h.owners.empty();
            }
            if (last && h.thr.joinable()) {
                h.stop.store(true, std::memory_order_release);
                h.cv.notify_one();
                h.thr.join();
                if (h.team_stop != nullptr) {
                    h.team_stop();
                }
                if (h.blocking && single_wait_blocking() != nullptr) {
                    single_wait_blocking()(false);
                }
                h.blocking = false;
                h.team = false;
                h.cpu.reset();
            }
            single_reg = false;
            if (single_batches > 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: single-graph host misses: %" PRIu64 " host batches, %" PRIu64 " expert rows, %.1f us per batch "
                    "(%.2f us per row), %" PRIu64 " above 1 ms, at most %.1f ms\n", tag, single_batches, single_rows, (double) single_us/single_batches,
                    single_rows > 0 ? (double) single_us/single_rows : 0.0, single_slow, single_max_us/1e3);
                LLAMA_LOG_INFO("llama_moe_cache%s: single-graph host misses: %" PRIu64 " of the process's host batches in one region "
                    "(LLAMA_MOE_SINGLE_FUSED)\n", tag, single_hub().fused_batches);
                single_timing_report();
            }
        }
        for (auto & mb : mboxes) {
            if (!mb) {
                continue;
            }
            for (auto & p : mb->plans) {
                if (p.galloc != nullptr) {
                    ggml_gallocr_free(p.galloc);
                    p.galloc = nullptr;
                }
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(mb->dev);
            auto fr = (ggml_backend_moe_mailbox_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_mailbox_free");
            mb->buf.reset();
            if (fr != nullptr) {
                fr(mb->dev, mb->host, mb->ctl);
            }
        }
        mboxes.clear();
    }

    ~impl() {
        // the weight store is destroyed first (llama_context): its destructor completes the reads ahead into the RAM slots, which are
        // still allocated then; wstore is not used here
        single_stop_all();
        profile_save();
        ap_save(true);
        if (ap_rec || ap_use) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-autoprof: %s; %zu ubatches / %llu routings counted, %zu profile experts read, target set %zu "
                "(rebuilt %zu times), %zu cold experts marked, offers: %llu target, %llu cold taken, %llu cold refused; page-cache fill %llu target / "
                "%llu cold\n", tag, ap_use ? "used" : "recorded only", aps.ubatches, (unsigned long long) aps.routings, aps.loaded, aps.targets,
                aps.rebuilds, aps.marked, (unsigned long long) aps.hot_taken, (unsigned long long) aps.cold_taken,
                (unsigned long long) aps.cold_refused, (unsigned long long) aps.pf_hot, (unsigned long long) aps.pf_cold);
        }
        if (resident_mode) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-mgpu2: resident prompt fill: %llu prompts, %llu layers, %llu experts wanted, %llu swapped in "
                "(%.2f MiB), %llu copies back chained into the promoted experts' RAM slots, %llu into fresh slots, %llu kept (RAM held them), "
                "%llu dropped, %llu read from the files, %.1f ms (LLAMA_MOE_RESIDENT_FILL=%d, MIN %d, MAX %lld, DEFER %d: %llu layer swaps in decode lookups, PASS %lld)\n", tag, (unsigned long long) rpfs.prompts,
                (unsigned long long) rpfs.layers, (unsigned long long) rpfs.wanted, (unsigned long long) rpfs.swapped, rpfs.bytes/1048576.0,
                (unsigned long long) rpfs.chained, (unsigned long long) rpfs.fresh, (unsigned long long) rpfs.kept, (unsigned long long) rpfs.dropped,
                (unsigned long long) rpfs.from_file, rpfs.us/1000.0, rpf_on ? 1 : 0, rpf_min, (long long) rpf_max, rpf_defer ? 1 : 0,
                (unsigned long long) rpf_deferred_layers, (long long) rpf_pass);
        }
        if (rps.used || rp_rec) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-mgpu2: resident device prefill %s (%zu profile entries read, %zu taken, %zu by id); decode routing "
                "%s: %llu routed experts in %llu host lookups\n", tag, rps.used ? "from the profile" : "by id", rps.read, rps.taken, rps.padded,
                rp_rec ? "recorded" : "not recorded", (unsigned long long) rps.counted, (unsigned long long) rps.lookups);
        }
        ua_shutdown();
        log_stats();
        if (core_mode != 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-ptres core: %s, %.1f MiB of experts, %.1f MiB of host pages dropped, %llu core slot checks, %llu lost, "
                "%llu bank releases refused\n", tag, core_on ? "on" : "off", core_bytes/1048576.0, core_dropped/1048576.0,
                (unsigned long long) core_checks, (unsigned long long) core_bad, (unsigned long long) core_refused);
        }
        ua_free();
        if (n_pad_zero > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4ttft: %llu padding pieces after uploaded runs written as zeros instead of read from the files "
                "(LLAMA_MOE_PROMPT_PAD_ZERO=0 off)\n", tag, (unsigned long long) n_pad_zero);
        }
        if (plr.layers > 0 || plr.over > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4ttft: prompt layer reads: %llu layers, %llu experts, %.1f MiB in %.1f ms, %llu past the scratch "
                "(LLAMA_MOE_PROMPT_LAYER_READ=1 opt-in)\n", tag, (unsigned long long) plr.layers, (unsigned long long) plr.experts, plr.bytes/1048576.0,
                plr.read_us/1000.0, (unsigned long long) plr.over);
        }
        plr_free();
        for (auto & d : d2h_devs) {
            if (d.be) {
                ggml_backend_synchronize(d.be);
            }
            for (auto * e : d.ev) {
                ggml_backend_event_free(e);
            }
            if (d.start) {
                ggml_backend_event_free(d.start);
            }
            if (d.be) {
                ggml_backend_free(d.be);
            }
        }
        d2h_devs.clear();
        for (const auto & [unmap, buf] : mapped) {
            unmap(buf);
        }
        llama_sysmem::pinned_file_add(-file_pinned);
    }

    // k19-core2: file-backed (model file) bytes this cache registered, published to llama_sysmem::pinned_file_bytes
    int64_t file_pinned = 0;
    void file_pin_add(size_t bytes) {
        file_pinned += (int64_t) bytes;
        llama_sysmem::pinned_file_add((int64_t) bytes);
    }

    using host_map_t   = void * (*)(ggml_backend_dev_t, void *, size_t, bool *);
    using host_unmap_t = void (*)(void *);
    using host_map_range_t    = void * (*)(ggml_backend_dev_t, void *, size_t, void **, bool *);
    using host_mapped_bytes_t = size_t (*)(void);

    // pinned expert ranges [LLAMA_MOE_PIN_RANGES]: where the whole file buffers cannot be mapped (more than the available memory could pin,
    // or the registration fails), the pages of the expert tensors alone are registered read-only, one range per layer (its expert tensors
    // and the small gaps between them), layer by layer while the budget lasts: min(LLAMA_MOE_PIN_MAX_FRAC (0.6) x MemTotal - host memory
    // pinned already (every mapping of the backend, both contexts, and the slots of the RAM tier), available - margin), or
    // LLAMA_MOE_PIN_MAX_MIB. A group whose layers are all pinned gets the device lookups; with only some layers pinned its misses and the
    // copies of prompts still read those layers at pinned speed. Each range holds whole tensors, so a copy of experts lies in one mapping.
    // LLAMA_MOE_PIN_RANGES: unset/1 = as the fallback, 2 = instead of the whole buffers (diagnostic), 0 = off (k9: host lookups, pageable)
    int    pin_mode = 1;
    size_t pin_bytes = 0;      // registered by the ranges of this cache
    size_t pin_layers = 0;     // layers pinned by ranges
    size_t pin_layers_want = 0;

    // pins the expert ranges of the layers of group g in order; returns true if every layer is pinned (host_dev set for all)
    bool pin_ranges(group & g, ggml_backend_dev_t dev, ggml_backend_reg_t reg, std::vector<std::vector<const void *>> & host_dev) {
        auto map_range = (host_map_range_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_map_range");
        auto mapped_b  = (host_mapped_bytes_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_bytes");
        auto unmap     = (host_unmap_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_unmap");
        host_dev.assign(g.layers.size(), {});
        if (map_range == nullptr || unmap == nullptr) {
            return false;
        }
        const uint64_t total  = llama_sysmem::total();
        const uint64_t avail  = llama_sysmem::available();
        double frac = 0.6;
        if (const char * env = getenv("LLAMA_MOE_PIN_MAX_FRAC")) {
            frac = std::max(0.0, std::min(0.95, atof(env)));
        }
        const uint64_t pinned = (mapped_b != nullptr ? mapped_b() : 0) + (host ? host->pinned_bytes() : 0);
        const uint64_t margin = UINT64_C(2) << 30;
        int64_t budget = INT64_MAX;
        if (const char * env = getenv("LLAMA_MOE_PIN_MAX_MIB")) {
            budget = std::max<int64_t>(0, atoll(env))*1048576 - (int64_t) pin_bytes;
        } else {
            if (total > 0) {
                budget = std::min<int64_t>(budget, (int64_t) (frac*(double) total) - (int64_t) pinned);
            }
            if (avail > 0) {
                budget = std::min<int64_t>(budget, (int64_t) avail - (int64_t) margin);
            }
        }
        const uintptr_t page = 4096;
        size_t n_ok = 0;
        size_t bytes = 0;
        pin_layers_want += g.layers.size();
        auto host_map = (host_map_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_map");
        for (size_t i = 0; i < g.layers.size(); ++i) {
            const layer & l = layers[g.layers[i]];
            // the ranges of the layer: runs of its mapped expert tensors with small gaps between them (one mapping per run: a copy of
            // experts never spans two), tensors in other host buffers (pinned already, e.g. LLAMA_MOE_PIN_EXPERTS) are mapped whole
            // as before, without a new pin
            bool fail = false;
            std::vector<const void *> devp(l.experts.size(), nullptr);
            // runs of one buffer only: the gaps between its tensors are mapped too (separate mappings may have holes between them)
            std::vector<std::tuple<const void *, uintptr_t, uintptr_t>> runs;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * t = l.experts[ip];
                if (strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)), "CPU_Mapped") == 0) {
                    runs.emplace_back((const void *) t->buffer, (uintptr_t) t->data, (uintptr_t) t->data + ggml_nbytes(t));
                    devp[ip] = t->data; // the device reads registered memory at its host address (checked by map_range)
                    continue;
                }
                bool reg_whole = false;
                void * base = ggml_backend_buffer_get_base(t->buffer);
                uint8_t * dp = host_map != nullptr ? (uint8_t *) host_map(dev, base, ggml_backend_buffer_get_size(t->buffer), &reg_whole) : nullptr;
                if (reg_whole) {
                    mapped.emplace_back(unmap, base);
                }
                if (dp == nullptr) {
                    fail = true;
                    break;
                }
                devp[ip] = dp + ((const uint8_t *) t->data - (const uint8_t *) base);
            }
            if (fail) {
                break;
            }
            std::sort(runs.begin(), runs.end());
            std::vector<std::pair<uintptr_t, uintptr_t>> merged;
            const void * buf_prev = nullptr;
            for (const auto & [buf, lo, hi] : runs) {
                if (!merged.empty() && buf == buf_prev && lo <= merged.back().second + (UINT64_C(16) << 20)) {
                    merged.back().second = std::max(merged.back().second, hi);
                } else {
                    merged.emplace_back(lo, hi);
                }
                buf_prev = buf;
            }
            size_t need = 0;
            for (const auto & r : merged) {
                need += ((r.second + page - 1) & ~(page - 1)) - (r.first & ~(page - 1));
            }
            if ((int64_t) (bytes + need) > budget) {
                break;
            }
            for (const auto & r : merged) {
                void * key = nullptr;
                bool registered = false;
                if (map_range(dev, (void *) r.first, r.second - r.first, &key, &registered) == nullptr) {
                    fail = true;
                    break;
                }
                if (registered) {
                    mapped.emplace_back(unmap, key);
                }
            }
            if (fail) {
                break;
            }
            bytes += need;
            host_dev[i] = devp;
            n_ok++;
        }
        pin_bytes  += bytes;
        pin_layers += n_ok;
        LLAMA_LOG_INFO("%s: MoE cache: pinned the expert ranges of %zu of %zu layers (%.0f MiB; budget %.0f MiB = min(%.2f x %.0f MiB - %.0f MiB pinned, "
            "%.0f MiB available - %.0f MiB)): %s (LLAMA_MOE_PIN_RANGES=0 off, LLAMA_MOE_PIN_MAX_MIB)\n", __func__, n_ok, g.layers.size(), bytes/1048576.0,
            budget == INT64_MAX ? -1.0 : budget/1048576.0, frac, total/1048576.0, pinned/1048576.0, avail/1048576.0, margin/1048576.0,
            n_ok == g.layers.size() ? "device lookups" : "host lookups, the pinned layers are copied at pinned speed");
        return n_ok == g.layers.size();
    }

    //
    // RAM tier
    //

    static size_t meminfo_avail() {
        return llama_weight_store::ram_available();
    }

    // WSL 2 (the GPU through WDDM) or Windows
    static bool host_is_wddm() {
#if defined(_WIN32)
        return true;
#else
        FILE * f = fopen("/proc/sys/kernel/osrelease", "r");
        if (f == nullptr) {
            return false;
        }
        char buf[256] = {};
        const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        buf[n] = 0;
        for (char * c = buf; *c; ++c) {
            *c = (char) tolower((unsigned char) *c);
        }
        return strstr(buf, "microsoft") != nullptr || strstr(buf, "wsl") != nullptr;
#endif
    }

    // the groups whose experts the weight store streams get a RAM tier when it is on and the budget holds at least one bundle
    void init_host_cache(int64_t host_mib) {
        host_group.assign(groups.size(), false);
        const char * env = getenv("LLAMA_MOE_HOST_CACHE");
        if (env != nullptr && atoi(env) == 0) {
            if (wstore != nullptr) {
                LLAMA_LOG_INFO("%s: MoE host cache off (LLAMA_MOE_HOST_CACHE=0)\n", __func__);
            }
            return;
        }
        if (host_mib == 0 || wstore == nullptr) {
            return;
        }
        size_t e_bytes = 0; // host experts of the streamed groups
        size_t v_bytes = 0; // their device slots
        size_t token_bytes = 0; // what one token routes over them
        ggml_backend_buffer_type_t buft = nullptr;
        std::vector<std::vector<size_t>> sizes(layers.size());
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            bool covered = true;
            for (int32_t il : g.layers) {
                for (const ggml_tensor * t : layers[il].experts) {
                    covered = covered && wstore->covers(t);
                }
            }
            if (!covered) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(devices[g.id].backend);
            ggml_backend_buffer_type_t b = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (b == nullptr) {
                continue;
            }
            buft = buft ? buft : b;
            host_group[ig] = true;
            e_bytes += g.host_bytes;
            v_bytes += (size_t) g.n_slots*g.expert_bytes;
            token_bytes += g.layers.size()*(size_t) n_expert_used*g.expert_bytes;
            for (int32_t il : g.layers) {
                for (const ggml_tensor * t : layers[il].experts) {
                    sizes[il].push_back(t->nb[2]);
                }
            }
        }
        if (buft == nullptr) {
            return;
        }
        const size_t avail = meminfo_avail();
        hst.avail_init = avail;
        const size_t mapped = std::max(e_bytes, wstore->bytes_mapped());
        int policy_env = -1;
        if (const char * env_p = getenv("LLAMA_MOE_HOST_CACHE_POLICY"); env_p != nullptr) {
            policy_env = strcmp(env_p, "inclusive") == 0 ? LLAMA_MOE_HOST_INCLUSIVE : LLAMA_MOE_HOST_EXCLUSIVE;
        }
        const bool force = env != nullptr && atoi(env) > 0;
        // k19-disktier: below the device minimum on a disk tier the tier goes on inclusive (see LLAMA_MOE_HOST_MIN_TOKENS);
        // LLAMA_MOE_HOST_DISK_INCLUSIVE=0: off there, as before
        const char * env_di = getenv("LLAMA_MOE_HOST_DISK_INCLUSIVE");
        const bool disk_incl = env_di == nullptr || atoi(env_di) != 0;
        double disk_frac = LLAMA_MOE_HOST_DISK_MIN_FRAC; // LLAMA_MOE_HOST_DISK_MIN_FRAC: the budget's minimum share of the streamed experts
        if (const char * e = getenv("LLAMA_MOE_HOST_DISK_MIN_FRAC")) {
            disk_frac = std::max(0.0, atof(e));
        }
        // k18-lowram: the tier never plans into a host reserve kept for the rest of the machine (a Studio user's desktop, browser and the
        // backend live there): LLAMA_MOE_HOST_RESERVE_MIB (default 6144, 0 = the auto budget as before). Only this path reaches it: the tier
        // exists where the weight store streams the experts, so every host that pins them loads as before. G4 Flash-Next UD-IQ4_XS 32 GB
        // at 48 GiB RAM without it: 39.7 GB of slots and MemAvailable down to 2.5 GiB during the run
        size_t reserve = (size_t) 6144 << 20;
        if (const char * e = getenv("LLAMA_MOE_HOST_RESERVE_MIB")) {
            reserve = (size_t) std::max<long long>(0, atoll(e)) << 20;
        }
        const size_t avail_plan = avail > reserve ? avail - reserve : 0;
        llama_moe_host_plan plan = llama_moe_host_decide(e_bytes, v_bytes, mapped, avail_plan, token_bytes, host_mib, policy_env, force, disk_incl, disk_frac);
        // where the whole reserve would cost the union of RAM and the device slots (the experts the device does not hold no longer fit),
        // but the budget without it holds them: the budget is that complement + 1 GiB, so resident mode stays and the reserve gets the
        // rest. G4 Flash-Next 32 GB at 48 GiB: the full 6 GiB reserve left 35.4 GB for 35.4 GB of complement, resident mode was lost and
        // the 8k / 16k cells read 12.0 / 18.3 GB from disk (TTFT 85 / 147 s, decode 25 / 34 tok/s against 63 / 96 s, 55 / 59 tok/s)
        // LLAMA_MOE_HOST_RESERVE_STRICT=1: the whole reserve even where it costs resident mode (the misses then come from the files:
        // with --weight-stream-io buffered from the page cache, which the host can reclaim)
        const bool reserve_strict = getenv("LLAMA_MOE_HOST_RESERVE_STRICT") != nullptr && atoi(getenv("LLAMA_MOE_HOST_RESERVE_STRICT")) != 0;
        if (reserve > 0 && host_mib < 0 && e_bytes > v_bytes && !reserve_strict) {
            const size_t need = e_bytes - v_bytes + ((size_t) 1 << 30);
            const llama_moe_host_plan full = llama_moe_host_decide(e_bytes, v_bytes, mapped, avail, token_bytes, host_mib, policy_env, force, disk_incl, disk_frac);
            if (full.on && full.budget >= need && (!plan.on || plan.budget < need)) {
                plan = full;
                plan.budget = need;
                plan.spare = need + v_bytes >= e_bytes;
                LLAMA_LOG_INFO("%s: MoE host cache: the %.1f MiB host reserve would leave the experts the device slots do not hold outside RAM: "
                        "the budget is that complement + 1 GiB (%.1f MiB), the reserve gets the rest\n", __func__, reserve/1048576.0, need/1048576.0);
            }
        }
        // k19-l4df2: on a disk tier (the mapped host weights exceed what is available, so the page cache cannot hold the streamed experts
        // the tier leaves out) the auto budget's planning room (max(4 GiB, 15%) + min(4 GiB, 1/16) of the available RAM, ~9.7 GB of a
        // 52 GB host) mostly buys page cache that cannot serve those experts: each device miss outside RAM is read from the disk. There
        // the exclusive tier plans into the available RAM minus the host reserve, the other mapped host weights and a fixed room
        // LLAMA_MOE_HOST_LEAN_ROOM_MIB (default 5632), up to the complement of the device slots + 1 GiB, filled on demand as before
        // (resident mode only where the budget reaches it under its own rule). L4 full card, Flash-Next UD-IQ4_XS from the disk: auto
        // 35.9 GB = 86% of the experts in RAM + device slots, 12% of device misses from the disk (~85% of a decode pass), 13.9 GB of
        // MemAvailable unused; the whole complement (44.9 GB, resident) froze the 54 GB VM at load, so the room stays.
        // LLAMA_MOE_HOST_LEAN=0 off
        {
            const char * e_lean = getenv("LLAMA_MOE_HOST_LEAN");
            size_t room_lean = (size_t) 5632 << 20;
            if (const char * e = getenv("LLAMA_MOE_HOST_LEAN_ROOM_MIB")) {
                room_lean = (size_t) std::max<long long>(0, atoll(e)) << 20;
            }
            const size_t comp  = e_bytes > v_bytes ? e_bytes - v_bytes : 0;
            const size_t other = mapped > e_bytes ? mapped - e_bytes : 0;
            const size_t keep  = other + room_lean;
            const size_t lean  = std::min(comp + ((size_t) 1 << 30), avail_plan > keep ? avail_plan - keep : 0);
            if ((e_lean == nullptr || atoi(e_lean) != 0) && host_mib < 0 && comp > 0 && mapped > avail && plan.on &&
                    plan.policy == LLAMA_MOE_HOST_EXCLUSIVE && plan.budget < lean) {
                LLAMA_LOG_INFO("%s: MoE host cache: disk tier: the budget plans into %.1f MiB available minus the %.1f MiB reserve, %.1f MiB "
                        "of other mapped weights and a %.1f MiB room: %.1f -> %.1f MiB (%.1f MiB of streamed experts outside RAM and the device "
                        "slots -> %.1f) (LLAMA_MOE_HOST_LEAN=0 off, LLAMA_MOE_HOST_LEAN_ROOM_MIB)\n", __func__, avail/1048576.0,
                        reserve/1048576.0, other/1048576.0, room_lean/1048576.0, plan.budget/1048576.0, lean/1048576.0,
                        (comp > plan.budget ? comp - plan.budget : 0)/1048576.0, (comp > lean ? comp - lean : 0)/1048576.0);
                plan.budget = lean;
                plan.spare  = lean + v_bytes >= e_bytes;
            }
        }
        if (reserve > 0 && host_mib < 0) {
            LLAMA_LOG_INFO("%s: MoE host cache: %.1f MiB available, %.1f MiB kept as the host reserve (LLAMA_MOE_HOST_RESERVE_MIB)\n", __func__,
                    avail/1048576.0, reserve/1048576.0);
        }
        if (!plan.on) {
            LLAMA_LOG_INFO("%s: MoE host cache off: %s (%.1f MiB of streamed experts, %.1f MiB in device slots = the experts of %.1f tokens, "
                    "minimum %d, budget %.1f MiB, MemAvailable %.1f MiB; LLAMA_MOE_HOST_CACHE=1 forces it)\n", __func__, plan.why,
                    e_bytes/1048576.0, v_bytes/1048576.0, plan.tokens, LLAMA_MOE_HOST_MIN_TOKENS, plan.budget/1048576.0, avail/1048576.0);
            host_group.assign(groups.size(), false);
            return;
        }
        const size_t budget = plan.budget;
        const int    policy = plan.policy;
        host_spare = policy == LLAMA_MOE_HOST_EXCLUSIVE && plan.spare;
        if (const char * e = getenv("LLAMA_MOE_HOST_CACHE_SPARE"); e != nullptr) {
            host_spare = policy == LLAMA_MOE_HOST_EXCLUSIVE && atoi(e) != 0;
        }
        int32_t n_expert = 0;
        for (const group & g : groups) {
            n_expert = std::max<int32_t>(n_expert, (int32_t) g.ref[0]->ne[2]);
        }
        // pin guard: pinned slots up to a cap (WSL / Windows: WDDM limits page-locked memory, 1 GiB by default), plain host memory past it
        // or once pinning fails (LLAMA_MOE_HOST_CACHE_PIN_MIB caps it, LLAMA_MOE_HOST_CACHE_PAGEABLE=0: the tier stops growing instead)
        size_t pin_cap = SIZE_MAX;
        const bool wddm = host_is_wddm();
        if (wddm) {
            pin_cap = (size_t) 1024 << 20;
        }
        if (const char * e = getenv("LLAMA_MOE_HOST_CACHE_PIN_MIB"); e != nullptr && atoll(e) >= 0) {
            pin_cap = (size_t) atoll(e) << 20;
        }
        ggml_backend_buffer_type_t buft_pageable = ggml_backend_cpu_buffer_type();
        if (const char * e = getenv("LLAMA_MOE_HOST_CACHE_PAGEABLE"); e != nullptr && atoi(e) == 0) {
            buft_pageable = nullptr;
        }
        host = std::make_unique<llama_moe_host_cache>(buft, budget, policy, n_expert, sizes, 0, buft_pageable, pin_cap);
        if (pin_cap != SIZE_MAX) {
            LLAMA_LOG_INFO("%s: MoE host cache: at most %.1f MiB pinned%s, %s past it\n", __func__, pin_cap/1048576.0, wddm ? " (WSL / WDDM)" : "",
                    buft_pageable ? "pageable" : "no more slots");
        }
        if (host->n_slots() == 0) {
            LLAMA_LOG_INFO("%s: MoE host cache off: %.1f MiB budget holds no expert (%.1f MiB available)\n", __func__, budget/1048576.0, avail/1048576.0);
            host.reset();
            host_group.assign(groups.size(), false);
            return;
        }
        if (const char * e = getenv("LLAMA_WEIGHT_STREAM_VERIFY")) {
            host_verify = atoi(e) != 0;
        }
        tier.m = this;
        wstore->set_tier(&tier);
        // k19-l4held: per-layer quota of the RAM tier (see llama_moe_host_cache::set_layer_quota): LLAMA_MOE_HOST_LAYER_QUOTA=1 =
        // the exclusive tier, 2 = also the inclusive one, 0 = off (also leaves the device prefill's RAM copies unmarked, see prefill)
        {
            const char * eq = getenv("LLAMA_MOE_HOST_LAYER_QUOTA");
            // default off (k19-l4held job l4h_x1: the quota left ~2.6k RAM slots empty and the chat's misses follow the share of experts held
            // nowhere, not how they spread over the layers: 1122-1190 misses vs 765-792 without it, chat TTFT 12.1 vs 7.0 s)
            const int    mq = eq != nullptr ? atoi(eq) : 0;
            host_quota_mode = mq;
            host->set_layer_quota(mq >= 2 || (mq == 1 && host->exclusive()));
            LLAMA_LOG_INFO("%s: k19-l4held: RAM tier layer quota %s (LLAMA_MOE_HOST_LAYER_QUOTA=%d; 0 off, 1 exclusive tier, 2 both)\n", __func__,
                host->layer_quota_on() ? "on" : "off", mq);
        }
        // resident mode (the layout of Strata's --resident-experts): when the RAM tier and the device slots can hold every streamed
        // expert (RAM slots + device slots), the device slots are prefilled at load (unless
        // LLAMA_MOE_CACHE_PREFILL=0) and the RAM tier is filled from the model files with every expert the device does not hold, so no
        // prompt or decode step reads the files afterwards. Low-RAM G4 (32 GiB RAM, 24 GB VRAM, Flash-Next IQ2_XS 68 GB, 2k prompt / decode):
        // 55.7 / 43.1 tok/s filled on demand vs 164.1 / 62.2 resident (Strata resident 176.7 / 61.9), +45 s of load.
        // Otherwise unchanged (the page-cache prefill below). LLAMA_MOE_RESIDENT=0 off, 1 forces it (fills what fits);
        // an explicit LLAMA_MOE_HOST_CACHE_PREFILL also leaves it off (0, 1 as before, 2 = the file fill without the device prefill rule)
        {
            size_t n_streamed = 0;
            size_t n_dev      = 0;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                if (host_group[ig]) {
                    n_streamed += groups[ig].layers.size()*(size_t) groups[ig].lru.n_expert;
                    n_dev      += (size_t) groups[ig].n_slots;
                }
            }
            // the file fill skips the experts the device holds under both policies, so the device slots count for both
            const size_t n_hold = (size_t) host->n_slots() + n_dev;
            const char * e_res = getenv("LLAMA_MOE_RESIDENT");
            const char * e_dp  = getenv("LLAMA_MOE_CACHE_PREFILL");
            const int    m_res = e_res != nullptr ? atoi(e_res) : -1;
            // "every expert" up to the slot rounding: the budget is split among the expert layouts in whole slots of 4 KiB-aligned areas,
            // which loses ~1% of the slots a byte count gives (low-RAM G4: 18.7 GB of device slots + 15.1 GB of RAM budget for 33.8 GB of
            // experts = 99.2% of the experts in slots; the few left are read on demand, 1.5 MiB in a whole run), so at least
            // LLAMA_MOE_RESIDENT_MIN_FRAC (0.98) of them
            double min_frac = 0.98;
            if (const char * e = getenv("LLAMA_MOE_RESIDENT_MIN_FRAC")) {
                min_frac = std::max(0.0, atof(e));
            }
            resident_mode = m_res != 0 && getenv("LLAMA_MOE_HOST_CACHE_PREFILL") == nullptr && n_streamed > 0 &&
                (m_res >= 1 || (double) n_hold >= min_frac*(double) n_streamed);
            if (resident_mode) {
                if (!prefilled_dev && (e_dp == nullptr || atoi(e_dp) != 0)) {
                    if (!rp_prefill()) { // k19-mgpu2: the hottest experts of the earlier launches' decodes (LLAMA_MOE_RESIDENT_PROF=1)
                        prefill();
                    }
                }
                prefill_deferred = false;
                LLAMA_LOG_INFO("%s: MoE cache resident mode: %zu RAM slots + %zu device slots for the %zu streamed experts: device slots %s, "
                        "the RAM tier is filled from the model files at load (LLAMA_MOE_RESIDENT=0 off)\n", __func__,
                        (size_t) host->n_slots(), n_dev, n_streamed, prefilled_dev ? "prefilled" : "not prefilled (LLAMA_MOE_CACHE_PREFILL=0)");
            } else if (m_res != 0 && getenv("LLAMA_MOE_HOST_CACHE_PREFILL") == nullptr) {
                LLAMA_LOG_INFO("%s: MoE cache resident mode off: %zu RAM slots + %zu device slots hold %.1f%% of the %zu streamed experts, "
                        "below %.0f%% (LLAMA_MOE_RESIDENT=1 forces it)\n", __func__, (size_t) host->n_slots(), n_dev,
                        100.0*n_hold/n_streamed, n_streamed, 100.0*min_frac);
            }
        }
        ap_init();
        rp_rec = resident_mode && rp_mode() != 0 && ap_rec && !ap_cnt.empty(); // k19-mgpu2: the decode routing goes into the profile
        prefill_resident();
        LLAMA_LOG_INFO("%s: MoE host cache on: %s, budget %.1f MiB (%s) for %.1f MiB of streamed experts (of %.1f MiB mapped), %.1f MiB in device slots, "
                "the experts of %.1f tokens, MemAvailable %.1f MiB, pinned on first use (%s)%s (LLAMA_MOE_HOST_CACHE=0 off)\n", __func__,
                policy == LLAMA_MOE_HOST_EXCLUSIVE ? (host_spare ? "exclusive (spare room: duplicates kept)" : "exclusive") : "inclusive", budget/1048576.0, plan.why,
                e_bytes/1048576.0, mapped/1048576.0, v_bytes/1048576.0, plan.tokens, avail/1048576.0,
                host->describe().c_str(), host_verify ? ", VERIFY" : "");
    }

    // k19-ramprof: the routing profile host_profile_file (LLAMA_MOE_HOST_PROFILE) fills the RAM tier in its order, most used first over every
    // layer (llama_moe_profile_ram_order). k19-ramprof2: read here (load), filled from the first batch on (see hprof_order). Returns the
    // entries (0: no profile, the page-cache fill runs as before)
    size_t prefill_host_profile() {
        std::vector<prof_entry> prof;
        if (!llama_moe_profile_load(host_profile_file, profile_n_expert(), prof)) {
            LLAMA_LOG_WARN("%s: MoE host cache: LLAMA_MOE_HOST_PROFILE: %s is missing or does not match this model's cache, the RAM tier is "
                    "filled from the page cache only\n", __func__, host_profile_file.c_str());
            return 0;
        }
        if (const char * e = getenv("LLAMA_MOE_HOST_PROFILE_READ_MIB")) {
            hprof_read_cap = (size_t) std::max(0.0, atof(e))*1048576;
        }
        hprof_read_left = hprof_read_cap;
        // inclusive tier: no reads during the load by default (L4 12 GB: 6 GiB of them made the load 10% longer, the copies capture more there)
        hprof_room = host->exclusive() ? 0.0 : 0.15;
        if (!host->exclusive()) {
            hprof_load_read = 0;
        }
        if (const char * e = getenv("LLAMA_MOE_HOST_PROFILE_LOAD_READ_MIB")) {
            hprof_load_read = (size_t) std::max(0.0, atof(e))*1048576;
        }
        if (const char * e = getenv("LLAMA_MOE_HOST_PROFILE_ROOM")) {
            hprof_room = std::min(1.0, std::max(0.0, atof(e)));
        }
        if (const char * e = getenv("LLAMA_MOE_HOST_PROFILE_INFLIGHT")) {
            hprof_inflight_max = std::max(1, atoi(e));
        }
        if (const char * e = getenv("LLAMA_MOE_HOST_PROFILE_PROMPT_FREE")) {
            hprof_prompt_free = std::min(1.0, std::max(0.0, atof(e)));
        }
        hprof_order = llama_moe_profile_ram_order(prof);
        hprof_ticket.assign(host->n_slots(), 0);
        LLAMA_LOG_INFO("%s: MoE host cache: profile %s: %zu experts, filled from the first batch on: the ones in the page cache copied, the others "
                "read in the background after the first prompt (LLAMA_MOE_HOST_PROFILE unset or 0 off, LLAMA_MOE_HOST_PROFILE_READ_MIB caps the reads, "
                "LLAMA_MOE_HOST_PROFILE_PROMPT_FREE = %.2f of the slots kept free for a request's reads)\n", __func__, host_profile_file.c_str(),
                prof.size(), hprof_prompt_free);
        return prof.size();
    }

    // k19-ramprof2: the profile experts in the page cache are copied into free slots in heat order (an exclusive tier skips the experts the
    // device slots hold); the others go to hprof_pending. Residency is checked per expert right before its copy (an answer taken earlier
    // goes stale while the slots being pinned evict file pages: the copies then faulted their experts back from the disk, 30 GB in
    // 107-124 s), and the pages of a copied expert are marked cold (MADV_COLD), so the kernel reclaims them, not the ones still to copy.
    // Placement only: a slot holds the bytes of the mapping, as a slot the first prompt fills.
    void hprof_copy() {
        hprof_copied = true;
        const std::vector<prof_entry> order = std::move(hprof_order);
        hprof_order.clear();
        hprof_pending.clear();
#if defined(__linux__) || defined(__APPLE__)
        const int64_t t0 = ggml_time_us();
        const size_t pg = (size_t) std::max(4096L, sysconf(_SC_PAGESIZE));
        std::vector<unsigned char> vec;
        auto resident = [&](int32_t il, int32_t e) -> bool {
            for (const ggml_tensor * w : layers[il].experts) {
                const uintptr_t a  = (uintptr_t) w->data + (size_t) e*w->nb[2];
                const uintptr_t p0 = a / pg * pg;
                const uintptr_t p1 = (a + w->nb[2] + pg - 1) / pg * pg;
                vec.resize((p1 - p0) / pg);
#if defined(__APPLE__)
                if (mincore((void *) p0, p1 - p0, (char *) vec.data()) != 0) {
#else
                if (mincore((void *) p0, p1 - p0, vec.data()) != 0) {
#endif
                    return false;
                }
                for (unsigned char v : vec) {
                    if ((v & 1) == 0) {
                        return false;
                    }
                }
            }
            return true;
        };
        size_t b_pc = 0;
        auto want = [&](int32_t il, int32_t e) -> bool {
            if (!host_layer(il) || (host->exclusive() && in_device(il, e)) || !hprof_has_room(il)) {
                return false;
            }
            if (resident(il, e)) {
                return true;
            }
            hprof_pending.push_back({ il, e, 0.0f });
            hps.load_skipped_uncached++;
            return false;
        };
        auto take = [&](size_t i, int32_t s) {
            const int32_t il = order[i].il;
            const int32_t e  = order[i].e;
            const layer & l = layers[il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const uint8_t * src = (const uint8_t *) w->data + (size_t) e*w->nb[2];
                host->set_lead(s, (int) ip, 0);
                memcpy(host->area(s, (int) ip), src, w->nb[2]);
                b_pc += w->nb[2];
#if defined(MADV_COLD)
                // the whole pages of the expert (a page shared with a neighbour stays as it is); not the ones the device slots hold
                if (!in_device(il, e)) {
                    const uintptr_t p0 = ((uintptr_t) src + pg - 1) / pg * pg;
                    const uintptr_t p1 = ((uintptr_t) src + w->nb[2]) / pg * pg;
                    if (p1 > p0) {
                        madvise((void *) p0, p1 - p0, MADV_COLD);
                    }
                }
#endif
            }
            host->publish(s);
            host->set_spec(s);
        };
        const size_t n = llama_moe_profile_reserve(*host, order, want, take);
        hps.load_copied = n;
        hps.load_copied_bytes = b_pc;
        hst.prefilled = n;
        if (hprof_read_cap == 0 || wstore == nullptr) {
            hprof_pending.clear();
        }
        LLAMA_LOG_INFO("%s: MoE host cache: profile %s: %zu of %zu profile experts copied from the page cache into RAM slots in %.1f s (%.1f MiB), "
                "most used first; %zu not in the page cache are read in the background after the first prompt, into the slots left free\n",
                __func__, host_profile_file.c_str(), n, order.size(), (ggml_time_us() - t0)/1e6, b_pc/1048576.0, hprof_pending.size());
#endif
    }

    // k19-ramprof2 diagnostics: the RAM-tier counters of the request that just ended (LLAMA_MOE_HOST_PROFILE only)
    void hprof_request_log(const char * when) {
        const auto & hs = host->st;
        hprof_snap now;
        now.hits = hst.hits; now.misses = hst.misses; now.read = hst.read; now.admitted = hst.admitted; now.evicted = hs.evicted;
        now.no_slot = hs.no_slot; now.spec_taken = hs.spec_taken; now.spec_used = hs.spec_used; now.spec_dropped = hs.spec_dropped; now.done = hps.done;
        const hprof_snap & o = hprof_last;
        LLAMA_LOG_INFO("llama_moe_cache: host profile, request %llu (%s): device misses %llu RAM hits / %llu RAM misses, %llu read into RAM, %llu admitted "
                "from prompts, %llu evicted, %llu without a slot; guesses %llu used, %llu taken, %llu dropped, %zu unused; %llu background reads done; "
                "%d slots free\n", (unsigned long long) hps.requests, when, (unsigned long long) (now.hits - o.hits), (unsigned long long) (now.misses - o.misses),
                (unsigned long long) (now.read - o.read), (unsigned long long) (now.admitted - o.admitted), (unsigned long long) (now.evicted - o.evicted),
                (unsigned long long) (now.no_slot - o.no_slot), (unsigned long long) (now.spec_used - o.spec_used),
                (unsigned long long) (now.spec_taken - o.spec_taken), (unsigned long long) (now.spec_dropped - o.spec_dropped), host->n_spec_slots(),
                (unsigned long long) (now.done - o.done), host->n_free());
        hprof_last = now;
    }

    // k19-ramprof2: the batch hook (prepare): see hprof_order
    void hprof_prepare(int64_t n_tokens) {
        const bool large = n_tokens > n_small_max;
        if (!hprof_copied) {
            hprof_copy();
            // the hottest of the rest are read while the load finishes (the draft model, the warmup), at most hprof_load_read bytes
            if (hprof_load_read > 0) {
                hprof_issue(hprof_load_read, 1 << 30, false);
            }
        }
        if (large) {
            if (!hprof_q.empty()) {
                hprof_cancel(false);
            }
            if (!hprof_prev_large) {
                // a request starts: room for its reads
                if (hprof_prompted) {
                    hprof_request_log("done");
                }
                hps.requests++;
                const int32_t want_free = (int32_t) (hprof_prompt_free*host->n_slots());
                const int32_t n_free = host->n_free();
                if (want_free > n_free && host->n_spec_slots() > 0) {
                    hps.dropped += host->drop_spec((size_t) (want_free - n_free));
                }
            }
            hprof_prompted = true;
        } else {
            hprof_reap();
            if (hprof_prompted) {
                hprof_issue(SIZE_MAX, hprof_inflight_max, true);
            }
        }
        hprof_prev_large = large;
    }

    // the layout of layer il has more free slots than the room the profile fill leaves (hprof_room)
    bool hprof_has_room(int32_t il) const {
        return hprof_room <= 0.0 || host->n_free_of(il) > (int32_t) (hprof_room*host->n_slots_of(il));
    }

    // k19-ramprof2: a background read of the profile is complete: the slot holds its expert (a guess, unless the device took the expert
    // meanwhile on an exclusive tier: then it is a duplicate, the first slot to go)
    void hprof_complete(int32_t s) {
        host->publish(s);
        hprof_ticket[s] = 0;
        hps.done++;
        const int32_t key = host->key(s);
        const int32_t il  = key / host->n_expert_of();
        const int32_t e   = key % host->n_expert_of();
        if (host->exclusive() && in_device(il, e)) {
            host->set_dup(s, true);
            hps.dup++;
        } else {
            host->set_spec(s);
        }
    }

    // the background reads that are done publish their slots, in issue order (never blocks)
    void hprof_reap() {
        while (!hprof_q.empty()) {
            const int32_t s = hprof_q.front();
            if (hprof_ticket[s] != 0) {
                if (!wstore->host_done(hprof_ticket[s])) {
                    break;
                }
                hprof_complete(s);
            }
            hprof_q.pop_front();
        }
    }

    // the pending entries are read into free slots (never a guess, never evicting), in heat order, at most hprof_inflight_max in flight
    // (a cancel stays cheap); the entries without a slot, already held, or (exclusive) in the device slots now are dropped. The first
    // call allocates (pins) the slot memory of every pending entry at once (then frees the slots): pinning costs ~0.7 s per GB, once,
    // instead of a chunk at a time inside later decode steps
    void hprof_issue(size_t max_bytes, int max_inflight, bool pin_all) {
        if (hprof_pending.empty() || wstore == nullptr || !host) {
            return;
        }
        auto skip = [&](const prof_entry & p) {
            return !host_layer(p.il) || (host->exclusive() && in_device(p.il, p.e)) || host->contains(p.il, p.e) || !hprof_has_room(p.il);
        };
        if (pin_all && !hprof_pinned) {
            hprof_pinned = true;
            const int64_t t0 = ggml_time_us();
            std::vector<int32_t> tmp;
            for (const prof_entry & p : hprof_pending) {
                if (skip(p)) {
                    continue;
                }
                const int32_t s = host->reserve(p.il, p.e, false, 0, false);
                if (s >= 0) {
                    tmp.push_back(s);
                }
            }
            for (const int32_t s : tmp) {
                host->abort(s);
            }
            LLAMA_LOG_INFO("%s: MoE host cache: profile: slot memory of %zu background reads allocated in %.1f ms\n", __func__, tmp.size(),
                    (ggml_time_us() - t0)/1000.0);
        }
        hps.rounds++;
        std::vector<llama_weight_store::host_read> rd;
        size_t bytes = 0;
        while (!hprof_pending.empty() && (int) hprof_q.size() < max_inflight && hprof_read_left > 0 && bytes < max_bytes) {
            const prof_entry p = hprof_pending.front();
            hprof_pending.pop_front();
            if (skip(p)) {
                continue;
            }
            const int32_t s = host->reserve(p.il, p.e, false, 0, false);
            if (s < 0) {
                hps.no_slot++;
                continue;
            }
            const layer & l = layers[p.il];
            rd.clear();
            size_t nb = 0;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                rd.push_back({ w, (size_t) p.e*w->nb[2], w->nb[2], host->area(s, (int) ip), 0 });
                nb += w->nb[2];
            }
            const uint64_t t = wstore->read_host_async(rd.data(), rd.size(), true);
            if (t == 0) {
                host->abort(s);
                hps.failed++;
                continue;
            }
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                host->set_lead(s, (int) ip, rd[ip].lead);
            }
            host->set_async(s, 1);
            hprof_ticket[s] = t;
            hprof_q.push_back(s);
            hps.issued++;
            hps.issued_bytes += nb;
            bytes += nb;
            hprof_read_left = nb >= hprof_read_left ? 0 : hprof_read_left - nb;
        }
    }

    // the background reads that did not start are cancelled (their slots are free again, their entries back in front of the pending ones);
    // the started ones complete as issued (wait_started: they are waited for, e.g. before the store is detached)
    void hprof_cancel(bool wait_started) {
        size_t n = 0;
        std::vector<prof_entry> back;
        std::deque<int32_t> keep;
        for (const int32_t s : hprof_q) {
            if (hprof_ticket[s] == 0) {
                continue;
            }
            if (!wstore->host_cancel(hprof_ticket[s])) {
                if (wait_started) {
                    wstore->host_wait(hprof_ticket[s]);
                    hprof_complete(s);
                } else {
                    keep.push_back(s);
                }
                continue;
            }
            const int32_t key = host->key(s);
            hprof_ticket[s] = 0;
            host->abort(s);
            back.push_back({ key / host->n_expert_of(), key % host->n_expert_of(), 0.0f });
            if (hprof_read_left != SIZE_MAX) {
                for (const ggml_tensor * w : layers[key / host->n_expert_of()].experts) {
                    hprof_read_left += w->nb[2];
                }
            }
            n++;
        }
        hprof_q.swap(keep);
        hprof_pending.insert(hprof_pending.begin(), back.begin(), back.end());
        hps.cancelled += n;
    }

    // every background read is cancelled or waited for (before the store is detached or the RAM tier is dropped)
    void hprof_stop() {
        if (host && wstore != nullptr && !hprof_q.empty()) {
            hprof_cancel(true);
        }
    }

    // at a lookup of layer il: its routed experts that are being read in the background are waited for (their reads go first)
    void hprof_wait_routed(int32_t il) {
        for (const int32_t e : ids) {
            if (e < 0) {
                continue;
            }
            const int32_t s = host->slot_of_key(il, e);
            if (s < 0 || hprof_ticket[s] == 0) {
                continue;
            }
            if (!wstore->host_done(hprof_ticket[s])) {
                const int64_t t0 = ggml_time_us();
                wstore->host_wait(hprof_ticket[s]);
                hps.waited_us += ggml_time_us() - t0;
                hps.waited++;
            }
            hprof_complete(s);
        }
    }

    // k19-autoprof (LLAMA_MOE_AUTOPROF, see llama-autoprof.h): a routing profile the cache learns itself. Recorded wherever a RAM tier
    // exists: the routing of every large ubatch of every prompt (llama_context reads the recorded ids: the prompt end, as k19-l4fill,
    // and the earlier ubatches at the start of the next one), counted per (layer, expert) on top of the previous launches' heats
    // (halved at load), written to <dir>/autoprof_moe_<model hash>.<n_layer>.<n_expert> (the moe-cache-profile format) at the
    // server's idle point when it changed and at exit. Used on the exclusive tier when the file was read at load:
    //  - the page-cache fill copies the resident experts hottest first over every layer (not in layer order);
    //  - the target set of each expert layout = its hottest experts that the device does not hold, as many as its RAM slots; at a
    //    request's first large batch the held experts outside it are marked as guesses (set_spec: a reservation takes them like free
    //    slots, coldest first), and every large batch counts the target experts held nowhere (ap_need);
    //  - a prompt's uploads offered to the tier (tier_offer) take a free slot for an expert outside the target set only while the
    //    layout keeps a free slot for each target expert still held nowhere (and mark it as a guess), target experts may also take a
    //    guess; the l4fill refill order is seeded with the profile (an idle server fills its free slots hottest first).
    // Placement only (which slot holds which expert's file bytes): exact. No file: nothing is used (the recording reads the ids only).
    int      ap_mode = 0;
    std::string ap_file;
    bool     ap_rec  = false;                    // recording (a RAM tier exists, LLAMA_MOE_AUTOPROF 1 / 2)
    bool     ap_use  = false;                    // a profile was read and the exclusive tier uses it
    std::vector<std::vector<float>>   ap_cnt;    // [n_layer][n_expert] heat: the file's (halved) + this launch's routing counts
    std::vector<std::vector<uint8_t>> ap_tgt;    // [n_layer][n_expert] in the target set
    std::vector<int64_t>  ap_need;               // [class] target experts held neither in RAM nor on the device
    std::vector<int64_t>  ap_dup;                // [class] RAM copies of device-held experts (a reservation takes them first)
    std::vector<std::vector<uint8_t>> ap_dev_pred; // [n_layer][n_expert] the device prefill's experts, before it ran (lowest ids per layer)
    std::vector<uint32_t> ap_layer_epoch;
    uint32_t ap_epoch = 0;
    uint64_t ap_new   = 0;                       // routings counted since the last write
    bool     ap_prev_large = false;
    struct {
        size_t   loaded = 0, ubatches = 0, saves = 0, save_fail = 0, targets = 0, marked = 0, rebuilds = 0;
        uint64_t routings = 0, cold_refused = 0, cold_taken = 0, hot_taken = 0, pf_hot = 0, pf_cold = 0, pf_stale = 0;
    } aps;

    // k19-mgpu2 (LLAMA_MOE_RESIDENT_PROF: unset / 0 = off, 2 = record only, 1 = record + use): the device slots of resident mode can
    // start from the decode routing of earlier launches. Resident mode (the exclusive RAM tier and the device slots hold every streamed expert, e.g. Flash-Next UD-IQ1_S on
    // Kaggle 2x T4) never seeds the device cache from prompts (k18-lowram) and filled each layer's lowest expert ids at load, so a decode
    // paid its working set one miss at a time: an upload from RAM plus the copy back of the victim, ~26 misses per token over the first
    // ~100 tokens of a chat (22 tok/s, then 33-34 once the CLOCK had converged). With it:
    //  - the host lookups of small batches (decode, verify windows; the host reads their routing anyway) add each routed expert to the
    //    autoprof heat (k19-autoprof file, previous launches at half weight), written at the server's idle point and at exit;
    //  - at load, with that file, each group's slots take its hottest (layer, expert) pairs over all its layers (Strata's --expert-profile
    //    order), the rest of the slots the lowest ids as before; the RAM tier takes every other expert (unchanged).
    // Placement only: every expert is computed by the same kernels wherever it was copied from; the text changes only through the route
    // tail skip (a skipped expert is a missing one, so it follows the residency; LLAMA_MOE_ROUTE_TAIL_SKIP=0: identical text). A first
    // launch (no file) fills by id as before and records. LLAMA_MOE_AUTOPROF=0 / 2 / 3 also gate it (no file / record only / read only).
    // The use is opt-in because of that text change (=1); the default records only (counting changes nothing computed), so a file is
    // in place when it is turned on.
    bool rp_rec = false; // host lookups count the routing of small batches into ap_cnt (resident mode, recording)
    struct {
        uint64_t counted = 0, lookups = 0;
        size_t   read = 0, taken = 0, padded = 0;
        bool     used = false;
    } rps;

    static int rp_mode() {
        static const int m = [] {
            const char * e = getenv("LLAMA_MOE_RESIDENT_PROF");
            // off unless asked: the held-out warm start gained nothing measurable (Kaggle 2x T4 fn1, inside the launch spread)
            const int v = e != nullptr && *e ? atoi(e) : 0;
            return v >= 0 && v <= 2 ? v : 0;
        }();
        return m;
    }

    // resident mode's device prefill from the autoprof file (see rp_rec); false = no file, the caller fills by id (prefill)
    bool rp_prefill() {
        if (rp_mode() != 1 || ap_mode == 0 || ap_file.empty() || !llama_autoprof::uses(ap_mode) || !profile_file.empty() || prefilled_dev) {
            return false;
        }
        std::vector<prof_entry> prof;
        if (!llama_moe_profile_load(ap_file, profile_n_expert(), prof)) {
            LLAMA_LOG_INFO("%s: k19-mgpu2: no routing profile at %s yet: resident device slots by expert id, this launch records the decode "
                "routing (LLAMA_MOE_RESIDENT_PROF=1)\n", __func__, ap_file.c_str());
            return false;
        }
        rps.read = prof.size();
        size_t n_fill  = 0;
        size_t n_bytes = 0;
        for (group & g : groups) {
            if (g.n_slots == 0 || g.layers.empty()) {
                continue;
            }
            size_t n_prof = 0;
            size_t n_pad  = 0;
            const std::vector<std::vector<int32_t>> pick = llama_moe_profile_pick(prof, g.layers, g.lru.n_expert,
                g.n_slots - g.lru.n_lent - g.lru.n_core, &n_prof, &n_pad);
            rps.taken  += n_prof;
            rps.padded += n_pad;
            for (size_t i = 0; i < g.layers.size(); ++i) {
                const int32_t il = g.layers[i];
                const std::vector<int32_t> & ids = pick[i];
                size_t n_hit = 0;
                if (ids.empty() || !g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
                    continue;
                }
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t expert_size = w->nb[2];
                    for (size_t j = 0; j < fills.size();) {
                        size_t m = 1;
                        while (j + m < fills.size() && fills[j + m].expert == fills[j].expert + (int32_t) m && fills[j + m].slot == fills[j].slot + (int32_t) m) {
                            m++;
                        }
                        ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) w->data + fills[j].expert*expert_size, fills[j].slot*expert_size, m*expert_size);
                        n_bytes += m*expert_size;
                        j += m;
                    }
                }
                n_fill += fills.size();
            }
        }
        if (n_fill == 0) {
            return false;
        }
        prefilled_dev = true;
        rps.used = true;
        LLAMA_LOG_INFO("%s: k19-mgpu2: resident device slots from the routing profile %s: %zu experts (%.2f MiB), %zu hottest of %zu profile "
            "entries + %zu by id (LLAMA_MOE_RESIDENT_PROF=1; 2 = record only)\n", __func__, ap_file.c_str(), n_fill, n_bytes/1048576.0, rps.taken, rps.read, rps.padded);
        return true;
    }

    // a host lookup of a small batch read its routing: each routed expert adds 1 to the heat (see rp_rec)
    void rp_count(int32_t il, const std::vector<int32_t> & v) {
        if (il < 0 || (size_t) il >= ap_cnt.size()) {
            return;
        }
        std::vector<float> & c = ap_cnt[il];
        for (const int32_t e : v) {
            if (e >= 0 && (size_t) e < c.size()) {
                c[e] += 1.0f;
                ap_new++;
                rps.counted++;
            }
        }
        rps.lookups++;
    }

    // k19-mgpu2 resident prompt fill (LLAMA_MOE_RESIDENT_FILL=1, opt-in). One GPU (not resident: the exclusive tier holds what
    // the device does not, the rest is read from the files) admits the experts a prompt routes, so its decode starts on the request's
    // working set (fn1 1x T4 chat: 5718 experts admitted, then no miss at all, 50 tok/s). Resident mode (every streamed expert in RAM or
    // in a device slot, e.g. fn1 on 2x T4) never did: prompts do not seed there (k18-lowram: a seed's copies back found no RAM slot and
    // dropped experts from both tiers), so a chat decoded its first ~100 tokens at ~27 tok/s paying ~25 misses per token.
    // At the end of a prompt each card swaps the experts its layers routed in the prompt's last large ubatch (most routed first, at most
    // the group's slots per layer) into its own device slots, as a chain on its stream: the victim of a slot is copied back into the
    // RAM slot the previous promotion just left (read by that upload, earlier on the same stream; the slot is re-keyed), so a layer
    // needs one free RAM slot however many it swaps and no expert ever leaves both tiers. Victims RAM still holds need no copy.
    // Placement only: every expert is computed by the same kernels wherever it came from; the text changes only through the route tail
    // skip (missing experts it drops follow the residency): with LLAMA_MOE_ROUTE_TAIL_SKIP=0 the text is the kill switch's.
    // Opt-in (LLAMA_MOE_RESIDENT_FILL=1): Kaggle 2x T4, fn1, 3 launches with the PLE row profile on both sides: decode chat +11% / 8k
    // +12% for +3 s chat / +6 s 8k first-token time (~6k experts moved per prompt); for a 256-token answer the wait costs more than
    // the decode saves. LLAMA_MOE_RESIDENT_FILL_DEFER=1 moves the swaps into decode (8k decode -15..-23%); a cap
    // (LLAMA_MOE_RESIDENT_FILL_MAX=1500) costs ~0.7 s and gains nothing measurable
    bool rpf_on = [] {
        const char * e = getenv("LLAMA_MOE_RESIDENT_FILL");
        return e != nullptr && atoi(e) != 0;
    }();
    std::vector<std::vector<int32_t>> rpf_counts; // [il] routing counts of the prompt's last large ubatch
    std::vector<int32_t> rpf_ids;
    std::vector<moe_cache_lru::fill> rpf_fills;
    std::vector<int32_t> rpf_victim;               // [slot] the key a planned fill evicts, -1 none
    struct {
        uint64_t prompts = 0, layers = 0, wanted = 0, swapped = 0, chained = 0, fresh = 0, kept = 0, dropped = 0, from_file = 0, bytes = 0;
        int64_t  us = 0;
    } rpfs;

    bool resident_fill_wanted() const {
        return rpf_on && host && host->exclusive() && resident_mode;
    }

    void set_resident_counts(int32_t il, const std::vector<int32_t> & counts) {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return;
        }
        rpf_counts.resize(layers.size());
        rpf_counts[il] = counts;
    }

    bool has_resident_counts() const {
        return std::any_of(rpf_counts.begin(), rpf_counts.end(), [](const std::vector<int32_t> & c) { return !c.empty(); });
    }

    // LLAMA_MOE_RESIDENT_FILL_MIN (1): an expert is swapped in only when the prompt's last large ubatch routed it at least this many times;
    // LLAMA_MOE_RESIDENT_FILL_MAX (0 = no cap): at most this many experts per prompt, the most routed first over all layers;
    // LLAMA_MOE_RESIDENT_FILL_DEFER (0): 1 = not at the prompt end (the first token waits for none of it) but in the decode's host lookups,
    // each layer's swaps just before its lookup, at most LLAMA_MOE_RESIDENT_FILL_PASS (256) experts per decode pass
    int32_t rpf_min   = [] { const char * e = getenv("LLAMA_MOE_RESIDENT_FILL_MIN");   return e != nullptr ? std::max(1, atoi(e)) : 1; }();
    int64_t rpf_max   = [] { const char * e = getenv("LLAMA_MOE_RESIDENT_FILL_MAX");   return e != nullptr ? std::max<int64_t>(0, atoll(e)) : 0; }();
    bool    rpf_defer = [] { const char * e = getenv("LLAMA_MOE_RESIDENT_FILL_DEFER"); return e != nullptr && atoi(e) != 0; }();
    int64_t rpf_pass  = [] { const char * e = getenv("LLAMA_MOE_RESIDENT_FILL_PASS");  return e != nullptr ? std::max<int64_t>(1, atoll(e)) : 256; }();
    std::vector<std::vector<int32_t>> rpf_pending; // [il] experts still to swap in (deferred mode), most routed first
    int64_t rpf_pass_left = 0;
    int32_t rpf_last_il   = -1;
    uint64_t rpf_deferred_layers = 0;

    // the experts of each layer to swap in, most routed first: count >= rpf_min, at most the group's slots per layer, at most rpf_max over all
    void resident_fill_select(std::vector<std::vector<int32_t>> & sel) {
        sel.assign(layers.size(), {});
        std::vector<std::pair<int32_t, std::pair<int32_t, int32_t>>> all; // (count, (il, e)) for the global cap
        for (group & g : groups) {
            if (g.n_slots == 0 || g.layers.empty()) {
                continue;
            }
            const int32_t n_expert  = g.lru.n_expert;
            const int32_t per_layer = std::max<int32_t>(1, (g.n_slots - g.lru.n_lent - g.lru.n_core)/(int32_t) g.layers.size());
            for (const int32_t il : g.layers) {
                if (!host_layer(il) || (size_t) il >= rpf_counts.size() || (int32_t) rpf_counts[il].size() != n_expert) {
                    continue;
                }
                const std::vector<int32_t> & c = rpf_counts[il];
                std::vector<int32_t> & ids = sel[il];
                for (int32_t e = 0; e < n_expert; ++e) {
                    if (c[e] >= rpf_min) {
                        ids.push_back(e);
                    }
                }
                std::stable_sort(ids.begin(), ids.end(), [&c](int32_t a, int32_t b) { return c[a] > c[b]; });
                if ((int32_t) ids.size() > per_layer) {
                    ids.resize(per_layer);
                }
                for (const int32_t e : ids) {
                    all.push_back({ c[e], { il, e } });
                }
            }
        }
        if (rpf_max > 0 && (int64_t) all.size() > rpf_max) {
            std::stable_sort(all.begin(), all.end(), [](const auto & x, const auto & y) { return x.first > y.first; });
            for (auto & v : sel) {
                v.clear();
            }
            for (int64_t k = 0; k < rpf_max; ++k) {
                sel[all[k].second.first].push_back(all[k].second.second); // still most routed first per layer
            }
        }
    }

    // swaps the experts ids (most routed first) of host layer il into its group's slots as one chain on the device's stream (see above);
    // returns false when nothing was planned
    bool resident_fill_layer(group & g, ggml_backend_t backend, int32_t il, const std::vector<int32_t> & ids_in) {
        const int32_t n_expert = g.lru.n_expert;
        if ((int32_t) rpf_victim.size() != g.n_slots) {
            rpf_victim.assign(g.n_slots, -1);
        }
        rpf_ids = ids_in;
        if (rpf_ids.empty()) {
            return false;
        }
        ua_lookup_guard(backend, g, il);
        host->begin_call();
        for (const int32_t e : rpf_ids) {
            const int32_t sl = host->find(il, e, false);
            if (sl >= 0) {
                host->pin(sl); // a fresh slot for a copy back is never one of the experts being promoted
            }
        }
        size_t n_hit = 0;
        evicted_tmp.clear();
        if (!g.lru.plan(il, rpf_ids.data(), rpf_ids.size(), rpf_fills, n_hit, &evicted_tmp) || rpf_fills.empty()) {
            return false;
        }
        for (const auto & [s, key] : evicted_tmp) {
            rpf_victim[s] = key;
        }
        const layer & l = layers[il];
        int32_t carry = -1; // the RAM slot the last promotion was uploaded from: free for a copy back once that upload ran
        for (const auto & f : rpf_fills) {
            const int32_t key = rpf_victim[f.slot];
            rpf_victim[f.slot] = -1;
            // 1. the victim leaves the slot first (RAM keeps it)
            if (key >= 0 && host_layer(key / n_expert)) {
                const int32_t vil = key / n_expert;
                const int32_t ve  = key % n_expert;
                const layer & vl = layers[vil];
                if (host->contains(vil, ve)) {
                    const int32_t h = host->held(vil, ve);
                    if (h >= 0) {
                        host->set_dup(h, false);
                    }
                    rpfs.kept++;
                } else {
                    int32_t r = -1;
                    if (carry >= 0 && host->rekey(carry, vil, ve)) {
                        r = carry;
                        rpfs.chained++;
                    } else {
                        r = host->reserve(vil, ve, false);
                        if (r < 0) {
                            r = host->reserve(vil, ve, true); // evicts an expert RAM alone holds (counted: it is read from the files later)
                            rpfs.dropped += r >= 0 ? 1 : 0;
                        }
                        rpfs.fresh += r >= 0 ? 1 : 0;
                    }
                    carry = r == carry ? -1 : carry;
                    if (r >= 0) {
                        for (size_t ip = 0; ip < vl.experts.size(); ++ip) {
                            const size_t nb = vl.experts[ip]->nb[2];
                            host->set_lead(r, (int) ip, 0);
                            ggml_backend_tensor_get_async(backend, g.banks[ip], host->area(r, (int) ip), (size_t) f.slot*nb, nb);
                            hst.demote_bytes += nb;
                        }
                        host->mark_write(r);
                        hst.demoted++;
                    } else {
                        hst.demote_skip++;
                        rpfs.dropped++;
                    }
                }
            }
            // 2. the promoted expert from its RAM slot (exclusive: it leaves RAM; with spare room its copy stays as a duplicate
            // until a later copy back re-keys it)
            const int32_t s = host->find(il, f.expert, false);
            if (s >= 0) {
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t nb = l.experts[ip]->nb[2];
                    ggml_backend_tensor_set_async(backend, g.banks[ip], host->data(s, (int) ip), (size_t) f.slot*nb, nb);
                    rpfs.bytes += nb;
                    hst.up_bytes += nb;
                }
                host->mark_read(s);
                host->set_dup(s, true); // the device holds it now; re-keyed by the next copy back (or released below)
                hst.promoted++;
                carry = s;
            } else {
                // not in RAM (the few experts slot rounding left to the files): read from the mapping
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t nb = w->nb[2];
                    ggml_backend_tensor_set_async(backend, g.banks[ip], (const uint8_t *) w->data + (size_t) f.expert*nb, (size_t) f.slot*nb, nb);
                    rpfs.bytes += nb;
                }
                rpfs.from_file++;
            }
            rpfs.swapped++;
        }
        if (carry >= 0 && !host_spare) {
            host->release(carry); // no spare room: the last promotion leaves RAM (exclusive), as upload_fills does
        }
        host->commit(backend);
        return true;
    }

    void resident_fill() {
        if (!resident_fill_wanted() || rpf_counts.empty()) {
            rpf_counts.clear();
            return;
        }
        const int64_t t0 = ggml_time_us();
        std::vector<std::vector<int32_t>> sel;
        resident_fill_select(sel);
        rpf_counts.clear();
        if (rpf_defer) {
            // the decode's host lookups swap them in (resident_fill_lookup); a new prompt replaces what is left
            rpf_pending   = std::move(sel);
            rpf_pass_left = rpf_pass;
            rpf_last_il   = -1;
            rpfs.prompts++;
            rpfs.us += ggml_time_us() - t0;
            return;
        }
        bool any = false;
        for (group & g : groups) {
            if (g.n_slots == 0 || g.layers.empty() || devices[g.id].released) {
                continue;
            }
            if (g.dev_newer) {
                download(g);
            }
            ggml_backend_t backend = devices[g.id].backend;
            bool used = false;
            for (const int32_t il : g.layers) {
                if ((size_t) il < sel.size() && !sel[il].empty()) {
                    rpfs.wanted += sel[il].size();
                    rpfs.layers++;
                    used = resident_fill_layer(g, backend, il, sel[il]) || used;
                }
            }
            if (used) {
                ggml_backend_synchronize(backend);
                any = true;
            }
        }
        rpfs.prompts += any ? 1 : 0;
        rpfs.us += ggml_time_us() - t0;
    }

    // deferred mode: in the host lookup of a small batch of layer il, before its plan: its pending swaps, within the pass budget
    void resident_fill_lookup(group & g, ggml_backend_t backend, int32_t il) {
        if ((size_t) il >= rpf_pending.size() || rpf_pending[il].empty()) {
            return;
        }
        if (il <= rpf_last_il) {
            rpf_pass_left = rpf_pass; // a new pass (the layers of a pass come in increasing order)
        }
        rpf_last_il = il;
        if (rpf_pass_left <= 0) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        std::vector<int32_t> & p = rpf_pending[il];
        const size_t n = std::min<size_t>(p.size(), (size_t) rpf_pass_left);
        std::vector<int32_t> ids(p.begin(), p.begin() + n);
        p.erase(p.begin(), p.begin() + n);
        rpf_pass_left -= (int64_t) n;
        rpfs.wanted += n;
        rpfs.layers++;
        rpf_deferred_layers++;
        resident_fill_layer(g, backend, il, ids);
        rpfs.us += ggml_time_us() - t0;
    }

    // on the device, or (before the device prefill ran) one of the experts it will put there: the lowest ids of each layer, an equal
    // share of the group's slots per layer (see prefill), so the page-cache fill does not copy experts the device is about to hold
    bool ap_on_dev(int32_t il, int32_t e) const {
        if (in_device(il, e)) {
            return true;
        }
        return !prefilled_dev && (size_t) il < ap_dev_pred.size() && (size_t) e < ap_dev_pred[il].size() && ap_dev_pred[il][e] != 0;
    }

    // the exclusive tier keeps no RAM copy of a device-held expert (a promotion takes it out of RAM), so those are not RAM targets and the
    // page-cache fill leaves them; the inclusive tier keeps what the device holds too (a device eviction is not copied back: only a RAM
    // copy saves its read), so every expert may be a target there
    bool ap_dev_skip(int32_t il, int32_t e) const {
        return host->exclusive() && ap_on_dev(il, e);
    }

    // k19-autoprof2 (LLAMA_MOE_AUTOPROF_INCL, opt-in =1): the inclusive RAM tier (a capped card whose device slots hold the experts of
    // fewer than LLAMA_MOE_HOST_MIN_TOKENS tokens on a disk tier, e.g. Colab L4 @12 GB with MTP) also uses the profile: the page-cache
    // fill hottest first, the target set (hottest experts of each layout, as many as its slots, device-held ones included), cold
    // experts marked as guesses at each request's first large batch (a miss read or a prompt upload takes them before the CLOCK evicts
    // anything), a prompt's uploads take a free slot for a cold expert only while the layout keeps one per target expert held nowhere.
    // Placement only (which RAM slot holds which expert's file bytes): exact. Opt-in: Colab L4 @12 GB fn4n MTP n2 against a base with
    // its PLE profile, 3 interleaved launches: decode after 8k +19%, chat TTFT -3%, but 8k TTFT +11% (8k request level). Unset / 0: the
    // inclusive tier records only, as before
    static bool ap_incl_env() {
        static const bool on = [] { const char * e = getenv("LLAMA_MOE_AUTOPROF_INCL"); return e != nullptr && atoi(e) != 0; }();
        return on;
    }

    bool ap_target(int32_t il, int32_t e) const {
        return (size_t) il < ap_tgt.size() && (size_t) e < ap_tgt[il].size() && ap_tgt[il][e] != 0;
    }

    // at the RAM tier's creation (before the page-cache fill): the file of the previous launches
    void ap_init() {
        ap_rec = false;
        ap_use = false;
        if (ap_mode == 0 || ap_file.empty() || !host) {
            return;
        }
        const std::vector<int32_t> ne = profile_n_expert();
        ap_cnt.assign(layers.size(), {});
        for (size_t il = 0; il < layers.size(); ++il) {
            ap_cnt[il].assign(ne[il] > 0 ? ne[il] : 0, 0.0f);
        }
        ap_layer_epoch.assign(layers.size(), 0);
        ap_rec = llama_autoprof::records(ap_mode);
        std::vector<prof_entry> prof;
        const bool have = llama_moe_profile_load(ap_file, ne, prof);
        if (have) {
            for (const prof_entry & p : prof) {
                if (p.heat > 0.0f && (size_t) p.e < ap_cnt[p.il].size()) {
                    ap_cnt[p.il][p.e] = 0.5f*p.heat; // the previous launches count half
                }
            }
        }
        aps.loaded = have ? prof.size() : 0;
        ap_dev_pred.assign(layers.size(), {});
        if (!prefilled_dev) {
            for (const group & g : groups) {
                if (g.n_slots == 0 || g.layers.empty()) {
                    continue;
                }
                const int32_t n_slots = g.n_slots - g.lru.n_lent - g.lru.n_core;
                const size_t  n_layer = g.layers.size();
                for (size_t i = 0; i < n_layer; ++i) {
                    const int32_t il = g.layers[i];
                    const int32_t n  = std::max<int32_t>(0, std::min<int32_t>(g.lru.n_expert, n_slots/(int32_t) n_layer + (i < (size_t) (n_slots % n_layer) ? 1 : 0)));
                    ap_dev_pred[il].assign(g.lru.n_expert, 0);
                    std::fill(ap_dev_pred[il].begin(), ap_dev_pred[il].begin() + n, 1);
                }
            }
        }
        ap_use = have && llama_autoprof::uses(ap_mode) && (host->exclusive() || ap_incl_env()) && !resident_mode && wstore != nullptr;
        if (ap_use) {
            ap_rebuild_target();
            // the refill of the free slots (k19-l4fill, exclusive tier) starts from the profile: one prompt's worth of counts per layer,
            // halved per prompt
            if (fill_mode != 0 && host->exclusive()) {
                fill_cnt.assign(layers.size(), {});
                fill_layer_epoch.assign(layers.size(), 0);
                for (size_t il = 0; il < layers.size(); ++il) {
                    if (!host_layer((int32_t) il)) {
                        continue;
                    }
                    double sum = 0.0;
                    for (float h : ap_cnt[il]) {
                        sum += h;
                    }
                    fill_cnt[il].assign(ap_cnt[il].size(), 0.0f);
                    for (size_t e = 0; e < ap_cnt[il].size() && sum > 0.0; ++e) {
                        fill_cnt[il][e] = (float) (ap_cnt[il][e]/sum*512.0*(double) n_expert_used);
                    }
                }
                fill_have = true;
                fill_order_dirty = true;
            }
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: k19-autoprof: routing profile %s %s (%zu experts)%s; LLAMA_MOE_AUTOPROF=0 off, 3 read only\n", tag,
            ap_file.c_str(), have ? "read" : "not found", aps.loaded,
            ap_use ? (host->exclusive() ? ": the RAM tier is filled hottest first" : ": the inclusive RAM tier is filled hottest first (LLAMA_MOE_AUTOPROF_INCL=1)") :
            have ? ": not used (needs a RAM tier of the weight store, LLAMA_MOE_AUTOPROF 1 / 3, LLAMA_MOE_AUTOPROF_INCL for the inclusive tier)" :
            ap_rec ? ": recording" : "");
    }

    // the target set: per expert layout its hottest experts (heat > 0) that the device does not hold, as many as its slots
    void ap_rebuild_target() {
        const int32_t ne_max = host->n_expert_of();
        ap_tgt.assign(layers.size(), {});
        int32_t n_class = 0;
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            n_class = std::max(n_class, host->class_of(il) + 1);
        }
        std::vector<std::vector<std::pair<float, int32_t>>> by_class(n_class);
        std::vector<int32_t> slots_of_class(n_class, 0);
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            ap_tgt[il].assign(il < (int32_t) ap_cnt.size() ? ap_cnt[il].size() : 0, 0);
            if (!host_layer(il) || ap_cnt[il].empty()) {
                continue;
            }
            const int32_t c = host->class_of(il);
            slots_of_class[c] = host->n_slots_of(il);
            for (int32_t e = 0; e < (int32_t) ap_cnt[il].size() && e < ne_max; ++e) {
                if (ap_cnt[il][e] > 0.0f && !ap_dev_skip(il, e)) {
                    by_class[c].emplace_back(ap_cnt[il][e], il*ne_max + e);
                }
            }
        }
        ap_need.assign(n_class, 0);
        ap_dup.assign(n_class, 0);
        aps.targets = 0;
        for (int32_t c = 0; c < n_class; ++c) {
            auto & v = by_class[c];
            std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
            const size_t n = std::min(v.size(), (size_t) std::max(0, slots_of_class[c]));
            for (size_t i = 0; i < n; ++i) {
                ap_tgt[v[i].second / ne_max][v[i].second % ne_max] = 1;
            }
            aps.targets += n;
        }
        aps.rebuilds++;
        ap_count_need();
    }

    void ap_count_need() {
        std::fill(ap_need.begin(), ap_need.end(), 0);
        std::fill(ap_dup.begin(), ap_dup.end(), 0);
        for (int32_t il = 0; il < (int32_t) ap_tgt.size(); ++il) {
            if (!host_layer(il)) {
                continue;
            }
            const int32_t c = host->class_of(il);
            for (int32_t e = 0; e < (int32_t) ap_tgt[il].size(); ++e) {
                const bool held = host->contains(il, e);
                if (ap_tgt[il][e] && !held && !ap_dev_skip(il, e)) {
                    ap_need[c]++;
                }
                if (held) {
                    const int32_t s = host->held(il, e);
                    ap_dup[c] += s >= 0 && host->is_dup(s) ? 1 : 0;
                }
            }
        }
    }

    // the held experts outside the target set become guesses, hottest first (the coldest is the first a reservation takes)
    void ap_mark_cold() {
        std::vector<std::pair<float, int32_t>> v;
        for (int32_t il = 0; il < (int32_t) ap_tgt.size(); ++il) {
            if (!host_layer(il)) {
                continue;
            }
            for (int32_t e = 0; e < (int32_t) ap_tgt[il].size(); ++e) {
                if (ap_tgt[il][e]) {
                    continue;
                }
                const int32_t s = host->held(il, e);
                if (s >= 0 && !host->is_spec(s) && !host->is_dup(s) && !host->is_async(s)) {
                    v.emplace_back(ap_cnt[il][e], s);
                }
            }
        }
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        for (const auto & p : v) {
            host->set_spec(p.second);
        }
        aps.marked += v.size();
    }

    // the batch hook: a request's first large batch rebuilds the target set from the counts so far and marks the cold experts; every
    // large batch counts the target experts held nowhere
    void ap_prepare(int64_t n_tokens) {
        const bool large = n_tokens > n_small_max;
        if (ap_use && large) {
            if (!ap_prev_large) {
                ap_rebuild_target();
                ap_mark_cold();
            } else {
                ap_count_need();
            }
        }
        ap_prev_large = large;
    }

    // an offered upload of expert e of layer il may take a slot: target experts always (also a guess), the others only a free slot
    // while the layout keeps one for each target expert held nowhere
    bool ap_offer_ok(int32_t il, int32_t e, bool & take_spec) {
        take_spec = true;
        if (!ap_use) {
            return true;
        }
        const int32_t c = host->class_of(il);
        if (ap_target(il, e)) {
            aps.hot_taken++;
            return true;
        }
        take_spec = false;
        // k19-autoprof2 (LLAMA_MOE_AUTOPROF_REFUSE=0, diagnostic): a cold expert takes any free slot (as a guess, see tier_offer), the
        // target experts then take guesses like free slots
        static const bool refuse = [] { const char * e = getenv("LLAMA_MOE_AUTOPROF_REFUSE"); return e == nullptr || atoi(e) != 0; }();
        // (a duplicate of a device-held expert is as good as a free slot; counted per large batch)
        if (refuse && c >= 0 && (size_t) c < ap_need.size() && (int64_t) host->n_free_of(il) + ap_dup[c] <= ap_need[c]) {
            aps.cold_refused++;
            return false;
        }
        aps.cold_taken++;
        return true;
    }

    // a target expert was admitted: one fewer held nowhere
    void ap_admitted(int32_t il, int32_t e) {
        const int32_t c = ap_use && ap_target(il, e) ? host->class_of(il) : -1;
        if (c >= 0 && (size_t) c < ap_need.size() && ap_need[c] > 0) {
            ap_need[c]--;
        }
    }

    void ap_counts_begin() {
        ap_epoch++;
        aps.ubatches++;
    }

    void ap_counts(const ggml_tensor * w, const std::vector<int32_t> & counts) {
        const auto it = bindings.find(w);
        if (!ap_rec || it == bindings.end()) {
            return;
        }
        const int32_t il = it->second.il;
        if (il < 0 || (size_t) il >= ap_cnt.size() || ap_layer_epoch[il] == ap_epoch) {
            return;
        }
        ap_layer_epoch[il] = ap_epoch;
        std::vector<float> & c = ap_cnt[il];
        for (size_t e = 0; e < counts.size() && e < c.size(); ++e) {
            c[e] += (float) counts[e];
            ap_new += (uint64_t) counts[e];
            aps.routings += (uint64_t) counts[e];
        }
    }

    // writes the profile when this launch counted something new (the idle point, exit)
    void ap_save(bool at_exit) {
        if (!ap_rec || ap_new == 0 || ap_file.empty()) {
            return;
        }
        std::vector<prof_entry> v;
        for (int32_t il = 0; il < (int32_t) ap_cnt.size(); ++il) {
            for (int32_t e = 0; e < (int32_t) ap_cnt[il].size(); ++e) {
                if (ap_cnt[il][e] > 0.0f) {
                    v.push_back({ il, e, ap_cnt[il][e] });
                }
            }
        }
        std::sort(v.begin(), v.end(), [](const prof_entry & a, const prof_entry & b) {
            return a.heat != b.heat ? a.heat > b.heat : a.il != b.il ? a.il < b.il : a.e < b.e;
        });
        std::string body = "moe-cache-profile 1 " + std::to_string(layers.size()) + "\n";
        char buf[64];
        for (const prof_entry & p : v) {
            snprintf(buf, sizeof(buf), "%d %d %.6g\n", p.il, p.e, p.heat);
            body += buf;
        }
        const bool ok = llama_autoprof::write_atomic(ap_file, body);
        aps.saves += ok ? 1 : 0;
        aps.save_fail += ok ? 0 : 1;
        if (ok) {
            ap_new = 0;
        }
        if (at_exit || aps.saves + aps.save_fail <= 2) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-autoprof: routing profile %s %s (%zu experts, %zu ubatches counted in this launch)\n", tag,
                ok ? "written to" : "NOT written to", ap_file.c_str(), v.size(), aps.ubatches);
        }
    }

    // k19-l4fill (see fill_mode): the refill of the slots the exclusive tier freed
    bool fill_active() const {
        return fill_mode != 0 && host && host->exclusive() && wstore != nullptr && hprof_ticket.empty() && !ahead_on();
    }

    // the routing counts of a prompt's last large ubatch (llama_context, once per prompt: fill_counts_begin, then each expert tensor)
    void fill_counts_begin() {
        fill_epoch++;
        if (fill_cnt.size() != layers.size()) {
            fill_cnt.assign(layers.size(), {});
            fill_layer_epoch.assign(layers.size(), 0);
        }
        for (auto & c : fill_cnt) {
            for (float & v : c) {
                v *= 0.5f;
            }
        }
        fls.prompts++;
    }

    void fill_counts(const ggml_tensor * w, const std::vector<int32_t> & counts) {
        const auto it = bindings.find(w);
        if (it == bindings.end() || fill_cnt.size() != layers.size()) {
            return;
        }
        const int32_t il = it->second.il;
        if (!host_layer(il) || fill_layer_epoch[il] == fill_epoch) {
            return;
        }
        fill_layer_epoch[il] = fill_epoch;
        std::vector<float> & c = fill_cnt[il];
        if (c.size() != counts.size()) {
            c.assign(counts.size(), 0.0f);
        }
        for (size_t e = 0; e < counts.size(); ++e) {
            c[e] += (float) counts[e];
        }
        fill_have = true;
        fill_order_dirty = true;
    }

    void fill_rebuild_order() {
        fill_order_dirty = false;
        hfill_order.clear();
        const int32_t ne = host->n_expert_of();
        std::vector<std::pair<float, int32_t>> v;
        for (int32_t il = 0; il < (int32_t) fill_cnt.size(); ++il) {
            const std::vector<float> & c = fill_cnt[il];
            for (int32_t e = 0; e < (int32_t) c.size() && e < ne; ++e) {
                if (c[e] > 0.0f) {
                    v.emplace_back(c[e], il*ne + e);
                }
            }
        }
        // most routed first; ties go to the lower layer and expert, so the order does not depend on the sort
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        hfill_order.reserve(v.size());
        for (const auto & p : v) {
            hfill_order.push_back(p.second);
        }
    }

    // a read is complete: the slot holds its expert, a guess until used (a duplicate if the device took the expert meanwhile)
    void fill_complete(int32_t s) {
        host->publish(s);
        fill_ticket[s] = 0;
        fls.done++;
        const int32_t key = host->key(s);
        const int32_t il  = key / host->n_expert_of();
        const int32_t e   = key % host->n_expert_of();
        if (in_device(il, e)) {
            host->set_dup(s, true);
            fls.dup++;
        } else {
            host->set_spec(s);
        }
    }

    // the reads that are done publish their slots (never blocks; out of issue order: a read still running does not hold back the ones
    // after it, which a prompt would otherwise read again)
    void fill_reap() {
        std::deque<int32_t> keep;
        for (const int32_t s : fill_q) {
            if (fill_ticket[s] == 0) {
                continue;
            }
            if (wstore->host_done(fill_ticket[s])) {
                fill_complete(s);
            } else {
                keep.push_back(s);
            }
        }
        fill_q.swap(keep);
    }

    // the reads that did not start are cancelled (their slots are free again); the started ones complete as issued (wait_started: waited for)
    void fill_cancel(bool wait_started) {
        std::deque<int32_t> keep;
        for (const int32_t s : fill_q) {
            if (fill_ticket[s] == 0) {
                continue;
            }
            if (!wstore->host_cancel(fill_ticket[s])) {
                if (wait_started) {
                    wstore->host_wait(fill_ticket[s]);
                    fill_complete(s);
                } else {
                    keep.push_back(s);
                }
                continue;
            }
            fill_ticket[s] = 0;
            host->abort(s);
            fls.cancelled++;
        }
        fill_q.swap(keep);
    }

    void fill_stop() {
        if (host && wstore != nullptr && !fill_q.empty()) {
            fill_cancel(true);
        }
    }

    // free slots (and duplicates) whose memory is allocated get the most routed experts that neither tier holds
    void fill_issue() {
        if (!fill_have) {
            return;
        }
        if (fill_order_dirty) {
            fill_rebuild_order();
        }
        if (fill_ticket.size() != (size_t) host->n_slots()) {
            fill_ticket.assign(host->n_slots(), 0);
        }
        if (!fill_store_set) {
            wstore->set_spec_parallel(fill_parallel);
            fill_store_set = true;
        }
        fls.rounds++;
        const int32_t ne = host->n_expert_of();
        // RAM copies of experts the device also holds that are not marked as duplicates (the page-cache fill at load copies whatever the
        // page cache holds, also experts of the device prefill: Colab L4 VMs whose page cache kept more of the model after the drop started
        // the chat with 761-1254 of them instead of ~236, and missed 908-910 experts instead of ~780) are marked: they go first, like free slots
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            if (!host_layer(il) || layers[il].experts.empty()) {
                continue;
            }
            for (int32_t e = 0; e < (int32_t) layers[il].experts[0]->ne[2]; ++e) {
                const int32_t s = host->held(il, e);
                if (s >= 0 && !host->is_dup(s) && !host->is_async(s) && in_device(il, e)) {
                    host->set_dup(s, true);
                    fls.dup_marked++;
                }
            }
        }
        std::vector<char> full(layers.size(), 0);
        std::vector<llama_weight_store::host_read> rd;
        for (const int32_t key : hfill_order) {
            const int32_t il = key / ne;
            const int32_t e  = key % ne;
            if (full[il] || !host_layer(il) || in_device(il, e) || host->contains(il, e)) {
                continue;
            }
            if (!host->can_take_alloc(il)) {
                full[il] = 1;
                continue;
            }
            const int32_t s = host->reserve(il, e, false, 0, false);
            if (s < 0) {
                full[il] = 1;
                continue;
            }
            const layer & l = layers[il];
            rd.clear();
            size_t nb = 0;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                rd.push_back({ w, (size_t) e*w->nb[2], w->nb[2], host->area(s, (int) ip), 0 });
                nb += w->nb[2];
            }
            const uint64_t t = wstore->read_host_async(rd.data(), rd.size(), true);
            if (t == 0) {
                host->abort(s);
                fls.failed++;
                continue;
            }
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                host->set_lead(s, (int) ip, rd[ip].lead);
            }
            host->set_async(s, 1);
            fill_ticket[s] = t;
            fill_q.push_back(s);
            fls.issued++;
            fls.issued_bytes += nb;
        }
    }

    // at a lookup of layer il: its routed experts that are being read are waited for (their reads go first)
    void fill_wait_routed(int32_t il) {
        for (const int32_t e : ids) {
            if (e < 0) {
                continue;
            }
            const int32_t s = host->slot_of_key(il, e);
            if (s < 0 || (size_t) s >= fill_ticket.size() || fill_ticket[s] == 0) {
                continue;
            }
            if (!wstore->host_done(fill_ticket[s])) {
                const int64_t t0 = ggml_time_us();
                wstore->host_wait(fill_ticket[s]);
                fls.waited_us += ggml_time_us() - t0;
                fls.waited++;
            }
            fill_complete(s);
        }
    }

    void fill_diag(const char * when) {
        const int32_t n_slots = host->n_slots();
        const int32_t n_free  = host->n_free();
        const int32_t n_fill  = host->n_filling();
        LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4fill RAM tier at %s: %d held (%d duplicates of device experts, %zu unused guesses), %d free, %d being "
            "filled, of %d slots; since load: promoted %llu with device evictions / %llu without / %llu by seeds, uploads into empty device slots %llu, "
            "demoted %llu into free slots or duplicates / %llu evicting / %llu kept (RAM held them) / %llu without a slot, admitted from prompts %llu, "
            "page-cache fill %llu; fill %s: %llu read, %llu done (%llu duplicates), %llu cancelled, %llu waited for (%.1f ms), %.1f MiB\n",
            tag, when, n_slots - n_free - n_fill, host->n_dup_slots(), host->n_spec_slots(), n_free, n_fill, n_slots,
            (unsigned long long) sss.prom_ev, (unsigned long long) sss.prom_noev, (unsigned long long) sss.prom_seed, (unsigned long long) sss.empty_fills,
            (unsigned long long) sss.dem_free, (unsigned long long) sss.dem_evict, (unsigned long long) sss.dem_kept, (unsigned long long) sss.dem_none,
            (unsigned long long) hst.admitted, (unsigned long long) hst.prefilled, fill_active() ? "on" : "off",
            (unsigned long long) fls.issued, (unsigned long long) fls.done, (unsigned long long) fls.dup, (unsigned long long) fls.cancelled,
            (unsigned long long) fls.waited, fls.waited_us/1000.0, fls.issued_bytes/1048576.0);
    }

    // the batch hook (prepare): the done reads publish their slots; LLAMA_MOE_HOST_FILL=1 (default): every batch cancels the reads that did
    // not start (they run between requests only, see fill_idle); =2: also issued at every small batch after a prompt (the decode), a
    // large batch cancels them
    void fill_prepare(int64_t n_tokens) {
        const bool large = n_tokens > n_small_max;
        const bool on = fill_active();
        if (on && !fill_q.empty()) {
            fill_reap();
        }
        if (fill_log && large != fill_prev_large) {
            fill_diag(large ? "a request's first large batch" : "the first small batch after a prompt");
        }
        if (on) {
            if ((large || fill_mode == 1) && !fill_q.empty()) {
                fill_cancel(false);
            }
            if (!large && fill_mode >= 2) {
                fill_issue();
            }
        }
        fill_prev_large = large;
    }

    // the server is idle (no request runs, none waits): the freed slots are refilled now, the reads run until the next batch
    void fill_idle() {
        ap_save(false); // k19-autoprof: the profile, when this launch counted something new since the last write
        if (!fill_active()) {
            return;
        }
        if (!fill_q.empty()) {
            fill_reap();
        }
        const uint64_t n0 = fls.issued;
        fill_issue();
        fls.idle_rounds++;
        if (fill_log && fls.issued > n0) {
            fill_diag("the server's idle point (refill issued)");
        }
    }

    // the experts whose pages are in the page cache (e.g. read while loading) are copied into free slots: pinning the slots would
    // otherwise evict those pages and the first prompt would read them again (LLAMA_MOE_HOST_CACHE_PREFILL=0 off)
    void prefill_resident() {
#if defined(__linux__) || defined(__APPLE__)
        // k19-ramprof: a routing profile fills the RAM tier first (the resident mode reads every expert anyway)
        const size_t n_prof = !resident_mode && !host_profile_file.empty() ? prefill_host_profile() : 0;
        hst.prefilled = 0; // k19-ramprof2: the profile copy at the first batch sets it
        const char * env_pf = getenv("LLAMA_MOE_HOST_CACHE_PREFILL");
        if (env_pf != nullptr && atoi(env_pf) == 0) {
            return;
        }
        // k19-ramprof2: the profile's copy at the first batch replaces the page-cache fill (it would take the slots for experts in layer order)
        if (n_prof > 0) {
            return;
        }
        // LLAMA_MOE_HOST_CACHE_PREFILL=2: also read the experts that are not in the page cache, layer by layer from the model files
        // (readahead of each layer's expert tensors first), skipping the experts the device slots hold: with the device cache prefilled
        // (LLAMA_MOE_CACHE_PREFILL=1) the first prompt then reads nothing from the files when RAM + VRAM hold every expert (low-RAM boxes)
        const bool from_file = (env_pf != nullptr && atoi(env_pf) >= 2) || resident_mode;
        // k19-l4ttft (LLAMA_MOE_HOST_PREFILL_GUESS=1, opt-in): what the page cache held at load is not a choice (whatever the load read),
        // so the experts copied from it are marked as unused guesses (set_spec, as the routing profile's): a prompt whose copies read
        // only the experts it routes (GGML_SCHED_PREFETCH_MIN_BATCH) then keeps its reads in the slots of the ones it did not use when
        // no slot is free. Placement only (what a slot holds is the expert's bytes); exclusive tier only
        static const bool guess = [] { const char * e = getenv("LLAMA_MOE_HOST_PREFILL_GUESS"); return e != nullptr && atoi(e) != 0; }();
        const bool mark = guess && !from_file && host->exclusive();
        const int64_t t0 = ggml_time_us();
        size_t n_file = 0;
        const size_t pg = (size_t) std::max(4096L, sysconf(_SC_PAGESIZE));
        std::vector<unsigned char> vec;
        std::vector<std::vector<bool>> res;
        size_t n_fill = 0;
        size_t n_bytes = 0;
        bool full = false;
        // k19-autoprof: with a profile the resident experts are collected over every layer and copied hottest first (below)
        const bool ap_order = ap_use && !from_file;
        std::vector<std::pair<float, int32_t>> ap_cand; // (heat, il*n_expert + e)
        auto copy_one = [&](int32_t il, int32_t e, int32_t s) {
            const layer & l = layers[il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                host->set_lead(s, (int) ip, 0);
                memcpy(host->area(s, (int) ip), (const uint8_t *) w->data + (size_t) e*w->nb[2], w->nb[2]);
                n_bytes += w->nb[2];
            }
            host->publish(s);
            if (mark) {
                host->set_spec(s);
            }
            n_fill++;
        };
        for (size_t ig = 0; ig < groups.size() && !full; ++ig) {
            if (!host_group[ig]) {
                continue;
            }
            for (int32_t il : groups[ig].layers) {
                const layer & l = layers[il];
                const int32_t n_expert = (int32_t) l.experts[0]->ne[2];
                const int32_t * dev_slots = il < (int32_t) groups[ig].lru.slot_map.size() ? groups[ig].lru.slot_map[il] : nullptr;
                if (from_file) {
                    for (const ggml_tensor * w : l.experts) {
                        const uintptr_t p0 = (uintptr_t) w->data / pg * pg;
                        madvise((void *) p0, (uintptr_t) w->data + ggml_nbytes(w) - p0, MADV_WILLNEED);
                    }
                }
                // residency of each expert of each tensor
                res.assign(l.experts.size(), std::vector<bool>(n_expert, false));
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const uintptr_t p0 = (uintptr_t) w->data / pg * pg;
                    const uintptr_t p1 = ((uintptr_t) w->data + ggml_nbytes(w) + pg - 1) / pg * pg;
                    vec.resize((p1 - p0) / pg);
#if defined(__APPLE__)
                    const bool ok = mincore((void *) p0, p1 - p0, (char *) vec.data()) == 0;
#else
                    const bool ok = mincore((void *) p0, p1 - p0, vec.data()) == 0;
#endif
                    if (!ok) {
                        return;
                    }
                    const size_t nb = w->nb[2];
                    for (int32_t e = 0; e < n_expert; ++e) {
                        const uintptr_t a = (uintptr_t) w->data + (size_t) e*nb;
                        bool all = true;
                        for (uintptr_t q = a / pg * pg; q < a + nb && all; q += pg) {
                            all = (vec[(q - p0) / pg] & 1) != 0;
                        }
                        res[ip][e] = all;
                    }
                }
                const std::vector<int32_t> order = fill_order(il, n_expert);
                for (int32_t k = 0; k < n_expert && !full; ++k) {
                    const int32_t e = order[k];
                    bool all = true;
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        all = all && res[ip][e];
                    }
                    const bool on_dev = dev_slots != nullptr && dev_slots[e] >= 0;
                    const bool file = from_file && !all && !on_dev;
                    all = all || file;
                    if (!all || host->contains(il, e) || (from_file && on_dev)) {
                        continue;
                    }
                    if (ap_order && ap_dev_skip(il, e)) {
                        continue;
                    }
                    if (ap_order) {
                        ap_cand.emplace_back((size_t) il < ap_cnt.size() && (size_t) e < ap_cnt[il].size() ? ap_cnt[il][e] : 0.0f,
                            il*host->n_expert_of() + e);
                        continue;
                    }
                    const int32_t s = host->reserve(il, e, false);
                    if (s < 0) {
                        // k19-l4held: this layer is at its quota, the next layer may still take slots
                        if (host->at_quota(il)) {
                            break;
                        }
                        // the slots of this layer's expert layout are used up; the file fill goes on with the layers of the other
                        // layouts (Flash-Next IQ2_XS has three: stopping at the first full one left 3137 RAM slots empty)
                        full = !from_file;
                        break;
                    }
                    copy_one(il, e, s);
                    n_file += file ? 1 : 0;
                }
            }
        }
        if (ap_order) {
            // hottest first (ties: layer, expert); an expert outside the target set only while its layout keeps a free slot for each
            // target expert held nowhere; then the copied cold experts become guesses
            std::sort(ap_cand.begin(), ap_cand.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
            const int32_t ne = host->n_expert_of();
            std::set<int32_t> full_class;
            for (const auto & cnd : ap_cand) {
                const int32_t il = cnd.second / ne;
                const int32_t e  = cnd.second % ne;
                const int32_t c  = host->class_of(il);
                if (full_class.count(c) > 0) {
                    continue;
                }
                const bool hot = ap_target(il, e);
                if (!hot && c >= 0 && (size_t) c < ap_need.size() && (int64_t) host->n_free_of(il) <= ap_need[c]) {
                    continue;
                }
                // still in the page cache right now (the residency above went stale while earlier copies pinned slots: a copy of
                // evicted pages would read the expert back from the disk one page at a time)
                bool still = true;
                for (const ggml_tensor * w : layers[il].experts) {
                    const uintptr_t a  = (uintptr_t) w->data + (size_t) e*w->nb[2];
                    const uintptr_t q0 = a / pg * pg;
                    const uintptr_t q1 = (a + w->nb[2] + pg - 1) / pg * pg;
                    vec.resize((q1 - q0) / pg);
#if defined(__APPLE__)
                    still = still && mincore((void *) q0, q1 - q0, (char *) vec.data()) == 0;
#else
                    still = still && mincore((void *) q0, q1 - q0, vec.data()) == 0;
#endif
                    for (size_t k = 0; k < vec.size() && still; ++k) {
                        still = (vec[k] & 1) != 0;
                    }
                }
                if (!still) {
                    aps.pf_stale++;
                    continue;
                }
                const int32_t s = host->reserve(il, e, false);
                if (s < 0) {
                    full_class.insert(c);
                    continue;
                }
                ap_admitted(il, e);
                copy_one(il, e, s);
                aps.pf_hot  += hot ? 1 : 0;
                aps.pf_cold += hot ? 0 : 1;
            }
            ap_mark_cold();
            LLAMA_LOG_INFO("%s: k19-autoprof: the page-cache fill copied %llu target + %llu other experts of %zu resident, hottest first (%llu no longer "
                "resident when their turn came, not copied)\n", __func__, (unsigned long long) aps.pf_hot, (unsigned long long) aps.pf_cold, ap_cand.size(),
                (unsigned long long) aps.pf_stale);
        }
        hst.prefilled = n_fill + n_prof;
        LLAMA_LOG_INFO("%s: MoE host cache: %zu experts (%.1f MiB) copied from the page cache into RAM slots in %.1f ms, %zu of them read from the files"
                " (LLAMA_MOE_HOST_CACHE_PREFILL=0 off, 2 also reads the experts that are not in the page cache)%s\n",
                __func__, n_fill, n_bytes/1048576.0, (ggml_time_us() - t0)/1000.0, n_file, mark ? "; k19-l4ttft: marked as guesses" : "");
#endif
    }

    //
    // layer-ahead reads of the RAM tier
    //


    void init_ahead(const llama_model & model) {
        const char * env = getenv("LLAMA_MOE_HOST_CACHE_AHEAD");
        ahead_depth = env != nullptr ? std::max(0, std::min(4, atoi(env))) : 0;
        if (!host || ahead_depth == 0) {
            ahead_depth = 0;
            return;
        }
        if (model.hparams.n_group_experts > 1) {
            LLAMA_LOG_INFO("%s: MoE host cache: no reads ahead (grouped expert ids)\n", __func__);
            ahead_depth = 0;
            return;
        }
        const char * env_k = getenv("LLAMA_MOE_HOST_CACHE_AHEAD_TOPK");
        ahead_topk   = env_k != nullptr ? std::max(1, atoi(env_k)) : (int) n_expert_used;
        const char * env_i = getenv("LLAMA_MOE_HOST_CACHE_AHEAD_INFLIGHT");
        ahead_max_inflight = env_i != nullptr ? std::max(1, atoi(env_i)) : (int) n_expert_used;
        ahead_gating = (int) model.hparams.expert_gating_func;
        const int64_t t0 = ggml_time_us();
        routers.assign(layers.size(), {});
        router_w.assign(layers.size(), nullptr);
        ahead_pred.assign(layers.size(), {});
        std::vector<uint8_t> raw;
        std::vector<float>   row;
        size_t n_layers = 0;
        for (size_t il = 0; il < layers.size() && il < model.layers.size(); ++il) {
            if (!host_layer((int32_t) il)) {
                continue;
            }
            const ggml_tensor * w = model.layers[il].ffn_gate_inp;
            const int64_t n_expert = layers[il].experts[0]->ne[2];
            if (w == nullptr || w->buffer == nullptr || ggml_n_dims(w) != 2 || w->ne[1] != n_expert || (ahead_n_embd != 0 && w->ne[0] != ahead_n_embd)) {
                continue;
            }
            const auto * tt = ggml_get_type_traits(w->type);
            if (w->type != GGML_TYPE_F32 && (tt == nullptr || tt->to_float == nullptr)) {
                continue;
            }
            ahead_n_embd = w->ne[0];
            raw.resize(ggml_nbytes(w));
            ggml_backend_tensor_get(w, raw.data(), 0, raw.size());
            ahead_router & r = routers[il];
            r.w.resize((size_t) n_expert*ahead_n_embd);
            r.ws.resize(n_expert);
            row.resize(ahead_n_embd);
            for (int64_t e = 0; e < n_expert; ++e) {
                const uint8_t * src = raw.data() + (size_t) e*w->nb[1];
                if (w->type == GGML_TYPE_F32) {
                    memcpy(row.data(), src, ahead_n_embd*sizeof(float));
                } else {
                    tt->to_float(src, row.data(), ahead_n_embd);
                }
                float amax = 0.0f;
                for (int64_t i = 0; i < ahead_n_embd; ++i) {
                    amax = std::max(amax, std::fabs(row[i]));
                }
                const float sc = amax > 0.0f ? amax/127.0f : 1.0f;
                r.ws[e] = sc;
                for (int64_t i = 0; i < ahead_n_embd; ++i) {
                    r.w[(size_t) e*ahead_n_embd + i] = (int8_t) std::lround(std::max(-127.0f, std::min(127.0f, row[i]/sc)));
                }
            }
            auto get_bias = [&](const ggml_tensor * b, std::vector<float> & out) {
                if (b != nullptr && b->buffer != nullptr && b->type == GGML_TYPE_F32 && ggml_nelements(b) == n_expert && ggml_is_contiguous(b)) {
                    out.resize(n_expert);
                    ggml_backend_tensor_get(b, out.data(), 0, n_expert*sizeof(float));
                }
            };
            get_bias(model.layers[il].ffn_gate_inp_b, r.b);
            get_bias(model.layers[il].ffn_exp_probs_b, r.pb);
            router_w[il] = w;
            ahead_router_bytes += r.w.size()*sizeof(int8_t) + (r.ws.size() + r.b.size() + r.pb.size())*sizeof(float);
            n_layers++;
        }
        if (n_layers == 0) {
            LLAMA_LOG_INFO("%s: MoE host cache: no reads ahead (no router of a streamed layer can be read)\n", __func__);
            ahead_depth = 0;
            routers.clear();
            return;
        }
        ahead_ticket.assign(host->n_slots(), 0);
        ahead_key.assign(host->n_slots(), -1);
        LLAMA_LOG_INFO("%s: MoE host cache: reads ahead of %d layer%s (top %d experts per token, LLAMA_MOE_HOST_CACHE_AHEAD=0 off), host routers of %zu layers = %.1f MiB in %.1f ms\n",
                __func__, ahead_depth, ahead_depth > 1 ? "s" : "", ahead_topk, n_layers, ahead_router_bytes/1048576.0, (ggml_time_us() - t0)/1000.0);
    }

    bool ahead_on() const {
        return ahead_depth > 0 && host && wstore != nullptr;
    }

    // the router input of layer il: src[1] of the matmul with its ffn_gate_inp that the selected experts sel derive from (nullptr if none)
    const ggml_tensor * ahead_router_input(int32_t il, const ggml_tensor * sel) const {
        if (il < 0 || (size_t) il >= router_w.size() || router_w[il] == nullptr) {
            return nullptr;
        }
        const ggml_tensor * stack[64];
        int n = 0;
        int visited = 0;
        stack[n++] = sel;
        while (n > 0 && visited < 64) {
            const ggml_tensor * t = stack[--n];
            visited++;
            if (t->op == GGML_OP_MUL_MAT && t->src[0] == router_w[il]) {
                return t->src[1];
            }
            for (int i = 0; i < GGML_MAX_SRC && n < 63; ++i) {
                if (t->src[i] != nullptr) {
                    stack[n++] = t->src[i];
                }
            }
            if (t->view_src != nullptr && n < 63) {
                stack[n++] = t->view_src;
            }
        }
        return nullptr;
    }

    // the experts of layer j the router input of layer ahead_il selects (top ahead_topk per token, union)
    void ahead_predict(int32_t j, std::vector<int32_t> & out) {
        out.clear();
        const ahead_router & r = routers[j];
        const int64_t n_expert = (int64_t) r.ws.size();
        const int64_t k = std::min<int64_t>(ahead_topk, n_expert);
        ahead_scores.resize(n_expert);
        ahead_xq.resize(ahead_n_embd);
        for (int64_t t = 0; t < ahead_nt; ++t) {
            const float * x = ahead_x.data() + t*ahead_n_embd;
            float amax = 0.0f;
            for (int64_t i = 0; i < ahead_n_embd; ++i) {
                amax = std::max(amax, std::fabs(x[i]));
            }
            if (!(amax > 0.0f) || !std::isfinite(amax)) {
                continue;
            }
            const float xs = amax/32767.0f;
            for (int64_t i = 0; i < ahead_n_embd; ++i) {
                ahead_xq[i] = (int16_t) std::lround(x[i]/xs);
            }
            const int16_t * xq = ahead_xq.data();
            for (int64_t e = 0; e < n_expert; ++e) {
                const int8_t * w = r.w.data() + (size_t) e*ahead_n_embd;
                int32_t acc = 0; // |w| <= 127, |x| <= 32767: no overflow below 2^31 / (127*32767) = 516 terms per partial sum
                int64_t i = 0;
                int64_t acc64 = 0;
                for (; i + 512 <= ahead_n_embd; i += 512) {
                    acc = 0;
                    for (int64_t j = 0; j < 512; ++j) {
                        acc += (int32_t) w[i + j]*xq[i + j];
                    }
                    acc64 += acc;
                }
                acc = 0;
                for (; i < ahead_n_embd; ++i) {
                    acc += (int32_t) w[i]*xq[i];
                }
                acc64 += acc;
                ahead_scores[e] = (float) acc64*r.ws[e]*xs + (r.b.empty() ? 0.0f : r.b[e]);
            }
            if (!r.pb.empty()) {
                // the selection bias is added to the probabilities
                if (ahead_gating == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX) {
                    float mx = -INFINITY;
                    for (int64_t e = 0; e < n_expert; ++e) {
                        mx = std::max(mx, ahead_scores[e]);
                    }
                    float sum = 0.0f;
                    for (int64_t e = 0; e < n_expert; ++e) {
                        ahead_scores[e] = expf(ahead_scores[e] - mx);
                        sum += ahead_scores[e];
                    }
                    for (int64_t e = 0; e < n_expert; ++e) {
                        ahead_scores[e] = ahead_scores[e]/sum + r.pb[e];
                    }
                } else {
                    for (int64_t e = 0; e < n_expert; ++e) {
                        ahead_scores[e] = 1.0f/(1.0f + expf(-ahead_scores[e])) + r.pb[e];
                    }
                }
            }
            ahead_rank.resize(n_expert);
            for (int64_t e = 0; e < n_expert; ++e) {
                ahead_rank[e] = { -ahead_scores[e], (int32_t) e };
            }
            std::nth_element(ahead_rank.begin(), ahead_rank.begin() + (k - 1), ahead_rank.end());
            std::sort(ahead_rank.begin(), ahead_rank.begin() + k);
            for (int64_t i = 0; i < k; ++i) {
                if (std::find(out.begin(), out.end(), ahead_rank[i].second) == out.end()) {
                    out.push_back(ahead_rank[i].second);
                }
            }
        }
    }

    // a read ahead completed: the slot holds its expert
    void ahead_complete(int32_t s) {
        host->publish(s);
        ahead_ticket[s] = 0;
        ahead_key[s] = host->key(s);
    }

    // the reads ahead that are done publish their slots (never blocks)
    void ahead_reap() {
        size_t o = 0;
        for (size_t i = 0; i < ahead_inflight.size(); ++i) {
            const int32_t s = ahead_inflight[i];
            if (wstore->host_done(ahead_ticket[s])) {
                ahead_complete(s);
            } else {
                ahead_inflight[o++] = s;
            }
        }
        ahead_inflight.resize(o);
    }

    // every read ahead is waited for (before a large batch, a detach of the store, or the end)
    void ahead_drain() {
        if (!host || ahead_inflight.empty() || wstore == nullptr) {
            return;
        }
        for (const int32_t s : ahead_inflight) {
            wstore->host_wait(ahead_ticket[s]);
            ahead_complete(s);
        }
        ahead_inflight.clear();
    }

    // the routed experts of layer il that are being read ahead: wait for those reads (their unstarted pieces go first); the reads
    // ahead of layers up to il that it does not route and that did not start are dropped (their slots are free again)
    void ahead_wait_routed(int32_t il) {
        bool any = false;
        for (const int32_t e : ids) {
            if (e < 0) {
                continue;
            }
            const int32_t s = host->slot_of_key(il, e);
            if (s < 0 || ahead_ticket[s] == 0) {
                continue;
            }
            if (!wstore->host_done(ahead_ticket[s])) {
                const int64_t t0 = ggml_time_us();
                wstore->host_wait(ahead_ticket[s]);
                ast.waited_us += ggml_time_us() - t0;
                ast.late++;
            }
            ahead_complete(s);
            any = true;
        }
        const int32_t n_expert = host->n_expert_of();
        for (const int32_t s : ahead_inflight) {
            if (ahead_ticket[s] == 0 || host->key(s) / n_expert > il) {
                continue;
            }
            if (wstore->host_cancel(ahead_ticket[s])) {
                ahead_ticket[s] = 0;
                host->abort(s);
                ast.cancelled++;
                any = true;
            }
        }
        if (any) {
            size_t o = 0;
            for (size_t i = 0; i < ahead_inflight.size(); ++i) {
                if (ahead_ticket[ahead_inflight[i]] != 0) {
                    ahead_inflight[o++] = ahead_inflight[i];
                }
            }
            ahead_inflight.resize(o);
        }
        // recall of the last prediction of the layer
        auto & pred = ahead_pred[il];
        if (!pred.empty()) {
            std::vector<int32_t> & seen = ahead_rank_ids();
            seen.clear();
            for (const int32_t e : ids) {
                if (e < 0 || std::find(seen.begin(), seen.end(), e) != seen.end()) {
                    continue;
                }
                seen.push_back(e);
                ast.routed++;
                ast.recalled += std::find(pred.begin(), pred.end(), e) != pred.end();
            }
            pred.clear();
        }
    }

    static std::vector<int32_t> & ahead_rank_ids() {
        static thread_local std::vector<int32_t> v;
        return v;
    }

    // at a lookup of layer il, before it synchronizes: the reads ahead for layers il .. ahead_il + depth from the router input of ahead_il
    void ahead_issue(int32_t il) {
        const int32_t p = ahead_il;
        for (int32_t j = std::max(il, p + 1); j <= p + ahead_depth && (size_t) j < layers.size(); ++j) {
            if (!host_layer(j) || routers[j].w.empty()) {
                continue;
            }
            std::vector<int32_t> & pred = ahead_pred[j];
            const int64_t t0 = ggml_time_us();
            ahead_predict(j, pred);
            ast.predict_us += ggml_time_us() - t0;
            ast.layers++;
            ast.predicted += pred.size();
            const layer & l = layers[j];
            for (const int32_t e : pred) {
                if (in_device(j, e) || host->contains(j, e)) {
                    ast.held++;
                    continue;
                }
                if ((int) ahead_inflight.size() >= ahead_max_inflight) {
                    ast.capped++;
                    continue;
                }
                // never a slot that a copy of the last few lookups may still read: the reservation would wait for the device
                const int32_t s = host->reserve(j, e, true, 4);
                if (s < 0) {
                    ast.no_slot++;
                    break;
                }
                ahead_reads.clear();
                size_t nb_all = 0;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    ahead_reads.push_back({ w, (size_t) e*w->nb[2], w->nb[2], host->area(s, (int) ip), 0 });
                    nb_all += w->nb[2];
                }
                const uint64_t t = wstore->read_host_async(ahead_reads.data(), ahead_reads.size(), true);
                if (t == 0) {
                    host->abort(s);
                    continue;
                }
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    host->set_lead(s, (int) ip, ahead_reads[ip].lead);
                }
                host->set_async(s, 1);
                ahead_ticket[s] = t;
                ahead_inflight.push_back(s);
                ast.issued++;
                ast.bytes += nb_all;
            }
        }
    }

    // k15-pred: upload ahead (LLAMA_MOE_UPLOAD_AHEAD, 0 = off). A prompt ubatch of n_small_max < n <= ua_max_tokens tokens copies the
    // experts each layer routes after reading its ids, so the uploads of layer p wait for its router and the link idles while the
    // device computes. Here the router of layer p+1 is applied (on the device) to the router input of layer p as soon as the ids of p
    // are read; after the last expert tensor of p is copied, the predicted experts of p+1 that the cache lacks are admitted to slots of
    // layer p+1 and the first expert tensor of each is uploaded into its slot on a second stream. When the real ids of p+1 are read,
    // the wrong guesses are evicted (only their first tensor was sent), and each further expert tensor of the right ones is uploaded
    // on that stream one tensor ahead of its copy. The copies read the slots after waiting for those uploads (an event on the
    // compute stream), so the weights the experts compute with are the same bytes: only the order of the uploads changes.
    // Each upload ahead waits (an event) for the misses the compute stream uploads for the copy issued before it, so the real misses go
    // first on the link. Victims (see ua_admit): free slots, then by the policy's order the slots of layer p+1 that no token predicts
    // and the slots of the layers already computed, then the later layers (never p, p+1's predicted ones, p+2, or a slot a pending copy
    // reads); a wrong guess frees its slot (the head of the list). Budget LLAMA_MOE_UPLOAD_AHEAD_BUDGET (1.5) x the misses of layer p.
    int     ua_mode   = 0;
    int     ua_topk   = 0;
    float   ua_budget = 1.5f;
    int     ua_log    = 0;
    int     ua_gating = 0;
    static constexpr int64_t ua_max_tokens = 255; // larger ubatches prefetch every expert (prefetch_min_batch = 256)
    static constexpr int     ua_n_events   = 16;
    struct ua_dev_t {
        ggml_backend_t up = nullptr; // second backend instance of the device: its stream uploads ahead
        std::vector<ggml_backend_event_t> ev;
        int ev_next = 0;
        ggml_backend_event_t pred_ev = nullptr;
        ggml_backend_event_t drain_ev = nullptr;
        ggml_backend_event_t mark_ev  = nullptr; // on the compute stream after the uploads of the last copy: the next upload ahead waits for it
        bool marked = false;
        ggml_backend_buffer_t buf_dev  = nullptr; // router logits [n_expert, ua_max_tokens] f32
        ggml_backend_buffer_t buf_host = nullptr;
        std::vector<float> host_vec;               // if no pinned host buffer
        float * host = nullptr;
        bool used = false;                         // uploads issued since the last drain
    };
    std::vector<ua_dev_t> ua_devs; // [device]
    struct ua_layer_t {
        uint64_t epoch = 0;
        std::vector<int32_t> experts, slots;  // admitted ahead of the routing (kept: the right ones after the routing is read)
        std::vector<int32_t> vkeys;           // [admitted] key of the expert its slot held (-1 = free)
        std::vector<int>     bank_ev;         // [bank] event of its upload, -1 = waited or none
        std::vector<uint8_t> bank_sent;       // [bank] uploaded for the admitted experts
        bool    resolved = false;             // the routing was read
    };
    std::vector<ua_layer_t> ua_l;
    std::vector<const ggml_tensor *> ua_router; // [n_layer] ffn_gate_inp on a device
    std::vector<std::vector<float>>  ua_b, ua_pb;
    ggml_context_ptr ua_ctx;
    uint64_t ua_epoch = 1;
    int32_t  ua_il    = -1; // layer of the current copies
    uint64_t ua_il_epoch = 0;
    int      ua_ncall = 0;  // expert tensors of ua_il copied so far
    int32_t  ua_pred_j = -1;
    int32_t  ua_pred_dev = -1;
    int64_t  ua_pred_nt = 0;
    int32_t  ua_miss_est = 0; // routed experts of ua_il the cache lacked before the uploads ahead
    std::vector<int32_t> ua_cnt;
    std::vector<float>   ua_w, ua_sc;
    std::vector<std::pair<float, int32_t>> ua_rank;
    std::vector<ggml_backend_copy_region> ua_regions;
    std::vector<int32_t> ua_want;
    std::vector<uint8_t> ua_prot;  // [expert] within the top k + ua_protect of some token: never a victim of its own layer
    int ua_protect = 4;
    std::vector<uint8_t> ua_busy;
    std::vector<int32_t> ua_donate;
    float ua_floor = 0.9f;
    bool  ua_keep_dec = false;
    bool  ua_dec_ref  = false;
    std::vector<std::pair<int64_t, int32_t>> ua_cand;
    int  ua_victims = 3;
    bool ua_verify  = false;
    uint64_t ua_verified = 0, ua_bad = 0;
    size_t   ua_up_last = 0;
    size_t   ua_copy_last = 0;
    uint64_t ua_dbg[5] = { 0, 0, 0, 0, 0 }; // copy_experts calls: all, dev_newer, no map, miss, hit (LLAMA_MOE_UPLOAD_AHEAD_LOG=3)
    bool ua_skip = false;        // no predictions in this ubatch (see prepare)
    int64_t ua_prev_tokens = 0;
    bool ua_off = false; // an upload could not be issued: no more uploads ahead (the admitted ones are dropped by the drain)
    struct ua_stats_t {
        uint64_t layers = 0, admitted = 0, hit = 0, waste = 0, lost = 0, miss_rest = 0, bytes = 0, waste_bytes = 0, fallback = 0, blocked = 0;
    } ua_st, ua_st_ub;

    void init_upload_ahead(const llama_model & model) {
        const char * env = getenv("LLAMA_MOE_UPLOAD_AHEAD");
        ua_mode = env != nullptr ? atoi(env) : 0; // opt-in: LLAMA_MOE_UPLOAD_AHEAD=1 (repeat-prompt decode loses 5-20% when on)
        const char * env_l0 = getenv("LLAMA_MOE_UPLOAD_AHEAD_LOG");
        ua_log = env_l0 != nullptr ? atoi(env_l0) : 0;
        if (ua_mode <= 0 || groups.empty()) {
            ua_mode = 0;
            return;
        }
        const char * env_k = getenv("LLAMA_MOE_UPLOAD_AHEAD_TOPK");
        ua_topk = env_k != nullptr ? std::max(1, atoi(env_k)) : (int) n_expert_used;
        const char * env_b = getenv("LLAMA_MOE_UPLOAD_AHEAD_BUDGET");
        ua_budget = env_b != nullptr ? (float) atof(env_b) : 1.5f;
        const char * env_v = getenv("LLAMA_MOE_UPLOAD_AHEAD_VICTIMS");
        ua_victims = env_v != nullptr ? atoi(env_v) : 3;
        const char * env_p = getenv("LLAMA_MOE_UPLOAD_AHEAD_PROTECT");
        ua_protect = env_p != nullptr ? std::max(0, atoi(env_p)) : 4;
        const char * env_f = getenv("LLAMA_MOE_UPLOAD_AHEAD_FLOOR");
        ua_floor = env_f != nullptr ? (float) atof(env_f) : 0.9f;
        const char * env_kd = getenv("LLAMA_MOE_UPLOAD_AHEAD_KEEP_DECODE");
        ua_keep_dec = env_kd != nullptr && atoi(env_kd) != 0;
        const char * env_dr = getenv("LLAMA_MOE_UPLOAD_AHEAD_DEC_REF");
        ua_dec_ref = env_dr != nullptr && atoi(env_dr) != 0;
        ua_verify  = getenv("LLAMA_MOE_UPLOAD_AHEAD_VERIFY") != nullptr && atoi(getenv("LLAMA_MOE_UPLOAD_AHEAD_VERIFY")) != 0;
        ua_gating = (int) model.hparams.expert_gating_func;
        ua_router.assign(layers.size(), nullptr);
        ua_b.assign(layers.size(), {});
        ua_pb.assign(layers.size(), {});
        ua_l.assign(layers.size(), {});
        int64_t n_expert_max = 0;
        size_t n_layers = 0;
        for (size_t il = 0; il < layers.size() && il < model.layers.size(); ++il) {
            if (layers[il].ig < 0 || host_layer((int32_t) il)) {
                continue;
            }
            const ggml_tensor * w = model.layers[il].ffn_gate_inp;
            const int64_t n_expert = layers[il].experts[0]->ne[2];
            if (w == nullptr || w->buffer == nullptr || ggml_backend_buffer_is_host(w->buffer) || ggml_n_dims(w) != 2 || w->ne[1] != n_expert) {
                continue;
            }
            auto get_bias = [&](const ggml_tensor * b, std::vector<float> & out) {
                if (b != nullptr && b->buffer != nullptr && b->type == GGML_TYPE_F32 && ggml_nelements(b) == n_expert && ggml_is_contiguous(b)) {
                    out.resize(n_expert);
                    ggml_backend_tensor_get(b, out.data(), 0, n_expert*sizeof(float));
                }
            };
            get_bias(model.layers[il].ffn_gate_inp_b, ua_b[il]);
            get_bias(model.layers[il].ffn_exp_probs_b, ua_pb[il]);
            ua_router[il] = w;
            ua_l[il].bank_ev.assign(layers[il].experts.size(), -1);
            ua_l[il].bank_sent.assign(layers[il].experts.size(), 0);
            n_expert_max = std::max(n_expert_max, n_expert);
            n_layers++;
        }
        if (n_layers < 2) {
            ua_mode = 0;
            return;
        }
        ua_devs.resize(devices.size());
        size_t n_dev = 0;
        for (size_t id = 0; id < devices.size(); ++id) {
            bool any = false;
            for (const group & g : groups) {
                any |= g.id == (int32_t) id && g.n_slots > 0;
            }
            ggml_backend_t backend = devices[id].backend;
            if (!any || !ggml_backend_tensor_copy_regions_async(backend, nullptr, 0)) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ua_dev_t & u = ua_devs[id];
            u.up = ggml_backend_dev_init(dev, nullptr);
            if (u.up == nullptr) {
                continue;
            }
            for (int i = 0; i < ua_n_events; ++i) {
                u.ev.push_back(ggml_backend_event_new(dev));
            }
            u.pred_ev  = ggml_backend_event_new(dev);
            u.drain_ev = ggml_backend_event_new(dev);
            u.mark_ev  = ggml_backend_event_new(dev);
            const size_t sz = (size_t) n_expert_max*ua_max_tokens*sizeof(float);
            u.buf_dev = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), sz);
            ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(dev);
            u.buf_host = hb != nullptr ? ggml_backend_buft_alloc_buffer(hb, sz) : nullptr;
            if (u.buf_host != nullptr) {
                u.host = (float *) ggml_backend_buffer_get_base(u.buf_host);
            } else {
                u.host_vec.resize(sz/sizeof(float));
                u.host = u.host_vec.data();
            }
            bool ok = u.buf_dev != nullptr && u.pred_ev != nullptr && u.drain_ev != nullptr && u.mark_ev != nullptr;
            for (auto * e : u.ev) {
                ok &= e != nullptr;
            }
            if (!ok) {
                ua_free_dev(u);
                continue;
            }
            n_dev++;
        }
        if (n_dev == 0) {
            ua_mode = 0;
            return;
        }
        ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead_custom(8, false), nullptr, true };
        ua_ctx.reset(ggml_init(ip));
        LLAMA_LOG_INFO("%s: MoE cache: uploads ahead of the routing for prompts of %" PRId64 "..%" PRId64 " tokens (next layer's router, top %d per token, "
                "budget %.1fx; LLAMA_MOE_UPLOAD_AHEAD=0 off), %zu layers\n", __func__, n_small_max + 1, ua_max_tokens, ua_topk, ua_budget, n_layers);
    }

    void ua_free_dev(ua_dev_t & u) {
        if (u.up != nullptr) {
            ggml_backend_synchronize(u.up);
        }
        for (auto * e : u.ev) {
            if (e != nullptr) {
                ggml_backend_event_free(e);
            }
        }
        u.ev.clear();
        if (u.pred_ev != nullptr) {
            ggml_backend_event_free(u.pred_ev);
        }
        if (u.drain_ev != nullptr) {
            ggml_backend_event_free(u.drain_ev);
        }
        if (u.mark_ev != nullptr) {
            ggml_backend_event_free(u.mark_ev);
        }
        u.pred_ev = u.drain_ev = u.mark_ev = nullptr;
        if (u.buf_dev != nullptr) {
            ggml_backend_buffer_free(u.buf_dev);
        }
        if (u.buf_host != nullptr) {
            ggml_backend_buffer_free(u.buf_host);
        }
        u.buf_dev = u.buf_host = nullptr;
        u.host = nullptr;
        if (u.up != nullptr) {
            ggml_backend_free(u.up);
        }
        u.up = nullptr;
    }

    // at the destruction: the context's backends may be gone, only the upload streams are waited for
    void ua_shutdown() {
        for (auto & u : ua_devs) {
            if (u.up != nullptr) {
                ggml_backend_synchronize(u.up);
            }
        }
        ua_st.layers += ua_st_ub.layers; ua_st.admitted += ua_st_ub.admitted; ua_st.hit += ua_st_ub.hit; ua_st.waste += ua_st_ub.waste;
        ua_st.lost += ua_st_ub.lost; ua_st.miss_rest += ua_st_ub.miss_rest; ua_st.bytes += ua_st_ub.bytes; ua_st.waste_bytes += ua_st_ub.waste_bytes;
        ua_st.fallback += ua_st_ub.fallback;
        ua_st_ub = {};
    }

    void ua_free() {
        for (auto & u : ua_devs) {
            ua_free_dev(u);
        }
        ua_devs.clear();
        ua_mode = 0;
    }

    bool ua_dev_ok(int32_t id) const {
        return ua_mode > 0 && id >= 0 && (size_t) id < ua_devs.size() && ua_devs[id].up != nullptr && !devices[id].released;
    }

    // the banks of device id are freed (release): the uploads ahead complete first, the admissions are dropped
    void ua_release(int32_t id) {
        if (!ua_dev_ok(id)) {
            return;
        }
        ggml_backend_synchronize(ua_devs[id].up);
        for (size_t j = 0; j < ua_l.size(); ++j) {
            if (ua_l[j].epoch != 0 && groups[layers[j].ig].id == id) {
                ua_drop(groups[layers[j].ig], (int32_t) j);
                ua_l[j].resolved = false;
                std::fill(ua_l[j].bank_ev.begin(), ua_l[j].bank_ev.end(), -1);
            }
        }
        ua_il = -1;
        ua_pred_j = -1;
    }

    // the router input of layer il: src[1] of the matmul with its router that the ids derive from
    const ggml_tensor * ua_router_input(int32_t il, const ggml_tensor * ids) const {
        const ggml_tensor * stack[64];
        int n = 0;
        int visited = 0;
        stack[n++] = ids;
        while (n > 0 && visited < 64) {
            const ggml_tensor * t = stack[--n];
            visited++;
            if (t->op == GGML_OP_MUL_MAT && t->src[0] == ua_router[il]) {
                return t->src[1];
            }
            for (int i = 0; i < GGML_MAX_SRC && n < 63; ++i) {
                if (t->src[i] != nullptr) {
                    stack[n++] = t->src[i];
                }
            }
            if (t->view_src != nullptr && n < 63) {
                stack[n++] = t->view_src;
            }
        }
        return nullptr;
    }

    // uploads bank ip of the experts admitted ahead to layer j on the upload stream (or on `on` when given)
    bool ua_send_bank(group & g, int32_t j, int ip, ggml_backend_t on) {
        ua_layer_t & L = ua_l[j];
        ua_dev_t & u = ua_devs[g.id];
        if (L.bank_sent[ip] || L.experts.empty()) {
            L.bank_sent[ip] = 1;
            return true;
        }
        const ggml_tensor * w = layers[j].experts[ip];
        ggml_tensor * bank = g.banks[ip];
        GGML_ASSERT(bank->nb[2] == w->nb[2]);
        const size_t nb = w->nb[2];
        ua_regions.clear();
        size_t bytes = 0;
        for (size_t i = 0; i < L.experts.size();) {
            size_t n = 1;
            while (i + n < L.experts.size() && L.experts[i + n] == L.experts[i] + (int32_t) n && L.slots[i + n] == L.slots[i] + (int32_t) n) {
                n++;
            }
            ua_regions.push_back({ w, (size_t) L.experts[i]*nb, bank, (size_t) L.slots[i]*nb, n*nb });
            bytes += n*nb;
            i += n;
        }
        if (on == nullptr && u.marked) {
            // after the misses the compute stream uploads for the copy just issued: they go first on the link
            ggml_backend_event_wait(u.up, u.mark_ev);
            u.marked = false;
        }
        if (!ggml_backend_tensor_copy_regions_async(on != nullptr ? on : u.up, ua_regions.data(), ua_regions.size())) {
            return false;
        }
        L.bank_sent[ip] = 1;
        ua_st_ub.bytes += bytes;
        if (on == nullptr) {
            const int e = u.ev_next++ % ua_n_events;
            ggml_backend_event_record(u.ev[e], u.up);
            L.bank_ev[ip] = e;
            u.used = true;
        } else {
            L.bank_ev[ip] = -1;
        }
        return true;
    }

    // drops the admissions of layer j (their slots are emptied and go to the head); the compute stream has waited for their uploads
    void ua_drop(group & g, int32_t j) {
        ua_layer_t & L = ua_l[j];
        const int32_t n_expert = g.lru.n_expert;
        for (size_t i = 0; i < L.experts.size(); ++i) {
            const int32_t s = L.slots[i];
            if (g.sd.il == j && !g.sd.old_slots.empty() && g.sd.old_slots[L.experts[i]] == s) {
                g.sd.old_slots[L.experts[i]] = -1; // the copies of the layer's other tensors must not read it
            }
            if (g.lru.key_of[s] == j*n_expert + L.experts[i]) {
                g.lru.slot_map[j][L.experts[i]] = -1;
                g.lru.key_of[s] = -1;
                g.lru.to_head(s);
            }
        }
        g.lru.gen++;
        L.experts.clear();
        L.slots.clear();
        L.vkeys.clear();
        L.epoch = 0;
    }

    // before a graph: the uploads ahead are complete on the compute stream, the admissions not resolved by a routing are dropped
    void ua_drain() {
        if (ua_mode <= 0) {
            if (ua_log > 1 && stats_upload.bytes != ua_up_last) {
                LLAMA_LOG_INFO("llama_moe_cache%s: uploads ahead off; copies uploaded %.1f MiB\n", tag, (stats_upload.bytes - ua_up_last)/1048576.0);
            }
            if (ua_log > 2) {
                for (const group & g : groups) {
                    const int64_t n_mapped = std::count_if(g.lru.key_of.begin(), g.lru.key_of.end(), [](int32_t k) { return k >= 0; });
                    LLAMA_LOG_INFO("llama_moe_cache%s: uploads ahead off: group of %zu layers: %" PRId64 " of %d slots mapped, %d lent, dev_newer %d; copied from the cache %.1f MiB, lends %d (max %d)\n", tag,
                        g.layers.size(), n_mapped, g.n_slots, g.lru.n_lent, (int) g.dev_newer, (stats_copy.bytes - ua_copy_last)/1048576.0, (int) n_lends, (int) lend_max);
                }
            }
            ua_up_last = stats_upload.bytes;
            ua_copy_last = stats_copy.bytes;
            return;
        }
        for (size_t id = 0; id < ua_devs.size(); ++id) {
            ua_dev_t & u = ua_devs[id];
            if (u.up == nullptr || !u.used) {
                continue;
            }
            ggml_backend_event_record(u.drain_ev, u.up);
            ggml_backend_event_wait(devices[id].backend, u.drain_ev);
            u.used = false;
        }
        for (size_t j = 0; j < ua_l.size(); ++j) {
            ua_layer_t & L = ua_l[j];
            if (L.epoch == 0) {
                continue;
            }
            group & g = groups[layers[j].ig];
            bool complete = L.resolved;
            for (size_t ip = 0; ip < L.bank_sent.size(); ++ip) {
                complete &= L.bank_sent[ip] != 0;
                L.bank_ev[ip] = -1;
            }
            if (!complete) {
                ua_st_ub.waste += L.experts.size();
                ua_drop(g, (int32_t) j);
            } else if (ua_verify) {
                // LLAMA_MOE_UPLOAD_AHEAD_VERIFY=1: every bank of the kept admissions holds the host bytes of its expert
                ggml_backend_synchronize(devices[g.id].backend);
                std::vector<uint8_t> buf;
                for (size_t i = 0; i < L.experts.size(); ++i) {
                    if (g.lru.key_of[L.slots[i]] != (int32_t) j*g.lru.n_expert + L.experts[i]) {
                        continue;
                    }
                    for (size_t ip = 0; ip < layers[j].experts.size(); ++ip) {
                        const ggml_tensor * w = layers[j].experts[ip];
                        const size_t nb = w->nb[2];
                        buf.resize(nb);
                        ggml_backend_tensor_get(g.banks[ip], buf.data(), (size_t) L.slots[i]*nb, nb);
                        ua_verified++;
                        ua_bad += memcmp(buf.data(), (const uint8_t *) w->data + (size_t) L.experts[i]*nb, nb) != 0;
                    }
                }
            }
            L.epoch = 0;
            L.resolved = false;
            L.experts.clear();
            L.slots.clear();
            L.vkeys.clear();
        }
        if (ua_verify) {
            for (group & g : groups) {
                if (g.n_slots > 0 && !g.dev_newer && ua_dev_ok(g.id)) {
                    verify(g, "after uploads ahead");
                }
            }
        }
        if (ua_verify && ua_verified > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: uploads ahead verified: %" PRIu64 " expert tensors, %" PRIu64 " mismatches\n", tag, ua_verified, ua_bad);
        }
        if (ua_log > 0 && (ua_st_ub.layers > 0 || (ua_log > 1 && stats_upload.bytes != ua_up_last))) {
            LLAMA_LOG_INFO("llama_moe_cache%s: uploads ahead: %" PRIu64 " layers, %" PRIu64 " admitted, %" PRIu64 " right, %" PRIu64 " wrong, %" PRIu64
                " hits lost, %" PRIu64 " misses left, %.1f MiB ahead, %.1f MiB wasted, %" PRIu64 " fallbacks; copies uploaded %.1f MiB\n", tag, ua_st_ub.layers, ua_st_ub.admitted,
                ua_st_ub.hit, ua_st_ub.waste, ua_st_ub.lost, ua_st_ub.miss_rest, ua_st_ub.bytes/1048576.0, ua_st_ub.waste_bytes/1048576.0,
                ua_st_ub.fallback, (stats_upload.bytes - ua_up_last)/1048576.0);
        }
        if (ua_log > 2) {
            for (const group & g : groups) {
                const int64_t n_mapped = std::count_if(g.lru.key_of.begin(), g.lru.key_of.end(), [](int32_t k) { return k >= 0; });
                {
                    std::string per;
                    for (int32_t il : g.layers) {
                        int c = 0;
                        for (int32_t e = 0; e < g.lru.n_expert; ++e) {
                            c += g.lru.slot_map[il][e] >= 0;
                        }
                        per += " " + std::to_string(c);
                    }
                    LLAMA_LOG_INFO("llama_moe_cache%s: uploads ahead: mapped per layer:%s\n", tag, per.c_str());
                }
                LLAMA_LOG_INFO("llama_moe_cache%s: uploads ahead: group of %zu layers: %" PRId64 " of %d slots mapped, %d lent, dev_newer %d, %" PRIu64 " reads blocked; copied from the cache %.1f MiB, lends %d (max %d); lookups %llu newer %llu nomap %llu miss %llu hit %llu\n", tag,
                    g.layers.size(), n_mapped, g.n_slots, g.lru.n_lent, (int) g.dev_newer, ua_st_ub.blocked, (stats_copy.bytes - ua_copy_last)/1048576.0, (int) n_lends, (int) lend_max,
                    (unsigned long long) ua_dbg[0], (unsigned long long) ua_dbg[1], (unsigned long long) ua_dbg[2], (unsigned long long) ua_dbg[3], (unsigned long long) ua_dbg[4]);
            }
        }
        ua_up_last = stats_upload.bytes;
        ua_copy_last = stats_copy.bytes;
        ua_st.layers += ua_st_ub.layers; ua_st.admitted += ua_st_ub.admitted; ua_st.hit += ua_st_ub.hit; ua_st.waste += ua_st_ub.waste;
        ua_st.lost += ua_st_ub.lost; ua_st.miss_rest += ua_st_ub.miss_rest; ua_st.bytes += ua_st_ub.bytes; ua_st.waste_bytes += ua_st_ub.waste_bytes;
        ua_st.fallback += ua_st_ub.fallback;
        ua_st_ub = {};
        ua_epoch++;
        ua_il = -1;
        ua_pred_j = -1;
    }

    // the copy of expert tensor w of a prompt ubatch, after its ids were read (counts: tokens per expert), before its copies
    void ua_begin(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * ids, const std::vector<int32_t> & counts) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return;
        }
        const binding & b = it->second;
        const int32_t il = b.il;
        group & g = groups[layers[il].ig];
        if (!ua_dev_ok(g.id) || ggml_backend_get_device(backend) != ggml_backend_get_device(devices[g.id].backend) || g.dev_newer ||
                (int64_t) counts.size() != g.lru.n_expert) {
            return;
        }
        ua_layer_t & L = ua_l[il];
        if (il != ua_il || ua_il_epoch != ua_epoch) {
            ua_il = il;
            ua_il_epoch = ua_epoch;
            ua_ncall = 0;
            ua_pred_j = -1;
            const int32_t n_expert = g.lru.n_expert;
            // the routing of il resolves its admissions: the wrong ones are dropped (the compute stream waits for their first upload
            // before it writes or reads any slot, see ua_wait: the copies of this tensor upload their misses first)
            if (L.epoch == ua_epoch && !L.resolved) {
                size_t o = 0;
                size_t n_wrong = 0;
                for (size_t i = 0; i < L.experts.size(); ++i) {
                    const int32_t e = L.experts[i];
                    const int32_t s = L.slots[i];
                    if (g.lru.key_of[s] != il*n_expert + e) {
                        continue; // reassigned since (not by a prompt copy: they never plan another layer)
                    }
                    const int32_t vk = L.vkeys[i];
                    if (vk >= 0 && vk / n_expert == il && counts[vk % n_expert] > 0) {
                        ua_st_ub.lost++;
                    }
                    if (counts[e] > 0) {
                        L.experts[o] = e;
                        L.slots[o] = s;
                        L.vkeys[o] = vk;
                        o++;
                        ua_st_ub.hit++;
                    } else {
                        g.lru.slot_map[il][e] = -1;
                        g.lru.key_of[s] = -1;
                        g.lru.to_head(s);
                        n_wrong++;
                    }
                }
                if (n_wrong > 0) {
                    g.lru.gen++;
                    size_t sent = 0;
                    for (size_t ip = 0; ip < L.bank_sent.size(); ++ip) {
                        sent += L.bank_sent[ip] ? layers[il].experts[ip]->nb[2] : 0;
                    }
                    ua_st_ub.waste += n_wrong;
                    ua_st_ub.waste_bytes += n_wrong*sent;
                }
                L.experts.resize(o);
                L.slots.resize(o);
                L.vkeys.resize(o);
                L.resolved = true;
            }
            const int32_t * slots = g.lru.slot_map[il];
            int32_t miss = 0;
            for (int32_t e = 0; e < n_expert; ++e) {
                miss += counts[e] > 0 && slots[e] < 0;
            }
            ua_st_ub.miss_rest += miss;
            ua_miss_est = miss + (L.epoch == ua_epoch ? (int32_t) L.experts.size() : 0);
            if (!ua_off && !ua_skip) {
                ua_predict(backend, g, il, ids);
            }
        }
        // the bank of this copy must hold the admitted experts: uploaded ahead (ua_wait) or, if not yet, on the compute stream now
        if (L.epoch == ua_epoch && L.resolved && b.ip < (int) L.bank_ev.size() && !L.bank_sent[b.ip]) {
            ua_st_ub.fallback++;
            if (!ua_send_bank(g, il, b.ip, backend)) {
                ua_drop(g, il); // the copies upload them instead
            }
        }
    }

    // the slot map of layer il is read by a lookup of the graph (not by a copy, see copy): its admissions ahead keep their slots, the
    // compute stream waits for the uploads ahead and uploads the banks not sent yet before the lookup reads them
    void ua_lookup_guard(ggml_backend_t backend, group & g, int32_t il) {
        if (ua_mode <= 0 || ua_l.empty() || (size_t) g.id >= ua_devs.size() || ua_devs[g.id].up == nullptr) {
            return;
        }
        ua_layer_t & L = ua_l[il];
        if (L.epoch != ua_epoch || L.experts.empty()) {
            return;
        }
        ua_dev_t & u = ua_devs[g.id];
        ggml_backend_event_record(u.drain_ev, u.up);
        ggml_backend_event_wait(backend, u.drain_ev);
        for (size_t ip = 0; ip < L.bank_sent.size(); ++ip) {
            L.bank_ev[ip] = -1;
            if (!L.bank_sent[ip] && !ua_send_bank(g, il, (int) ip, backend)) {
                ua_drop(g, il);
                return;
            }
        }
        ua_st_ub.fallback++;
        L.resolved = true;
    }

    // the uploads ahead of the layer of w that the compute stream has not waited for
    bool ua_pending(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        if (ua_mode <= 0 || it == bindings.end()) {
            return false;
        }
        const ua_layer_t & L = ua_l[it->second.il];
        return L.epoch == ua_epoch && std::any_of(L.bank_ev.begin(), L.bank_ev.end(), [](int e) { return e >= 0; });
    }

    // the uploads of the misses of w are queued on the compute stream: the next upload ahead goes after them
    void ua_mark(ggml_backend_t backend, const ggml_tensor * w) {
        const auto it = bindings.find(w);
        if (ua_mode <= 0 || ua_off || it == bindings.end()) {
            return;
        }
        const group & g = groups[layers[it->second.il].ig];
        if (!ua_dev_ok(g.id) || ua_devs[g.id].mark_ev == nullptr) {
            return;
        }
        ggml_backend_event_record(ua_devs[g.id].mark_ev, backend);
        ua_devs[g.id].marked = true;
    }

    // the compute stream waits for them: before the copies of w read the cache and before its seed writes slots
    void ua_wait(ggml_backend_t backend, const ggml_tensor * w) {
        const auto it = bindings.find(w);
        if (ua_mode <= 0 || it == bindings.end()) {
            return;
        }
        const int32_t il = it->second.il;
        ua_layer_t & L = ua_l[il];
        if (L.epoch != ua_epoch) {
            return;
        }
        const group & g = groups[layers[il].ig];
        for (size_t ip = 0; ip < L.bank_ev.size(); ++ip) {
            if (L.bank_ev[ip] >= 0) {
                ggml_backend_event_wait(backend, ua_devs[g.id].ev[L.bank_ev[ip]]);
                L.bank_ev[ip] = -1;
            }
        }
    }

    // the slots of layer il, bank ip may be read: no admission ahead of the routing whose bank is not uploaded
    bool ua_readable(int32_t il, int32_t ip) const {
        if (ua_mode <= 0 || ua_l.empty()) {
            return true;
        }
        const ua_layer_t & L = ua_l[il];
        return L.epoch != ua_epoch || L.experts.empty() || (L.resolved && L.bank_sent[ip]);
    }

    // the router of layer il+1 on the router input of layer il, on the device; read after the copies of il are issued
    void ua_predict(ggml_backend_t backend, group & g, int32_t il, const ggml_tensor * ids) {
        const int32_t j = il + 1;
        if ((size_t) j >= layers.size() || layers[j].ig != layers[il].ig || ua_router[j] == nullptr || ua_router[il] == nullptr || host_layer(j) ||
                g.host_read) {
            return; // (a graph that reads the group through lookups uploads its misses at each lookup)
        }
        const ggml_tensor * x = ua_router_input(il, ids);
        const ggml_tensor * wj = ua_router[j];
        if (x == nullptr || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != wj->ne[0] || ggml_nrows(x) != x->ne[1] ||
                x->ne[1] < 1 || x->ne[1] > ua_max_tokens || x->buffer == nullptr || x->data == nullptr || ggml_backend_buffer_is_host(x->buffer) ||
                !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(x->buffer)) ||
                !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(wj->buffer))) {
            return;
        }
        ua_dev_t & u = ua_devs[g.id];
        ggml_context * ctx = ua_ctx.get();
        ggml_reset(ctx);
        ggml_tensor * xv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1]);
        xv->buffer = x->buffer;
        xv->data   = x->data;
        ggml_tensor * l = ggml_mul_mat(ctx, const_cast<ggml_tensor *>(wj), xv);
        if (ggml_backend_tensor_alloc(u.buf_dev, l, ggml_backend_buffer_get_base(u.buf_dev)) != GGML_STATUS_SUCCESS) {
            return;
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(gf, l);
        if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) {
            return;
        }
        ggml_backend_tensor_get_async(backend, l, u.host, 0, ggml_nbytes(l));
        ggml_backend_event_record(u.pred_ev, backend);
        ua_pred_j   = j;
        ua_pred_dev = g.id;
        ua_pred_nt  = x->ne[1];
    }

    // after the copies (and the seed) of expert tensor w: the next bank of its layer's admissions, and after the last tensor of the
    // layer the admissions of the next layer from the prediction with their first bank
    void ua_end(ggml_backend_t backend, const ggml_tensor * w) {
        GGML_UNUSED(backend);
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return;
        }
        const binding & b = it->second;
        const int32_t il = b.il;
        if (il != ua_il || ua_il_epoch != ua_epoch) {
            return;
        }
        group & g = groups[layers[il].ig];
        if (!ua_dev_ok(g.id) || ua_off) {
            return; // (a bank not sent is uploaded on the compute stream before its copy, see ua_begin)
        }
        const int n_banks = (int) layers[il].experts.size();
        ua_ncall++;
        ua_layer_t & L = ua_l[il];
        if (L.epoch == ua_epoch && L.resolved) {
            // one bank ahead of its copy (the lowest one not sent)
            for (int ip = 0; ip < n_banks; ++ip) {
                if (!L.bank_sent[ip] && ip != b.ip) {
                    if (!ua_send_bank(g, il, ip, nullptr)) {
                        ua_drop(g, il);
                    }
                    break;
                }
            }
        }
        if (ua_ncall == n_banks && ua_pred_j == il + 1) {
            ua_admit(g, il);
        }
    }

    void ua_admit(group & g, int32_t il) {
        const int32_t j = ua_pred_j;
        ua_pred_j = -1;
        ua_dev_t & u = ua_devs[ua_pred_dev];
        ggml_backend_event_synchronize(u.pred_ev);
        const int32_t n_expert = g.lru.n_expert;
        const int64_t nt = ua_pred_nt;
        const int64_t k  = std::min<int64_t>(ua_topk, n_expert);
        ua_cnt.assign(n_expert, 0);
        ua_prot.assign(n_expert, 0);
        ua_w.assign(n_expert, 0.0f);
        ua_sc.resize(n_expert);
        ua_rank.resize(n_expert);
        const auto & bj  = ua_b[j];
        const auto & pbj = ua_pb[j];
        for (int64_t t = 0; t < nt; ++t) {
            const float * lg = u.host + t*n_expert;
            float mx = -INFINITY;
            for (int32_t e = 0; e < n_expert; ++e) {
                ua_sc[e] = lg[e] + (bj.empty() ? 0.0f : bj[e]);
                mx = std::max(mx, ua_sc[e]);
            }
            if (ua_gating == LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID) {
                for (int32_t e = 0; e < n_expert; ++e) {
                    ua_sc[e] = 1.0f/(1.0f + expf(-ua_sc[e]));
                }
            } else {
                float sum = 0.0f;
                for (int32_t e = 0; e < n_expert; ++e) {
                    ua_sc[e] = expf(ua_sc[e] - mx);
                    sum += ua_sc[e];
                }
                for (int32_t e = 0; e < n_expert; ++e) {
                    ua_sc[e] /= sum;
                }
            }
            for (int32_t e = 0; e < n_expert; ++e) {
                ua_rank[e] = { -(ua_sc[e] + (pbj.empty() ? 0.0f : pbj[e])), e };
            }
            const int64_t kp = std::min<int64_t>(k + ua_protect, n_expert);
            std::nth_element(ua_rank.begin(), ua_rank.begin() + (kp - 1), ua_rank.end());
            if (kp > k) {
                std::nth_element(ua_rank.begin(), ua_rank.begin() + (k - 1), ua_rank.begin() + kp);
            }
            for (int64_t i = 0; i < kp; ++i) {
                const int32_t e = ua_rank[i].second;
                if (i < k) {
                    ua_cnt[e]++;
                    ua_w[e] += ua_sc[e];
                }
                ua_prot[e] = 1;
            }
        }
        // the predicted experts the cache lacks, most tokens first
        int32_t * slots = g.lru.slot_map[j];
        std::vector<int32_t> & want = ua_want;
        want.clear();
        for (int32_t e = 0; e < n_expert; ++e) {
            if (ua_cnt[e] > 0) {
                want.push_back(e);
            }
        }
        want.erase(std::remove_if(want.begin(), want.end(), [&](int32_t e) { return slots[e] >= 0; }), want.end());
        std::sort(want.begin(), want.end(), [&](int32_t a, int32_t c) {
            return ua_cnt[a] != ua_cnt[c] ? ua_cnt[a] > ua_cnt[c] : ua_w[a] != ua_w[c] ? ua_w[a] > ua_w[c] : a < c;
        });
        const size_t budget = (size_t) std::max(0.0f, std::ceil(ua_budget*(float) ua_miss_est));
        if (want.size() > budget) {
            want.resize(budget);
        }
        if (want.empty()) {
            return;
        }
        // never a slot the copies of layer il read (its slots before the seed's plan)
        const int32_t * old = g.sd.il == il ? g.sd.old_slots.data() : g.lru.slot_map[il];
        std::vector<uint8_t> & busy = ua_busy;
        busy.assign(g.lru.n_slots, 0);
        for (int32_t e = 0; e < n_expert; ++e) {
            if (old != nullptr && old[e] >= 0) {
                busy[old[e]] = 1;
            }
        }
        // classes: 0 free slots; 1 the slots of layer j whose expert no token predicts within its top k + LLAMA_MOE_UPLOAD_AHEAD_PROTECT
        // (4) and the slots of the layers this ubatch computed already; 2 the layers after j+1. Within a class the policy's order (lowest
        // counter, then least recent). LLAMA_MOE_UPLOAD_AHEAD_VICTIMS=1: classes 0 and the layer j part of 1 only, 2: classes 0 and 1
        auto & cand = ua_cand;
        cand.clear();
        int64_t pos = 0;
        for (int32_t s = g.lru.head; s >= 0; s = g.lru.next[s], ++pos) {
            if (busy[s]) {
                continue;
            }
            const int32_t key = g.lru.key_of[s];
            if (ua_keep_dec && key >= 0 && !g.ua_dec_key.empty() && g.ua_dec_key[s] == key) {
                continue; // the last decode used it: the next one likely does too
            }
            const int64_t v = key >= 0 && g.lru.grow_q16 > 0 ? (int64_t) g.lru.scores[key] : 0;
            int64_t cls;
            if (key < 0) {
                cls = 0;
            } else if (key / n_expert == j) {
                if (ua_prot[key % n_expert]) {
                    continue;
                }
                cls = 1;
            } else if (ua_victims >= 4 && key / n_expert != il) {
                cls = 1; // r19 LLAMA_MOE_UPLOAD_AHEAD_VICTIMS=4: every other layer in one class, the policy's order alone (as a real miss)
            } else if (ua_victims >= 2 && ua_victims <= 3 && key / n_expert < il) {
                cls = 1;
            } else if (ua_victims == 3 && key / n_expert > j + 1) {
                cls = 2;
            } else {
                continue;
            }
            cand.push_back({ (cls << 58) + (v << 24) + pos, s });
        }
        if (cand.empty()) {
            return;
        }
        // another layer gives slots only down to LLAMA_MOE_UPLOAD_AHEAD_FLOOR (0.9) of its share: without the floor the later layers of a
        // ubatch took every slot of the first ones (0 of 99 left in layers 0-4 after a 184-token ubatch), and the large ubatch after it
        // re-seeded those layers by evicting the experts it was about to copy (uploads 17.1 instead of 11.4 GB)
        {
            const int32_t share = std::max<int32_t>(1, g.n_slots / (int32_t) std::max<size_t>(1, g.layers.size()));
            const int32_t floor_n = (int32_t) std::ceil(ua_floor*(float) share);
            ua_donate.assign(layers.size(), 0);
            for (int32_t q : g.layers) {
                int32_t c = 0;
                for (int32_t e = 0; e < n_expert; ++e) {
                    c += g.lru.slot_map[q][e] >= 0;
                }
                ua_donate[q] = std::max(0, c - floor_n);
            }
            std::sort(cand.begin(), cand.end());
            size_t o = 0;
            for (size_t i = 0; i < cand.size() && o < want.size(); ++i) {
                const int32_t key = g.lru.key_of[cand[i].second];
                const int32_t q = key >= 0 ? key / n_expert : -1;
                if (q >= 0 && q != j) {
                    if (ua_donate[q] <= 0) {
                        continue;
                    }
                    ua_donate[q]--;
                }
                cand[o++] = cand[i];
            }
            cand.resize(o);
        }
        if (cand.empty()) {
            return;
        }
        const size_t n = std::min(want.size(), cand.size());
        cand.resize(n);
        std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & c) { return a.second < c.second; });
        want.resize(n);
        std::sort(want.begin(), want.end());
        ua_layer_t & L = ua_l[j];
        L.experts.clear();
        L.slots.clear();
        L.vkeys.clear();
        for (size_t i = 0; i < n; ++i) {
            const int32_t s = cand[i].second;
            const int32_t key = g.lru.key_of[s];
            if (key >= 0) {
                g.lru.slot_map[key / n_expert][key % n_expert] = -1;
            }
            g.lru.key_of[s] = j*n_expert + want[i];
            slots[want[i]] = s;
            g.lru.touch(s);
            L.experts.push_back(want[i]);
            L.slots.push_back(s);
            L.vkeys.push_back(key);
        }
        g.lru.gen++;
        L.epoch = ua_epoch;
        L.resolved = false;
        std::fill(L.bank_sent.begin(), L.bank_sent.end(), 0);
        std::fill(L.bank_ev.begin(), L.bank_ev.end(), -1);
        ua_st_ub.layers++;
        ua_st_ub.admitted += n;
        if (!ua_send_bank(g, j, 0, nullptr)) {
            ua_drop(g, j);
            ua_off = true;
            LLAMA_LOG_WARN("llama_moe_cache%s: uploads ahead off: the experts cannot be copied asynchronously\n", tag);
        }
    }

    bool host_layer(int32_t il) const {
        return host && il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0 && host_group[layers[il].ig] && host->caches(il);
    }

    bool in_device(int32_t il, int32_t e) const {
        const group & g = groups[layers[il].ig];
        return g.lru.slot_map[il] != nullptr && g.lru.slot_map[il][e] >= 0;
    }

    // VERIFY: the bytes of expert e of tensor ip of layer il in slot s against the mapping (copied from it if they differ)
    void host_check(int32_t s, int32_t il, int32_t e, int ip, size_t off, size_t n) {
        const ggml_tensor * w = layers[il].experts[ip];
        const uint8_t * ref = (const uint8_t *) w->data + (size_t) e*w->nb[2] + off;
        uint8_t * p = host->data(s, ip) + off;
        hst.verify_regions++;
        hst.verify_bytes += n;
        if (memcmp(p, ref, n) != 0) {
            if (hst.verify_mismatch++ < 8) {
                LLAMA_LOG_ERROR("%s: VERIFY: RAM tier slot %d (layer %d, expert %d, tensor %d) differs from the mapping\n", __func__, s, il, e, ip);
            }
            memcpy(p, ref, n);
        }
    }

    int    host_quota_mode = 1;   // k19-l4held: LLAMA_MOE_HOST_LAYER_QUOTA
    size_t n_prefill_dup   = 0;   // k19-l4held: RAM copies of device-prefilled experts marked as duplicates

    static bool pad_zero_on() {
        static const bool on = [] { const char * e = getenv("LLAMA_MOE_PROMPT_PAD_ZERO"); return e == nullptr || atoi(e) != 0; }();
        return on;
    }
    uint64_t n_pad_zero = 0;

    void tier_runs(const ggml_tensor * src, size_t off, size_t size, bool lease, std::vector<llama_weight_tier::run> & out) {
        const auto it = bindings.find(src);
        if (it == bindings.end() || !host_layer(it->second.il)) {
            out.push_back({ off, size, nullptr });
            return;
        }
        const int32_t il = it->second.il;
        const int     ip = it->second.ip;
        const size_t  nb = src->nb[2];
        const size_t  end = off + size;
        for (size_t o = off; o < end; ) {
            const int32_t e  = (int32_t) (o / nb);
            const size_t  e1 = std::min(end, (size_t) (e + 1)*nb);
            // planning (under the store lock) does not wait for copies back into RAM, the upload does
            // k19-l4ttft: the experts the prompt layer read put in the scratch (uploads only)
            const uint8_t * pp = !lease && e < src->ne[2] ? plr_find(il, ip, e, o - (size_t) e*nb) : nullptr;
            if (pp != nullptr) {
                // the RAM tier is offered these bytes as the upload of a demand read offered them (same call, same order)
                tier_offer(src, o, e1 - o, pp);
                out.push_back({ o, e1 - o, pp, false });
                o = e1;
                continue;
            }
            const int32_t sl = e >= src->ne[2] ? -1 : lease ? host->held(il, e) : host->find(il, e, false);
            // k19-l4ttft (LLAMA_MOE_PROMPT_PAD_ZERO, default 1; 0 = read them): the copy of a run of routed experts ends with the first
            // min(expert, 512) bytes of the next expert, which the prompt does not route, as padding for the CUDA MMQ kernels (they read
            // past the last row of an expert: the bytes only have to be finite). Where neither RAM nor the scratch holds that expert, the
            // upload read them from the file: one 4 KiB read per run end, ~3900 per 42-token chat on the L4 disk tier (~1.1 s of its TTFT).
            // Zeros instead (a zero block is finite in every quant type); nothing is offered to the tier for them
            if (!lease && sl < 0 && pad_zero_on() && e < src->ne[2] && o == (size_t) e*nb && end - o <= 512 && end - o < nb) {
                static const uint8_t zeros[512] = {};
                out.push_back({ o, end - o, zeros, false });
                n_pad_zero++;
                o = end;
                continue;
            }
            if (sl >= 0) {
                if (lease) {
                    host->lease(sl);
                } else {
                    if (host_verify) {
                        host_check(sl, il, e, ip, o - (size_t) e*nb, e1 - o);
                    }
                    host->mark_read(sl);
                }
                out.push_back({ o, e1 - o, host->data(sl, ip) + (o - (size_t) e*nb), false });
                o = e1;
                continue;
            }
            // planning: the experts in the device slots are copied there (copy_experts takes every cached run of a tiered layer); the
            // first min(expert, 512) bytes after an uploaded run are still read ahead: the upload of a run copies them as padding
            const bool dev = lease && e < src->ne[2] && in_device(il, e);
            auto add = [&](size_t a0, size_t a1, bool d) {
                if (a1 <= a0) {
                    return;
                }
                if (!out.empty() && out.back().ptr == nullptr && out.back().device == d && out.back().off + out.back().size == a0) {
                    out.back().size += a1 - a0;
                } else {
                    out.push_back({ a0, a1 - a0, nullptr, d });
                }
            };
            if (dev && o == (size_t) e*nb && (out.empty() || !out.back().device)) {
                const size_t pad = std::min({ nb, (size_t) 512, e1 - o });
                add(o, o + pad, false);
                add(o + pad, e1, true);
            } else {
                add(o, e1, dev);
            }
            o = e1;
        }
    }

    // the uploads of a prompt fill free slots (never evicting): from the first expert tensor of a layer on, for the experts the device does not
    // cache when the tier is exclusive
    void tier_offer(const ggml_tensor * src, size_t off, size_t n, const uint8_t * p) {
        const auto it = bindings.find(src);
        if (it == bindings.end() || !host_layer(it->second.il)) {
            return;
        }
        const int32_t il = it->second.il;
        const int     ip = it->second.ip;
        const size_t  nb = src->nb[2];
        const size_t  end = std::min(off + n, (size_t) src->ne[2]*nb);
        for (size_t o = off; o < end; ) {
            const int32_t e  = (int32_t) (o / nb);
            const size_t  e1 = std::min(end, (size_t) (e + 1)*nb);
            const size_t  a  = o - (size_t) e*nb;
            int32_t sl = -1;
            if (host->contains(il, e)) {
                // held: nothing to do; being filled: continue it
                sl = host->find(il, e, false) >= 0 ? -1 : host->slot_of_key(il, e);
            } else if (ip == 0 && a == 0 && (e1 - o == nb || e1 - o > 512) && (!host->exclusive() || !in_device(il, e))) {
                // not from the bytes of padding uploaded after a run of experts (min(expert, 512)): that expert is not uploaded
                // k19-l4held rank mode: an expert the prompt routes may take the slot of an unrouted one of its layer at the quota
                const std::vector<bool> * rk = rank_any && (size_t) il < rank_mask.size() && (int32_t) rank_mask[il].size() > e &&
                    rank_mask[il][e] ? &rank_mask[il] : nullptr;
                // k19-autoprof: an expert outside the target set takes a free slot only while the layout keeps one per target expert
                // held nowhere, and never a guess
                bool take_spec = true;
                if (ap_offer_ok(il, e, take_spec)) {
                    sl = host->reserve(il, e, false, 0, take_spec, rk);
                    if (sl >= 0) {
                        ap_admitted(il, e);
                    }
                }
            }
            if (sl >= 0 && host->filling(sl) && host->fill_piece(sl, ip, a, p + (o - off), e1 - o)) {
                hst.admitted++;
                // k19-autoprof: a cold expert is a guess (the first a reservation takes)
                if (ap_use && !ap_target(il, e)) {
                    host->set_spec(sl);
                }
            }
            o = e1;
        }
    }

    // k19-l4ttft: prompt layer reads (LLAMA_MOE_PROMPT_LAYER_READ=1, opt-in: measured no faster on the L4 disk tier, chat TTFT 8.32 vs 6.81 s
    // median with it off, and its pinned scratch cost the 8k after it 103 vs 87 s; the per-byte read rate is the loop device's either way). A prompt ubatch whose expert copies read the routing (the copy callback's routed-only path, ubatches below the
    // prefetch size) used to read each expert tensor's misses on its own when its split started: the reads of the gate, up and down
    // tensors of a layer were three waits with the device work between them, the disk idle in between. Colab L4 full card, Qwen3.8-Flash-Next
    // UD-IQ4_XS from the VM disk (k19_l4ttft): the 42-token chat read 1.5-1.8 GB in 8.3-8.8 s with the disk busy 64-69% of it. Here
    // the copy of the layer's first expert tensor (the one that reads the routing) reads the routed experts of the layer that neither the
    // device slots nor the RAM tier hold, every expert tensor of each, in one batch into a pinned scratch (unbuffered, the store's IO
    // threads); the copies of all of the layer's expert tensors then upload those experts from it (tier_runs returns the scratch as held
    // bytes). The same bytes as the reads they replace, and each upload from the scratch offers them to the RAM tier as the upload of
    // a demand read did (free slots only, the same calls in the same order), so what each tier holds is unchanged: placement and text
    // as without it.
    // Scratch LLAMA_MOE_PROMPT_LAYER_READ_MIB (512), pinned at the first use only if llama_sysmem::available_reclaimable() keeps
    // LLAMA_MOE_PROMPT_LAYER_READ_FLOOR_MIB (4096) after it, else pageable; experts past it take the old per-tensor path.
    // LLAMA_MOE_PROMPT_DIAG=1 (log only): one line per prompt with the routed / device / RAM / read experts and the read time per layer
    struct plr_t {
        ggml_backend_buffer_t buf = nullptr;
        std::vector<uint8_t> pageable; // when the pinned scratch was refused
        uint8_t * base = nullptr;
        size_t    size = 0;
        bool      tried = false;
        int32_t   il = -1;                    // the layer the scratch holds
        std::vector<std::vector<int64_t>> off; // [ip][e] offset of the expert's bytes in the scratch, -1 none
        std::vector<int32_t> filled;          // experts of il in the scratch
        // per prompt (diag) and per process
        uint64_t p_layers = 0, p_used = 0, p_dev = 0, p_ram = 0, p_miss = 0, p_read = 0, p_over = 0, p_bytes = 0, p_us = 0;
        std::vector<int32_t> p_miss_layer;
        int64_t  p_t0 = 0;
        uint64_t p_sync_us = 0, p_up_us = 0, p_ncopy = 0; // host time in the routing reads (device work before them) and the copies' issue
        uint64_t p_x_us[3] = { 0, 0, 0 };                  // k19-l4held: whole copy callbacks, seeds, upload-ahead waits
        llama_weight_store::stats p_ws;                    // the store's counters at the first layer
        uint64_t layers = 0, experts = 0, bytes = 0, read_us = 0, over = 0, prompts = 0;
    } plr;
    static int plr_mode() {
        static const int m = [] { const char * e = getenv("LLAMA_MOE_PROMPT_LAYER_READ"); return e == nullptr ? 0 : atoi(e); }();
        return m;
    }
    static bool plr_diag() {
        static const bool d = [] { const char * e = getenv("LLAMA_MOE_PROMPT_DIAG"); return e != nullptr && atoi(e) != 0; }();
        return d;
    }

    bool plr_alloc(ggml_backend_t backend) {
        if (plr.tried) {
            return plr.base != nullptr;
        }
        plr.tried = true;
        const char * em = getenv("LLAMA_MOE_PROMPT_LAYER_READ_MIB");
        const char * ef = getenv("LLAMA_MOE_PROMPT_LAYER_READ_FLOOR_MIB");
        const size_t want  = (size_t) std::max<int64_t>(16, em != nullptr ? atoll(em) : 512) << 20;
        const size_t floor = (size_t) std::max<int64_t>(0, ef != nullptr ? atoll(ef) : 4096) << 20;
        const uint64_t recl = llama_sysmem::available_reclaimable();
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t hb = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (hb != nullptr && recl >= want + floor) {
            plr.buf = ggml_backend_buft_alloc_buffer(hb, want);
        }
        if (plr.buf != nullptr) {
            plr.base = (uint8_t *) ggml_backend_buffer_get_base(plr.buf);
            plr.size = want;
        } else if (recl >= want + floor / 2) {
            // pageable: slower uploads, but still one batch of reads per layer
            plr.pageable.resize(want + 4096);
            plr.base = (uint8_t *) (((uintptr_t) plr.pageable.data() + 4095) & ~(uintptr_t) 4095);
            plr.size = want;
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4ttft: prompt layer reads: %s (%.0f MiB scratch, %.0f MiB reclaimable, floor %.0f MiB; "
            "LLAMA_MOE_PROMPT_LAYER_READ=1 opt-in)\n", tag, plr.base == nullptr ? "off, no room" : plr.buf != nullptr ? "on, pinned" : "on, pageable",
            want/1048576.0, recl/1048576.0, floor/1048576.0);
        return plr.base != nullptr;
    }

    void plr_free() {
        if (plr.buf != nullptr) {
            ggml_backend_buffer_free(plr.buf);
            plr.buf = nullptr;
        }
        plr.pageable.clear();
        plr.pageable.shrink_to_fit();
        plr.base = nullptr;
        plr.il   = -1;
    }

    void plr_clear() {
        if (plr.il < 0) {
            return;
        }
        for (auto & v : plr.off) {
            for (int32_t e : plr.filled) {
                v[e] = -1;
            }
        }
        plr.filled.clear();
        plr.il = -1;
    }

    // the scratch bytes [o, o + n) of expert tensor ip of layer il, nullptr if the scratch does not hold that expert
    const uint8_t * plr_find(int32_t il, int ip, int32_t e, size_t a) const {
        if (plr.il != il || ip < 0 || ip >= (int) plr.off.size() || e < 0 || e >= (int32_t) plr.off[ip].size() || plr.off[ip][e] < 0) {
            return nullptr;
        }
        return plr.base + plr.off[ip][e] + a;
    }

    // k19-l4held rank mode (see llama_moe_cache::prompt_rank_copy)
    std::vector<std::vector<bool>> rank_mask; // [il] experts the current ubatch routes (empty: none)
    bool     rank_any = false;
    uint64_t rank_layers = 0, rank_routed = 0, rank_extra = 0, rank_skipped = 0;
    static int rank_mode() {
        static const int m = [] { const char * e = getenv("LLAMA_MOE_PROMPT_RANK"); return e == nullptr ? 0 : atoi(e); }();
        return m;
    }
    void rank_clear() {
        if (rank_any) {
            for (auto & m : rank_mask) {
                m.clear();
            }
            rank_any = false;
        }
    }
    bool prompt_rank_copy(const ggml_tensor * w, std::vector<bool> & used) {
        if (rank_mode() == 0 || !host) {
            return false;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end() || !host_layer(it->second.il)) {
            return false;
        }
        const int32_t il = it->second.il;
        const int32_t n_expert = (int32_t) w->ne[2];
        if ((int32_t) used.size() != n_expert) {
            return false;
        }
        if (rank_mask.size() != layers.size()) {
            rank_mask.resize(layers.size());
        }
        rank_mask[il] = used;
        rank_any = true;
        // the layer's room under its quota goes to the routed experts held nowhere first, then to the unrouted ones (lowest ids)
        int64_t room = host->layer_quota_on() ? (int64_t) host->quota_of(il) - host->n_keys_of(il) : INT64_MAX/2;
        int64_t n_routed_miss = 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (used[e] && !in_device(il, e) && !host->contains(il, e)) {
                n_routed_miss++;
            }
        }
        int64_t extra = room - n_routed_miss;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (used[e] || in_device(il, e) || host->contains(il, e)) {
                continue;
            }
            if (extra > 0) {
                used[e] = true;
                extra--;
                rank_extra++;
            } else {
                rank_skipped++;
            }
        }
        rank_layers++;
        rank_routed += n_routed_miss;
        return true;
    }

    void prompt_layer_read(ggml_backend_t backend, const ggml_tensor * w, const std::vector<bool> & used) {
        const int mode = plr_mode();
        const bool diag = plr_diag();
        if ((mode == 0 && !diag) || wstore == nullptr) {
            return;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end() || !host_layer(it->second.il)) {
            return;
        }
        const int32_t il = it->second.il;
        const layer & l = layers[il];
        const int32_t n_expert = (int32_t) w->ne[2];
        if ((int32_t) used.size() != n_expert) {
            return;
        }
        if (plr.p_t0 == 0) {
            plr.p_t0 = ggml_time_us();
            plr.p_ws = wstore->get_stats();
            plr.p_miss_layer.assign(layers.size(), -1);
            // k19-l4held: the RAM tier as the prompt finds it (free slots, duplicates of device-held experts)
            if (diag && host) {
                uint64_t n_dup = 0, n_held = 0;
                for (size_t jl = 0; jl < layers.size(); ++jl) {
                    if (!host_layer((int32_t) jl) || layers[jl].experts.empty()) {
                        continue;
                    }
                    for (int32_t e = 0; e < (int32_t) layers[jl].experts[0]->ne[2]; ++e) {
                        const bool r = host->held((int32_t) jl, e) >= 0;
                        n_held += r;
                        n_dup  += r && in_device((int32_t) jl, e);
                    }
                }
                LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4held prompt start: RAM tier %llu held (%llu also on the device) of %d slots, %d free\n", tag,
                    (unsigned long long) n_held, (unsigned long long) n_dup, host->n_slots(), host->n_free());
            }
        }
        std::vector<int32_t> miss;
        uint64_t n_used = 0, n_dev = 0, n_ram = 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (!used[e]) {
                continue;
            }
            n_used++;
            if (in_device(il, e)) {
                n_dev++;
            } else if (host->contains(il, e)) {
                n_ram++;
            } else {
                miss.push_back(e);
            }
        }
        plr.p_layers++;
        plr.p_used += n_used;
        plr.p_dev  += n_dev;
        plr.p_ram  += n_ram;
        plr.p_miss += miss.size();
        if (il < (int32_t) plr.p_miss_layer.size()) {
            plr.p_miss_layer[il] = (int32_t) miss.size();
        }
        if (mode == 0 || miss.empty()) {
            plr_clear();
            return;
        }
        if (!plr_alloc(backend)) {
            return;
        }
        // the uploads from the scratch of the previous layer were queued on this stream (the routing read synchronized it already)
        ggml_backend_synchronize(backend);
        plr_clear();
        if (plr.off.size() < l.experts.size()) {
            plr.off.resize(l.experts.size());
        }
        for (auto & v : plr.off) {
            if ((int32_t) v.size() < n_expert) {
                v.assign(n_expert, -1);
            }
        }
        std::vector<llama_weight_store::host_read> rd;
        rd.reserve(miss.size()*l.experts.size());
        size_t pos = 0;
        size_t n_fit = 0;
        for (int32_t e : miss) {
            size_t need = 0;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                need += (l.experts[ip]->nb[2] + 8192 + 4095) & ~(size_t) 4095;
            }
            if (pos + need > plr.size) {
                break;
            }
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * t = l.experts[ip];
                const size_t nb = t->nb[2];
                rd.push_back({ t, (size_t) e*nb, nb, plr.base + pos, 0 });
                pos += (nb + 8192 + 4095) & ~(size_t) 4095;
            }
            n_fit++;
        }
        plr.p_over += miss.size() - n_fit;
        plr.over   += miss.size() - n_fit;
        if (n_fit == 0) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        if (!wstore->read_host(rd.data(), rd.size(), true)) {
            return; // nothing held: the copies read on demand as before
        }
        const int64_t dt = ggml_time_us() - t0;
        size_t k = 0;
        uint64_t bytes = 0;
        for (size_t i = 0; i < n_fit; ++i) {
            const int32_t e = miss[i];
            for (size_t ip = 0; ip < l.experts.size(); ++ip, ++k) {
                plr.off[ip][e] = (int64_t) (rd[k].area - plr.base) + (int64_t) rd[k].lead;
                bytes += rd[k].size;
            }
            plr.filled.push_back(e);
        }
        plr.il = il;
        plr.p_read  += n_fit;
        plr.p_bytes += bytes;
        plr.p_us    += dt;
        plr.layers++;
        plr.experts += n_fit;
        plr.bytes   += bytes;
        plr.read_us += dt;
    }

    // the copy callback's host time of a prompt ubatch's expert tensor: the routing read (sync_us: waits for the device work queued
    // before it) and the copies' issue (up_us: the plan, the RAM uploads and the per-tensor demand reads), LLAMA_MOE_PROMPT_DIAG only
    void prompt_diag_time(int64_t sync_us, int64_t up_us) {
        if (plr.p_t0 == 0) {
            return;
        }
        plr.p_sync_us += sync_us;
        plr.p_up_us   += up_us;
        plr.p_ncopy++;
    }
    void prompt_diag_extra(int k, int64_t us) {
        if (plr.p_t0 == 0 || k < 0 || k > 2) {
            return;
        }
        plr.p_x_us[k] += us;
    }

    // the end of a prompt (the first small batch after large ones): the scratch is dropped, the diagnostic line
    void plr_prompt_end() {
        plr_clear();
        if (plr.p_t0 == 0) {
            return;
        }
        plr.prompts++;
        if (plr_diag()) {
            std::string ml;
            for (size_t il = 0; il < plr.p_miss_layer.size(); ++il) {
                if (plr.p_miss_layer[il] >= 0) {
                    ml += (ml.empty() ? "" : " ") + std::to_string(plr.p_miss_layer[il]);
                }
            }
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4ttft prompt %llu: %llu layers, routed %llu = device %llu + RAM %llu + missing %llu; read in layer "
                "batches %llu experts (%.1f MiB) in %.1f ms, %llu past the scratch; %.1f ms since the first layer; missing per layer: %s\n", tag,
                (unsigned long long) plr.prompts, (unsigned long long) plr.p_layers, (unsigned long long) plr.p_used, (unsigned long long) plr.p_dev,
                (unsigned long long) plr.p_ram, (unsigned long long) plr.p_miss, (unsigned long long) plr.p_read, plr.p_bytes/1048576.0, plr.p_us/1000.0,
                (unsigned long long) plr.p_over, (ggml_time_us() - plr.p_t0)/1000.0, ml.c_str());
            const llama_weight_store::stats w = wstore != nullptr ? wstore->get_stats() : plr.p_ws;
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4ttft prompt %llu: %llu copies: routing reads waited %.1f ms, copies issued in %.1f ms; store: "
                "%.1f MiB uploaded from the RAM tier, %.1f MiB demand reads (%llu regions), %.1f MiB host reads, %.1f MiB ring reads, waited %.1f ms\n", tag,
                (unsigned long long) plr.prompts, (unsigned long long) plr.p_ncopy, plr.p_sync_us/1000.0, plr.p_up_us/1000.0,
                (w.tier_uploaded - plr.p_ws.tier_uploaded)/1048576.0, (w.demand_bytes - plr.p_ws.demand_bytes)/1048576.0,
                (unsigned long long) (w.demand_regions - plr.p_ws.demand_regions), (w.host_read_bytes - plr.p_ws.host_read_bytes)/1048576.0,
                (w.plan_bytes - plr.p_ws.plan_bytes)/1048576.0, (w.wait_us - plr.p_ws.wait_us)/1000.0);
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4held prompt %llu: copy callbacks %.1f ms in all, seeds %.1f ms, upload-ahead %.1f ms\n", tag,
                (unsigned long long) plr.prompts, plr.p_x_us[0]/1000.0, plr.p_x_us[1]/1000.0, plr.p_x_us[2]/1000.0);
            // where the experts are after the prompt: device / RAM / both / neither, per layer (neither) and the layer quota's counters
            if (host) {
                uint64_t n_d = 0, n_r = 0, n_b = 0, n_n = 0;
                std::string nl, rl;
                for (size_t il = 0; il < layers.size(); ++il) {
                    if (!host_layer((int32_t) il) || layers[il].experts.empty()) {
                        continue;
                    }
                    const int32_t ne = (int32_t) layers[il].experts[0]->ne[2];
                    int32_t none = 0, ram = 0;
                    for (int32_t e = 0; e < ne; ++e) {
                        const bool d = in_device((int32_t) il, e);
                        const bool r = host->held((int32_t) il, e) >= 0;
                        n_d += d && !r; n_r += r && !d; n_b += d && r; n_n += !d && !r;
                        none += !d && !r;
                        ram  += r;
                    }
                    nl += (nl.empty() ? "" : " ") + std::to_string(none);
                    rl += (rl.empty() ? "" : " ") + std::to_string(ram);
                }
                LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4held prompt %llu: held after it: device only %llu, RAM only %llu, both %llu, neither %llu; "
                    "quota %s (refused %llu, same-layer %llu (%llu ranked), slack %llu); rank mode: %llu layers, %llu routed misses, %llu extra staged, %llu not staged; "
                    "neither per layer: %s; RAM per layer: %s\n", tag, (unsigned long long) plr.prompts,
                    (unsigned long long) n_d, (unsigned long long) n_r, (unsigned long long) n_b, (unsigned long long) n_n,
                    host->layer_quota_on() ? "on" : "off", (unsigned long long) host->quota_refused, (unsigned long long) host->quota_same,
                    (unsigned long long) host->quota_ranked, (unsigned long long) host->quota_slack, (unsigned long long) rank_layers,
                    (unsigned long long) rank_routed, (unsigned long long) rank_extra, (unsigned long long) rank_skipped, nl.c_str(), rl.c_str());
            }
        }
        plr.p_x_us[0] = plr.p_x_us[1] = plr.p_x_us[2] = 0;
        plr.p_sync_us = plr.p_up_us = plr.p_ncopy = 0;
        plr.p_layers = plr.p_used = plr.p_dev = plr.p_ram = plr.p_miss = plr.p_read = plr.p_over = plr.p_bytes = plr.p_us = 0;
        plr.p_t0 = 0;
    }

    // k19-d2hstream: the copies back of a host lookup (exclusive tier) go out on a second stream of the device, in chunks with an event
    // each, and the uploads into the evicted slots of a chunk wait for its event only: the D2H of chunk c+1 runs beside the H2D of chunk
    // c (PCIe is full duplex; on one stream every copy back finished before the first upload began). The side stream starts where the
    // compute stream is (event), every upload into a slot waits for that slot's copy back, and the compute stream has waited for every
    // chunk before the RAM tier commits, so the bytes, slots and their order per slot are unchanged: exact by construction.
    // LLAMA_MOE_D2H_STREAM=1: on (k19-d2h2: opt-in; unset / 0: on the compute stream as before); LLAMA_MOE_D2H_STREAM_CHUNKS=N (default 4, 2..16)
    struct d2h_dev_t {
        bool tried = false;
        ggml_backend_t be = nullptr;
        ggml_backend_event_t start = nullptr;
        std::vector<ggml_backend_event_t> ev;
    };
    std::vector<d2h_dev_t> d2h_devs;
    // (k19-d2h2: opt-in, LLAMA_MOE_D2H_STREAM=1; no gain on Kaggle 2x T4, being measured on the L4's full-duplex link)
    bool d2h_on = [] {
        const char * e = getenv("LLAMA_MOE_D2H_STREAM");
        return e != nullptr && atoi(e) != 0;
    }();
    int d2h_nchunk = [] {
        const char * e = getenv("LLAMA_MOE_D2H_STREAM_CHUNKS");
        return e != nullptr ? std::min(16, std::max(2, atoi(e))) : 4;
    }();
    struct { uint64_t calls = 0, chunks = 0, uploads = 0, early = 0, fail = 0; } d2hs;
    std::vector<int8_t> d2h_slot_chunk; // [slot of the group being uploaded] chunk of its copy back, -1 = none
    std::vector<std::pair<int32_t, int32_t>> d2h_sub;

    d2h_dev_t * d2h_get(int32_t id, ggml_backend_t backend) {
        if (d2h_devs.size() < devices.size()) {
            d2h_devs.resize(devices.size());
        }
        d2h_dev_t & d = d2h_devs[id];
        if (!d.tried) {
            d.tried = true;
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            d.be = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
            bool ok = d.be != nullptr;
            if (ok) {
                d.start = ggml_backend_event_new(dev);
                ok = d.start != nullptr;
                for (int i = 0; i < d2h_nchunk && ok; ++i) {
                    d.ev.push_back(ggml_backend_event_new(dev));
                    ok = d.ev.back() != nullptr;
                }
            }
            if (!ok) {
                for (auto * e : d.ev) {
                    if (e) {
                        ggml_backend_event_free(e);
                    }
                }
                d.ev.clear();
                if (d.start) {
                    ggml_backend_event_free(d.start);
                    d.start = nullptr;
                }
                if (d.be) {
                    ggml_backend_free(d.be);
                    d.be = nullptr;
                }
                d2hs.fail++;
                LLAMA_LOG_WARN("%s: MoE cache: no second stream / events on device %d: copies back stay on the compute stream\n", __func__, id);
            }
        }
        return d.be != nullptr ? &d : nullptr;
    }

    // exclusive: the experts evicted from device slots are copied back into RAM before the slots are overwritten (same backend, in order)
    void demote(ggml_backend_t backend, group & g, const std::vector<std::pair<int32_t, int32_t>> & ev) {
        const int32_t n_expert = g.lru.n_expert;
        for (const auto & [s, key] : ev) {
            const int32_t il = key / n_expert;
            const int32_t e  = key % n_expert;
            if (!host_layer(il)) {
                continue;
            }
            if (host->contains(il, e)) {
                // RAM kept a copy when it was promoted: now its only copy, no copy back
                sss.dem_kept++;
                const int32_t h = host->held(il, e);
                if (h >= 0) {
                    host->set_dup(h, false);
                    const layer & l = layers[il];
                    if (copyback_skip) {
                        hst.demote_kept++;
                        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                            hst.demote_kept_bytes += l.experts[ip]->nb[2];
                        }
                    } else {
                        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                            const size_t nb = l.experts[ip]->nb[2];
                            ggml_backend_tensor_get_async(backend, g.banks[ip], host->data(h, (int) ip), (size_t) s*nb, nb);
                            hst.demote_forced_bytes += nb;
                        }
                        host->mark_write(h);
                        hst.demote_forced++;
                    }
                }
                continue;
            }
            // k19-l4held rank mode: at the layer's quota the copy back takes an expert the prompt does not route first
            const std::vector<bool> * rk = rank_any && (size_t) il < rank_mask.size() && (int32_t) rank_mask[il].size() > e ? &rank_mask[il] : nullptr;
            const uint64_t ev0 = host->st.evicted - host->st.dup_taken;
            const int32_t r = host->reserve(il, e, true, 0, true, rk);
            if (r < 0) {
                hst.demote_skip++;
                sss.dem_none++;
                continue;
            }
            if (host->st.evicted - host->st.dup_taken > ev0) {
                sss.dem_evict++;
            } else {
                sss.dem_free++;
            }
            const layer & l = layers[il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const size_t nb = l.experts[ip]->nb[2];
                host->set_lead(r, (int) ip, 0);
                ggml_backend_tensor_get_async(backend, g.banks[ip], host->area(r, (int) ip), (size_t) s*nb, nb);
                hst.demote_bytes += nb;
            }
            host->mark_write(r);
            hst.demoted++;
        }
    }

    // uploads the experts the device plan admitted to layer il: from the RAM tier where it holds them (exclusive: they leave it), else read into a
    // RAM slot first (inclusive), else read by the weight store (or copied from the mapping); returns the bytes uploaded
    // strata-gap: the uploads of missing experts (the refill of lent slots after a prompt, seeds, host lookups) go out as one batched copy
    // per call (cudaMemcpyBatchAsync) instead of one cudaMemcpyAsync per expert run: the refill after an 8k Flash-Next prompt is ~3800
    // copies of 0.3-1.7 MB, which reached ~40 GB/s busy (B200 nsys) against 52 GB/s for the batched prompt copies. Same bytes, same slots,
    // same stream order. LLAMA_MOE_FILL_BATCH=0: one copy per run, as before
    bool hprof_on = false; // upload_fills of a profiled host lookup (LLAMA_MOE_HOST_PROF)
    std::vector<ggml_backend_copy_region> fill_regions;
    bool fill_batch = [] {
        const char * env = getenv("LLAMA_MOE_FILL_BATCH");
        return env == nullptr || atoi(env) != 0;
    }();
    size_t n_fill_batches = 0;
    void flush_fill_regions(ggml_backend_t backend) {
        if (fill_regions.empty()) {
            return;
        }
        if (ggml_backend_tensor_copy_regions_async(backend, fill_regions.data(), fill_regions.size())) {
            n_fill_batches++;
        } else {
            for (const auto & r : fill_regions) {
                ggml_backend_tensor_set_async(backend, r.dst, (const uint8_t *) r.src->data + r.src_offset, r.dst_offset, r.size);
            }
        }
        fill_regions.clear();
    }

    size_t upload_fills(ggml_backend_t backend, int32_t il, const std::vector<moe_cache_lru::fill> & fl, const std::vector<std::pair<int32_t, int32_t>> * ev) {
        const layer & l = layers[il];
        group & g = groups[l.ig];
        GGML_ASSERT(!devices[g.id].released && "MoE cache: an upload while the banks are released");
        size_t bytes = 0;
        const std::vector<moe_cache_lru::fill> * rest = &fl;
        if (host_layer(il)) {
            sss.empty_fills += fl.size() > (ev != nullptr ? ev->size() : 0) ? fl.size() - (ev != nullptr ? ev->size() : 0) : 0;
            fills_rest.clear();
            host_reads.clear();
            host_pending.clear();
            for (const auto & f : fl) {
                const int32_t s = host->find(il, f.expert, false);
                if (s >= 0) {
                    host->pin(s); // the reads into RAM below never evict a fill that RAM holds
                }
            }
            // exclusive: the evicted experts move to RAM before their device slots are overwritten (same backend, in order)
            int64_t tp = hprof_on ? ggml_time_us() : 0;
            d2h_dev_t * dd = nullptr;
            int n_ck = 0;     // chunks of copies back on the side stream (0: on the compute stream)
            int waited = -1;  // the last chunk event the compute stream waits for
            if (ev != nullptr && host->exclusive()) {
                if (d2h_on && ev->size() >= 2 && (dd = d2h_get(g.id, backend)) != nullptr) {
                    n_ck = (int) std::min<size_t>((size_t) d2h_nchunk, ev->size());
                    if ((int32_t) d2h_slot_chunk.size() < g.n_slots) {
                        d2h_slot_chunk.assign(g.n_slots, -1);
                    }
                    ggml_backend_event_record(dd->start, backend);
                    ggml_backend_event_wait(dd->be, dd->start);
                    const size_t n_ev = ev->size();
                    for (int c = 0; c < n_ck; ++c) {
                        d2h_sub.assign(ev->begin() + c*n_ev/n_ck, ev->begin() + (c + 1)*n_ev/n_ck);
                        for (const auto & pr : d2h_sub) {
                            d2h_slot_chunk[pr.first] = (int8_t) c;
                        }
                        demote(dd->be, g, d2h_sub);
                        ggml_backend_event_record(dd->ev[c], dd->be);
                    }
                    d2hs.calls++;
                    d2hs.chunks += n_ck;
                } else {
                    demote(backend, g, *ev);
                }
            }
            auto d2h_wait = [&](int upto) {
                for (; waited < upto; ++waited) {
                    ggml_backend_event_wait(backend, dd->ev[waited + 1]);
                }
            };
            if (hprof_on) {
                const int64_t t = ggml_time_us();
                hprof.demote_us += t - tp;
                tp = t;
            }
            for (const auto & f : fl) {
                const int32_t s = host->find(il, f.expert);
                if (s >= 0) {
                    if (n_ck > 0) {
                        const int c = f.slot >= 0 && f.slot < (int32_t) d2h_slot_chunk.size() ? d2h_slot_chunk[f.slot] : -1;
                        d2h_wait(c);
                        d2hs.uploads++;
                        d2hs.early += waited < n_ck - 1;
                    }
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const size_t nb = l.experts[ip]->nb[2];
                        if (host_verify) {
                            host_check(s, il, f.expert, (int) ip, 0, nb);
                        }
                        ggml_backend_tensor_set_async(backend, g.banks[ip], host->data(s, (int) ip), (size_t) f.slot*nb, nb);
                        bytes += nb;
                        hst.up_bytes += nb;
                    }
                    host->mark_read(s);
                    hst.hits++;
                    if (!ahead_key.empty() && ahead_key[s] >= 0) {
                        ast.used += ahead_key[s] == host->key(s);
                        ahead_key[s] = -1;
                    }
                    if (host->exclusive()) {
                        if (host_spare) {
                            // promoted: the copy stays in its slot as the first to go (a device eviction then needs no copy back)
                            host->set_dup(s, true);
                        } else {
                            host->release(s);
                        }
                        hst.promoted++;
                        if (ev != nullptr) {
                            sss.prom_ev++;
                        } else {
                            sss.prom_noev++;
                        }
                    }
                    continue;
                }
                hst.misses++;
                // exclusive with spare room: into a free slot only (a copy of an expert the device is about to hold never evicts another)
                const int32_t r = !host->exclusive() ? host->reserve(il, f.expert) : host_spare ? host->reserve(il, f.expert, false) : -1;
                if (r >= 0) {
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const ggml_tensor * w = l.experts[ip];
                        host_reads.push_back({ w, (size_t) f.expert*w->nb[2], w->nb[2], host->area(r, (int) ip), 0 });
                    }
                    host_pending.emplace_back(r, f);
                    continue;
                }
                fills_rest.push_back(f);
            }
            if (n_ck > 0) {
                // every other write into the slots (reads from the files, the rest below) and the RAM commit after all copies back
                d2h_wait(n_ck - 1);
                for (const auto & pr : *ev) {
                    d2h_slot_chunk[pr.first] = -1;
                }
            }
            if (!host_pending.empty()) {
                if (wstore->read_host(host_reads.data(), host_reads.size(), true)) {
                    size_t k = 0;
                    for (const auto & [r, f] : host_pending) {
                        for (size_t ip = 0; ip < l.experts.size(); ++ip, ++k) {
                            const size_t nb = l.experts[ip]->nb[2];
                            host->set_lead(r, (int) ip, host_reads[k].lead);
                            ggml_backend_tensor_set_async(backend, g.banks[ip], host->data(r, (int) ip), (size_t) f.slot*nb, nb);
                            bytes += nb;
                            hst.read_bytes += nb;
                        }
                        host->publish(r);
                        host->mark_read(r);
                        if (host->exclusive()) {
                            host->set_dup(r, true);
                        }
                        hst.read++;
                    }
                } else {
                    for (const auto & [r, f] : host_pending) {
                        host->abort(r);
                        fills_rest.push_back(f);
                    }
                }
            }
            if (hprof_on) {
                const int64_t t = ggml_time_us();
                hprof.ram_us += t - tp;
                tp = t;
            }
            host->commit(backend);
            if (hprof_on) {
                hprof.commit_us += ggml_time_us() - tp;
            }
            rest = &fills_rest;
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        // with the weight store, the reads of all of them from the model files are issued at once, then the uploads
        ws_regions.clear();
        bool ws = wstore != nullptr && !rest->empty();
        for (size_t ip = 0; ip < l.experts.size() && ws; ++ip) {
            ws = wstore->covers(l.experts[ip]);
        }
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            const auto & fr = *rest;
            for (size_t i = 0; i < fr.size();) {
                size_t n = 1;
                while (i + n < fr.size() && fr[i + n].expert == fr[i].expert + (int32_t) n && fr[i + n].slot == fr[i].slot + (int32_t) n) {
                    n++;
                }
                if (ws) {
                    ws_regions.push_back({ w, fr[i].expert*expert_size, bank, fr[i].slot*expert_size, n*expert_size });
                } else if (fill_batch) {
                    fill_regions.push_back({ w, fr[i].expert*expert_size, bank, fr[i].slot*expert_size, n*expert_size });
                } else {
                    ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fr[i].expert*expert_size, fr[i].slot*expert_size, n*expert_size);
                }
                bytes += n*expert_size;
                i += n;
            }
        }
        const int64_t t_ws = hprof_on && ws ? ggml_time_us() : 0;
        const bool ws_ok = !ws || wstore->upload_batch(backend, ws_regions.data(), ws_regions.size());
        if (hprof_on && ws) {
            hprof.disk_us += ggml_time_us() - t_ws;
            hprof.n_disk  += rest->size();
        }
        if (!ws_ok) {
            for (const auto & r : ws_regions) {
                ggml_backend_tensor_set_async(backend, r.dst, (const uint8_t *) r.src->data + r.src_off, r.dst_off, r.size);
            }
        }
        flush_fill_regions(backend);
        return bytes;
    }

    // the device slots that are lent (or overwritten by a seed) lose their experts: the RAM tier keeps a copy (synchronous, once per prompt)
    // k19-l4held (LLAMA_MOE_SHADOW_ASYNC=1, opt-in; unset: synchronous, as before): the copies back are queued on backend (the stream that then writes
    // the device slots, so each copy reads the slot before it is overwritten) and the RAM slots are published when they complete (mark_write
    // + commit, as demote does) instead of a device sync and one synchronous copy per expert tensor. A prompt that seeds the device slots
    // of an exclusive tier (the 42-token chat on the Colab L4 full card: ~90 evictions per layer) waited for each layer's device work and
    // then ~270 synchronous 2-3 MB copies per layer
    // opt-in (LLAMA_MOE_SHADOW_ASYNC=1): Colab L4 full card, fn4nd chat after the 640-token request (job l4h_c2, 3 launches each, md5 equal):
    // seeds 47-102 ms vs 1090-1103 ms synchronous, but chat TTFT 8.70 [7.40-9.43] vs 7.44 [7.34-9.03] s: in 2 of 3 launches the RAM tier
    // held 785-1004 duplicates of device experts at the chat's start (234-236 otherwise) and the chat missed 862-914 experts vs 777-787;
    // not explained, so not a default
    static bool shadow_async() {
        static const bool on = [] { const char * e = getenv("LLAMA_MOE_SHADOW_ASYNC"); return e != nullptr && atoi(e) != 0; }();
        return on;
    }
    uint64_t n_shadow_async = 0;
    // async only from a seed (allow_async): lend / retire / release hand the slots to the graph allocation or free them right after
    void shadow_slots(ggml_backend_t backend, group & g, const std::vector<std::pair<int32_t, int32_t>> & ev, bool allow_async = false) {
        if (!host || ev.empty()) {
            return;
        }
        const bool as = allow_async && shadow_async();
        // the copies into the device slots may be on the compute backend or the copy queue of the device: both done (async: the copies
        // back are queued behind the work of backend, only another queue is waited for)
        if (!as) {
            ggml_backend_synchronize(backend);
        }
        if (devices[g.id].backend != backend) {
            ggml_backend_synchronize(devices[g.id].backend);
        }
        if (as) {
            demote(backend, g, ev);
            host->commit(backend);
            n_shadow_async += ev.size();
            return;
        }
        const int32_t n_expert = g.lru.n_expert;
        for (const auto & [s, key] : ev) {
            const int32_t il = key / n_expert;
            const int32_t e  = key % n_expert;
            if (!host_layer(il)) {
                continue;
            }
            if (host->contains(il, e)) {
                sss.dem_kept++;
                const int32_t h = host->held(il, e);
                if (h >= 0) {
                    host->set_dup(h, false);
                    const layer & l = layers[il];
                    if (copyback_skip) {
                        hst.demote_kept++;
                        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                            hst.demote_kept_bytes += l.experts[ip]->nb[2];
                        }
                    } else {
                        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                            const size_t nb = l.experts[ip]->nb[2];
                            ggml_backend_tensor_get(g.banks[ip], host->data(h, (int) ip), (size_t) s*nb, nb);
                            hst.demote_forced_bytes += nb;
                        }
                        hst.demote_forced++;
                    }
                }
                continue;
            }
            // k19-l4held rank mode: at the layer's quota the copy back takes an expert the prompt does not route first
            const std::vector<bool> * rk = rank_any && (size_t) il < rank_mask.size() && (int32_t) rank_mask[il].size() > e ? &rank_mask[il] : nullptr;
            const uint64_t ev0 = host->st.evicted - host->st.dup_taken;
            const int32_t r = host->reserve(il, e, true, 0, true, rk);
            if (r < 0) {
                hst.demote_skip++;
                sss.dem_none++;
                continue;
            }
            if (host->st.evicted - host->st.dup_taken > ev0) {
                sss.dem_evict++;
            } else {
                sss.dem_free++;
            }
            const layer & l = layers[il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const size_t nb = l.experts[ip]->nb[2];
                host->set_lead(r, (int) ip, 0);
                ggml_backend_tensor_get(g.banks[ip], host->area(r, (int) ip), (size_t) s*nb, nb);
                hst.demote_bytes += nb;
            }
            host->publish(r);
            hst.demoted++;
        }
    }

    // the device of the group can map host memory and run the lookup of the largest batch
    bool lookup_capable(const group & g) const {
        ggml_backend_t backend = devices[g.id].backend;
        if (!llama_moe_cache_device_lookup_supported(ggml_backend_get_device(backend))) {
            return false;
        }
        ggml_init_params params = {
            /*.mem_size   =*/ 4*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        // the largest batch whose experts fit in the slots (see use_device)
        const int64_t n_tokens = std::max<int64_t>(1, std::min<int64_t>(max_batch, g.n_slots/n_expert_used));
        ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
        std::vector<const void *> host(g.banks.size(), nullptr);
        std::vector<ggml_tensor *> banks = op_banks(g);
        ggml_tensor * op = ggml_moe_cache_lookup(ctx.get(), ids, g.map, g.state, banks.data(), host.data(), (int) banks.size(), g.layers[0]);
        if (g.scores != nullptr) {
            ggml_moe_cache_set_scores(op, g.scores, g.lru.grow_q16, true);
        }
        return ggml_backend_supports_op(backend, op);
    }

    static bool stream_device_lookup() {
        const char * env = getenv("LLAMA_WEIGHT_STREAM_DEVICE_LOOKUP");
        return env != nullptr && atoi(env) != 0;
    }

    // for each group: checks that its device supports the lookup and maps the host experts into its address space
    // the groups of devices without the lookup op or the host_map function (backend capability) keep the host lookups
    void init_device_lookup() {

        // device pointer of each mapped host buffer, per device (the device address of a host buffer may differ between devices)
        std::map<std::pair<ggml_backend_dev_t, ggml_backend_buffer_t>, uint8_t *> dev_base;

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_backend_t backend = devices[g.id].backend;
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (!lookup_capable(g)) {
                continue;
            }
            // served by the RAM tier: only its slots are registered memory, the expert buffers are never mapped for the device
            if (ig < host_group.size() && host_group[ig]) {
                LLAMA_LOG_INFO("%s: group %zu: host lookups, its experts are streamed from the model files through the RAM tier\n", __func__, ig);
                continue;
            }
            // experts streamed from the model files (they do not fit in RAM): mapping them for the device would page-lock the file
            // mapping, the host lookups read the misses with the weight store instead (LLAMA_WEIGHT_STREAM_DEVICE_LOOKUP=1 maps them)
            if (wstore != nullptr && !stream_device_lookup()) {
                bool streamed = false;
                for (int32_t il : g.layers) {
                    for (const ggml_tensor * t : layers[il].experts) {
                        streamed = streamed || wstore->covers(t);
                    }
                }
                if (streamed) {
                    LLAMA_LOG_INFO("%s: group %zu: host lookups, its experts are streamed from the model files\n", __func__, ig);
                    continue;
                }
            }
            auto host_map   = (host_map_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_map");
            auto host_unmap = (host_unmap_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_unmap");

            bool use_ranges = pin_mode == 2;
            // registering a memory-mapped file pins all of its pages, which must then fit in system memory: when the file buffers this
            // group would register exceed the available memory (llama_sysmem::available, minus a margin), registering fails only after
            // faulting in most of them (OOM after 83 s for 37 GiB of experts on a 31 GiB host), so it is not tried
            // (LLAMA_MOE_CACHE_MAP_CHECK=0 tries anyway); buffers in anonymous memory are resident already and are not counted
            {
                const char * env = getenv("LLAMA_MOE_CACHE_MAP_CHECK");
                if (env == nullptr || atoi(env) != 0) {
                    std::set<ggml_backend_buffer_t> bufs;
                    uint64_t need = 0;
                    for (int32_t il : g.layers) {
                        for (const ggml_tensor * t : layers[il].experts) {
                            if (dev_base.count({ dev, t->buffer }) == 0 && bufs.insert(t->buffer).second &&
                                    strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)), "CPU_Mapped") == 0) {
                                need += ggml_backend_buffer_get_size(t->buffer);
                            }
                        }
                    }
                    const uint64_t avail  = need > 0 ? llama_sysmem::available() : 0;
                    const uint64_t margin = std::max<uint64_t>(UINT64_C(2) << 30, need/16);
                    if (need > 0 && avail > 0 && need + margin > avail) {
                        LLAMA_LOG_WARN("%s: %s: the host experts are memory-mapped files of %.0f MiB, more than the %.0f MiB of system memory "
                            "available (less a %.0f MiB margin) could pin: %s for %zu layers (LLAMA_MOE_CACHE_MAP_CHECK=0 tries anyway)\n",
                            __func__, ggml_backend_name(backend), need/1048576.0, avail/1048576.0, margin/1048576.0,
                            pin_mode != 0 ? "the expert ranges are pinned instead" : "the MoE cache uses host lookups", g.layers.size());
                        if (pin_mode == 0) {
                            continue;
                        }
                        use_ranges = true;
                    }
                }
            }

            // the mappings this group adds are appended to mapped, the ones shared with earlier groups stay untouched
            const size_t n_mapped = mapped.size();
            std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_t>> added;

            bool ok = !use_ranges;
            std::vector<std::vector<const void *>> host_dev(g.layers.size());
            for (size_t i = 0; i < g.layers.size() && ok; ++i) {
                for (const ggml_tensor * t : layers[g.layers[i]].experts) {
                    auto it = dev_base.find({ dev, t->buffer });
                    if (it == dev_base.end() && core_on && core_holes_in((uintptr_t) ggml_backend_buffer_get_base(t->buffer), ggml_backend_buffer_get_size(t->buffer))) {
                        // k19-ptres: the pages of the core experts stay out of the registration (the device never reads them: they never leave the slots)
                        void * base = ggml_backend_buffer_get_base(t->buffer);
                        void * dev_ptr = core_map_buffer(dev, host_map, host_unmap, base, ggml_backend_buffer_get_size(t->buffer));
                        if (dev_ptr != nullptr) {
                            added.emplace_back(dev, t->buffer);
                        }
                        it = dev_base.emplace(std::make_pair(dev, t->buffer), (uint8_t *) dev_ptr).first;
                    }
                    if (it == dev_base.end()) {
                        void * base = ggml_backend_buffer_get_base(t->buffer);
                        bool registered = false;
                        void * dev_ptr = host_map(dev, base, ggml_backend_buffer_get_size(t->buffer), &registered);
                        if (registered) {
                            mapped.emplace_back(host_unmap, base);
                            added.emplace_back(dev, t->buffer);
                            hst.registered += ggml_backend_buffer_get_size(t->buffer);
                            if (strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)), "CPU_Mapped") == 0) {
                                file_pin_add(ggml_backend_buffer_get_size(t->buffer));
                            }
                        }
                        it = dev_base.emplace(std::make_pair(dev, t->buffer), (uint8_t *) dev_ptr).first;
                    }
                    if (it->second == nullptr) {
                        ok = false;
                        break;
                    }
                    host_dev[i].push_back(it->second + ((const uint8_t *) t->data - (const uint8_t *) ggml_backend_buffer_get_base(t->buffer)));
                }
            }
            if (!ok) {
                // release the mappings registered for this group only, so that the host experts do not stay page-locked until the cache is destroyed
                for (size_t i = n_mapped; i < mapped.size(); ++i) {
                    mapped[i].first(mapped[i].second);
                }
                mapped.resize(n_mapped);
                for (const auto & key : added) {
                    dev_base.erase(key);
                }
                if (!use_ranges) {
                    LLAMA_LOG_WARN("%s: %s cannot map the host experts (pageable or memory-mapped), %s for %zu layers\n",
                        __func__, ggml_backend_name(backend), pin_mode != 0 ? "pinning their ranges" : "the MoE cache uses host lookups", g.layers.size());
                }
                if (pin_mode == 0 || !pin_ranges(g, dev, reg, host_dev)) {
                    continue;
                }
            }
            for (size_t i = 0; i < g.layers.size(); ++i) {
                layers[g.layers[i]].host_dev = std::move(host_dev[i]);
            }
            g.dev_lookup = true;
            LLAMA_LOG_INFO("%s: %s: MoE cache lookups of small batches run on the device for %zu layers (LLAMA_MOE_CACHE_DEVICE=0 to disable)\n",
                __func__, ggml_backend_name(backend), g.layers.size());
        }
        if (core_on) {
            core_drop();
        }
    }

    // every routed expert of a ubatch of n_tokens fits the slots of the group (the plain lookup's condition)
    bool slots_fit(const group & g, int64_t n_tokens) const {
        return std::min<int64_t>(n_tokens*n_expert_used, g.ref[0]->ne[2]) <= g.n_slots - g.lru.n_core;
    }

    // k19-hostrow (LLAMA_MOE_HYBRID_WIDE=1, default off): a hybrid group's small batch of at most wide_max tokens whose experts do not
    // fit the slots. fn4 UD-IQ4_XS at 8 GB on G4: 23 slots per group serve 2 tokens, so a verify window of 3-8 tokens read every one
    // of its experts on the CPU (no fetch share over PCIe, no route tail skip): ~12 ms per extra row against ~16 ms for a whole token.
    // With it the window takes the hybrid assign like a single token (hits on the device, the fetch share of the misses copied into
    // the unpinned slots, the rest on the host). Not exact against the CPU path it replaces (another split of the experts between
    // device and host); off in the exact modes (LLAMA_MTP_EXACT=1, LLAMA_MOE_CPU_MISS=0)
    bool wide_ok(const group & g, int64_t n_tokens) const {
        return hybrid_wide && g.counts != nullptr && (hybrid || cm) && n_tokens <= wide_max;
    }

    bool use_device(const group & g, int64_t n_tokens) const {
        return !force_large && g.dev_lookup && g.n_slots > 0 && n_tokens > 0 && (n_tokens <= max_batch || cm_serves(n_tokens)) &&
            (slots_fit(g, n_tokens) || wide_ok(g, n_tokens)) && !(place8_single && (hybrid || cm) && n_tokens > 1);
    }

    // k15-lfm2: llama_moe_cpu_miss_drop_slow. Once calibrated: where the cache holds at least LLAMA_MOE_CPU_MISS_DROP_FRAC (0.75) of the
    // host experts' bytes, the host has at most LLAMA_MOE_CPU_MISS_DROP_THREADS (4) small-batch threads and its time per expert is at
    // least LLAMA_MOE_CPU_MISS_DROP_RATIO (0.5) of an upload, the rule is turned off as a failed calibration does (the misses are
    // fetched). Kaggle T4x2 LFM2.5 16 GB (2 threads, host 640-660 us per expert, upload 520-940 us, 86% cached): the rule itself
    // computed nothing on the host there, while the pinned small-batch host path of the default drafts (spill mode 3) cost 2.5 ms
    // per token (62 vs 72 tok/s). LLAMA_MOE_CPU_MISS_DROP=0: never
    bool cm_drop_slow() {
        static const bool   on      = [] { const char * e = getenv("LLAMA_MOE_CPU_MISS_DROP"); return e == nullptr || atoi(e) != 0; }();
        static const double frac    = [] { const char * e = getenv("LLAMA_MOE_CPU_MISS_DROP_FRAC"); return e != nullptr ? atof(e) : 0.75; }();
        static const double ratio   = [] { const char * e = getenv("LLAMA_MOE_CPU_MISS_DROP_RATIO"); return e != nullptr ? atof(e) : 0.5; }();
        static const int    threads = [] { const char * e = getenv("LLAMA_MOE_CPU_MISS_DROP_THREADS"); return e != nullptr ? atoi(e) : 4; }();
        if (!on || !cm || !cm_cal || cm_force || cm_frozen || cache_frac < frac || single_threads <= 0 || single_threads > threads ||
                cm_t1 < ratio*cm_tup) {
            return false;
        }
        cm     = false;
        hybrid = false;
        cm_gen++;
        LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses off: %.1f%% of the host experts cached, %d host threads, host %.1f us per expert vs "
            "upload %.1f us: the missing experts are fetched (LLAMA_MOE_CPU_MISS_DROP=0: keep the rule)\n", tag, 100.0*cache_frac,
            single_threads, cm_t1, cm_tup);
        return true;
    }

    // k13: see cm_frozen; only once calibrated (the warm-up decode calibrates), returns false otherwise or without CPU misses
    // keep_fetch (cm_frozen_fetch, llama_moe_pin_small): pin the decisions only
    bool cm_freeze(bool keep_fetch = false) {
        if (!cm || !cm_cal || cm_frozen) {
            return cm_frozen && cm_frozen_fetch == keep_fetch;
        }
        if (!cm_pinned) {
            cm_trial_saved = cm_trial;
        }
        cm_pinned     = cm_pinned || keep_fetch;
        cm_frozen     = true;
        cm_frozen_fetch = keep_fetch || (getenv("LLAMA_MOE_CPU_MISS_FROZEN_FETCH") != nullptr && atoi(getenv("LLAMA_MOE_CPU_MISS_FROZEN_FETCH")) != 0);
        cm_trial      = false;
        cmt_done      = -1;
        cmt_wait      = -1;
        cmt_t_prev    = -1;
        cmw_on        = false;
        cm_small_ok   = true;
        cm_small_live = true;
        // k19-l4poly: on a host-team host the short prompts (k17-ttft, see cm_prompt_off) keep the cost model's verdict of the
        // calibration (persisted per build and host, so the same in every launch); LLAMA_MOE_PIN_PROMPT_SHORT=0 off
        const bool keep_short = cm_pin_short() && single_team_host();
        for (int64_t n = 1; n <= cm_max; ++n) {
            // batches above the frozen sizes (short prompts) keep the plain cache: their calibrated host path is timing-driven, and
            // a frozen cache gives the same text in every run
            const bool short_n = keep_short && n > max_batch && n <= cm_short_max;
            cm_on[n] = n <= std::min<int64_t>({ cm_frozen_max, max_batch }) ? 1 : short_n ? cm_on[n] : 0;
        }
        cm_gen++;
        if (cm_pin_lifted_once) {
            return true; // a pin restored after cm_pin_set(false): logged once at the first pin
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: small batches of up to %d tokens frozen: %s, "
            "no timing trials (LLAMA_MOE_CPU_MISS_FROZEN=0 off)\n", tag, (int) std::min<int64_t>({ cm_frozen_max, max_batch, cm_max }),
            cm_frozen_fetch ? "host path pinned on, the fetch share and admissions kept (LLAMA_MOE_CPU_MISS_FROZEN_FETCH=1)" :
                              "nothing fetched, the host computes every miss");
        return true;
    }

    // k15-lfm2: lift / restore the pin of llama_moe_pin_small (cm_freeze(true)) between requests. Unpinned, the small batches follow
    // the rule again (cost model, the timing trials where this cache runs them, after cmt_gap small batches); pinned again, as at the
    // first pin. Only a cache the pin froze (cm_pinned) takes part. Returns the pinned state
    bool cm_pinned       = false; // frozen by pin_small (not by freeze_small)
    bool cm_pin_lifted   = false;
    bool cm_pin_lifted_once = false;
    bool cm_trial_saved  = true;
    bool cm_pin_set(bool on) {
        if (!cm_pinned) {
            return cm_frozen;
        }
        if (on && cm_pin_lifted) {
            cm_pin_lifted = false;
            cm_frozen     = false; // cm_freeze(true) pins again from here
            cm_freeze(true);
        } else if (!on && !cm_pin_lifted && cm_frozen) {
            cm_pin_lifted   = true;
            cm_pin_lifted_once = true;
            cm_frozen       = false;
            cm_frozen_fetch = false;
            cm_trial        = cm_trial_saved;
            cm_small_ok     = true;
            cm_small_live   = true;
            cmt_done        = -1;
            cmt_t_prev      = -1;
            cmt_wait        = cm_trial && !cm_trials_skipped ? cmt_gap : -1;
            cm_decide();
            cm_gen++;
            if (pin_log_n++ < 4) {
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: small-batch pin lifted for this request (the rule decides%s)\n", tag,
                    cmt_wait > 0 ? ", a timing trial after the next small batches" : "");
            }
        }
        return cm_frozen;
    }
    int pin_log_n = 0;

    // k15-lfm2 LLAMA_MOE_CPU_MISS_PER_REQUEST (opt-in): the rule's decisions of a request do not depend on the requests before it. At each
    // request the per-size decisions and the miss rate go back to those of the first request (the calibrated prior), the miss-rate
    // window starts over (its first window is the short one, so the request decides from its own small batches), and after the first
    // timing trial no further one runs (one trial verdict per launch: later trials flip on differences of a few percent, G4 LFM2.5
    // 12 GB 16k 222 vs 291 tok/s from one 3.10 vs 3.01 ms trial). Without CPU misses (pinned, LLAMA_MOE_CPU_MISS=0) nothing changes
    bool cm_req_snap = false;
    std::vector<uint8_t> cm_on_snap;
    double cm_rate_snap = -1.0;
    bool   cm_measured_snap = false;
    int    cm_req_n = 0;
    // =1 also empties the cache at each request (hybrid_reset, as the fixed-share hybrid path does): which experts the host computes
    // depends on what is cached, which the earlier requests left (G4 LFM2.5 8 GB 8k: 189 vs 221 tok/s and another text by the order
    // of the requests alone, with the same rule decisions); the prompt's seed fills it again. =2: the decisions only
    static int cm_per_request() {
        static const int on = [] { const char * e = getenv("LLAMA_MOE_CPU_MISS_PER_REQUEST"); return e != nullptr ? atoi(e) : 0; }();
        return on;
    }
    bool cm_request_resets() const {
        return cm_per_request() == 1 && cm && cm_cal && !cm_frozen;
    }
    bool cm_request_begin() {
        const int on = cm_per_request();
        if (on == 0 || !cm || !cm_cal || cm_frozen) {
            return false;
        }
        if (on == 1) {
            hybrid_reset(true);
        }
        if (cm_force) {
            return on == 1; // LLAMA_MOE_CPU_MISS=2: no decisions to restart
        }
        cm_req_n++;
        if (!cm_req_snap) {
            cm_on_snap       = cm_on;
            cm_rate_snap     = cm_rate;
            cm_measured_snap = cm_measured;
            cm_req_snap      = true;
        } else {
            cm_on       = cm_on_snap;
            cm_rate     = cm_rate_snap;
            cm_measured = cm_measured_snap;
        }
        cm_have0 = false;
        cm_nb    = 0;
        cm_win   = cm_win1;
        if (cmt_n > 0 && cmt_done < 0) {
            cmt_wait = -1;
        }
        cm_gen++;
        if (cm_req_n <= 2) {
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: request %d starts from the launch's decisions%s (LLAMA_MOE_CPU_MISS_PER_REQUEST)\n", tag,
                cm_req_n, on == 1 ? " and an empty cache" : "");
        }
        return on == 1;
    }

    // a batch of more than max_batch tokens (a short prompt) that the host path serves: it is looked up on the device like a small batch
    // short prompts take the host path only once a trial found it faster for the small batches on this host and model (without a
    // trial: the model alone). G4 gpt-oss-20b Q2_K 8 GB and T4 Qwen3.6 UD-Q4_K_XL 8 GB without a draft: trials off, the 95-token chat
    // prompt -19% / -26% on the host path from the prior; where the trials kept it (Qwen3.6 8 GB MTP: G4 +18%, T4 +6%) it gained
    // k11 (LLAMA_MOE_CPU_MISS_TRIAL_FRAC, default 0.20; 0 = the trials of k10 everywhere): a cache that holds less than this share of
    // the host experts runs no warm phase and no timing trials (see cm_calibrate for the cells). Round 19
    // (LLAMA_MOE_CPU_MISS_TRIAL_FRAC_SINGLE, default 0.50): a cache whose host misses go through the single-graph mailbox (see
    // single_weigh) skips them up to this share, since the trials judged the graph split the mailbox replaces. G4 8 GB / 32 GB, k10 on
    // the same VM: no trials below 50% gave Qwen3.6 UD-Q3_K_M 8 GB (23.8%) +17% decode over the 20% rule and Flash-Next UD-IQ4_XS 32 GB
    // (33.5%; "3 trials, small batches off") +9..+19% 2k / 8k, while gemma-4 8 GB (GELU experts: graph split, no mailbox) lost 3-12%
    // without its trials, so it keeps them. An explicit LLAMA_MOE_CPU_MISS_TRIAL_FRAC sets both. The mailbox form is known from the
    // first graph with the hybrid path, usually after the calibration: single_note_form calls this again (late) then, once, if no
    // trial has completed
    bool cm_trials_late_done = false;
    bool cm_trials_skipped   = false;
    bool cm_skip_trials(bool late) {
        static const double trial_frac_env = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_TRIAL_FRAC");
            return e != nullptr ? std::max(0.0, atof(e)) : -1.0;
        }();
        const double trial_frac_single = llama_moe_cache::trial_frac_single();
        if (late && (cm_trials_late_done || !cm || !cm_cal || trial_frac_env >= 0.0 || cmt_n > 0 || cm_trials_skipped)) {
            return false;
        }
        const bool mailbox = single_weigh() && single_usable(1);
        const double trial_frac = trial_frac_env >= 0.0 ? trial_frac_env : (mailbox ? std::max(0.20, trial_frac_single) : 0.20);
        // k15-lfm2: with LLAMA_MOE_CPU_MISS_PER_REQUEST no timing trials (the cost model only), so no verdict depends on which request
        // ran the first trial (G4 LFM2.5 12 GB: chat / 8k / 16k other texts by request order); LLAMA_MOE_CPU_MISS_PER_REQUEST_TRIALS=1
        // keeps the first trial
        static const bool per_req_no_trials = [] {
            const char * e  = getenv("LLAMA_MOE_CPU_MISS_PER_REQUEST");
            const char * et = getenv("LLAMA_MOE_CPU_MISS_PER_REQUEST_TRIALS");
            return e != nullptr && atoi(e) != 0 && !(et != nullptr && atoi(et) != 0);
        }();
        const bool skip = cm_trial && !cm_force && ((cache_frac >= 0.0 && cache_frac < trial_frac) || per_req_no_trials);
        if (late) {
            cm_trials_late_done = true;
        }
        if (skip) {
            cm_trials_skipped = true;
            cmt_done      = -1;
            cmt_wait      = -1;
            cmt_t_prev    = -1;
            cmw_on        = false;
            cm_small_ok   = true;
            cm_small_live = true;
            cm_gen++;
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: no timing trials at %.1f%% of the host experts cached (%s%.0f%%%s%s): small "
                "batches follow the cost model, short prompts stay on the plain cache (LLAMA_MOE_CPU_MISS_TRIAL_FRAC=0 restores the trials)\n",
                tag, 100.0*cache_frac, per_req_no_trials ? "LLAMA_MOE_CPU_MISS_PER_REQUEST; trial share " : "below ", 100.0*trial_frac,
                mailbox ? ", single-graph mailbox" : "", late ? ", from now on" : "");
        }
        return skip;
    }

    bool cm_prompt_ok() const {
        return cmt_host_ok && (!cm_trial || (cmt_n > 0 && cm_small_ok));
    }

    // k15 (port of k11-spill arm P, 43a25ef811): the ubatches of a prompt batch (tokens without outputs, its last partial ubatch
    // included) stay off the CPU-miss host path, so the first token after a prompt carries no host-computed expert. On by default
    // only while the small batches are pinned (llama_moe_pin_small, spill mode 3: G4 LFM2.5 8 GB first-token KLD came from the
    // prompt's small checkpoint tail ubatch taking the pinned host path). LLAMA_MOE_PIN_PROMPT_DEV=0: off there;
    // LLAMA_MOE_CPU_MISS_PROMPT_DEV=1: also without the pin (test arm)
    bool cm_prompt_now = false; // set by the context around each decode (set_prompt_batch)
    int64_t cm_prompt_all = 0;  // k17-ttft: tokens of that prompt batch (0 outside prompt batches)

    // k17-ttft short prompts (LLAMA_MOE_CPU_MISS_SHORT, default 128; 0 = off, the k15 default): a prompt batch of max_batch + 1 ..
    // cm_short_max tokens in all (a chat turn, not the tail of a long prompt: G4 r22b gemma-4 2k decode -18% after its last partial
    // ubatch took the host path and admitted less) takes the host path in each ubatch the cost model turns on (cm_on: calibrated host
    // us per expert and per further token, upload us, the split and the activations at cm_margin_p, the measured miss rate), at the
    // rule's fetch share but at most 1/4 as k11's prompt path; no timing trials, so no trial ubatch changes what the cache admits.
    // As Strata's own default (q36: --hybrid-rows 128, every miss of a batch of up to 128 rows on the CPU pool, q36_engine.cu
    // moe_hybrid_cpu; Flash-Next: STRATA_PREFILL_CPU_SHARE=auto below 1024 tokens): the host's expert products are ggml-cpu's
    // (Q8_K activations), not the device's bit for bit, so the text changes (KLD-gated). Before it, the small-batch trials (which
    // never ran on these placements: "0 trials") kept every prompt above 32 tokens on the uploads (G4 q4x 12 GB: 4.8 GB for a
    // 42-token chat prompt, 90 ms of H2D at the link ceiling)
    const int64_t cm_short_max = [] {
        const char * e = getenv("LLAMA_MOE_CPU_MISS_SHORT");
        return e != nullptr ? std::max<int64_t>(0, std::min<int64_t>(cm_max_limit, atoll(e))) : (int64_t) 128;
    }();
    bool short_live() const {
        return cm_short_max > 0 && cm_prompt_now && cm_prompt_all > max_batch && cm_prompt_all <= cm_short_max;
    }
    // only where the small-batch timing trials were skipped (below LLAMA_MOE_CPU_MISS_TRIAL_FRAC of the host experts cached, 50% with the
    // single-graph mailbox): where they run, their verdict already gates prompts (cm_prompt_ok). G4 k17g4a q4x 16 GB (61% cached, trials
    // run): the rule without this gate took chat_s onto the host path at 358 vs 550 tok/s and the 2k decode after it fell 342 -> 306 tok/s;
    // at 8 / 12 GB (trials skipped) chat +18% / +30%
    // and only where the host is clearly faster than the uploads at the batch's tokens per expert: host us per missing expert
    // t_cpu(n) = cm_t1 + (tbar(n) - 1)*cm_dt at most cm_short_ratio (LLAMA_MOE_CPU_MISS_SHORT_RATIO, 0.67) of an upload, so the host's 3/4
    // of the misses take at most half the time their uploads would. The cost model alone (cm_on) put these G4 cells on the host path:
    // Qwen3.6 UD-Q4_K_XL 8 / 12 GB (t_cpu / t_up 0.60: TTFT -15% / -23%) but also Flash-Next UD-IQ1_S 24 GB (0.75: TTFT +11%, 2k decode
    // -8%; its re-calibration measured the host 46% slower than at start-up) and gemma-4 Q3_K_M 8 GB (1.3: TTFT +29%, 2k decode -20%)
    const double cm_short_ratio = [] {
        const char * e = getenv("LLAMA_MOE_CPU_MISS_SHORT_RATIO");
        return e != nullptr ? std::max(0.0, atof(e)) : 0.67;
    }();
    bool short_fast(int64_t n_tokens) const {
        for (const group & g : groups) {
            if (g.dev_lookup && g.n_slots > 0 && g.counts != nullptr && cm_tup > 0.0) {
                const int64_t n_expert = g.ref[0]->ne[2];
                const double  D        = cm_distinct(n_tokens, n_expert);
                const double  tbar     = std::max(1.0, (double) n_tokens*std::min<int64_t>(n_expert_used, n_expert)/std::max(1.0, D));
                return cm_t1 + (tbar - 1.0)*cm_dt <= cm_short_ratio*cm_tup;
            }
        }
        return false;
    }
    bool short_eligible(int64_t n_tokens) const {
        return short_live() && cm_trials_skipped && cm && cm_cal && !cm_force && n_tokens > max_batch && n_tokens <= std::min(cm_max, cm_short_max) &&
            short_fast(n_tokens) &&
            n_tokens > cm_prompt_force && cm_on[n_tokens] != 0 && cmt_host_ok && !cm_prompt_off();
    }
    // only where the cache holds at most LLAMA_MOE_PIN_PROMPT_SHORT_MAX_FRAC (0.25) of the host experts: srv_kld short192 against the
    // base (l4p_f3), LFM2.5 4 GB (14.6% cached) first-token KLD mean / p99 / top-1 0.099 / 0.84 / 0.885 inside the -fa off band
    // (0.154 / 1.32 / 0.854); 8 GB (40%) 0.144 / 1.53 / 0.849, p99 and top-1 just outside it (1.16 / 0.854)
    bool cm_pin_short() const {
        static const bool pin_short = [] {
            const char * e = getenv("LLAMA_MOE_PIN_PROMPT_SHORT");
            return e == nullptr || atoi(e) != 0;
        }();
        static const double max_frac = getenv("LLAMA_MOE_PIN_PROMPT_SHORT_MAX_FRAC") != nullptr ?
            std::max(0.0, atof(getenv("LLAMA_MOE_PIN_PROMPT_SHORT_MAX_FRAC"))) : 0.25;
        return pin_short && cache_frac >= 0.0 && cache_frac <= max_frac;
    }
    bool cm_prompt_off() const {
        static const bool pin_dev = [] {
            const char * e = getenv("LLAMA_MOE_PIN_PROMPT_DEV");
            return e == nullptr || atoi(e) != 0;
        }();
        static const bool all_dev = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_PROMPT_DEV");
            return e != nullptr && atoi(e) != 0;
        }();
        // k19-l4poly (LLAMA_MOE_PIN_PROMPT_SHORT, default 1; 0 = off): on a host-team host (single_team_host) a short prompt (the k17-ttft
        // rule, at most LLAMA_MOE_CPU_MISS_SHORT tokens in all) keeps its host path beside the pinned small batches: the pin is for the
        // small checkpoint tail of a long prompt (k15), while a chat turn's prompt otherwise uploads the misses of every layer. Colab L4
        // LFM2.5 4 / 8 GB with n-gram drafting (pinned): a 38-token prompt 875 / 739 ms, 498 ms at 8 GB without drafting (not pinned);
        // PolyStrata 526 ms
        const bool pinned = cm_frozen && cm_frozen_fetch && pin_dev && !(cm_pin_short() && short_live() && single_team_host());
        return cm_prompt_now && cm && (pinned || all_dev);
    }

    // r22b: a prompt ubatch the cost model puts on the host path, outside the forced and the trial-gated rules
    bool pt_eligible(int64_t n_tokens) const {
        return pt_auto && cm && cm_cal && !cm_force && n_tokens > max_batch && n_tokens <= cm_max && n_tokens > cm_prompt_force &&
            (cm_on[n_tokens] != 0 || pt_any) && cmt_host_ok && !cm_prompt_ok();
    }

    bool cm_serves(int64_t n_tokens) const {
        return !force_large && cm && cm_cal && !cm_prompt_off() && n_tokens > max_batch && n_tokens <= cm_max && (n_tokens <= cm_prompt_force || (cm_on[n_tokens] != 0 && cm_prompt_ok()) ||
            (pt_eligible(n_tokens) && pt_arm != 0) || short_eligible(n_tokens));
    }

    void cm_dec_init() {
        if (cm_dec_on >= 0) {
            return;
        }
        const char * e  = getenv("LLAMA_MOE_CPU_MISS_DECODE");
        const char * em = getenv("LLAMA_MOE_CPU_MISS_DECODE_MAX");
        const char * es = getenv("LLAMA_MOE_CPU_MISS_DECODE_SHARE");
        cm_dec_on    = e != nullptr && atoi(e) != 0 ? 1 : 0;
        cm_dec_max   = std::max<int64_t>(1, std::min<int64_t>(single_max(), em != nullptr && atoi(em) > 0 ? atoi(em) : single_max()));
        cm_dec_share = es != nullptr ? (float) std::min(1.0, std::max(0.0, atof(es))) : 0.125f;
    }
    // k19-flint: a decode batch of n_tokens on the fixed-share host path (LLAMA_MOE_CPU_MISS_DECODE)
    bool cm_decode(int64_t n_tokens) const {
        return cm_dec_on > 0 && cm && cm_cal && !cm_force && !cm_frozen && n_tokens >= 1 && n_tokens <= cm_dec_max && n_tokens <= max_batch &&
            n_tokens <= cm_max && single_form_state >= 0 && single_usable(n_tokens);
    }

    // the host computes the misses of a batch of n_tokens: the fetch share, 1 = every missing expert is fetched
    float cm_share_of(int64_t n_tokens) const {
        if (cm_prompt_off()) {
            return 1.0f;
        }
        if (cm_frozen && !cm_frozen_fetch && cm && cm_cal && n_tokens >= 1 && n_tokens <= std::min<int64_t>({ cm_frozen_max, max_batch, cm_max })) {
            return 0.0f;
        }
        if (cm_decode(n_tokens)) {
            return cm_dec_share;
        }
        if (cm && cm_cal && n_tokens > max_batch && n_tokens <= cm_prompt_force) {
            // default: at most 1/4 fetched. The rule's share (capped at 3/4) left 71 ms of zero-copy fetch kernels in a 38-token LFM2.5 8 GB
            // prompt of 122 ms; G4 k11pgW at 1/4: LFM2.5 chat 444 tok/s (339 at the rule's share), Qwen3.6 360 (329); 1/8: LFM2.5 415
            return cm_prompt_share > 0.0f ? cm_prompt_share : std::min(cm_share[n_tokens], 0.25f);
        }
        if (short_eligible(n_tokens) && !cm_prompt_ok()) {
            return cm_prompt_share > 0.0f ? cm_prompt_share : std::min(cm_share[n_tokens], 0.25f);
        }
        if (pt_eligible(n_tokens)) {
            // the fetch share of k11's prompt path: at most 1/4 (its fetched misses are read by the zero-copy kernel, slower than the DMA
            // the model prices: G4 LFM2.5 8 GB chat 444 tok/s at 1/4 vs 339 at the rule's share)
            return pt_arm == 0 ? 1.0f : (cm_prompt_share > 0.0f ? cm_prompt_share : std::min(cm_share[n_tokens], 0.25f));
        }
        if (!cm || !cm_cal || n_tokens < 1 || n_tokens > cm_max || cm_on[n_tokens] == 0 || (n_tokens <= max_batch ? !cm_small_live : !cm_prompt_ok())) {
            return 1.0f;
        }
        return cm_share[n_tokens];
    }

    bool cm_any_on() const {
        if (!cm || !cm_cal) {
            return false;
        }
        if (cm_decode(1)) {
            return true;
        }
        for (int64_t n = 1; n <= cm_max; ++n) {
            if (n > max_batch && pt_eligible(n) && pt_state != 0) {
                return true;
            }
            if (cm_on[n] != 0 && (n > max_batch ? cm_prompt_ok() : cm_small_live)) {
                return true;
            }
        }
        return false;
    }

    // distinct experts of a layer that a batch of n tokens routes to (independent uniform routing), and the tokens of each
    double cm_distinct(int64_t n, int64_t n_expert) const {
        const double k = (double) std::min<int64_t>(n_expert_used, n_expert);
        return (double) n_expert*(1.0 - std::pow(1.0 - k/n_expert, (double) n));
    }

    ggml_tensor * lookup(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_ubatch) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        if (!use_device(g, n_ubatch) || !slots_fit(g, n_ubatch) || ids->ne[1] > n_ubatch || ids->ne[0] > n_expert_used || l.host_dev.size() != g.banks.size()) {
            return nullptr;
        }
        std::vector<ggml_tensor *> banks = op_banks(g);
        std::vector<const void *>  host  = l.host_dev;
        ggml_tensor * op = ggml_moe_cache_lookup(ctx, ids, g.map, g.state, banks.data(), host.data(), (int) banks.size(), il);
        if (g.scores != nullptr) {
            ggml_moe_cache_set_scores(op, g.scores, g.lru.grow_q16, il == g.layers[0]);
        }
        if (g.lru.n_core > 0) {
            ggml_moe_cache_set_core(op, g.lru.n_core);
        }
        return op;
    }

    // k19-q4xcache LLAMA_MOE_OVERLAP_SMALL_PLAIN (default 1; 0 = off, the plain lookup as before): a verify window of 2..N tokens
    // (N = LLAMA_MOE_CACHE_FETCH_OVERLAP_SMALL, default 4) whose host path is off took the plain lookup, which copies its missing
    // experts in the compute stream before any expert of the layer is computed (Kaggle T4 Qwen3.6-35B-A3B UD-Q4_K_XL, MTP n2, nsys:
    // moe_cache_fetch 10.8 ms of a 42.0 ms pass at full card, 16.5 of 47.7 at 12 GB, serial). It now takes the hybrid assign at fetch
    // share 1 with the overlap of k19-overlap: the copies run on the side stream beside the hits (cached before the lookup), then the
    // fetched experts are computed. Exact: the assign at share 1 picks the lookup's slots and victims, admits every miss and leaves the
    // use counters unwritten as the lookup does (ggml_moe_cache_assign_lookup_counts), the CPU-miss rule sees the same counters, and the
    // multi-token mmvq kernels give each (token, slot) entry its own sums (each entry is computed once, by one side; the other side's
    // row is +0)
    static bool overlap_plain_on() {
        static const bool on = [] {
            const char * e = getenv("LLAMA_MOE_OVERLAP_SMALL_PLAIN");
            return e == nullptr || atoi(e) != 0;
        }();
        return on;
    }
    bool overlap_plain_batch(int64_t n_ubatch) const {
        return overlap_plain_on() && overlap_small_batch(n_ubatch);
    }
    mutable uint64_t n_ov_plain = 0; // layer graphs built with it (engagement)
    mutable uint64_t n_ov_plain_wide = 0; // k19-bigverify: of them, windows of 5+ tokens (overlap_plain_wide)

    // the layer can use the hybrid lookup for a ubatch of this size; plain (k19-q4xcache): only for the overlap, in place of the plain lookup
    bool hybrid_layer(int32_t il, int64_t n_ubatch, bool * plain = nullptr) const {
        if (plain != nullptr) {
            *plain = false;
        }
        bool widened = !(hybrid || overlap_batch(n_ubatch));
        if (force_large || (widened && !overlap_plain_batch(n_ubatch)) || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return false;
        }
        // CPU misses off for this batch size: the graph of k9 (plain lookup, or the fetch overlap of single tokens)
        if (cm && cm_share_of(n_ubatch) >= 1.0f && !overlap_batch(n_ubatch)) {
            if (!overlap_plain_batch(n_ubatch)) {
                return false;
            }
            widened = true;
        }
        const group & g = groups[layers[il].ig];
        const bool ok = g.counts != nullptr && use_device(g, n_ubatch) && layers[il].host_dev.size() == g.banks.size();
        if (plain != nullptr) {
            *plain = ok && widened;
        }
        return ok;
    }

    // while the hybrid path is off but may be used (probe, trials), the lookup fetches every missing expert (share 1, same results
    // as ggml_moe_cache_lookup) and keeps the use counters up to date
    ggml_tensor * hybrid_assign(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_ubatch, bool & host) const {
        host = false;
        bool plain = false;
        if (!hybrid_layer(il, n_ubatch, &plain) || ids->ne[1] > n_ubatch || ids->ne[0] > n_expert_used) {
            return nullptr;
        }
        float share = 1.0f;
        if (plain) {
            // k19-q4xcache: in place of the plain lookup (see overlap_plain_batch): every miss is fetched, nothing goes to the host
        } else if (cm) {
            share = cm_share_of(n_ubatch);
            host  = share < 1.0f;
        } else {
            host  = hybrid_active();
            share = host ? hybrid_share : 1.0f;
        }
        const group & g = groups[layers[il].ig];
        if (plain && !slots_fit(g, n_ubatch)) {
            return nullptr; // the plain lookup refuses it too
        }
        if (!slots_fit(g, n_ubatch)) {
            // wide window (wide_ok): only with the host path, which computes the misses the slots cannot take
            if (!host) {
                return nullptr;
            }
            static std::atomic<bool> logged{false};
            if (!logged.exchange(true)) {
                LLAMA_LOG_INFO("llama_moe_cache: k19-hostrow wide hybrid windows: a small batch of %" PRId64 " tokens takes the hybrid assign beyond "
                    "the %d slots of its group (the misses they cannot take run on the host; LLAMA_MOE_HYBRID_WIDE=0: the CPU path)\n",
                    n_ubatch, (int) g.n_slots);
            }
        }
        ggml_tensor * op = ggml_moe_cache_assign(ctx, ids, g.map, g.state, g.counts, share, il);
        if (g.lru.n_core > 0) {
            ggml_moe_cache_set_core(op, g.lru.n_core);
        }
        if (plain) {
            ggml_moe_cache_assign_lookup_counts(op);
            if (n_ubatch > 4) {
                n_ov_plain_wide++;
            }
            if (n_ov_plain++ == 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: k19-q4xcache: verify windows of %" PRId64 " tokens with the host path off copy their missing "
                    "experts beside the cached ones (LLAMA_MOE_OVERLAP_SMALL_PLAIN=0 off)\n", tag, n_ubatch);
            }
        }
        if (host && split_frozen()) {
            // k13: frozen small batches admit nothing between prompts, not even to the empty slots a lent graph left: the n-gram
            // drafts of a verify window (rejected ones too) would admit experts a single token decode never sees
            ggml_moe_cache_assign_freeze(op);
        } else if (g.scores != nullptr) {
            ggml_moe_cache_set_scores(op, g.scores, g.lru.grow_q16, il == g.layers[0]);
        }
        return op;
    }

    // k16: route tail skip, Strata's STRATA_ROUTE_TAIL_SKIP (src/program/generate.cpp, src/core/expert_source.cpp at 4b9e05a2): a missing
    // expert that every token of a decode / verify window routes at rank >= 7 (of the router's top k, in routing order) is neither copied
    // nor computed; its contribution is zero and the other experts keep their router weights. Off by default since k19-tailskip-off (see
    // the end of tail_skip_init); LLAMA_MOE_ROUTE_TAIL_SKIP=N opts in. The previous default (rank 7) applied where Strata has it on by
    // default: CUDA (and HIP RDNA4, decided by the backend), unless every expert is on the device (no cache, nothing to skip), 90% or
    // more of the experts are on the device (resident layers + cache slots, LLAMA_MOE_ROUTE_TAIL_SKIP_MAX_CACHED=<percent>), or the
    // experts are NVFP4. The graph also leaves it off for windows of several sequences (Strata: off with --batch slots), prompt ubatches
    // (not every row an output), more than 8 tokens (LLAMA_MOE_ROUTE_TAIL_SKIP_MAX_TOKENS; Strata's verify window) and experts with
    // biases (the bias would still be added). LLAMA_MOE_ROUTE_TAIL_SKIP=0: off (the exact path); =N: rank N, also where the default is off.
    // Changes the output (Strata: KL 0.003-0.006, top-1 ~98%, docs/DETAILS.md)
    void tail_skip_init(const llama_model & model, layer_set set) {
        pf_quiet = model.hparams.no_alloc; // k19-g4pref: the fit's dry instances do not log the prefetch decision
        {
            const char * e       = getenv("LLAMA_MOE_HYBRID_WIDE");
            const char * e_exact = getenv("LLAMA_MTP_EXACT");
            const char * e_miss  = getenv("LLAMA_MOE_CPU_MISS");
            const bool exact = (e_exact != nullptr && atoi(e_exact) != 0) || (e_miss != nullptr && atoi(e_miss) == 0);
            const int  v     = e != nullptr ? atoi(e) : 0;
            hybrid_wide = v > 0 && !exact;
            wide_max    = v > 1 ? std::min<int64_t>(n_small_max, v) : 8;
            const char * e8  = getenv("LLAMA_MOE_PLACE8_UB");
            const char * e8s = getenv("LLAMA_MOE_PLACE8_SINGLE");
            const char * e8m = getenv("LLAMA_MOE_PLACE8_MID"); // k19-placemid: the 8 GB guard only (k19-nanfix is merged)
            place8_single = e8 != nullptr && atoi(e8) > 0 && (e8s == nullptr || atoi(e8s) != 0) && (e8m == nullptr || atoi(e8m) == 0);
            if (hybrid_wide && !model.hparams.no_alloc) {
                LLAMA_LOG_INFO("%s: k19-hostrow: hybrid small batches of up to %" PRId64 " tokens may outnumber the slots (LLAMA_MOE_HYBRID_WIDE)\n",
                    __func__, wide_max);
            }
        }
        const char * env = getenv("LLAMA_MOE_ROUTE_TAIL_SKIP");
        if (const char * e = getenv("LLAMA_MOE_ROUTE_TAIL_SKIP_MAX_TOKENS")) {
            tail_max_tokens = std::max<int64_t>(1, std::min<int64_t>(n_small_max, atoll(e)));
        }
        tail_rank = 0;
        tail_dflt = false;
        if (!device_lookup || groups.empty()) {
            return;
        }
        if (env != nullptr) {
            tail_rank = std::max(0, std::min(255, atoi(env)));
            tail_rank = tail_rank < (int32_t) n_expert_used ? tail_rank : 0;
            if (tail_rank > 0 && !model.hparams.no_alloc) {
                LLAMA_LOG_INFO("%s: route tail skip: rank %d (LLAMA_MOE_ROUTE_TAIL_SKIP): a missing expert that every token of a decode / "
                    "verify window of at most %" PRId64 " tokens routes at rank %d or lower is neither fetched nor computed (changes the output)\n",
                    __func__, tail_rank, tail_max_tokens, tail_rank);
            }
            return;
        }
        // k15-pp: off by default in the exact modes, where the text must not depend on what the cache holds: batch-invariant verify
        // (LLAMA_MTP_EXACT=1) or the miss path pinned to uploads (LLAMA_MOE_CPU_MISS=0). G4 fn4 24 GB and local fn1, pinned + strict
        // exact: with the default rank-7 skip the text followed the cache size, the drafter and the prefetch depth (a verify window
        // skips only where every token routes the expert at rank >= 7, a decode decides per token); with it off, drafter = no draft
        // and one md5 at every cache size. LLAMA_MOE_ROUTE_TAIL_SKIP=N still sets it; LLAMA_MOE_ROUTE_TAIL_SKIP_EXACT=1 keeps the default
        // rule in the exact modes (the kill switch of k15-fn's ab146924b36, which this replaces)
        {
            const char * e_exact = getenv("LLAMA_MTP_EXACT");
            const char * e_miss  = getenv("LLAMA_MOE_CPU_MISS");
            const char * e_keep  = getenv("LLAMA_MOE_ROUTE_TAIL_SKIP_EXACT");
            const bool exact = (e_exact != nullptr && atoi(e_exact) != 0) || (e_miss != nullptr && atoi(e_miss) == 0);
            if (exact && !(e_keep != nullptr && atoi(e_keep) != 0)) {
                if (!model.hparams.no_alloc) {
                    LLAMA_LOG_INFO("%s: route tail skip off: exact mode (LLAMA_MTP_EXACT=1 or LLAMA_MOE_CPU_MISS=0; LLAMA_MOE_ROUTE_TAIL_SKIP=N "
                        "sets it, LLAMA_MOE_ROUTE_TAIL_SKIP_EXACT=1 keeps the default rule)\n", __func__);
                }
                return;
            }
        }
        // Strata's share: experts on the device (resident layers and cache slots) over all experts of the cached set
        double n_all = 0.0;
        double n_dev = 0.0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            if (!llama_moe_cache_in_set(model, set, il)) {
                continue;
            }
            const auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty()) {
                continue;
            }
            n_all += (double) experts[0]->ne[2];
            if (!std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                n_dev += (double) experts[0]->ne[2];
            }
        }
        bool nvfp4 = false;
        for (const group & g : groups) {
            n_dev += (double) g.n_slots;
            for (const ggml_tensor * t : g.ref) {
                nvfp4 = nvfp4 || t->type == GGML_TYPE_NVFP4;
            }
        }
        const double frac = n_all > 0.0 ? std::min(1.0, n_dev/n_all) : 1.0;
        const char * e_max = getenv("LLAMA_MOE_ROUTE_TAIL_SKIP_MAX_CACHED");
        const double max_frac = e_max != nullptr ? atof(e_max)/100.0 : 0.90;
        const int32_t rank = 7;
        if (model.hparams.no_alloc) {
            return;
        }
        if (rank >= (int32_t) n_expert_used) {
            LLAMA_LOG_INFO("%s: route tail skip off: %u routed experts per token (rank %d)\n", __func__, n_expert_used, rank);
        } else if (nvfp4) {
            LLAMA_LOG_INFO("%s: route tail skip off by default with NVFP4 experts (LLAMA_MOE_ROUTE_TAIL_SKIP=%d turns it on)\n", __func__, rank);
        } else if (frac >= max_frac) {
            LLAMA_LOG_INFO("%s: route tail skip off by default: %.1f%% of the experts are on the device (at least %.0f%%, "
                "LLAMA_MOE_ROUTE_TAIL_SKIP_MAX_CACHED; LLAMA_MOE_ROUTE_TAIL_SKIP=%d turns it on)\n", __func__, 100.0*frac, 100.0*max_frac, rank);
        } else {
            // k19-tailskip-off: off by default. The rule drops routed experts without renormalising, and its output error grows with
            // context: fn1 1x T4 long48 decode KLD 0.0268 vs a 0.0182 band (top-1 -1 pt), G4 Flash-Next 12 GB ctx 262144 on the
            // shipping q8_0 placement 0.0211 with it vs 0.0050 without (band 0.0043). It stays available as an explicit opt-in
            // (LLAMA_MOE_ROUTE_TAIL_SKIP=N); the exact path is the default
            LLAMA_LOG_INFO("%s: route tail skip off by default (%.1f%% of the experts on the device; approximate, changes the output: "
                "LLAMA_MOE_ROUTE_TAIL_SKIP=%d turns it on)\n", __func__, 100.0*frac, rank);
        }
    }

    void tail_skip(ggml_tensor * op, int32_t il, int64_t n_tokens) const {
        if (tail_rank <= 0 || op == nullptr || n_tokens > tail_max_tokens || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return;
        }
        const group & g = groups[layers[il].ig];
        // the skips are counted in the use counters' statistics row (the assign reads them anyway, the lookup gets them as a source)
        // (an assign reads g.counts as src[3], a lookup gets it as an extra source)
        ggml_tensor * stat = g.counts != nullptr && g.counts->ne[0] >= 6 ? g.counts : nullptr;
        ggml_moe_cache_set_tail_skip(op, tail_rank, tail_dflt, stat);
    }

    // k19-l4full: route tail skip for host lookups (see llama_moe_cache::tail_skip_host). A layer streamed through the weight store
    // (and the RAM tier) is looked up by the copy callback of its slot map get_rows, not by the device op, so the k16 rule announced
    // at init ("rank 7 by default") never ran there: full Colab L4 + fn4nd on the VM disk (dt_l6) skipped 0 experts while Strata
    // skipped 58% of its decode misses with the same rule. The rule is the device op's (moe_cache_tail_skip): an entry at rank >= the
    // threshold whose expert is not in a device slot before this lookup and that no entry of the window routes at a lower rank is
    // left out of the plan (not admitted, not fetched, no copy back), its slot stays -1 and the product's rows are 0. The default is
    // taken on CUDA devices only (their MUL_MAT_ID zeroes the rows of slot -1; the device op takes it on NVIDIA and RDNA4, a host
    // lookup can run on other backends), an explicit LLAMA_MOE_ROUTE_TAIL_SKIP=N on CUDA / ROCm. LLAMA_MOE_ROUTE_TAIL_SKIP_HOST=0 off
    static constexpr int32_t tail_host_tag = 0x7a110000;
    static bool tail_host_on() {
        static const bool on = [] {
            const char * e = getenv("LLAMA_MOE_ROUTE_TAIL_SKIP_HOST");
            return e == nullptr || atoi(e) != 0;
        }();
        return on;
    }
    uint64_t tail_host_skipped = 0; // entries left out by host lookups
    uint64_t tail_host_experts = 0; // distinct (layer, expert) per lookup left out
    std::vector<uint32_t> tail_head_gen; // [n_expert] generation of the experts some entry routes at a rank below the threshold
    uint32_t tail_gen = 0;
    std::vector<int32_t>  ids_kept;      // the entries of a host lookup the plan sees

    void tail_skip_host(ggml_tensor * op, int32_t il, int64_t n_tokens) const {
        if (!tail_host_on() || tail_rank <= 0 || op == nullptr || op->op != GGML_OP_GET_ROWS || n_tokens > tail_max_tokens || il < 0 ||
                il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return;
        }
        const group & g = groups[layers[il].ig];
        if (g.dev_lookup || g.n_slots == 0) {
            return;
        }
        const char * name = ggml_backend_name(devices[g.id].backend);
        const bool cuda = name != nullptr && strncmp(name, "CUDA", 4) == 0;
        const bool rocm = name != nullptr && strncmp(name, "ROCm", 4) == 0;
        if (!(cuda || (rocm && !tail_dflt))) {
            return;
        }
        op->op_params[0] = tail_host_tag | tail_rank;
    }

    // the entries of a host lookup's ids that its plan sees: all, or without the route tail skip's entries (see tail_skip_host)
    void tail_filter_host(const ggml_tensor * lookup, const group & g, int32_t il, const std::vector<int32_t> & ids,
            const int32_t *& out, size_t & n_out) {
        out   = ids.data();
        n_out = ids.size();
        const int32_t tag  = lookup->op_params[0];
        const int32_t rank = (tag & 0xffff0000) == tail_host_tag ? (tag & 0xff) : 0;
        const int64_t neu  = n_expert_used;
        if (rank <= 0 || neu <= rank || ids.empty() || ids.size() % (size_t) neu != 0 || (int64_t) ids.size() > tail_max_tokens*neu ||
                g.lru.slot_map[il] == nullptr) {
            return;
        }
        const int32_t * sm = g.lru.slot_map[il];
        const int32_t n_expert = g.lru.n_expert;
        if ((int32_t) tail_head_gen.size() < n_expert) {
            tail_head_gen.assign(n_expert, 0);
        }
        if (++tail_gen == 0) {
            std::fill(tail_head_gen.begin(), tail_head_gen.end(), 0);
            tail_gen = 1;
        }
        bool any = false;
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert) {
                return; // the plan asserts on it, as before
            }
            if ((int64_t) (i % neu) < rank) {
                tail_head_gen[e] = tail_gen;
            }
        }
        for (size_t i = 0; i < ids.size() && !any; ++i) {
            const int32_t e = ids[i];
            any = (int64_t) (i % neu) >= rank && sm[e] < 0 && tail_head_gen[e] != tail_gen;
        }
        if (!any) {
            return;
        }
        ids_kept.clear();
        const uint32_t gen_skip = ++tail_gen; // marks the distinct skipped experts (never 0: the head marks used the previous value)
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t e = ids[i];
            if ((int64_t) (i % neu) >= rank && sm[e] < 0 && tail_head_gen[e] != gen_skip - 1) {
                tail_host_skipped++;
                if (tail_head_gen[e] != gen_skip) {
                    tail_head_gen[e] = gen_skip;
                    tail_host_experts++;
                }
                continue;
            }
            ids_kept.push_back(e);
        }
        out   = ids_kept.data();
        n_out = ids_kept.size();
    }

    // k19-l4full LLAMA_MOE_HOST_PROF (diagnostic, default 0 = off; 1 = host timers of the small-batch host lookups, 2 = also waits for
    // the lookup's copies before the device computes the layer, which splits the next lookup's sync into compute and copies; 2 changes
    // timing, never what is computed)
    static int host_prof_mode() {
        static const int m = [] {
            const char * e = getenv("LLAMA_MOE_HOST_PROF");
            return e != nullptr ? atoi(e) : 0;
        }();
        return m;
    }
    struct {
        uint64_t n = 0, sync_us = 0, plan_us = 0, up_us = 0, demote_us = 0, ram_us = 0, disk_us = 0, commit_us = 0, copy_us = 0;
        uint64_t n_disk = 0, n_skip_lookups = 0;
    } hprof;

    // k19-ramevict: missing experts the hybrid assign left to the host because of its admission margin (LLAMA_MOE_CACHE_ADMIT_MARGIN)
    uint64_t admit_rejected() const {
        uint64_t n = 0;
        for (const group & g : groups) {
            if (g.counts == nullptr || g.counts->data == nullptr || !g.dev_lookup || g.counts->ne[0] < 8) {
                continue;
            }
            uint64_t cnt = 0;
            ggml_backend_tensor_get(g.counts, &cnt, ((size_t) g.counts->ne[0]*(g.counts->ne[1] - 1) + 6)*sizeof(int32_t), sizeof(cnt));
            n += cnt;
        }
        return n;
    }

    uint64_t tail_skipped() const {
        uint64_t n = tail_host_skipped;
        for (const group & g : groups) {
            if (g.counts == nullptr || g.counts->data == nullptr || !g.dev_lookup || g.counts->ne[0] < 6) {
                continue;
            }
            uint64_t cnt = 0;
            ggml_backend_tensor_get(g.counts, &cnt, ((size_t) g.counts->ne[0]*(g.counts->ne[1] - 1) + 4)*sizeof(int32_t), sizeof(cnt));
            n += cnt;
        }
        return n;
    }

    // k13: frozen small batches of the CPU-miss rule (cm_freeze): the host computes every missing expert of a small batch, nothing is
    // fetched and the assign admits nothing, not even to free slots (LLAMA_MOE_CACHE_FETCH=0 keeps its k12 behaviour: it fills free
    // slots, so it is not frozen)
    bool split_frozen() const {
        return cm && cm_frozen && !cm_frozen_fetch;
    }

    // k19-g4pref cross-layer prefetch: in decode / verify windows of at most LLAMA_MOE_PREFETCH_MAX_TOKENS (default 8) tokens, a cache
    // layer whose previous MoE layer is a cache layer on the same device (both on the device lookup) admits, beside the graph, the N
    // experts with the largest logits of its router run on the previous layer's MoE input, at most MAX per layer and window; its own
    // lookup waits for those copies. LLAMA_MOE_PREFETCH=N (1..32) on with N (MAX = LLAMA_MOE_PREFETCH_MAX, default 0 = no cap), =0 off.
    // Unset: on by default (N = 10, MAX = 1, LLAMA_MOE_PREFETCH_MAX still sets MAX; LLAMA_MOE_PREFETCH_MIN_TOKENS: windows of fewer
    // tokens do not prefetch, default 1) unless LLAMA_MOE_PREFETCH_AUTO=0, which common sets beside an MTP drafter (the Studio MTP n2
    // shape at depth 4: 48 GB 8k passes 1-5% longer on 5 of 6 prompts, tok/s geomean 0.90; prefetching only 6+ row windows instead cost
    // DFlash2 6.5% in chat and 3.5% in 8k, its EV-auto rule then picks wider windows), where every device holding cache slots is a CUDA
    // device of compute capability 12.0 and the cache holds at least LLAMA_MOE_PREFETCH_AUTO_MIN_FRAC (0.60) of the host experts'
    // bytes, or from LLAMA_MOE_PREFETCH_AUTO_LO_MIN_FRAC (0.10) to less than LLAMA_MOE_PREFETCH_AUTO_LO_FRAC (0.20) of them (N =
    // LLAMA_MOE_PREFETCH_AUTO_LO, default 0 = not in that band; 10 was measured); 8 GB (3.7% cached, single-token decode) lost 1-3.5%.
    // The 16 GB band is opt-in: with DFlash2 it passed (below), with the Studio MTP n2 drafter srv_kld long48 put the first token outside
    // the -fa off band (mean / p99 0.0040 / 0.049 vs 0.0034 / 0.022: the prompt's small tail batch takes the prefetch there). Off in between (32 GB: 41%, chat +7% but 8k -6..-15%) and on other devices (A100 80 GB at 48 GB: level).
    // 16 GB (15.5% cached), G4: chat level / +4.5% (two VMs), 8k +7.7 / +7.9% (other texts: host path); srv_kld long48 inside the band
    // (first-token mean / p99 0.0030 / 0.017 vs 0.0044 / 0.022); reference-scored 1 glitch in 5120 tokens vs 3 with it off. 24 GB
    // (27.1%) is left out: chat level, 8k +5.5..7.6% (other text), but srv_kld long48 put one position above the band's max (0.230 vs
    // 0.093; first-token stats inside)
    // Colab G4 (RTX PRO 6000) Qwen3.8-Flash-Next UD-IQ4_XS DFlash2 n6, 3 interleaved launches per arm: N=10 MAX=1 at 48 GB (70.7%
    // cached) chat 278.4 -> 292.4 and 286.8 -> 303.5 tok/s on two VMs (same 256-token text), 8k 175.2 -> 197.1 (texts vary per launch
    // there in every arm); 16 GB chat +4.5%, 8k +7.9% (other texts: host path); 32 GB (41%) chat +7..9% but 8k -6..-15% (more 3-row
    // windows); A100 80 GB at 48 GB level. 78-91% of the prefetched experts are hit by the next layer. srv_kld long48 at 48 GB inside
    // the -fa off band (first-token mean 0.0002 vs 0.0067), reference-scored texts 0 glitches in 8192 tokens. The text follows the
    // residency only through the route tail skip and the host path: LLAMA_MOE_ROUTE_TAIL_SKIP=0 gives the same text with it on and off
    struct pf_cfg_t {
        int n        = 0;
        int max_fill = 0;
        int min_tok  = 1; // windows of fewer tokens do not prefetch
    };
    mutable bool     pf_resolved = false;
    mutable pf_cfg_t pf_cfg;
    bool             pf_quiet    = false; // a no_alloc (fit) instance: no log

    // the compute capability of the CUDA device of a backend (major*100 + minor*10), -1 if it is not one
    static int pf_backend_cc(ggml_backend_t backend) {
        ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
        ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto cc_fn = reg != nullptr ? (int (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
        return cc_fn != nullptr ? cc_fn(dev) : -1;
    }

    pf_cfg_t prefetch_cfg() const {
        if (pf_resolved) {
            return pf_cfg;
        }
        const char * e  = getenv("LLAMA_MOE_PREFETCH");
        const char * em = getenv("LLAMA_MOE_PREFETCH_MAX");
        const char * et = getenv("LLAMA_MOE_PREFETCH_MIN_TOKENS");
        pf_cfg_t c;
        if (e != nullptr) {
            c.n        = std::max(0, std::min(32, atoi(e)));
            c.max_fill = em != nullptr ? std::max(0, atoi(em)) : 0;
            c.min_tok  = et != nullptr ? std::max(1, atoi(et)) : 1;
            pf_cfg = c;
            pf_resolved = true;
            return c;
        }
        if (cache_frac < 0.0) {
            return c; // the slots are not sized yet: decided later
        }
        // (k19-g4pref-optin: the rule only with LLAMA_MOE_PREFETCH_AUTO=1; srv_kld long48 put one position of the 48 GB DFlash2 cell above
        // the -fa off band's max in 2 of 3 runs, 0.135 / 0.159 vs 0.116, and the 512-token chat texts differ from it off after ~300 tokens)
        // k19-g4tail: LLAMA_MOE_PREFETCH_AUTO=2 (set by the server beside an MTP drafter, see its k19-g4tail rule): only the band where
        // the placed cache holds LLAMA_MOE_G4TAIL_FRAC_LO .. _HI (0.10 .. 0.20) of the host experts, on compute capability 12.0
        // devices with device lookups: top 10 predicted experts, at most 1 copied per layer (generation and verify windows only)
        const char * ea_g4t = getenv("LLAMA_MOE_PREFETCH_AUTO");
        if (ea_g4t != nullptr && atoi(ea_g4t) == 2) {
            const double flo = getenv("LLAMA_MOE_G4TAIL_FRAC_LO") ? atof(getenv("LLAMA_MOE_G4TAIL_FRAC_LO")) : 0.10;
            const double fhi = getenv("LLAMA_MOE_G4TAIL_FRAC_HI") ? atof(getenv("LLAMA_MOE_G4TAIL_FRAC_HI")) : 0.20;
            int  cmin = INT_MAX;
            int  cmax = -1;
            bool anyg = false;
            for (const group & g : groups) {
                if (g.n_slots == 0 || !g.dev_lookup) {
                    continue;
                }
                const int cc = pf_backend_cc(devices[g.id].backend);
                cmin = std::min(cmin, cc);
                cmax = std::max(cmax, cc);
                anyg = true;
            }
            const bool on = anyg && cmin == 1200 && cmax == 1200 && cache_frac >= flo && cache_frac < fhi;
            if (on) {
                c.n        = 10;
                c.max_fill = em != nullptr ? std::max(0, atoi(em)) : 1;
                c.min_tok  = et != nullptr ? std::max(1, atoi(et)) : 1;
            }
            if (!pf_quiet) {
                LLAMA_LOG_INFO("llama_moe_cache%s: k19-g4tail: cross-layer prefetch beside the MTP drafter %s: %.1f%% of the host experts "
                    "cached (band %.2f .. %.2f), %s (top %d predicted experts, at most %d per layer, generation and verify windows only; "
                    "LLAMA_MOE_G4TAIL=0 off)\n", tag, on ? "on" : "off", 100.0*cache_frac, flo, fhi, !anyg ? "no device lookups" :
                    !(cmin == 1200 && cmax == 1200) ? "not a compute capability 12.0 device" : on ? "in the band" : "outside the band", c.n, c.max_fill);
            }
            pf_cfg = c;
            pf_resolved = true;
            return c;
        }
        if (const char * ea = getenv("LLAMA_MOE_PREFETCH_AUTO"); ea == nullptr || atoi(ea) == 0) {
            if (!pf_quiet) {
                LLAMA_LOG_INFO("llama_moe_cache%s: k19-g4pref: cross-layer prefetch off by default: LLAMA_MOE_PREFETCH_AUTO=0 (set beside an "
                    "MTP drafter; LLAMA_MOE_PREFETCH=N on)\n", tag);
            }
            pf_cfg = c;
            pf_resolved = true;
            return c;
        }
        int  cc_min = INT_MAX;
        int  cc_max = -1;
        bool any    = false;
        for (const group & g : groups) {
            if (g.n_slots == 0 || !g.dev_lookup) {
                continue;
            }
            const int cc = pf_backend_cc(devices[g.id].backend);
            cc_min = std::min(cc_min, cc);
            cc_max = std::max(cc_max, cc);
            any    = true;
        }
        const double hi_frac = getenv("LLAMA_MOE_PREFETCH_AUTO_MIN_FRAC") ? atof(getenv("LLAMA_MOE_PREFETCH_AUTO_MIN_FRAC")) : 0.60;
        const double lo_frac = getenv("LLAMA_MOE_PREFETCH_AUTO_LO_FRAC") ? atof(getenv("LLAMA_MOE_PREFETCH_AUTO_LO_FRAC")) : 0.20;
        const int    lo_n    = getenv("LLAMA_MOE_PREFETCH_AUTO_LO") ? std::max(0, std::min(32, atoi(getenv("LLAMA_MOE_PREFETCH_AUTO_LO")))) : 0;
        const double lo_min  = getenv("LLAMA_MOE_PREFETCH_AUTO_LO_MIN_FRAC") ? atof(getenv("LLAMA_MOE_PREFETCH_AUTO_LO_MIN_FRAC")) : 0.10;
        const bool   cc_ok   = any && cc_min == 1200 && cc_max == 1200;
        const char * why     = nullptr;
        if (!any) {
            why = "no device lookups";
        } else if (!cc_ok) {
            why = "not a compute capability 12.0 device";
        } else if (cache_frac >= hi_frac) {
            c.n = 10;
        } else if (lo_n > 0 && cache_frac < lo_frac && cache_frac >= lo_min) {
            c.n = lo_n;
        } else {
            why = lo_n > 0 && cache_frac < lo_min ? "the cached share is below LLAMA_MOE_PREFETCH_AUTO_LO_MIN_FRAC" :
                  lo_n > 0 ? "the cached share is in the band between LLAMA_MOE_PREFETCH_AUTO_LO_FRAC and _MIN_FRAC"
                           : "the cached share is below LLAMA_MOE_PREFETCH_AUTO_MIN_FRAC";
        }
        if (c.n > 0) {
            c.max_fill = em != nullptr ? std::max(0, atoi(em)) : 1;
            c.min_tok  = et != nullptr ? std::max(1, atoi(et)) : 1;
        }
        if (!pf_quiet && any) {
            if (c.n > 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: k19-g4pref: cross-layer prefetch on by default: compute capability 12.0, %.1f%% of the host "
                    "experts cached: top %d predicted experts per token, at most %d per layer (LLAMA_MOE_PREFETCH=0 off)\n", tag,
                    100.0*cache_frac, c.n, c.max_fill);
            } else {
                LLAMA_LOG_INFO("llama_moe_cache%s: k19-g4pref: cross-layer prefetch off by default: %s (compute capability %d, %.1f%% of the "
                    "host experts cached; LLAMA_MOE_PREFETCH=N on)\n", tag, why, cc_max, 100.0*cache_frac);
            }
        }
        pf_cfg = c;
        pf_resolved = true;
        return c;
    }
    mutable uint64_t n_prefetch_ops = 0; // engagement: prefetch ops built

    // k19-g4tail: this batch is a prompt's and the prefetch is on: no prefetch op (see prefetch_ok); also a graph key (hybrid_mode)
    bool pf_prompt_off() const {
        // (LLAMA_MOE_G4TAIL=0 too: the int3 behaviour of an explicit LLAMA_MOE_PREFETCH)
        static const bool pf_prompt = [] {
            const char * e = getenv("LLAMA_MOE_PREFETCH_PROMPT");
            const char * k = getenv("LLAMA_MOE_G4TAIL");
            return e != nullptr ? atoi(e) != 0 : k != nullptr && atoi(k) == 0;
        }();
        return cm_prompt_now && !pf_prompt && pf_resolved && pf_cfg.n > 0;
    }

    bool prefetch_ok(int32_t il_prev, int32_t il, int64_t n_ubatch) const {
        static const int64_t max_tok = getenv("LLAMA_MOE_PREFETCH_MAX_TOKENS") != nullptr ?
            std::max<int64_t>(1, atoll(getenv("LLAMA_MOE_PREFETCH_MAX_TOKENS"))) : 8;
        if (force_large || n_ubatch < 1 || n_ubatch > max_tok || n_ubatch > n_small_max || prefetch_cfg().n <= 0 ||
                n_ubatch < prefetch_cfg().min_tok) {
            return false;
        }
        // (the groups are per expert type, so neighbouring layers are often in different groups; both on the same device)
        if (il_prev < 0 || il <= il_prev || il >= (int32_t) layers.size() || layers[il].ig < 0 || layers[il_prev].ig < 0) {
            return false;
        }
        const group & g  = groups[layers[il].ig];
        const group & gp = groups[layers[il_prev].ig];
        if (g.id != gp.id || !use_device(g, n_ubatch) || !slots_fit(g, n_ubatch) || layers[il].host_dev.size() != g.banks.size() ||
                !use_device(gp, n_ubatch) || layers[il_prev].host_dev.size() != gp.banks.size()) {
            return false;
        }
        // frozen small batches admit nothing (k13), and the prefetch admits
        if (cm && cm_frozen) {
            return false;
        }
        // k19-g4tail: generation and verify windows only (every token an output). A prompt's small tail batch took the prefetch and
        // moved experts between the host and the device for the first generated token (G4 16 GB long48 first-token KLD outside the
        // band, plans/k19_g4pref.md round 5). LLAMA_MOE_PREFETCH_PROMPT=1: prompt tails too (as before). A prompt tail graph has fewer
        // outputs than tokens, so it is never reused for a verify window of the same size or the reverse
        if (pf_prompt_off()) {
            return false;
        }
        return true;
    }

    ggml_tensor * prefetch(ggml_context * ctx, int32_t il, ggml_tensor * logits, ggml_tensor * dep, int64_t n_ubatch) const {
        GGML_UNUSED(n_ubatch);
        const pf_cfg_t cfg = prefetch_cfg();
        const int max_fill = cfg.max_fill;
        const int n_top = std::min<int>(cfg.n, (int) logits->ne[0]);
        if (n_top <= 0 || n_top*logits->ne[1] > 256) {
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        std::vector<ggml_tensor *> banks = op_banks(g);
        std::vector<const void *>  host  = l.host_dev;
        ggml_tensor * op = ggml_moe_cache_prefetch(ctx, logits, g.map, g.state, banks.data(), host.data(), (int) banks.size(), il, n_top,
            max_fill, g.lru.n_core, g.scores, dep);
        if (n_prefetch_ops++ == 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-g4pref: cross-layer prefetch of the top %d predicted experts per token (max %d per layer, "
                "0 = no cap; LLAMA_MOE_PREFETCH=0 off)\n", tag, n_top, max_fill);
        }
        return op;
    }

    ggml_tensor * hybrid_fetch_experts(ggml_context * ctx, int32_t il, ggml_tensor * res, bool ov) const {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        std::vector<ggml_tensor *> banks = op_banks(g);
        std::vector<const void *>  host  = l.host_dev;
        return ggml_moe_cache_fetch(ctx, res, banks.data(), host.data(), (int) banks.size(), ov);
    }

    bool hybrid_active() const {
        if (cm) {
            return cm_any_on();
        }
        return hybrid && hybrid_live && hybrid_share < 1.0f;
    }

    // fixed share (LLAMA_MOE_CACHE_FETCH, or the default share from the threads): which missing experts the host computes depends on the cached experts, the use counters and
    // the rounding residual of the share, which all carry over from earlier batches; the context calls this before a batch whose
    // sequences are all empty (a new request) and after a state is read, so the output does not depend on earlier requests:
    // the cache starts empty as in a new process (the plain path computes the same with any cache contents, it is not reset)
    void hybrid_reset(bool force = false) {
        if (!force && (!hybrid || !hybrid_fixed)) {
            return;
        }
        seed_counts.clear();
        for (group & g : groups) {
            if (g.n_slots == 0 || !g.dev_lookup || g.counts == nullptr || g.counts->data == nullptr) {
                continue;
            }
            // k19-fixB: released banks (llama_moe_cache::release, the tiny-ubatch rule) have no memory to clear; the caller restores
            // them first (llama_context::tiny_rel_leave), this only keeps a missed caller from a memset on a tensor without a buffer
            if (devices[g.id].released) {
                LLAMA_LOG_WARN("%s: MoE cache banks of device %d are released: not reset\n", __func__, g.id);
                continue;
            }
            for (int32_t il : g.layers) {
                if ((size_t) il < refill.size()) {
                    refill[il].clear();
                }
            }
            if (g.lru.n_lent > 0) {
                unlend(g);
            }
            ggml_backend_synchronize(devices[g.id].backend);
            seed_end(g);
            for (int32_t il : g.layers) {
                std::fill(g.lru.slot_map[il], g.lru.slot_map[il] + g.lru.n_expert, -1);
            }
            g.lru.clear();
            // the banks as allocated (cleared), kernels may read a little past the end of an expert
            // (k17-small: the slots left and the spare slot; the rows shrink retired are not backed)
            for (ggml_tensor * bank : g.banks) {
                // (k17-small: the slots left and the spare slot, not the retired rows; k19-ptres: the core keeps its experts)
                ggml_backend_tensor_memset(bank, 0, (size_t) g.lru.n_core*bank->nb[2], (size_t) (g.n_slots + 1 - g.lru.n_core)*bank->nb[2]);
            }
            // the use counters and the residual; the count of experts computed on the host (statistics, stat[0..1]) is kept
            const size_t n_cnt = (size_t) g.counts->ne[0]*(g.counts->ne[1] - 1);
            ggml_backend_tensor_memset(g.counts, 0, 0, n_cnt*sizeof(int32_t));
            // (k16: the count of route tail skips, stat[4..5], is kept too)
            ggml_backend_tensor_memset(g.counts, 0, (n_cnt + 2)*sizeof(int32_t), 2*sizeof(int32_t));
            if (g.counts->ne[0] > 6) {
                ggml_backend_tensor_memset(g.counts, 0, (n_cnt + 6)*sizeof(int32_t), ((size_t) g.counts->ne[0] - 6)*sizeof(int32_t));
            }
            // the device state is replaced by the empty host state before the next device lookup (see prepare)
            g.dev_newer = false;
            g.dev_gen   = UINT64_MAX;
        }
    }

    // the overlap only for single token ubatches: their expert matrix products (mul_mat_vec_q) compute every slot on its own, so
    // splitting the slots between two chains gives the same results; larger ubatches may pick kernels whose tiling depends on the slots
    bool overlap_batch(int64_t n_ubatch) const {
        return overlap && ov_live && n_ubatch == 1;
    }

    // k19-overlap LLAMA_MOE_CACHE_FETCH_OVERLAP_SMALL=N (1..GGML_MOE_MAILBOX_MAX_TOKENS; default 4, 1 = off): ubatches of 2..N tokens that
    // take the hybrid path (the CPU-miss rule has the host path on for them, see hybrid_layer) copy their admitted experts on the side
    // stream beside the hits (cached before the lookup), then compute the admitted experts, as single tokens do. Every (token, slot) entry
    // is computed once by the multi-token mmvq kernels, which give each entry its own sums, so the output is the serial fetch's.
    // Independent of the single-token rule (ov_live). Colab G4 Flash-Next UD-IQ4_XS MTP n2 (3-row windows; ov_r2, md5 equal): 32 GB
    // chat / 8k +5.1 / +5.8%, 24 GB +2.9 / +4.0% (device beside the host 100 -> 83 us per layer). DFlash2 7-row windows (N = 8): 32 GB
    // +3.6 / +5.9% but 24 GB -2.5 / -3.6%, so the default stops at 4. Ubatches whose host path is off (48 GB) keep the plain lookup:
    // taking the hybrid lookup there for the overlap changed the text (ov_r2), so hybrid_layer does not widen for it
    // k19-g4pass (default on; LLAMA_MOE_G4PASS=0 or an explicit LLAMA_MOE_CACHE_FETCH_OVERLAP_SMALL: off): a window of 5..8 rows that
    // publishes its host experts to the single-graph mailbox (single_usable: a many-core host, see single_max) takes the overlap too.
    // The 24 GB loss above was the k19-overlap round 2 bin (before LEAN, 2 launches). On the mailbox the admitted experts' copies
    // (moe_cache_fetch, 8.0 of 29.8 ms per traced w7 pass at 16 GB) ran serially before the hits while the host computed; beside the
    // hits they shorten the device side of every host layer. Colab G4 Flash-Next UD-IQ4_XS DFlash2 n6 16 GB, same text: w7 24.9 -> 23.8
    // ms, chat 179.1 -> 186.6 tok/s, 8k 126.9 -> 136.5 (gp_p1, 2 launches). Windows on the eager split (fewer CPUs, smaller caches)
    // keep the 4-token limit. Only below LLAMA_MOE_G4PASS_OV_MAX_FRAC (default 0.20) of the host experts cached, the share where the
    // wide mailbox is new (see single_max) and the copies are largest. gp_ab1 / gp_ab2 (G4, 3-4 interleaved launches, same text): 16 GB
    // chat 167.2 -> 178.3 (launches 2+: 166-168 -> 174.9-186.1), 8k 120.6 -> 134.8; above the share it was mixed: 24 GB (27.5%) chat
    // 203.7 -> 199.3 median over 7 launches (head 189-211, its host batches slower in the slow launches), 8k +2.4%; 32 GB (42%) chat /
    // 8k +4.3 / +2.9% (3 launches). LLAMA_MOE_G4PASS_OV_MAX_FRAC=1: every mailbox window
    static bool g4pass_on() {
        static const bool on = [] {
            const char * e = getenv("LLAMA_MOE_G4PASS");
            return e == nullptr || atoi(e) != 0;
        }();
        return on;
    }
    bool overlap_small_batch(int64_t n_ubatch) const {
        static const char * e = getenv("LLAMA_MOE_CACHE_FETCH_OVERLAP_SMALL");
        static const int64_t mx = e != nullptr ? std::max<int64_t>(1, std::min<int64_t>(GGML_MOE_MAILBOX_MAX_TOKENS, atoll(e))) : (int64_t) 4;
        if (!overlap || n_ubatch <= 1) {
            return false;
        }
        static const double ov_max_frac = getenv("LLAMA_MOE_G4PASS_OV_MAX_FRAC") != nullptr ? atof(getenv("LLAMA_MOE_G4PASS_OV_MAX_FRAC")) : 0.20;
        if (e == nullptr && n_ubatch > mx && g4pass_on() && cache_frac >= 0.0 && cache_frac < ov_max_frac && single_usable(n_ubatch)) {
            n_ov_wide_host++;
            return true;
        }
        if (e == nullptr && n_ubatch > mx && g4pass_on() && g4mid_ov_band() && single_usable(n_ubatch)) {
            n_ov_wide_mid++;
            return true;
        }
        return n_ubatch <= mx || overlap_plain_wide(n_ubatch);
    }
    mutable uint64_t n_ov_wide_host = 0; // k19-g4pass: overlap_small_batch calls answered by the mailbox rule above (engagement)
    mutable uint64_t n_ov_wide_mid  = 0; // k19-g4mid: the same, answered by the band rule (g4mid_ov_band)

    // k19-g4mid (default on; LLAMA_MOE_G4MID=0 off): the side-stream copy overlap of a 5..8-row mailbox window (k19-g4pass above) also where
    // the cache holds LLAMA_MOE_G4MID_OV_MIN_FRAC (0.35) .. LLAMA_MOE_G4MID_OV_MAX_FRAC (0.60) of the host experts (G4 fn4n 32 GB: 41.4%).
    // There a wide window's device side (the admitted experts' copies, then the hits) is the longer of the two per layer: Colab G4
    // Flash-Next UD-IQ4_XS DFlash2 n6 32 GB chat, GGML_MOE_MAILBOX_TIMING: device beside the host 126.2 us, host 115.4 us per batch, merge
    // wait 23.5 us; nsys (node trace, 0am): moe_cache_fetch of a 7-row window 110-125 us per layer, serial before the hits. Beside the hits
    // the copies leave the host as the longer side, so the pass shortens by the device's excess (w7 21.7 -> 20.1-21.5 ms; gm_p1 / gp_ab1 /
    // gp_ab3, same text in every cell: chat +1.4..+4.3%, 8k +2.7..+3.0%; gm_s3, 3 launches: device beside the host 123.4 -> 106.0 us, host
    // 114-123 us, chat 206.0 -> 214.0, 8k 176.2 -> 180.4; gm_ab32, 3 interleaved launches vs the base build: chat 203.6 -> 207.5, 8k
    // 174.7 -> 180.5, with LLAMA_SPEC_PREWARM_CACHED_FRAC=0.3 chat 223.1 -> 233.1, 8k 189.7 -> 194.9, md5 equal in every cell; 48 GB
    // (70.7%) is outside the band, its windows have the host path off). Not below the band by default: at 24 GB (27.5% cached) the device
    // side and the host batch are already level (144.9 vs 145.6-163.7 us), so with the overlap the pass follows the host batch time, which varies by
    // launch on the G4 host (145-170 us, in both arms): gp_ab1 + gp_ab2 chat median -2.2% over 7 launches (spread 10.8%), gm_s3 +3.9% chat
    // / +2.5% 8k over 3 (host 144-146 us in every overlapped launch). LLAMA_MOE_G4MID_OV_MIN_FRAC=0.2 takes 24 GB in
    static bool g4mid_on() {
        static const bool on = [] {
            const char * e = getenv("LLAMA_MOE_G4MID");
            return e == nullptr || atoi(e) != 0;
        }();
        return on;
    }
    bool g4mid_ov_band() const {
        static const double lo = getenv("LLAMA_MOE_G4MID_OV_MIN_FRAC") != nullptr ? atof(getenv("LLAMA_MOE_G4MID_OV_MIN_FRAC")) : 0.35;
        static const double hi = getenv("LLAMA_MOE_G4MID_OV_MAX_FRAC") != nullptr ? atof(getenv("LLAMA_MOE_G4MID_OV_MAX_FRAC")) : 0.60;
        return g4mid_on() && cache_frac >= 0.0 && cache_frac >= lo && cache_frac < hi;
    }

    // k19-cohfix (open): Colab G4 Qwen3.8-Flash-Next UD-IQ4_XS @48 GB, DFlash2 n6, 8k cell: the last row of a 7-row verify can pick " One"
    // where a full-card reference gives " America" p 0.97 (" One" outside its top 10); LLAMA_MOE_OVERLAP_PLAIN_MAX=4 (this overlap off)
    // gives " America" in the same window. Not pinned (CUDA graphs off, GGML_CUDA_GRAPH_CHUNK_SIZE=0 and GGML_CUDA_MOE_FETCH_OVERLAP_Z=0
    // clear it too) and not shown to be more frequent with it (reference rescore, 2 VMs: 1 wrong token in 14.3k with the default, 0 in
    // 24.6k with =4), so the default stays 8 (plans/k19_coherence.md)
    // k19-bigverify LLAMA_MOE_OVERLAP_PLAIN_MAX=N (default 8, at most GGML_MOE_MAILBOX_MAX_TOKENS; <= 4 = off): verify windows of 5..N tokens
    // whose host path is off (the plain lookup, k19-q4xcache) take the overlap too: their missing experts are copied on the side stream
    // beside the cached ones instead of in the compute stream before any expert of the layer. Windows with the host path on keep the 4-token
    // limit above (DFlash2 7-row windows on the hybrid path: -2.5 / -3.6% at 24 GB, k19-overlap). Same results as the plain lookup (see
    // overlap_plain_on). Colab G4 Qwen3.8-Flash-Next UD-IQ4_XS DFlash2 n6 at a 48 GB cap (70.7% of the host experts cached, host path off),
    // bv_s1g: 208.1 -> 224.5 tok/s chat, same text
    bool overlap_plain_wide(int64_t n_ubatch) const {
        static const int64_t mxp = [] {
            const char * e = getenv("LLAMA_MOE_OVERLAP_PLAIN_MAX");
            return e != nullptr ? std::max<int64_t>(1, std::min<int64_t>(GGML_MOE_MAILBOX_MAX_TOKENS, atoll(e))) : std::min<int64_t>(GGML_MOE_MAILBOX_MAX_TOKENS, 8);
        }();
        if (n_ubatch > mxp || !overlap_plain_on() || force_large) {
            return false;
        }
        // the host path is off for this size: what hybrid_assign would treat as the plain lookup
        return cm ? cm_share_of(n_ubatch) >= 1.0f : !hybrid_active();
    }

    // k19-l4c deferred admission (LLAMA_MOE_FETCH_DEFER, default 1; 0 = off, 2 = also on hosts without the host team): a single token
    // ubatch on the overlapped hybrid path with the single-graph mailbox hands the experts its assign admits to the host as well, and
    // their copies to the slots run beside the rest of the graph, waited for only at its end (ggml_moe_cache_set_defer). Before, the
    // layer waited for the copies (zero-copy reads at the link speed, ~470 us per 5.4 MiB expert on Colab L4) to compute the admitted
    // experts on the device. nsys (l4c_p1, i3f, Colab L4, chat, ms per pass): LFM2.5 8 GB 6.7 of 21.2 waiting for copies (4.1 for the host),
    // LFM2.5 4 GB 12.4 of 27.5, Flint 16 GB 5.0 of 22.8; PolyStrata stages its experts with DMA beside the compute and its CPU computes the
    // misses of the token. Only on hosts whose host experts run on the host team (single_team_host: Colab L4 / A100 class; not the G4
    // EPYC, whose host has no team, nor a 4-vCPU T4). The admissions and the cache state are the same as without it; the admitted
    // experts are computed on the host (int8 activations) instead of the device, so the output changes: KLD-gated
    bool fetch_defer(int64_t n_ubatch) const {
        const int mode = fetch_defer_mode();
        // LLAMA_MOE_FETCH_DEFER_MAX_TOKENS (default 1): small batches of up to that many tokens (the mailbox's sizes) defer too (verify
        // windows; a single token needs the overlap, which keeps the hits apart from the admitted experts)
        static const int64_t max_tok = getenv("LLAMA_MOE_FETCH_DEFER_MAX_TOKENS") != nullptr ?
            std::max<int64_t>(1, atoll(getenv("LLAMA_MOE_FETCH_DEFER_MAX_TOKENS"))) : 1;
        if (mode == 0 || n_ubatch < 1 || n_ubatch > max_tok || (n_ubatch == 1 && !overlap_batch(n_ubatch)) || !single_req || single_off) {
            return false;
        }
        // only where the cache holds more than LLAMA_MOE_FETCH_DEFER_MIN_FRAC (0.25) and at most LLAMA_MOE_FETCH_DEFER_MAX_FRAC (0.5) of the
        // host experts (fetch_defer_frac): a host-bound cache (Colab L4 LFM2.5 4 GB, 16% cached) already keeps the host team busy for
        // the whole step, the admitted experts on top of it cost more than the copies they save (l4c_a2: chat 42.9 vs 44.7 tok/s);
        // LFM2.5 8 GB (40%) +4% alone, +8-12% with the team of 10 waiting blocked; Flint 16 GB (56%): see fetch_defer_frac
        const bool on = (mode == 2 || single_team_host()) && fetch_defer_frac();
        static std::atomic<bool> logged{false};
        if (on && !logged.exchange(true)) {
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-l4c deferred admission: single token ubatches compute the admitted experts on the host, their "
                "copies to the slots finish beside the graph (LLAMA_MOE_FETCH_DEFER=0 off)\n", tag);
        }
        return on;
    }

    bool ov_pending() const {
        return ov_rule || (ov_auto && (ov_done >= 0 || ov_wait >= 0));
    }

    // the next single token tick of the rule reads the device counters (the backends must be synchronized)
    bool ov_rule_reads() const {
        return ov_rule && ((!ov_have0 && ov_seen >= ov_rule_skip) || (ov_have0 && ov_n1 + 1 >= ov_win));
    }

    uint64_t ov_miss_bytes() const {
        uint64_t b = 0;
        for (const group & g : groups) {
            if (!g.dev_lookup || g.n_slots == 0 || g.state == nullptr || g.state->data == nullptr) {
                continue;
            }
            uint64_t cnt[2] = { 0, 0 };
            ggml_backend_tensor_get(g.state, cnt, 2*sizeof(int32_t), sizeof(cnt));
            b += cnt[1]*g.expert_bytes;
        }
        return b;
    }

    void ov_rule_tick(int64_t n_tokens) {
        if (n_tokens != 1) {
            ov_have0 = false; // the lookups of larger small batches are not counted
            ov_n1    = 0;
            return;
        }
        if (ov_seen < ov_rule_skip) {
            ov_seen++;
            return;
        }
        if (!ov_have0) {
            ov_m0    = ov_miss_bytes();
            ov_have0 = true;
            ov_n1    = 0;
            return;
        }
        if (++ov_n1 < ov_win) {
            return;
        }
        size_t n_layers = 0;
        for (const group & g : groups) {
            if (g.dev_lookup && g.n_slots > 0) {
                n_layers += g.layers.size();
            }
        }
        const uint64_t m1  = ov_miss_bytes();
        const double   per = n_layers > 0 ? (double) (m1 - ov_m0)/((double) ov_n1*n_layers) : 0.0;
        const bool     on  = ov_live ? per >= ov_rule_hyst*ov_min : per >= ov_min;
        ov_nwin++;
        if (ov_nwin <= 4 || on != ov_live) {
            LLAMA_LOG_INFO("llama_moe_cache: fetch overlap (window %d, %d tokens): %.3f MiB of missing experts per layer and token, threshold %.2f MiB -> %s\n",
                ov_nwin, ov_n1, per/1024.0/1024.0, (ov_live ? ov_rule_hyst : 1.0)*ov_min/1024.0/1024.0, on ? "on" : "off");
        }
        ov_live = on;
        ov_m0   = m1;
        ov_n1   = 0;
        ov_win  = ov_rule_win2;
    }

    // overlap auto trial, called before the graph of every ubatch while a trial runs or waits; while a trial runs, after the previous
    // ubatch finished (the time from one single token ubatch to the next is credited to the path of the first)
    void ov_tick(int64_t n_tokens) {
        if (ov_rule) {
            ov_rule_tick(n_tokens);
            return;
        }
        if (!ov_pending()) {
            return;
        }
        const bool single = n_tokens == 1;
        if (ov_done < 0) {
            if (!single || --ov_wait > 0) {
                return;
            }
            ov_wait   = -1;
            ov_done   = 0;
            ov_t_prev = -1;
        }
        const int64_t now = ggml_time_us();
        if (ov_t_prev >= 0 && single) {
            ov_t[ov_arm].push_back(now - ov_t_prev);
        }
        ov_t_prev = -1;
        if (!single) {
            return; // the next interval would include this ubatch
        }
        if (ov_done >= 4*ov_block) {
            auto median = [](std::vector<int64_t> v) {
                std::sort(v.begin(), v.end());
                return v.empty() ? 0.0 : (double) v[v.size()/2];
            };
            ov_done = -1;
            const double t_plain = median(ov_t[0]);
            const double t_ov    = median(ov_t[1]);
            const bool   on      = ov_t[0].size() >= 4 && ov_t[1].size() >= 4 && t_ov < (1.0 - ov_margin)*t_plain;
            ov_n++;
            LLAMA_LOG_INFO("llama_moe_cache: fetch overlap auto (trial %d): %.3f ms per token without, %.3f ms with (medians of %zu and %zu) -> %s\n",
                ov_n, t_plain/1e3, t_ov/1e3, ov_t[0].size(), ov_t[1].size(), on ? "on" : "off");
            ov_t[0].clear();
            ov_t[1].clear();
            ov_live = on;
            ov_wait = ov_gap;
            ov_gap  = std::min(2*ov_gap, ov_gap_max);
            return;
        }
        const int blk = ov_done/ov_block;
        const int arm = blk == 1 || blk == 2;
        if (arm != ov_arm || ov_done == 0) {
            ov_skipn = ov_skip;
        }
        ov_arm  = arm;
        ov_live = arm != 0;
        if (ov_skipn == 0) {
            ov_t_prev = now;
        }
        ov_skipn = std::max(0, ov_skipn - 1);
        ov_done++;
    }

    int hybrid_mode() const {
        if (cm) {
            // the decisions of every batch size are in the graph: a new decision rebuilds the graphs
            return 3 + (ov_live ? 4 : 0) + 8*(int) (cm_gen & 0xfffff) + (cm_prompt_off() ? (1 << 28) : 0) + (short_live() ? (1 << 29) : 0) +
                (pf_prompt_off() ? (1 << 30) : 0);
        }
        // a single-graph failure (single_off) rebuilds the graphs without their mailboxes: a reused graph would keep merging from a
        // mailbox whose control block says failed (wrong rows in every later ubatch with a pinned fetch share, LLAMA_MOE_CACHE_FETCH)
        return (!hybrid ? 0 : hybrid_active() ? 2 : 1) + (ov_live ? 4 : 0) + (single_off ? 8 : 0) + (pf_prompt_off() ? 16 : 0);
    }

    // auto trial, called before the graph of every ubatch while a trial runs or waits, after the previous ubatch finished
    void tick(int64_t n_tokens) {
        if (!hybrid || (win_done < 0 && win_wait < 0)) {
            return;
        }
        const int64_t now = ggml_time_us();
        bool small = false;
        for (const group & g : groups) {
            small = small || (g.counts != nullptr && use_device(g, n_tokens) && layers[g.layers[0]].host_dev.size() == g.banks.size());
        }
        if (win_done < 0) {
            if (!small || --win_wait > 0) {
                return;
            }
            win_wait   = -1;
            win_done   = 0;
            win_t_prev = -1;
        }
        if (win_t_prev >= 0 && small) {
            win_t[win_arm].push_back(now - win_t_prev);
        }
        win_t_prev = -1;
        if (!small) {
            return; // the next interval would include this ubatch
        }
        auto median = [](std::vector<int64_t> v) {
            std::sort(v.begin(), v.end());
            return v.empty() ? 0.0 : (double) v[v.size()/2];
        };
        // a hybrid path much slower than the plain cache (e.g. other copies on the bus) ends the trial early
        if (win_t[0].size() >= 4 && win_t[1].size() >= 4 &&
                *std::min_element(win_t[1].begin(), win_t[1].end()) > (1.0 + 8*hybrid_margin)*median(win_t[0])) {
            win_done = 5*hybrid_block;
        }
        if (win_done >= 5*hybrid_block) {
            win_done = -1;
            const double t_plain = median(win_t[0]);
            const double t_hyb   = median(win_t[1]);
            const bool   on      = win_t[0].size() >= 3 && win_t[1].size() >= 3 && t_hyb < (1.0 - hybrid_margin)*t_plain;
            win_n++;
            LLAMA_LOG_INFO("llama_moe_cache: hybrid auto (trial %d): %.2f ms per small batch with the plain cache, %.2f ms with the hybrid path "
                "(fetch share %.3f, medians of %zu and %zu) -> hybrid path %s\n",
                win_n, t_plain/1e3, t_hyb/1e3, hybrid_share, win_t[0].size(), win_t[1].size(), on ? "on" : "off");
            if (getenv("LLAMA_MOE_CACHE_TRIAL_LOG") != nullptr) {
                for (int a = 0; a < 2; ++a) {
                    std::string str;
                    for (int64_t t : win_t[a]) {
                        str += " " + std::to_string(t/100/10.0).substr(0, 5);
                    }
                    LLAMA_LOG_INFO("llama_moe_cache: hybrid auto (trial %d) %s ms:%s\n", win_n, a ? "hybrid" : "plain", str.c_str());
                }
            }
            win_t[0].clear();
            win_t[1].clear();
            hybrid_live = on;
            hybrid      = on;
            if (on) {
                win_wait = win_gap;
                win_gap  = std::min(2*win_gap, hybrid_gap_max);
            }
            return;
        }
        const int blk = win_done/hybrid_block;
        const int arm = blk == 2 || blk == 3;
        if (arm != win_arm || win_done == 0) {
            win_skip = hybrid_skip; // the graph is rebuilt and captured again, the host threads wake up
        }
        win_arm     = arm;
        hybrid_live = arm != 0;
        if (blk > 0 && win_skip == 0) {
            win_t_prev = now;
        }
        win_skip = std::max(0, win_skip - 1);
        win_done++;
    }

    // deterministic share (LLAMA_MOE_CACHE_HYBRID unset or 2): share = k/(k + n_threads), k = 6, rounded to 1/8, at least 1/8; the
    // host path is off above 3/4; the timed probe (LLAMA_MOE_CACHE_HYBRID=3) found 0.25 with 24 threads (G4: 58-91 GB/s on the host,
    // 31 GB/s upload) and 0.5 / 0.625 with 6 threads (L4: 13 / 10 GB/s), and the rounded share flipped between launches, which changed
    // the greedy output; this one depends only on the thread count.
    // k18-lfm: at most LLAMA_MOE_CACHE_HYBRID_SHARE_CAP (0.25) with LLAMA_MOE_CACHE_HYBRID_SHARE_CAP_THREADS (6) or more decode threads.
    // L4 (6 threads, 12 GB/s upload, ~470 us per uploaded expert): LFM2.5 UD-Q4_K_XL 4 GB (4.8% cached) at the old 0.5 vs 0.25 decoded
    // chat / 2k / 8k 28.4 / 36.0 / 32.0 vs 37.1 / 52.2 / 51.0 tok/s (every miss on the host: 28.5 / 39.0 / 34.8); 24 threads give 0.25
    // either way. LLAMA_MOE_CACHE_HYBRID_SHARE_CAP=1 restores k/(k + n)
    void set_threads(int n_threads, bool quiet) {
        if (!hybrid || !hybrid_det) {
            return;
        }
        static const double share_cap = [] {
            const char * e = getenv("LLAMA_MOE_CACHE_HYBRID_SHARE_CAP");
            return e != nullptr ? std::min(1.0, std::max(0.125, atof(e))) : 0.25;
        }();
        static const int share_cap_threads = [] {
            const char * e = getenv("LLAMA_MOE_CACHE_HYBRID_SHARE_CAP_THREADS");
            return e != nullptr ? std::max(1, atoi(e)) : 6;
        }();
        const int    n     = std::max(1, n_threads);
        const double share = n >= share_cap_threads ? std::min(share_cap, hybrid_k/(hybrid_k + n)) : hybrid_k/(hybrid_k + n);
        const float  q     = std::max(0.125f, (float) std::round(share*8.0)/8.0f);
        if (q > 0.75f) {
            hybrid       = false;
            hybrid_det   = false;
            hybrid_fixed = false;
            hybrid_live  = false;
            hybrid_share = 1.0f;
        } else {
            hybrid_share = q;
            hybrid_live  = true;
        }
        if (!quiet) {
            if (hybrid) {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid MoE cache: the host computes the missing experts that are not fetched, fetch share = %.3f "
                    "(from %d decode threads; LLAMA_MOE_CACHE_FETCH=0..1 sets it)\n", hybrid_share, n);
            } else {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid MoE cache off: %d decode threads would fetch a share of %.3f (at most 0.75)\n", n, q);
            }
        }
    }

    // ---- CPU misses [LLAMA_MOE_CPU_MISS] ------------------------------------------------------------------------------------------
    // calibration, once before the first small batch, with the threads of small batches: the host times one layer's expert tensors for 1
    // expert x 1 token, k experts x 1 token and k experts x T tokens (other experts every repetition, as they are read in generation),
    // the device times uploads of k experts from the host experts and a small synchronous read (the round trip of a host split)
    // k15-idlewait: set by cm_calibrate when cm_cal_abort stopped a re-calibration (nothing changed, it stays pending)
    bool (*cm_cal_abort)(void *) = nullptr;
    void * cm_cal_abort_data     = nullptr;
    bool   cm_cal_aborted        = false;

    // k19-determinism (LLAMA_MOE_CPU_MISS_CAL_PERSIST, default 1): the start-up calibration is kept in a small file per device, host
    // threads and expert form (LLAMA_MOE_CPU_MISS_CAL_DIR, else $LLAMA_CACHE, else $XDG_CACHE_HOME/llama.cpp, else ~/.cache/llama.cpp), and
    // every later launch reads it at the same point instead of timing the host again; the re-calibration after warm-up then changes nothing.
    // A per-launch sample decided some batch sizes on a noisy timing: G4 Qwen3.6-35B-A3B UD-Q4_K_XL + MTP 12 GB, host split 30.7 vs 34.7 us
    // put single token batches on the host path in one launch and not in the other (7931 vs 7388 host batches, other text at temperature 0
    // in 1 of 3 launches). The first launch decides from the numbers it writes, so all launches on a machine decide alike. =0: timed in
    // every launch (the live rule, with the re-calibration); =2: timed again and the file rewritten (after a hardware or load change)
    static int cm_persist_mode() {
        static const int mode = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_CAL_PERSIST");
            return e != nullptr ? atoi(e) : 1;
        }();
        return mode;
    }
    bool cm_from_file = false; // this launch decides from a persisted calibration
    // k19-determinism (option a): no file yet. This launch runs the live rule, and the re-calibration after warm-up (the host as it is during
    // decode) writes the file; later launches decide from it from the first token
    bool cm_persist_pending = false;
    std::string cm_pend_dir, cm_pend_key, cm_pend_path;

    // noise of sample sets: the spread of the per-set host times ((max - min) / median) or the interquartile spread within a set
    static bool cm_cal_noisy(std::vector<double> t1s, double iqr, double limit, double & spread) {
        if (t1s.empty()) {
            spread = 1e9;
            return true;
        }
        std::sort(t1s.begin(), t1s.end());
        const double m = t1s[t1s.size()/2];
        spread = std::max(iqr, m > 0.0 ? (t1s.back() - t1s.front())/m : 1e9);
        return spread > limit;
    }
    static double cm_cal_max_spread() {
        static const double v = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_CAL_MAX_SPREAD");
            return e != nullptr ? std::max(0.0, atof(e)) : 0.25;
        }();
        return v;
    }

    // the build identity for the calibration key: computed lazily, once per process, on the first calibration file read or write (a load
    // without a MoE cache with CPU misses never gets here); the module files are hashed only for an unknown / dirty commit, timed, logged once
    static const std::string & cm_build_id() {
        static const std::string id = [] {
            std::string mods[2];
            mods[0] = llama_build_id_module_path((const void *) &llama_backend_init);
            ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            ggml_backend_reg_t reg = cpu != nullptr ? ggml_backend_dev_backend_reg(cpu) : nullptr;
            mods[1] = llama_build_id_module_path(reg != nullptr ? ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads") : nullptr);
            const int64_t t0 = ggml_time_us();
            std::string r = llama_build_id(ggml_commit(), mods, 2);
            if (r != ggml_commit()) {
                std::error_code ec;
                const double mb0 = mods[0].empty() ? 0.0 : (double) std::filesystem::file_size(mods[0], ec) / 1048576.0;
                const double mb1 = mods[1].empty() ? 0.0 : (double) std::filesystem::file_size(mods[1], ec) / 1048576.0;
                LLAMA_LOG_INFO("llama_moe_cache: CPU-miss calibration key: commit \"%s\", so the build is named by its modules: %s (%.1f MiB) + %s (%.1f MiB) hashed in %.1f ms\n",
                    ggml_commit(), mods[0].empty() ? "?" : mods[0].c_str(), mb0, mods[1].empty() ? "?" : mods[1].c_str(), mb1, (ggml_time_us() - t0) / 1000.0);
            }
            return r;
        }();
        return id;
    }

    // (k19-autoprof: the same directory keeps the self-learned profiles, see llama-autoprof.h)
    static std::string cm_cal_dir() {
        return llama_autoprof::cal_dir();
    }

    // what the timings depend on: the device, the host threads and CPU count, the cache (main / MTP) and the expert form
    std::string cm_cal_key(const group & g, const layer & l) const {
        // the format version and the build: an upgraded binary times the host again instead of reusing older numbers (a build without
        // git metadata reports commit "unknown": then the content of libllama and the CPU backend module stands in, llama-build-id.h)
        std::string k = "moe-cpu-miss-cal v2|build=" + cm_build_id();
        ggml_backend_dev_t dev = devices[g.id].backend != nullptr ? ggml_backend_get_device(devices[g.id].backend) : nullptr;
        k += std::string("|dev=") + (dev != nullptr ? ggml_backend_dev_description(dev) : "?");
        k += "|threads=" + std::to_string(single_threads) + "|ncpu=" + std::to_string(std::thread::hardware_concurrency());
        k += std::string("|cache=") + (tag[0] != '\0' ? "mtp" : "main");
        k += "|n_expert=" + std::to_string(g.ref[0]->ne[2]) + "|n_used=" + std::to_string(n_expert_used);
        k += "|bytes=" + std::to_string(g.expert_bytes);
        for (const ggml_tensor * w : l.experts) {
            k += std::string("|") + ggml_type_name(w->type) + ":" + std::to_string(w->ne[0]) + "x" + std::to_string(w->ne[1]);
        }
        for (char & c : k) {
            c = c == '\n' || c == '\r' ? ' ' : c;
        }
        return k;
    }

    static std::string cm_cal_path(const std::string & dir, const std::string & key) {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : key) {
            h ^= c;
            h *= 1099511628211ull;
        }
        char name[64];
        snprintf(name, sizeof(name), "moe_cpu_miss_cal_%016llx.txt", (unsigned long long) h);
#if defined(_WIN32)
        return dir + "\\" + name;
#else
        return dir + "/" + name;
#endif
    }

    // the file: the key on the first line, then t1 dt tup split (us)
    static bool cm_cal_read(const std::string & path, const std::string & key, double v[4]) {
        FILE * f = fopen(path.c_str(), "r");
        if (f == nullptr) {
            return false;
        }
        std::string line;
        char buf[4096];
        bool ok = fgets(buf, sizeof(buf), f) != nullptr;
        if (ok) {
            line = buf;
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                line.pop_back();
            }
            ok = line == key && fscanf(f, "%lf %lf %lf %lf", &v[0], &v[1], &v[2], &v[3]) == 4 &&
                 v[0] > 0.0 && v[1] >= 0.0 && v[2] > 0.0 && v[3] >= 0.0;
        }
        fclose(f);
        return ok;
    }

    // written to a temporary file and then put in place in one step, so a reader never sees half a file. Two servers that time the host
    // at once: the first file in place wins (a hard link fails if the file exists) and the other launch reads it and decides from it too,
    // so both decide alike; replace (=2) renames over the old file. Returns false if nothing usable is in place (v keeps the own numbers)
    static bool cm_cal_write(const std::string & dir, const std::string & path, const std::string & key, double v[4], bool replace) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        const std::string tmp = path + ".tmp" + std::to_string((long long) ggml_time_us());
        FILE * f = fopen(tmp.c_str(), "w");
        if (f == nullptr) {
            return false;
        }
        const bool ok = fprintf(f, "%s\n%.17g %.17g %.17g %.17g\n", key.c_str(), v[0], v[1], v[2], v[3]) > 0;
        if (fclose(f) != 0 || !ok) {
            std::remove(tmp.c_str());
            return false;
        }
        bool placed = false;
        if (replace) {
            std::filesystem::rename(tmp, path, ec);
            placed = !ec;
        } else {
#if defined(__linux__) || defined(__APPLE__)
            placed = link(tmp.c_str(), path.c_str()) == 0;
#else
            if (!std::filesystem::exists(path, ec)) {
                std::filesystem::rename(tmp, path, ec);
                placed = !ec;
            }
#endif
        }
        std::remove(tmp.c_str());
        if (placed) {
            return true;
        }
        // another launch put its numbers in place first: decide from those
        double w[4];
        if (cm_cal_read(path, key, w)) {
            std::copy(w, w + 4, v);
            return true;
        }
        return false;
    }

    // k19-determinism: one line at load naming the calibration file this cache will use (read, or written by its first small batch)
    void cm_cal_log_file(bool quiet) const {
        if (quiet || !cm || cm_persist_mode() == 0) {
            return;
        }
        for (const group & g : groups) {
            if (g.dev_lookup && g.n_slots > 0 && g.counts != nullptr && !g.layers.empty()) {
                const std::string dir = cm_cal_dir();
                if (dir.empty()) {
                    LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: no calibration directory (LLAMA_MOE_CPU_MISS_CAL_DIR / LLAMA_CACHE / HOME unset): timed in every launch\n", tag);
                    return;
                }
                const std::string key  = cm_cal_key(g, layers[g.layers[0]]);
                const std::string path = cm_cal_path(dir, key);
                double v[4];
                const bool have = cm_persist_mode() == 1 && cm_cal_read(path, key, v);
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: calibration file %s (%s; LLAMA_MOE_CPU_MISS_CAL_PERSIST=0: timed in every launch)\n", tag,
                    path.c_str(), have ? "found, read at the first small batch" : "not found: the first small batch times the host and writes it");
                return;
            }
        }
    }

    // k19-ptres: the next expert id from nx on (mod n_expert) that is not in the core: the timings read host copies, which the core
    // has none of (with no core: nx++ % n_expert, as before)
    int64_t not_core(const group & g, int32_t il, int64_t & nx, int64_t n_expert) const {
        const int32_t * cs = g.lru.n_core > 0 ? g.lru.slot_map[il] : nullptr;
        for (int64_t k = 0; k < n_expert; ++k) {
            const int64_t e = nx++ % n_expert;
            if (cs == nullptr || cs[e] < 0 || cs[e] >= g.lru.n_core) {
                return e;
            }
        }
        return nx++ % n_expert;
    }

    void cm_calibrate(ggml_backend_t backend_cpu) {
        cm_cal_aborted = false;
        const bool hybrid_probe_was = hybrid_probe;
        auto aborted = [&]() {
            if (cm_cal && cm_cal_abort != nullptr && cm_cal_abort(cm_cal_abort_data)) {
                cm_cal_aborted = true;
                hybrid_probe   = hybrid_probe_was;
                return true;
            }
            return false;
        };
        hybrid_probe = false;
        const int64_t t_start = ggml_time_us();
        // k13-fn: the second call is the re-calibration after warm-up (see cm_recal_wait): host timings only, see below
        const bool recal = cm_cal;
        const group * gp = nullptr;
        for (const group & g : groups) {
            if (g.dev_lookup && g.n_slots > 0 && g.counts != nullptr) {
                gp = &g;
                break;
            }
        }
        if (gp == nullptr || backend_cpu == nullptr) {
            cm     = false;
            hybrid = false;
            return;
        }
        const group & g = *gp;
        const layer & l = layers[g.layers[0]];
        const int64_t n_expert = g.ref[0]->ne[2];
        // ka experts and kb > ka experts of one token (the slope is the time of one expert, the rest the cost of a host graph),
        // ka experts of T tokens each (the time of a further token)
        const int     ka       = (int) std::max<int64_t>(1, std::min<int64_t>(n_expert_used, n_expert/2));
        const int     kb       = (int) std::min<int64_t>(4*ka, n_expert);
        const int     T        = 8;
        const int     n_rep    = 11;

        // k19-determinism: a persisted calibration replaces the timing (start-up), and the re-calibration changes nothing
        const int persist = cm_persist_mode();
        if (recal && cm_from_file) {
            return;
        }
        // LLAMA_MOE_CPU_MISS_CAL_VALUES="t1 dt tup split" (us; tests): decide from these numbers, no timing, no file, no re-calibration
        if (!recal) {
            const char * fv = getenv("LLAMA_MOE_CPU_MISS_CAL_VALUES");
            double v[4];
            if (fv != nullptr && sscanf(fv, "%lf %lf %lf %lf", &v[0], &v[1], &v[2], &v[3]) == 4 && v[0] > 0.0 && v[2] > 0.0) {
                cm_t1 = v[0]; cm_dt = v[1]; cm_tup = v[2]; cm_split = v[3];
                cm_from_file = true;
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: calibration from LLAMA_MOE_CPU_MISS_CAL_VALUES (%s)\n", tag, fv);
                cm_calibrate_finish(false, g, t_start);
                return;
            }
        }
        std::string cal_dir, cal_key, cal_path;
        if (!recal && persist != 0) {
            cal_dir = cm_cal_dir();
            if (!cal_dir.empty()) {
                cal_key  = cm_cal_key(g, l);
                cal_path = cm_cal_path(cal_dir, cal_key);
                double v[4];
                if (persist == 1 && cm_cal_read(cal_path, cal_key, v)) {
                    cm_t1    = v[0];
                    cm_dt    = v[1];
                    cm_tup   = v[2];
                    cm_split = v[3];
                    cm_from_file = true;
                    LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: calibration read from %s (LLAMA_MOE_CPU_MISS_CAL_PERSIST=0: timed in every launch)\n",
                        tag, cal_path.c_str());
                    cm_calibrate_finish(false, g, t_start);
                    return;
                }
            }
        }

        ggml_init_params params = {
            /*.mem_size   =*/ (8 + 6*l.experts.size())*ggml_tensor_overhead() + 3*ggml_graph_overhead_custom(16, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        ggml_tensor * ids_a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, ka, 1);
        ggml_tensor * ids_b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, kb, 1);
        ggml_tensor * ids_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, ka, T);
        ggml_cgraph * gf_a  = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_cgraph * gf_b  = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_cgraph * gf_t  = ggml_new_graph_custom(ctx.get(), 16, false);
        std::vector<ggml_tensor *> xs;
        for (ggml_tensor * w : l.experts) {
            ggml_tensor * x1 = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, w->ne[0], 1, 1);
            ggml_tensor * xt = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, w->ne[0], 1, T);
            xs.push_back(x1);
            xs.push_back(xt);
            ggml_build_forward_expand(gf_a, ggml_mul_mat_id(ctx.get(), w, x1, ids_a));
            ggml_build_forward_expand(gf_b, ggml_mul_mat_id(ctx.get(), w, x1, ids_b));
            ggml_build_forward_expand(gf_t, ggml_mul_mat_id(ctx.get(), w, xt, ids_t));
        }
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend_cpu));
        if (!buf) {
            cm     = false;
            hybrid = false;
            return;
        }
        for (ggml_tensor * x : xs) {
            std::vector<float> v(ggml_nelements(x));
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] = 0.01f*(float) ((i*7919) % 13) - 0.06f;
            }
            ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
        }
        int64_t next = 0;
        // distinct experts within a graph, other experts every repetition (read from memory as in generation)
        auto set_ids = [&](ggml_tensor * ids, int n, int rows) {
            std::vector<int32_t> v((size_t) n*rows);
            for (int i = 0; i < n; ++i) {
                v[i] = (int32_t) not_core(g, g.layers[0], next, n_expert);
            }
            for (int r = 1; r < rows; ++r) {
                std::copy(v.begin(), v.begin() + n, v.begin() + (size_t) r*n);
            }
            ggml_backend_tensor_set(ids, v.data(), 0, v.size()*sizeof(int32_t));
        };
        auto median = [](std::vector<int64_t> v) {
            std::sort(v.begin(), v.end());
            return (double) v[v.size()/2];
        };
        // k19-determinism: interquartile range over the median of a sample set (the noise of a start-up calibration)
        auto rel_iqr = [](std::vector<int64_t> v) {
            if (v.size() < 4) {
                return 0.0;
            }
            std::sort(v.begin(), v.end());
            const double q1 = (double) v[v.size()/4], q3 = (double) v[(3*v.size())/4], m = (double) v[v.size()/2];
            return m > 0.0 ? (q3 - q1)/m : 0.0;
        };
        double cal_spread = 0.0;
        auto timed = [&](ggml_cgraph * gf) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_graph_compute(backend_cpu, gf);
            return ggml_time_us() - t0;
        };

        // k19-calib (LLAMA_MOE_CPU_MISS_CAL_ROBUST, default 1; 0 = the single sample set of before): one sample set of 11 repetitions
        // (about 50 ms) is a launch lottery on a small host: Colab L4 LFM2.5 (12 vCPUs, 6 decode threads) timed the further token of an
        // expert at ~73 or ~130 us per launch for the same bin (u13g_l4 / u12_l4b / i3h_l4, both bins), the slow value moved the fetch
        // shares and the per-launch decode by ~3%. Robust: 3 untimed warm-up repetitions, then 3 (start-up) / 5 (re-calibration) sample
        // sets, a set during which other processes kept CPUs busy retried after a pause; each graph time is the fastest clean set median
        // (a slow set is a busy host, never a faster path), the file keeps the median host time per expert (see below). LLAMA_MOE_CPU_MISS_CAL_LOG=1: every sample, the process CPU time per wall time, the
        // other processes' busy CPUs, context switches and page faults of each set (the diagnosis of k19-calib)
        static const bool cal_robust = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_CAL_ROBUST");
            return e == nullptr || atoi(e) != 0;
        }();
        static const bool cal_log = getenv("LLAMA_MOE_CPU_MISS_CAL_LOG") != nullptr && atoi(getenv("LLAMA_MOE_CPU_MISS_CAL_LOG")) != 0;
        const int n_sets_rb = !cal_robust ? 1 : [&] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_CAL_ROBUST_SETS");
            return e != nullptr ? std::max(1, std::min(32, atoi(e))) : (recal ? 5 : 3);
        }();
        const int n_warm = cal_robust ? 3 : 1;
        int n_set_log = 0;
        // one sample set: n_warm untimed repetitions, then n_rep timed ones of the three graphs
        // k19-calib: a set during which other processes kept more than LLAMA_MOE_CPU_MISS_CAL_BUSY_MAX CPUs (0.75) busy is contended (kcal_l4:
        // every slow-mode set had 1.0-1.5 other CPUs busy, the fast ones at most 0.5): it is retried after a pause, up to 6 more sets
        static const double busy_max = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_CAL_BUSY_MAX");
            return e != nullptr ? atof(e) : 0.75;
        }();
        double set_others = -1.0;
        auto sample_set = [&](std::vector<int64_t> & ta, std::vector<int64_t> & tb, std::vector<int64_t> & tt) {
            set_others = -1.0;
#if defined(__linux__)
            auto proc_busy = []() {   // (busy, total) jiffies of the whole host
                long long v[10] = {0};
                FILE * f = fopen("/proc/stat", "r");
                if (f != nullptr) {
                    if (fscanf(f, "cpu %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]) < 4) {
                        v[0] = 0;
                    }
                    fclose(f);
                }
                long long tot = 0;
                for (long long x : v) {
                    tot += x;
                }
                return std::make_pair(tot - v[3] - v[4], tot);
            };
            rusage ru0 = {}, ru1 = {};
            std::pair<long long, long long> ps0 = {0, 0};
            const bool measure = cal_log || cal_robust;
            if (measure) {
                getrusage(RUSAGE_SELF, &ru0);
                ps0 = proc_busy();
            }
#endif
            const int64_t tw0 = ggml_time_us();
            for (int r = 0; r < n_warm + n_rep; ++r) {
                if (aborted()) {
                    return false;
                }
                set_ids(ids_a, ka, 1);
                const int64_t da = timed(gf_a);
                set_ids(ids_b, kb, 1);
                const int64_t db = timed(gf_b);
                set_ids(ids_t, ka, T);
                const int64_t dt = timed(gf_t);
                if (r >= n_warm) {
                    ta.push_back(da);
                    tb.push_back(db);
                    tt.push_back(dt);
                }
            }
#if defined(__linux__)
            if (measure) {
                const double wall = (double) (ggml_time_us() - tw0);
                getrusage(RUSAGE_SELF, &ru1);
                const std::pair<long long, long long> ps1 = proc_busy();
                const double cpu_us = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec)*1e6 + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) +
                                      (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec)*1e6 + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec);
                const long long d_tot = ps1.second - ps0.second, d_busy = ps1.first - ps0.first;
                const double ncpu = (double) std::thread::hardware_concurrency();
                const double host_busy = d_tot > 0 ? ncpu*(double) d_busy/(double) d_tot : -1.0;
                set_others = host_busy >= 0.0 ? host_busy - cpu_us/std::max(1.0, wall) : -1.0;
            }
            if (cal_log) {
                const double wall = (double) (ggml_time_us() - tw0);
                const double cpu_us = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec)*1e6 + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) +
                                      (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec)*1e6 + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec);
                const double ncpu = (double) std::thread::hardware_concurrency();
                const double host_busy = set_others >= 0.0 ? set_others + cpu_us/std::max(1.0, wall) : -1.0;
                std::string sa, sb, st;
                for (size_t i = 0; i < ta.size(); ++i) {
                    sa += (i ? " " : "") + std::to_string(ta[i]);
                    sb += (i ? " " : "") + std::to_string(tb[i]);
                    st += (i ? " " : "") + std::to_string(tt[i]);
                }
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU-miss calibration samples (%s set %d, %d warm-up): a [%s] b [%s] t [%s] us; process CPU %.2f per wall, "
                    "host busy %.2f of %.0f CPUs, others %.2f; ctx switches %ld vol / %ld invol, page faults %ld minor / %ld major; t+%.3f s\n",
                    tag, recal ? "re-cal" : "start-up", n_set_log, n_warm, sa.c_str(), sb.c_str(), st.c_str(), cpu_us/std::max(1.0, wall),
                    host_busy, ncpu, host_busy - cpu_us/std::max(1.0, wall), ru1.ru_nvcsw - ru0.ru_nvcsw, ru1.ru_nivcsw - ru0.ru_nivcsw,
                    ru1.ru_minflt - ru0.ru_minflt, ru1.ru_majflt - ru0.ru_majflt, ggml_time_us()/1e6);
            }
#endif
            GGML_UNUSED(tw0);
            n_set_log++;
            return true;
        };

        // a busy host gives inconsistent times (more experts not slower, an implausible bandwidth): measured again, at most 3 times
        double d_a = 0.0, d_b = 0.0, d_t = 0.0, slope = 0.0;
        double slope_med = 0.0;   // k19-calib: the median of the clean sets' own host times per expert (what the file keeps, see below)
        bool ok = false;
        int attempt = 0;
        for (; attempt < 3 && !ok; ++attempt) {
            std::vector<double> slopes, slopes_all;
            double ma = 0.0, mb = 0.0, mt = 0.0, iqr = 0.0;     // over the clean sets
            double ca = 0.0, cb = 0.0, ct = 0.0, ciqr = 0.0;   // over every set (all contended: these)
            int n_clean = 0, n_all = 0, n_busy = 0;
            bool last_busy = false;
            const int n_extra = cal_robust ? 6 : 0;
            while (n_clean < n_sets_rb && n_all < n_sets_rb + n_extra && !(n_all >= n_sets_rb && n_busy == 0)) {
                // back to back as before unless the last set was contended (a pause re-places the pool's threads: on G4 it also turned the
                // re-calibration's decode-time 34-36 us per expert into the idle 23, which decodes 24 GB chat 2-3% slower, kcal3_g4)
                if (last_busy) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                }
                std::vector<int64_t> ta, tb, tt;
                if (!sample_set(ta, tb, tt)) {
                    return;
                }
                const double a = median(ta), b = median(tb), t = median(tt), q = std::max(rel_iqr(ta), rel_iqr(tb));
                ca = n_all == 0 ? a : std::min(ca, a);
                cb = n_all == 0 ? b : std::min(cb, b);
                ct = n_all == 0 ? t : std::min(ct, t);
                ciqr = n_all == 0 ? q : std::min(ciqr, q);
                slopes_all.push_back(kb > ka ? (b - a)/(kb - ka) : a/ka);
                n_all++;
                last_busy = cal_robust && set_others > busy_max;
                if (last_busy) {
                    n_busy++;
                    continue;
                }
                ma = n_clean == 0 ? a : std::min(ma, a);
                mb = n_clean == 0 ? b : std::min(mb, b);
                mt = n_clean == 0 ? t : std::min(mt, t);
                iqr = n_clean == 0 ? q : std::min(iqr, q);
                slopes.push_back(slopes_all.back());
                n_clean++;
            }
            if (n_clean == 0) {
                ma = ca; mb = cb; mt = ct; iqr = ciqr;
                slopes = slopes_all;
            }
            std::sort(slopes.begin(), slopes.end());
            slope_med = slopes.empty() ? 0.0 : slopes[slopes.size()/2];
            if (cal_log && n_busy > 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU-miss calibration: %d of %d sets contended (other processes above %.2f busy CPUs), %d clean\n",
                    tag, n_busy, n_all, busy_max, n_clean);
            }
            d_a = ma;
            d_b = mb;
            d_t = mt;
            cal_spread = iqr;
            slope = kb > ka ? (d_b - d_a)/(kb - ka) : d_a/ka;
            const double gbs = slope > 0.0 ? g.expert_bytes/slope/1e3 : 0.0;
            ok = slope > 0.0 && gbs >= 0.2 && gbs <= 400.0 && d_t >= 0.9*d_a;
        }
        if (cal_log) {
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU-miss calibration %s: %d set(s) (robust %d), a %.1f b %.1f t %.1f us -> host %.1f us per expert, "
                "%.2f us per further token (ka %d kb %d T %d, %d attempt(s))\n", tag, recal ? "re-cal" : "start-up", n_sets_rb, cal_robust ? 1 : 0,
                d_a, d_b, d_t, std::max(slope, 0.05), std::max(0.0, (d_t - d_a)/((double) ka*(T - 1))), ka, kb, T, attempt);
        }
        if (!ok && recal) {
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: re-calibration inconsistent, the start-up calibration stays\n", tag);
            return;
        }
        if (!ok) {
            LLAMA_LOG_WARN("llama_moe_cache%s: CPU misses off: the host timings are inconsistent (%.1f us for %d experts, %.1f us for %d, %.1f us for %d x %d tokens; "
                "a busy host), the missing experts are fetched\n", tag, d_a, ka, d_b, kb, d_t, ka, T);
            cm     = false;
            hybrid = false;
            return;
        }

        if (aborted()) {
            return;
        }
        // uploads into the spare slot at the end of the banks (never looked up), then a small synchronous read
        std::vector<int64_t> tu, trt;
        ggml_backend_t backend = devices[g.id].backend;
        ggml_backend_synchronize(backend);
        // the uploads of ka experts are queued together and waited for once, as the fetches of a layer (one copy latency, not ka)
        for (int r = 0; r <= n_rep; ++r) {
            const int64_t t0 = ggml_time_us();
            for (int i = 0; i < ka; ++i) {
                const int64_t e = not_core(g, g.layers[0], next, n_expert);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t nb = l.experts[ip]->nb[2];
                    ggml_backend_tensor_set_async(backend, g.banks[ip], (const uint8_t *) l.experts[ip]->data + e*nb, (size_t) g.n_slots*nb, nb);
                }
            }
            ggml_backend_synchronize(backend);
            if (r > 0) {
                tu.push_back(ggml_time_us() - t0);
            }
        }
        for (int r = 0; r <= n_rep; ++r) {
            int32_t tmp[16];
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_get(g.state, tmp, 0, sizeof(tmp));
            if (r > 0) {
                trt.push_back(ggml_time_us() - t0);
            }
        }

        if (recal && cm_persist_pending) {
            // option (a): the host as it is during decode, the median of LLAMA_MOE_CPU_MISS_CAL_SETS (3) sample sets (this one and more)
            // k19-calib: with the robust sampling the sets were taken above (the fastest set median of each quantity): one entry here
            static const int n_sets = cal_robust ? 1 : [] {
                const char * e = getenv("LLAMA_MOE_CPU_MISS_CAL_SETS");
                return e != nullptr ? std::max(1, atoi(e)) : 3;
            }();
            std::vector<double> t1s = { std::max(slope, 0.05) }, dts = { std::max(0.0, (d_t - d_a)/((double) ka*(T - 1))) }, das = { d_a };
            double iqr = cal_spread;
            for (int k = 1; k < n_sets; ++k) {
                std::vector<int64_t> ta, tb, tt;
                for (int r = 0; r <= n_rep; ++r) {
                    if (aborted()) {
                        return;
                    }
                    set_ids(ids_a, ka, 1);
                    const int64_t da = timed(gf_a);
                    set_ids(ids_b, kb, 1);
                    const int64_t db = timed(gf_b);
                    set_ids(ids_t, ka, T);
                    const int64_t dt = timed(gf_t);
                    if (r > 0) {
                        ta.push_back(da);
                        tb.push_back(db);
                        tt.push_back(dt);
                    }
                }
                const double a = median(ta), b = median(tb), t = median(tt);
                t1s.push_back(std::max(kb > ka ? (b - a)/(kb - ka) : a/ka, 0.05));
                dts.push_back(std::max(0.0, (t - a)/((double) ka*(T - 1))));
                das.push_back(a);
                iqr = std::max(iqr, std::max(rel_iqr(ta), rel_iqr(tb)));
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; };
            double spread = 0.0;
            const bool noisy = cm_cal_noisy(t1s, iqr, cm_cal_max_spread(), spread);
            const double t1w = med(t1s), dtw = med(dts), daw = med(das);
            // k19-calib: robust: the file keeps the faster further-token time of the start-up and the re-calibration (the bimodal quantity:
            // a slow draw of either no longer decides every later launch on this host) and, as before, the host time per expert of the
            // host as it is after warm-up (the median of the re-calibration's clean sets: on G4 Flash-Next 24 GB chat the decode-time 34-36
            // us per expert decoded 2.5-3% faster than the start-up 23 on its own text, kcal_g4 / kcal2_g4)
            const double t1p = cal_robust ? std::max(slope_med, 0.05) : t1w, dtp = cal_robust ? std::min(dtw, cm_dt) : dtw;
            double v[4] = { t1p, dtp, std::max(median(tu)/ka, 0.05), std::max(0.0, daw - ka*t1w) + 2.0*median(trt) };
            cm_persist_pending = false;
            if (noisy) {
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: calibration not kept, the host timings spread %.0f%% (above %.0f%%, "
                    "LLAMA_MOE_CPU_MISS_CAL_MAX_SPREAD): the next launch times the host again\n", tag, 100.0*spread, 100.0*cm_cal_max_spread());
            } else {
                const bool w = cm_cal_write(cm_pend_dir, cm_pend_path, cm_pend_key, v, persist == 2);
                if (w) {
                    cmt_ps_flush(); // k19-cmtpersist: verdicts timed before the file existed
                }
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: calibration %s %s: host %.1f us per expert (%.2f us per further token), upload %.1f us, "
                    "host split %.1f us (re-calibration, %zu sets, spread %.0f%%)\n", tag, w ? "written to" : "could not be written to",
                    cm_pend_path.c_str(), v[0], v[1], v[2], v[3], t1s.size(), 100.0*spread);
            }
        }
        if (recal) {
            // k13-fn: one start-up sample set (11 repetitions of one layer, before any request) can be slow as a whole: on the file
            // tier (G4 Flash-Next UD-IQ4_XS 16 GB / 64 GiB RAM) 2 of 3 launches measured 33.3 instead of 23.1 us per expert, the fetch
            // share went from 0.25 to 0.5 and 8k decode lost 4-9%. After warm-up the host is timed again; if the two disagree by more
            // than 20%, the faster host time is kept (a slow sample comes from a cold page cache or a busy host, not a slower path)
            const double t1_new = std::max(slope, 0.05);
            const double dt_new = std::max(0.0, (d_t - d_a)/((double) ka*(T - 1)));
            const double rel    = std::fabs(t1_new - cm_t1)/std::max(cm_t1, 1e-3);
            // k15-lfm2 (cm_stable): an agreeing re-calibration also keeps the faster of the two and decides again below (before, it
            // returned with the start-up decisions, which the next rule window, up to cm_win2 small batches later, then changed)
            if (rel <= cm_recal_tol && cm_stable) {
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: re-calibration %.1f us per expert agrees with %.1f (within %.0f%%), host %.1f "
                    "us per expert kept\n", tag, t1_new, cm_t1, 100.0*cm_recal_tol, std::min(cm_t1, t1_new));
                // k19-calib: the further token is its own quantity (the bimodal one), the faster of the two is kept on its own
                cm_dt = cal_robust ? std::min(cm_dt, dt_new) : t1_new < cm_t1 ? dt_new : cm_dt;
                cm_t1 = std::min(cm_t1, t1_new);
            } else if (rel <= cm_recal_tol && !(cal_robust && dt_new < (1.0 - cm_recal_tol)*cm_dt)) {
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: re-calibration %.1f us per expert agrees with %.1f (within %.0f%%)\n", tag,
                    t1_new, cm_t1, 100.0*cm_recal_tol);
                return;
            } else if (rel <= cm_recal_tol) {
                // k19-calib: the host time agrees but the further token is more than cm_recal_tol faster: the start-up set was the slow mode
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: re-calibration %.2f us per further token vs %.2f at start-up: kept\n", tag,
                    dt_new, cm_dt);
                cm_dt = dt_new;
                cm_t1 = std::min(cm_t1, t1_new);
            } else {
                const double t1_old = cm_t1;
                cm_t1 = std::min(cm_t1, t1_new);
                cm_dt = cal_robust ? std::min(cm_dt, dt_new) : t1_new < t1_old ? dt_new : cm_dt;
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: re-calibration %.1f us per expert vs %.1f at start-up (%.0f%% apart): host %.1f us "
                    "per expert kept\n", tag, t1_new, t1_old, 100.0*rel, cm_t1);
            }
        } else {
            cm_t1    = std::max(slope, 0.05);
            cm_dt    = std::max(0.0, (d_t - d_a)/((double) ka*(T - 1)));
            cm_tup   = std::max(median(tu)/ka, 0.05);
            cm_split = std::max(0.0, d_a - ka*cm_t1) + 2.0*median(trt);
            if (!cal_path.empty()) {
                // option (a): the start-up sample decides this launch only (it is often faster than the host during decode: G4 Flash-Next
                // 23.3-23.9 us per expert at start-up in most launches, 34.6-35.5 re-timed after warm-up in all); the re-calibration writes
                cm_persist_pending = true;
                cm_pend_dir  = cal_dir;
                cm_pend_key  = cal_key;
                cm_pend_path = cal_path;
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: calibrating; later launches will use %s (written by the re-calibration after "
                    "warm-up; LLAMA_MOE_CPU_MISS_CAL_PERSIST=0: timed in every launch)\n", tag, cal_path.c_str());
            }
        }
        GGML_UNUSED(attempt);
        cm_calibrate_finish(recal, g, t_start);
    }

    // the decisions from the host / upload timings (cm_t1, cm_dt, cm_tup, cm_split), measured or read from the calibration file
    void cm_calibrate_finish(bool recal, const group & g, int64_t t_start) {
        const int64_t n_expert = g.ref[0]->ne[2];
        {
            int64_t n_embd = 0;
            for (const ggml_tensor * t : g.ref) {
                n_embd = std::max<int64_t>(n_embd, t->ne[0]);
            }
            cm_act = 2.0*n_embd*sizeof(float)/std::max(1e-3, g.expert_bytes/cm_tup);
        }

        cm_share.assign(cm_max + 1, 1.0f);
        // cm_stable: a re-calibration keeps the decisions it decides again from (the hysteresis of the sizes that are on, the frozen sizes)
        if (!recal || !cm_stable || cm_on.size() != (size_t) cm_max + 1) {
            cm_on.assign(cm_max + 1, 0);
        }
        // k11 LLAMA_MOE_CPU_MISS_FETCH_SCALE=s (opt-in, unset = 1): the fetched share scaled by s before the levels, with a level of 1/16
        // below 1/8. The balance above treats a fetch as running beside the device work, but the fetch kernel reads host memory in the
        // device stream (G4 q4x + MTP 8 GB nsys: 45% of the decode window), so the device side of the balance is slower than cm_tup says;
        // G4 with a fixed share of 0.1 instead of 0.25: decode +13.5 / +16.1 / +4.7% (chat / 2k / 8k)
        static const double fetch_scale = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_FETCH_SCALE");
            return e != nullptr ? std::max(0.0, atof(e)) : 1.0;
        }();
        for (int64_t n = 1; n <= cm_max; ++n) {
            const double D    = cm_distinct(n, n_expert);
            const double tbar = std::max(1.0, (double) n*std::min<int64_t>(n_expert_used, n_expert)/D);
            const double tc   = cm_t1 + (tbar - 1.0)*cm_dt;
            const double raw  = tc/(tc + cm_tup)*fetch_scale;
            // log levels: two calibrations a little apart rarely fall on two sides of a level. k15-lfm2: small batches (at most
            // max_batch tokens) take 1/4 up to cm_small_half instead of the geometric midpoint 0.354 (see cm_small_half)
            const double half = n <= max_batch ? cm_small_half_eff() : 0.354;
            cm_share[n] = raw <= 0.177 ? 0.125f : raw <= half ? 0.25f : raw <= 0.612 ? 0.5f : raw <= 0.80 ? 0.75f : 1.0f;
            if (fetch_scale != 1.0 && raw <= 0.088) {
                cm_share[n] = 0.0625f;
            }
        }
        if (recal) {
            cm_gen++;
            if (!cm_force) {
                cm_decide();
            } else {
                for (int64_t n = 1; n <= cm_max; ++n) {
                    cm_share[n] = std::min(cm_share[n], 0.75f);
                    cm_on[n]    = 1;
                }
            }
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: fetch share now %.3f (1 token), %.3f (4)\n", tag, cm_share[1], cm_share[std::min<int64_t>(4, cm_max)]);
            return;
        }
        cm_dec_init();
        cm_cal = true;
        cm_recal_wait = cm_from_file ? -1 : cm_recal_after; // k19-determinism: no re-calibration from a persisted one
        // the first prompt and tokens run before any measured miss rate: a prior from the cached share c of the experts, (1 - c)^4
        // (decode miss rates measured on a G4: Qwen3.6-35B-A3B UD-Q4_K_XL c = 0.17 -> 0.29-0.47, c = 0.39 -> 0.16; gpt-oss-20b Q2_K
        // c = 0.54 -> 0.007), the first measured window corrects it without the hysteresis
        if (!cm_force) {
            cm_rate = std::max(0.01, std::min(1.0, std::pow(1.0 - std::max(0.0, std::min(1.0, cache_frac)), 4.0)));
            cm_decide();
        }
        // k11 (LLAMA_MOE_CPU_MISS_TRIAL_FRAC, default 0.20; 0 = the trials of k10 everywhere): a cache that holds less than this share
        // of the host experts runs no warm phase and no timing trials, so small batches follow the cost model from the calibration on;
        // short prompts stay on the plain cache (they need a trial that kept the host path, see cm_prompt_ok). G4 Qwen3.6 UD-Q4_K_XL
        // 8 GB + MTP (17% cached): the trials turned the draft context's host path off (0.467 vs 0.501 ms per small batch) and ran their
        // blocks without it, decode -6..-15% against the cost model alone. The cells the trials were made for keep them: Qwen3.6 12 GB
        // (39% cached, the split costs more than modelled) and gpt-oss-20b Q2_K 8 GB (54%). Round 9 (G4 / L4 / T4, k10 on the same VM):
        // Qwen3.6 UD-Q4_K_XL 8 GB + MTP (16.8%) decode +5.6..+13.2% G4, +10.7..+21.0% L4; Q3_K_M 8 GB (23.8%) -1..-2% on G4, T4 UD-Q4_K_XL
        // 8 GB without a draft (22.2%) inside its spread: both stay above 0.20
        const bool trial_skip = cm_skip_trials(false);
        if (cm_trial && !cm_force && !trial_skip) {
            // warm phase first (see cmw_on): small batches off, the first trial starts in cm_tick
            cmt_done      = -1;
            cmt_t_prev    = -1;
            cmw_on        = true;
            cmw_prev      = -1.0;
            cmw_n         = 0;
            cm_small_live = false;
            cm_gen++;
        }
        if (cm_force) {
            for (int64_t n = 1; n <= cm_max; ++n) {
                cm_share[n] = std::min(cm_share[n], 0.75f);
                cm_on[n]    = 1;
            }
            cm_gen++;
        }
        int64_t n_max_share = 0;
        for (int64_t n = 1; n <= cm_max; ++n) {
            n_max_share = cm_share[n] < 1.0f ? n : n_max_share;
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: host %.1f us per expert (%.2f GB/s, %.2f us per further token), upload %.1f us per expert "
            "(%.2f GB/s), host split %.1f us; fetch share %.3f (1 token), %.3f (%d), %.3f (%" PRId64 "), host path possible up to %" PRId64 " tokens; "
            "decided from a prior miss rate, then from the measured one after %d small batches (%.1f ms)\n", tag,
            cm_t1, g.expert_bytes/cm_t1/1e3, cm_dt, cm_tup, g.expert_bytes/cm_tup/1e3, cm_split, cm_share[1], cm_share[std::min<int64_t>(4, cm_max)],
            (int) std::min<int64_t>(4, cm_max), cm_share[cm_max], cm_max, n_max_share, cm_skip + cm_win1, (ggml_time_us() - t_start)/1e3);
        if (cm_dec_on > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: k19-flint decode host path (LLAMA_MOE_CPU_MISS_DECODE=1): batches of 1..%" PRId64 " tokens "
                "fetch %.3f of their missing experts (the most used) and the host computes the rest through the mailbox, no rule or trial "
                "(%s; changes the output, KLD-gated)\n", tag, cm_dec_max, cm_dec_share,
                cm_decode(1) ? "mailbox usable" : "not used here: no single-graph mailbox, frozen small batches or LLAMA_MOE_CPU_MISS=2");
        }
    }

    static bool cmt_persist_on() {
        static const bool on = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_PERSIST");
            return e == nullptr || atoi(e) != 0;
        }();
        return on && cm_persist_mode() != 0;
    }
    static double cmt_hyst() {
        static const double h = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_TRIAL_HYST");
            return e != nullptr ? std::max(0.0, atof(e)) : 0.03;
        }();
        return cmt_persist_on() ? h : 0.0;
    }
    // k19-cmtpersist: the calibration file and this cache's tag, and the verdicts recorded there (once, before the first trial ends)
    void cmt_ps_setup() {
        if (cmt_ps_state >= 0) {
            return;
        }
        cmt_ps_state = 0;
        if (!cmt_persist_on()) {
            return;
        }
        const group * gp = nullptr;
        for (const group & g : groups) {
            if (g.dev_lookup && g.n_slots > 0 && g.counts != nullptr && !g.layers.empty()) {
                gp = &g;
                break;
            }
        }
        const std::string dir = gp != nullptr ? cm_cal_dir() : std::string();
        if (dir.empty()) {
            return;
        }
        long long slots = 0;
        for (const group & g : groups) {
            slots += g.n_slots;
        }
        cmt_ps_key   = cm_cal_key(*gp, layers[gp->layers[0]]);
        cmt_ps_path  = cm_cal_path(dir, cmt_ps_key);
        cmt_ps_tag   = llama_moe_tp::tag_of(slots, (int) layers.size(), (long long) max_batch);
        cmt_ps_state = 1;
        std::vector<llama_moe_tp::rec> r;
        if (llama_moe_tp::read(cmt_ps_path, cmt_ps_key, cmt_ps_tag, r)) {
            cmt_rec      = r;
            cmt_rec_file = r.size();
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: %zu trial verdict(s) recorded for %s in %s%s (LLAMA_MOE_CPU_MISS_PERSIST=0: timed in every launch)\n",
            tag, cmt_rec_file, cmt_ps_tag.c_str(), cmt_ps_path.c_str(), cmt_rec_file > 0 ? ": replayed in order" : ": this launch times and records them");
    }
    // writes the verdicts not yet in the file (needs the calibration file: a launch that has none yet writes them after its re-calibration)
    void cmt_ps_flush() {
        if (cmt_ps_state != 1 || !cmt_rec_dirty) {
            return;
        }
        if (llama_moe_tp::write(cmt_ps_path, cmt_ps_key, cmt_ps_tag, cmt_rec, (long long) ggml_time_us())) {
            cmt_rec_dirty = false;
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: %zu trial verdict(s) written to %s\n", tag, cmt_rec.size(), cmt_ps_path.c_str());
        }
    }

    // true if the host path took at least cmt_host_bad times as long as the plain cache
    static bool cmt_host_slow(double t_on, double t_off) {
        return t_off > 0.0 && t_on >= cmt_host_bad*t_off;
    }

    bool cmt_pending() const {
        return cm && cm_cal && cm_trial && (cmt_done >= 0 || cmt_wait >= 0);
    }

    // a small batch whose size the model lets the host path take (the trial times only those)
    bool cmt_counted(int64_t n_tokens) const {
        return n_tokens >= 1 && n_tokens <= max_batch && n_tokens <= cm_max && cm_on[n_tokens] != 0 && !cm_decode(n_tokens);
    }

    // the ubatch timed by the trial was computed (backends synchronized): its time from the tick (graph build + compute; the time of
    // the other contexts, e.g. an MTP draft between two verify batches, is not in it)
    bool cmt_timed() const {
        return cmt_t_prev >= 0;
    }
    void cmt_end() {
        if (cmt_t_prev >= 0) {
            cmt_t[cmt_arm].push_back(ggml_time_us() - cmt_t_prev);
            cmt_t_prev = -1;
        }
    }

    // r22b prompt timing (see pt_auto): before the graph of a ubatch, and at its end (backends synchronized)
    bool pt_pending() const {
        return pt_auto && cm && cm_cal && !cm_force;
    }
    void pt_tick(int64_t n_tokens) {
        pt_t_prev = -1;
        if (!pt_eligible(n_tokens)) {
            return;
        }
        int arm;
        if (pt_state < 0) {
            // alternate, on first, until pt_need of each
            arm = pt_t[1].size() <= pt_t[0].size() ? 1 : 0;
        } else {
            pt_count++;
            arm = pt_count % pt_every == 0 ? 1 - pt_state : pt_state;
        }
        if (arm != pt_arm) {
            cm_gen++; // graph key
        }
        pt_arm    = arm;
        pt_t_prev = ggml_time_us();
        pt_n_prev = n_tokens;
    }
    bool pt_timed() const {
        return pt_t_prev >= 0;
    }
    void pt_end() {
        if (pt_t_prev < 0) {
            return;
        }
        auto & v = pt_t[pt_arm];
        v.push_back((double) (ggml_time_us() - pt_t_prev)/(double) std::max<int64_t>(1, pt_n_prev));
        if ((int) v.size() > pt_keep) {
            v.erase(v.begin());
        }
        pt_t_prev = -1;
        const bool trial_done = pt_state < 0 && (int) pt_t[0].size() >= pt_need && (int) pt_t[1].size() >= pt_need;
        const bool recheck    = pt_state >= 0 && pt_arm != pt_state;
        if (!trial_done && !recheck) {
            return;
        }
        auto median = [](std::vector<double> x) {
            std::sort(x.begin(), x.end());
            return x.empty() ? 0.0 : x[x.size()/2];
        };
        const double t_off = median(pt_t[0]);
        const double t_on  = median(pt_t[1]);
        const int    st    = t_on < (1.0 - cmt_margin)*t_off ? 1 : 0;
        pt_n_dec++;
        if (pt_state < 0 || st != pt_state || cm_dbg) {
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses, short prompts: %.1f us per token without the host path, %.1f with it (medians of %zu and %zu) "
                "-> host path %s for prompts of %" PRId64 "-%" PRId64 " tokens where the cost model allows (LLAMA_MOE_CPU_MISS_PROMPT_AUTO=0 off)\n",
                tag, t_off, t_on, pt_t[0].size(), pt_t[1].size(), st ? "on" : "off", max_batch + 1, cm_max);
        }
        if (st != pt_arm) {
            cm_gen++;
        }
        pt_state = st;
        pt_arm   = st;
        pt_count = 0;
    }

    // before the graph of every ubatch while a trial runs or waits
    void cmt_tick(int64_t n_tokens) {
        if (!cmt_pending()) {
            return;
        }
        const bool counted = cmt_counted(n_tokens);
        const int64_t now = ggml_time_us();
        cmt_t_prev = -1;
        if (cmt_done < 0) {
            if (!counted || --cmt_wait > 0) {
                return;
            }
            cmt_wait   = -1;
            cmt_done   = 0;
        }
        if (!counted) {
            return;
        }
        auto median = [](std::vector<int64_t> v) {
            std::sort(v.begin(), v.end());
            return v.empty() ? 0.0 : (double) v[v.size()/2];
        };
        // k19-cmtpersist: a recorded trial runs its recorded length and takes its recorded verdict (no timing decides)
        cmt_ps_setup();
        const bool replay = cmt_ps_state == 1 && (size_t) cmt_n < cmt_rec_file;
        const int  end_at = replay ? std::min(cmt_rec[cmt_n].len, 5*cmt_block) : 5*cmt_block;
        cmt_len_now = cmt_done;
        // the host path much slower than without it ends the trial early
        if (!replay && cmt_t[0].size() >= 4 && cmt_t[1].size() >= 4 &&
                *std::min_element(cmt_t[1].begin(), cmt_t[1].end()) > (1.0 + 8*cmt_margin)*median(cmt_t[0])) {
            cmt_done = 5*cmt_block;
        }
        if (cmt_done >= end_at) {
            cmt_done = -1;
            const double t_off = median(cmt_t[0]);
            const double t_on  = median(cmt_t[1]);
            bool on, host_ok;
            int  gap_next;
            if (replay) {
                const llama_moe_tp::rec & r = cmt_rec[cmt_n];
                on       = r.on != 0;
                host_ok  = r.host_ok != 0;
                gap_next = std::max(1, r.gap);
            } else {
                // hysteresis only where the verdicts are persisted, against this launch's (or the record's) last verdict
                on       = llama_moe_tp::decide(t_on, t_off, cmt_t[0].size(), cmt_t[1].size(), cmt_n > 0, cm_small_ok, cmt_margin, cmt_hyst());
                host_ok  = !cmt_host_slow(t_on, t_off);
                // a clear loss is checked again much later: each trial runs 2 blocks on the slower path, and their fewer admissions leave
                // the cache colder for a while (G4 Qwen3.6 UD-Q4_K_XL 12 GB, gpt-oss-20b 8 GB: -4..-8% decode with the path off but re-tried)
                gap_next = std::min(on || t_on < 1.05*t_off ? 2*cmt_gap : std::max(4*cmt_gap, 512), cmt_gap_max);
                if (cmt_ps_state == 1) {
                    llama_moe_tp::rec r;
                    r.on = on; r.host_ok = host_ok; r.gap = gap_next; r.len = cmt_len_now; r.ratio = t_off > 0.0 ? t_on/t_off : 0.0;
                    cmt_rec.push_back(r);
                    cmt_rec_dirty = true;
                }
            }
            cmt_n++;
            if (cmt_n <= 4 || on != cm_small_ok || cm_dbg) {
                if (replay) {
                    LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses trial %d: replayed from the calibration file -> %s for small batches "
                        "(recorded t_on / t_off %.3f; this launch %.3f / %.3f ms, medians of %zu and %zu)\n",
                        tag, cmt_n, on ? "on" : "off", cmt_rec[cmt_n - 1].ratio, t_off/1e3, t_on/1e3, cmt_t[0].size(), cmt_t[1].size());
                } else {
                    LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses trial %d: %.3f ms per small batch without the host path, %.3f ms with it (medians of %zu and %zu) -> %s for small batches%s%s\n",
                        tag, cmt_n, t_off/1e3, t_on/1e3, cmt_t[0].size(), cmt_t[1].size(), on ? "on" : "off",
                        !host_ok ? ", off for short prompts (the host is slower than calibrated)" : "",
                        cmt_n > 1 && cmt_hyst() > 0.0 ? " (hysteresis, LLAMA_MOE_CPU_MISS_TRIAL_HYST)" : "");
                }
            }
            cmt_t[0].clear();
            cmt_t[1].clear();
            cm_gen++; // the small batches and the short prompts (cm_prompt_ok) may change
            cm_small_ok   = on;
            cm_small_live = on;
            if (host_ok != cmt_host_ok) {
                cm_gen++;
            }
            cmt_host_ok = host_ok;
            cmt_wait = cmt_gap;
            cmt_gap  = gap_next;
            cmt_ps_flush();
            return;
        }
        const int blk = cmt_done/cmt_block;
        const int arm = blk == 2 || blk == 3;
        if (arm != cmt_arm || cmt_done == 0) {
            cmt_skipn = cmt_skip;
        }
        if ((arm != 0) != cm_small_live) {
            cm_gen++;
        }
        cmt_arm       = arm;
        cm_small_live = arm != 0;
        if (blk > 0 && cmt_skipn == 0) {
            cmt_t_prev = now;
        }
        cmt_skipn = std::max(0, cmt_skipn - 1);
        cmt_done++;
    }

    // reads the device counters of the lookups: distinct hits, fetched misses, misses left to the host
    void cm_counters(uint64_t c[3]) const {
        c[0] = c[1] = c[2] = 0;
        for (const group & g : groups) {
            if (!g.dev_lookup || g.n_slots == 0 || g.state == nullptr || g.state->data == nullptr) {
                continue;
            }
            uint64_t cnt[2] = { 0, 0 };
            ggml_backend_tensor_get(g.state, cnt, 2*sizeof(int32_t), sizeof(cnt));
            c[0] += cnt[0];
            c[1] += cnt[1];
            if (g.counts != nullptr && g.counts->data != nullptr) {
                uint64_t h = 0;
                ggml_backend_tensor_get(g.counts, &h, (size_t) g.counts->ne[0]*(g.counts->ne[1] - 1)*sizeof(int32_t), sizeof(h));
                c[2] += h;
            }
        }
    }

    // LLAMA_MOE_CACHE_STATS_REQ=1 (diagnostic, read-only): the cumulative lookup counters at each request begin, so the hit rate of one
    // request's decode is the difference to the next line (or to the exit summary); the read waits for the device, between requests
    void stats_request_snapshot() {
        static const bool on = [] { const char * e = getenv("LLAMA_MOE_CACHE_STATS_REQ"); return e != nullptr && atoi(e) != 0; }();
        if (!on) {
            return;
        }
        static int n_req = 0;
        uint64_t c[3] = { 0, 0, 0 };
        if (device_lookup) {
            cm_counters(c);
        }
        LLAMA_LOG_INFO("llama_moe_cache%s: stats at request %d: device hits = %llu, device misses = %llu, host computed = %llu, "
            "small hits = %zu, small misses = %zu, large hits = %zu, large misses = %zu, tail skipped = %llu\n", tag, n_req++,
            (unsigned long long) c[0], (unsigned long long) c[1], (unsigned long long) c[2], stats_small.hits, stats_small.misses,
            stats_large.hits, stats_large.misses, (unsigned long long) (tail_rank > 0 ? tail_skipped() : 0));
    }

    // the batch runs device lookups (and moves the counters)
    bool cm_counted(int64_t n_tokens) const {
        return n_tokens >= 1 && (n_tokens <= max_batch || cm_serves(n_tokens));
    }

    bool cm_tick_reads(int64_t n_tokens) const {
        return cm && cm_cal && !cm_force && !cm_paused && cm_counted(n_tokens) && cm_seen >= cm_skip && (!cm_have0 || cm_nb + 1 >= cm_win);
    }

    // before the graph of every ubatch (backends synchronized when cm_tick_reads)
    void cm_tick(int64_t n_tokens) {
        if (!cm || !cm_cal || cm_paused || !cm_counted(n_tokens)) {
            return;
        }
        if (cm_share_of(n_tokens) < 1.0f) {
            (n_tokens > max_batch ? cm_n_prompt : cm_n_small)++;
        }
        if (cm_recal_wait > 0 && n_tokens <= max_batch && --cm_recal_wait == 0) {
            cm_recal_pending = true; // served when the server goes idle (llama_context::restore_pp_plan): cm_calibrate re-times the host
            cm_recal_wait    = -1;
        }
        if (cm_force) {
            return;
        }
        if (cm_seen < cm_skip) {
            cm_seen++;
            return;
        }
        if (!cm_have0) {
            cm_counters(cm_c0);
            cm_have0 = true;
            cm_nb    = 0;
            return;
        }
        if (++cm_nb < cm_win) {
            return;
        }
        uint64_t c1[3];
        cm_counters(c1);
        const double hits = (double) (c1[0] - cm_c0[0]);
        const double miss = (double) (c1[1] - cm_c0[1]) + (double) (c1[2] - cm_c0[2]);
        std::copy(c1, c1 + 3, cm_c0);
        cm_nb  = 0;
        cm_win = cm_win2;
        cm_nwin++;
        if (hits + miss <= 0.0) {
            return;
        }
        cm_rate = miss/(hits + miss);
        cm_decide();
        cm_measured = true;
        if (cmw_on) {
            cmw_n += cm_win1;
            const bool flat = cmw_prev >= 0.0 && cm_rate >= cmw_flat*cmw_prev;
            if (flat || cmw_n >= cmw_max) {
                cmw_on        = false;
                cmt_done      = 0;
                cmt_t_prev    = -1;
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: warm phase over after %d small batches (miss rate %.1f%%, before %.1f%%), "
                    "the first trial starts\n", tag, cmw_n, 100.0*cm_rate, 100.0*cmw_prev);
            } else {
                cm_win = cm_win1;
            }
            cmw_prev = cm_rate;
        }
    }

    // per batch size: on where the expected time saved per layer beats the cost of the host split
    void cm_decide() {
        // k13: frozen small batches keep every per-size decision as it was at the freeze (a decision from the miss rate of a window
        // would follow the drafts: their verify windows look up other tokens than single token decodes)
        if (cm_frozen) {
            return;
        }
        const group * gp = nullptr;
        for (const group & g : groups) {
            if (g.dev_lookup && g.n_slots > 0 && g.counts != nullptr) {
                gp = &g;
                break;
            }
        }
        if (gp == nullptr || cm_rate < 0.0) {
            return;
        }
        const int64_t n_expert = gp->ref[0]->ne[2];
        bool changed = false;
        int64_t n_on = 0, lo = 0, hi = 0;
        for (int64_t n = 1; n <= cm_max; ++n) {
            uint8_t on = 0;
            if (cm_share[n] < 1.0f) {
                const double D    = cm_distinct(n, n_expert);
                const double tbar = std::max(1.0, (double) n*std::min<int64_t>(n_expert_used, n_expert)/D);
                const double tc   = cm_t1 + (tbar - 1.0)*cm_dt;
                const double f    = cm_share[n];
                const double m    = cm_rate*D;
                const double gain = m*(cm_tup - std::max(f*cm_tup, (1.0 - f)*tc));
                // k15-np (LLAMA_MOE_SINGLE_MIN_MISSES, default 0 = off): the mailbox's cheaper split (single_split_frac) only where a
                // batch misses at least this many distinct experts per layer at the measured miss rate. Off by default: G4 k15g4n at 1.5
                // still turned the host path on at q3m 12 GB and q4x 16 GB + MTP, whose first windows miss 7-12% (m(3) 1.8-2.8) before
                // settling at 4-6%, against 13.5%+ (3.4) where the host path wins; the cache-share gate on the early note (llama-graph.cpp,
                // LLAMA_MOE_SINGLE_NOTE_PLAIN_MAX_FRAC) separates those cells instead. Kept as an override
                static const double mb_min_misses = [] {
                    const char * e = getenv("LLAMA_MOE_SINGLE_MIN_MISSES");
                    return e != nullptr ? std::max(0.0, atof(e)) : 0.0;
                }();
                const bool mb_cheap = n <= max_batch && single_weigh() && single_usable(n) && m >= mb_min_misses;
                // k19-l4poly: on a host whose host experts run on the host team (a small host, see single_team_host), where the timing
                // trials run, a small batch is a candidate at LLAMA_MOE_CPU_MISS_SPLIT_SCALE_TEAM (0.5) times the weighed split instead of
                // cm_split_scale (3, set on G4 for the graph split's unmodelled cost): the trial then decides it. Colab L4 Flint 16 GB
                // (55.8% cached, trials on): 1-token batches gain ~60 us per layer against a weighed split of 1.5 x 3 x 0.2 x 200-330 us,
                // so decode fetched every miss over PCIe, while a fixed host share of 1/4 decoded +13% (k19-flint); the scale 0.5 on the
                // same VM: chat +8.4%, 2k +32% (l4p_b2). LLAMA_MOE_CPU_MISS_SPLIT_SCALE_TEAM_MAILBOX=1: mailbox sizes only. An explicit
                // LLAMA_MOE_CPU_MISS_SPLIT_SCALE applies to every size as before
                static const bool team_mb_only = getenv("LLAMA_MOE_CPU_MISS_SPLIT_SCALE_TEAM_MAILBOX") != nullptr &&
                    atoi(getenv("LLAMA_MOE_CPU_MISS_SPLIT_SCALE_TEAM_MAILBOX")) != 0;
                const double scale = (mb_cheap || !team_mb_only) && !cm_split_scale_env && cm_trial && !cm_trials_skipped && !cm_force &&
                    single_team_host() ? cm_split_scale_team() : cm_split_scale;
                const double need = (n <= max_batch ? cm_margin*scale*cm_split*(mb_cheap ? single_split_frac : 1.0) :
                                     cm_margin_p*(cm_split + n*cm_act))*
                                    (cm_on[n] && cm_measured ? cm_hyst : 1.0);
                // a batch above max_batch runs the host path in the groups whose slots hold its experts (see use_device), the
                // other groups copy its experts as a prompt does
                bool fits = false;
                for (const group & g : groups) {
                    if (g.dev_lookup && g.n_slots > 0) {
                        fits = fits || std::min<int64_t>(n*n_expert_used, g.ref[0]->ne[2]) <= g.n_slots;
                    }
                }
                on = gain > need && (n <= max_batch || fits) ? 1 : 0;
            }
            if (cm_frozen && n <= std::min<int64_t>(cm_frozen_max, max_batch)) {
                on = 1;
            }
            changed = changed || on != cm_on[n];
            cm_on[n] = on;
            if (on) {
                n_on++;
                lo = lo == 0 ? n : lo;
                hi = n;
            }
        }
        if (changed) {
            cm_gen++;
        }
        if (changed || cm_dbg || cm_nwin <= 2) {
            LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses (window %d): miss rate %.1f%% of the distinct experts looked up -> host path %s",
                tag, cm_nwin, 100.0*cm_rate, n_on == 0 ? "off" : "on");
            if (n_on > 0) {
                LLAMA_LOG_CONT(" for %" PRId64 " batch sizes in [%" PRId64 ", %" PRId64 "] (fetch share %.3f at %" PRId64 ")", n_on, lo, hi, cm_share[lo], lo);
            }
            LLAMA_LOG_CONT("\n");
        }
    }

    // fetch share: the host and the device finish at the same time when share*t_upload = (1 - share)*t_host per missing expert
    // t_host: one token through the expert tensors of a cached layer on the host backend, t_upload: copy of an expert from host memory
    void probe(ggml_backend_t backend_cpu) {
        if (cm) {
            cm_calibrate(backend_cpu);
            return;
        }
        hybrid_probe = false;
        const group * gp = nullptr;
        for (const group & g : groups) {
            if (g.dev_lookup && g.n_slots > 0 && g.counts != nullptr) {
                gp = &g;
                break;
            }
        }
        if (gp == nullptr || backend_cpu == nullptr) {
            hybrid = false;
            return;
        }
        const group & g = *gp;
        const layer & l = layers[g.layers[0]];
        const int64_t n_expert = g.ref[0]->ne[2];
        const int     k        = (int) std::max<int64_t>(2, std::min<int64_t>(n_expert_used, n_expert));
        const int     n_rep    = 7;

        ggml_init_params params = {
            /*.mem_size   =*/ (4 + 3*l.experts.size())*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(16, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        ggml_tensor * ids_k = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, k, 1);
        ggml_tensor * ids_1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
        ggml_cgraph * gf_k  = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_cgraph * gf_1  = ggml_new_graph_custom(ctx.get(), 16, false);
        std::vector<ggml_tensor *> xs;
        for (ggml_tensor * w : l.experts) {
            ggml_tensor * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, w->ne[0], 1, 1);
            xs.push_back(x);
            ggml_build_forward_expand(gf_k, ggml_mul_mat_id(ctx.get(), w, x, ids_k));
            ggml_build_forward_expand(gf_1, ggml_mul_mat_id(ctx.get(), w, x, ids_1));
        }
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend_cpu));
        if (!buf) {
            hybrid = false;
            return;
        }
        for (ggml_tensor * x : xs) {
            std::vector<float> v(ggml_nelements(x), 0.01f);
            ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
        }

        // different experts in every repetition, so they are read from memory as in generation
        int64_t next = 0;
        auto set_ids = [&](ggml_tensor * ids, int n) {
            std::vector<int32_t> v(n);
            for (int i = 0; i < n; ++i) {
                v[i] = (int32_t) not_core(g, g.layers[0], next, n_expert);
            }
            ggml_backend_tensor_set(ids, v.data(), 0, n*sizeof(int32_t));
        };
        auto median = [](std::vector<int64_t> v) {
            std::sort(v.begin(), v.end());
            return (double) v[v.size()/2];
        };
        std::vector<int64_t> tk, t1, tu;
        for (int r = 0; r <= n_rep; ++r) {
            set_ids(ids_k, k);
            int64_t t0 = ggml_time_us();
            ggml_backend_graph_compute(backend_cpu, gf_k);
            const int64_t dk = ggml_time_us() - t0;
            set_ids(ids_1, 1);
            t0 = ggml_time_us();
            ggml_backend_graph_compute(backend_cpu, gf_1);
            const int64_t d1 = ggml_time_us() - t0;
            if (r > 0) {
                tk.push_back(dk);
                t1.push_back(d1);
            }
        }

        // upload into the spare slot at the end of the banks (never looked up)
        ggml_backend_t backend = devices[g.id].backend;
        ggml_backend_synchronize(backend);
        for (int r = 0; r <= n_rep; ++r) {
            const int64_t t0 = ggml_time_us();
            for (int i = 0; i < k; ++i) {
                const int64_t e = not_core(g, g.layers[0], next, n_expert);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t nb = l.experts[ip]->nb[2];
                    ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) l.experts[ip]->data + e*nb, (size_t) g.n_slots*nb, nb);
                }
            }
            if (r > 0) {
                tu.push_back(ggml_time_us() - t0);
            }
        }

        const double t_k  = median(tk);
        const double t_1  = median(t1);
        const double slope = (t_k - t_1)/(k - 1);
        const double t_host = std::max(slope > 0.0 ? slope : t_k/k, 1e-3);
        const double t_up   = std::max(median(tu)/k, 1e-3);

        // rounded to 1/8 so close measurements give the same share; the host path is not used when it would take less than 1/4
        const double share = t_host/(t_host + t_up);
        float q = (float) std::round(share*8.0)/8.0f;
        if (q > 0.75f) {
            q = 1.0f;
        }
        q = std::max(q, 0.125f);
        hybrid_share = q;
        if (hybrid_share >= 1.0f) {
            hybrid = false;
        } else if (hybrid_auto) {
            win_done   = 0;
            win_t_prev = -1;
        } else {
            hybrid_live = true;
        }

        LLAMA_LOG_INFO("llama_moe_cache: hybrid probe: host %.2f GB/s (%.1f us per expert, %.1f us for %d), upload %.2f GB/s (%.1f us per expert) "
            "-> fetch share %.3f%s\n",
            g.expert_bytes/t_host/1e3, t_host, t_k, k, g.expert_bytes/t_up/1e3, t_up, hybrid_share,
            !hybrid ? " (hybrid path off)" : hybrid_auto ? ", trial of the first small batches" : "");
    }

    // header of the device state, see ggml_moe_cache_lookup
    static constexpr int64_t n_hdr = 8;

    void upload(group & g) {
        ggml_backend_t backend = devices[g.id].backend;
        ggml_backend_synchronize(backend);

        const int64_t n_expert = g.ref[0]->ne[2];
        for (int32_t il : g.layers) {
            ggml_backend_tensor_set(g.map, g.lru.slot_map[il], il*n_expert*sizeof(int32_t), n_expert*sizeof(int32_t));
        }
        // hand and epoch, the counters are kept
        const int32_t hdr[2] = { 0, 0 };
        ggml_backend_tensor_set(g.state, hdr, 0, sizeof(hdr));
        if (g.counts != nullptr) {
            // the number of filled slots, see ggml_moe_cache_assign
            const int32_t filled = (int32_t) std::count_if(g.lru.key_of.begin(), g.lru.key_of.end(), [](int32_t k) { return k >= 0; });
            ggml_backend_tensor_set(g.state, &filled, 6*sizeof(int32_t), sizeof(filled));
        }
        std::vector<int32_t> buf(3*(size_t) g.n_slots, 0);
        std::copy(g.lru.key_of.begin(), g.lru.key_of.end(), buf.begin());
        // k15-pred r20 (LLAMA_MOE_UPLOAD_AHEAD_DEC_REF=1): the slots the last decode used, if they still hold that expert, start with the
        // reference bit set, so the decode after a prompt does not evict its own working set first (the hand starts at slot 0 with every bit
        // clear; after a prompt with uploads ahead the decode missed 1400 times per 32 tokens vs 850 off, 90 and fewer with this and KEEP_DECODE)
        if (ua_dec_ref && ua_mode > 0 && !g.ua_dec_key.empty()) {
            for (int32_t s = 0; s < g.n_slots; ++s) {
                if (g.lru.key_of[s] >= 0 && g.ua_dec_key[s] == g.lru.key_of[s]) {
                    buf[g.n_slots + s] = 1;
                }
            }
        }
        ggml_backend_tensor_set(g.state, buf.data(), n_hdr*sizeof(int32_t), buf.size()*sizeof(int32_t));
        if (g.scores != nullptr) {
            ggml_backend_tensor_set(g.scores, g.lru.scores.data(), 0, g.lru.scores.size()*sizeof(int32_t));
            ggml_backend_tensor_set(g.state, &g.lru.w, 7*sizeof(int32_t), sizeof(int32_t));
        }

        g.dev_gen = g.lru.gen;
    }

    // debug: every slot the host state maps must hold the bytes of its expert
    void verify(group & g, const char * tag) {
        ggml_backend_synchronize(devices[g.id].backend);
        const int32_t n_expert = g.lru.n_expert;
        std::vector<uint8_t> buf;
        size_t n_checked = 0;
        size_t n_bad = 0;
        for (int32_t s = 0; s < g.n_slots; ++s) {
            const int32_t key = g.lru.key_of[s];
            if (key < 0) {
                continue;
            }
            const int32_t il = key / n_expert;
            const int32_t e  = key % n_expert;
            const layer & l = layers[il];
            for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const size_t sz = w->nb[2];
                buf.resize(sz);
                ggml_backend_tensor_get(g.banks[ip], buf.data(), (size_t) s*g.banks[ip]->nb[2], sz);
                if (memcmp(buf.data(), (const uint8_t *) w->data + (size_t) e*sz, sz) != 0) {
                    if (n_bad < 8) {
                        size_t first = 0;
                        while (first < sz && buf[first] == ((const uint8_t *) w->data)[(size_t) e*sz + first]) {
                            first++;
                        }
                        LLAMA_LOG_WARN("%s: verify %s: slot %d (layer %d, expert %d, bank %zu) differs from byte %zu of %zu (n_lent = %d of %d)\n",
                            __func__, tag, s, il, e, ip, first, sz, g.lru.n_lent, g.n_slots);
                    }
                    n_bad++;
                }
            }
            n_checked++;
        }
        LLAMA_LOG_INFO("%s: verify %s: %zu slots checked, %zu bank slots differ\n", __func__, tag, n_checked, n_bad);
    }

    void download(group & g) {
        ggml_backend_synchronize(devices[g.id].backend);

        std::vector<int32_t> buf(2*(size_t) g.n_slots);
        ggml_backend_tensor_get(g.state, buf.data(), n_hdr*sizeof(int32_t), buf.size()*sizeof(int32_t));
        std::vector<int32_t> sc;
        int32_t w = 0;
        if (g.scores != nullptr) {
            sc.resize(g.lru.scores.size());
            ggml_backend_tensor_get(g.scores, sc.data(), 0, sc.size()*sizeof(int32_t));
            ggml_backend_tensor_get(g.state, &w, 7*sizeof(int32_t), sizeof(int32_t));
        }
        g.lru.load(g.layers, buf.data(), buf.data() + g.n_slots, g.scores != nullptr ? sc.data() : nullptr, w);
        if (ua_mode > 0 && (ua_keep_dec || ua_dec_ref)) {
            g.ua_dec_key.assign(g.n_slots, -1);
            for (int32_t s = 0; s < g.n_slots; ++s) {
                if (buf[s] >= 0 && buf[g.n_slots + s] != 0) {
                    g.ua_dec_key[s] = buf[s];
                }
            }
        }

        g.dev_gen   = g.lru.gen;
        g.dev_newer = false;
        if (verify_enabled) {
            verify(g, "after device lookups");
        }
    }

    // LLAMA_MOE_CACHE_STATE_LOG=1 (debug, k11 3l): a hash of the host cache state (slots, LRU order, scores) at the first small batch
    // after large ones, i.e. what a prompt left for the decode
    bool st_after_large = false;
    void state_log() {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](int64_t v) { h ^= (uint64_t) v; h *= 1099511628211ull; };
        bool stale = false;
        for (const group & g : groups) {
            stale = stale || g.dev_newer;
            for (int32_t k : g.lru.key_of) { mix(k); }
            for (int32_t k : g.lru.prev)   { mix(k); }
            for (int32_t k : g.lru.next)   { mix(k); }
            mix(g.lru.head); mix(g.lru.tail); mix(g.lru.w);
            for (int32_t k : g.lru.scores) { mix(k); }
        }
        LLAMA_LOG_INFO("llama_moe_cache: state after the prompt: %016llx%s\n", (unsigned long long) h, stale ? " (device newer)" : "");
    }

    void prepare(int64_t n_tokens) {
        ua_drain();
        // k19-l4ttft: a small batch ends the prompt (the scratch of the prompt layer reads is not read again)
        if (n_tokens <= n_small_max && (plr.il >= 0 || plr.p_t0 != 0)) {
            plr_prompt_end();
        }
        // k19-ramprof2: the profile fill (see hprof_order)
        if (host && wstore != nullptr && !hprof_ticket.empty()) {
            hprof_prepare(n_tokens);
        }
        // k19-autoprof: the target set at a request's first large batch (see ap_use)
        if (host) {
            ap_prepare(n_tokens);
        }
        // k19-l4fill: the refill of the freed RAM slots (see fill_mode)
        if (host) {
            fill_prepare(n_tokens);
        }
        // a medium ubatch right after a large one is the tail of the same prompt: the large ubatch just seeded the cache with the
        // experts this text routes, and evicting them for guesses costs more than the overlap gains (G4 q4x 12 GB multi-turn tails +8%)
        ua_skip = ua_prev_tokens > ua_max_tokens;
        ua_prev_tokens = n_tokens;
        static const bool st_log = getenv("LLAMA_MOE_CACHE_STATE_LOG") != nullptr && atoi(getenv("LLAMA_MOE_CACHE_STATE_LOG")) != 0;
        if (st_log) {
            if (n_tokens > n_small_max) {
                st_after_large = true;
            } else if (st_after_large) {
                st_after_large = false;
                state_log();
            }
        }
        ahead_epoch++;
        // a large batch plans the reads of its uploads with the RAM tier as it is: every read ahead is complete first
        if (ahead_on() && n_tokens > n_small_max) {
            ahead_drain();
            ahead_il = -1;
        }
        for (group & g : groups) {
            if (g.n_slots == 0 || !g.dev_lookup) {
                continue;
            }
            if (use_device(g, n_tokens)) {
                // the device lookup and the hybrid assign treat every slot as a cache entry: the lent slots come back first
                // the scheduler already took them back when it allocated the graph of this small batch, this only guards the order
                if (g.lru.n_lent > 0) {
                    unlend(g);
                }
                if (g.dev_gen != g.lru.gen) {
                    GGML_ASSERT(!g.dev_newer);
                    upload(g);
                }
                g.dev_newer = true;
            } else if (g.dev_newer) {
                download(g);
            }
        }
        // k13 diagnostic (LLAMA_MOE_CACHE_STATE_LOG=1): a hash of every group's slot map and state as the ubatch of n_tokens starts,
        // logged when it changed; verify windows and single token decodes must see the same sequence of states
        static const bool state_log = getenv("LLAMA_MOE_CACHE_STATE_LOG") != nullptr && atoi(getenv("LLAMA_MOE_CACHE_STATE_LOG")) != 0;
        if (state_log) {
            static uint64_t n_ub = 0;
            n_ub++;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                group & g = groups[ig];
                if (g.n_slots == 0 || !g.dev_lookup || g.map == nullptr) {
                    continue;
                }
                std::vector<uint8_t> buf;
                uint64_t h[3] = { 1469598103934665603ULL, 1469598103934665603ULL, 1469598103934665603ULL };
                ggml_tensor * ts[3] = { g.map, g.state, g.counts };
                ggml_backend_synchronize(devices[g.id].backend);
                for (int k = 0; k < 3; ++k) {
                    if (ts[k] == nullptr || ts[k]->buffer == nullptr) {
                        continue;
                    }
                    buf.resize(ggml_nbytes(ts[k]));
                    if (g.dev_newer || k > 0) {
                        ggml_backend_tensor_get(ts[k], buf.data(), 0, buf.size());
                    } else {
                        // host state newer: the slot maps of the LRU
                        buf.clear();
                        for (int32_t il : g.layers) {
                            const uint8_t * p = (const uint8_t *) g.lru.slot_map[il];
                            buf.insert(buf.end(), p, p + g.lru.n_expert*sizeof(int32_t));
                        }
                    }
                    for (uint8_t b : buf) {
                        h[k] = (h[k] ^ b)*1099511628211ULL;
                    }
                }
                static std::vector<uint64_t> last;
                if (last.size() < 3*groups.size()) {
                    last.resize(3*groups.size(), 0);
                }
                if (h[0] != last[3*ig] || h[1] != last[3*ig + 1]) { // (the counts carry statistics that every lookup writes)
                    LLAMA_LOG_INFO("moe_cache_state: ub %llu n %lld group %zu dev %d map %016llx state %016llx counts %016llx\n", (unsigned long long) n_ub,
                        (long long) n_tokens, ig, (int) g.dev_newer, (unsigned long long) h[0], (unsigned long long) h[1], (unsigned long long) h[2]);
                    last[3*ig] = h[0]; last[3*ig + 1] = h[1]; last[3*ig + 2] = h[2];
                }
            }
        }
    }

    // the scheduler places the prefetched copies of a large batch: they go to the last slots of the largest group of the device, which
    // are evicted while lent, so the device does not keep a staging arena that only large batches use; n_copies = 0 takes the slots back
    void lend(ggml_backend_t backend, ggml_backend_sched_lend_copy * copies, int n_copies, int depth, bool dry, int64_t n_tokens) {
        const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.backend == backend; });
        // released (see release): no slots to lend, nothing to take back
        if (it_dev == devices.end() || !it_dev->buf_bank || it_dev->released) {
            return;
        }
        const int32_t id = (int32_t) (it_dev - devices.begin());

        // k15-lfm2: a graph of in-place prompts lends one layer of slots for its missing experts; its other copies (if any) stay in the
        // scheduler's arena
        if (inplace_graph) {
            prompt_lend(backend, id, dry);
            return;
        }

        if (n_copies == 0) {
            if (!dry) {
                for (group & g : groups) {
                    if (g.id == id && g.lru.n_lent > 0) {
                        unlend(g);
                    }
                }
            }
            return;
        }

        if (!lend_auto && !dry && devices[id].lend_auto < 0) {
            devices[id].lend_auto = lend_mode != 0 ? 1 : 0;
            LLAMA_LOG_INFO("%s: %s: MoE cache lending %s (LLAMA_MOE_CACHE_LEND=%d)\n", __func__, ggml_backend_name(backend), lend_mode != 0 ? "on" : "off", lend_mode);
        }
        if (lend_mode == 0) {
            return;
        }
        if (!lend_multi && std::count_if(devices.begin(), devices.end(), [](const device & d) { return d.host_bytes > 0 && d.buf_bank; }) > 1) {
            if (!lend_off_log) {
                LLAMA_LOG_INFO("%s: MoE cache on several devices: the cache does not lend slots to the staging copies (LLAMA_MOE_CACHE_LEND_MULTI=1 to force)\n", __func__);
                lend_off_log = true;
            }
            return;
        }

        // the largest group of the device
        group * gl = nullptr;
        bool host_read = false;
        for (group & g : groups) {
            if (g.id == id && g.n_slots > 0 && (gl == nullptr || (size_t) g.n_slots*g.expert_bytes > (size_t) gl->n_slots*gl->expert_bytes)) {
                gl = &g;
            }
            host_read = host_read || (g.id == id && g.host_read);
        }
        // small batches looked up on the device may use every slot of the group
        if (gl == nullptr || use_device(*gl, n_tokens)) {
            return;
        }
        // a host lookup (e.g. the last layer of a prompt cropped to its outputs) reads the banks of its group as weights: the scheduler treats
        // the lent buffer as a compute buffer only in the splits that read copies placed there (a lookup reads the banks in a split of its own),
        // and the copies keep away from the slots the lookup may read past (see place); the graph allocator in lent memory (LEND=2) is live in
        // every split, it is not lent to such a graph
        const bool lend_graph_now = lend_mode == 2 || graph_lend_scope;
        if (host_read && lend_graph_now) {
            return;
        }
        group & g = *gl;
        const bool guard = g.host_read;

        // the graph allocator borrows only with LLAMA_MOE_CACHE_LEND=2 (opt-in, see plans/impl_moe_vram_lend.md), by default it keeps
        // its own buffer
        std::vector<ggml_backend_sched_lend_copy> only;
        ggml_backend_sched_lend_copy * all = copies;
        const int n_all = n_copies;
        // the graph allocation comes first, as n_graph entries with a negative ordinal (parts in different regions, see ggml_backend_sched_lend_copy)
        int n_graph = 0;
        while (n_graph < n_copies && copies[n_graph].src == nullptr && copies[n_graph].ordinal < 0) {
            n_graph++;
        }
        const size_t graph_all = n_graph > 0 ? copies[0].size : 0; // bytes of the whole graph allocation
        if (!lend_graph_now && n_graph > 0) {
            only.assign(copies + n_graph, copies + n_copies);
            copies   = only.data();
            n_copies = (int) only.size();
            n_graph  = 0;
            if (n_copies == 0) {
                return;
            }
        }
        // k15-pp (LLAMA_MOE_LM_LEND_GRAPH_ONLY=0 off): a merged layer-major graph whose growth does not fit lends from the cache even where
        // lending is off for the staging copies (lend_decide: they are a small share of the cache): only the graph allocation, for that graph,
        // and the lent slots are refilled after it, so the decode finds the cache as it was. Split instead, each half streamed every host
        // expert again: G4 Qwen3.8-Flash-Next UD-IQ4_XS at 32 GB (19 GB cache, lending off at 6.7%), 8k prompt, 4 ubatches of 2048 were
        // short of 10.5 MiB (755 MiB growth, 744.6 MiB free) and ran as two graphs
        static const bool graph_only_env = getenv("LLAMA_MOE_LM_LEND_GRAPH_ONLY") == nullptr || atoi(getenv("LLAMA_MOE_LM_LEND_GRAPH_ONLY")) != 0;
        bool graph_only = false;
        if (graph_only_env && graph_lend_scope && lend_auto && devices[id].lend_auto == 0 && n_graph > 0) {
            only.assign(copies, copies + n_graph);
            copies     = only.data();
            n_copies   = n_graph;
            graph_only = true;
        }
        const size_t graph_need = n_graph > 0 ? graph_all : 0;
        for (int i = 0; i < n_graph; ++i) {
            copies[i].buffer = nullptr;
            copies[i].offset = 0;
            copies[i].size   = 0;
        }

        // the copies of `depth` consecutive splits are live together, the graph allocation all the time
        size_t need  = 0;
        size_t split = 0; // copies of the largest split
        for (int i = n_graph; i < n_copies; ++i) {
            size_t live = 0;
            size_t own  = 0;
            for (int j = i; j < n_copies && copies[j].ordinal < copies[i].ordinal + depth; ++j) {
                live += copies[j].size;
                own  += copies[j].ordinal == copies[i].ordinal ? copies[j].size : 0;
            }
            need  = std::max(need, live);
            split = std::max(split, own);
        }
        need += graph_need;

        // a prompt that seeds from its own ids holds the copies of a split in the graph allocation (see llama_context::sched_reserve_seed)
        const size_t seed  = seed_own ? split : 0;
        const size_t graph = graph_all + seed;
        // a measurement records the copies the device lends for the memory it frees (set below once the slots hold them)
        const bool save_rec = dry && !graph_only && lend_save && (!lend_auto || lend_saves(id, need, seed));
        if (lend_auto && !graph_only && !lend_decide(id, need, graph, seed, dry)) {
            return;
        }

        // the smallest number of slots that holds every copy (and the graph allocation): at most half of the slots for the copies alone
        // (large batches also read and seed the cache), and with host lookups in the group the slots they need, plus the slot kept empty in
        // front of the copies; with the graph allocation (never with host lookups, see above) every slot but those of one layer (a batch
        // that looks up its experts in the cache needs up to all experts of a layer resident at once)
        const int32_t n_keep = std::min<int32_t>(g.n_slots / 2, g.lru.n_expert + 1);
        const int32_t n_max0 = guard       ? (int32_t) std::min<int64_t>(g.n_slots / 2, g.n_slots - g.host_need - 1)
                             : n_graph > 0 ? std::max<int32_t>(g.n_slots / 2, g.n_slots - n_keep)
                             :               g.n_slots / 2;
        // k19-ptres: never into the core
        const int32_t n_max  = g.lru.n_core > 0 ? std::min<int32_t>(n_max0, g.n_slots - g.lru.n_core - 2) : n_max0;
        int32_t n = (int32_t) std::min<size_t>((need + g.expert_bytes - 1) / g.expert_bytes, (size_t) n_max + 1);
        const int32_t step = std::max<int32_t>(1, n / 16);
        std::vector<size_t> offsets(n_copies);
        std::vector<size_t> part_sizes(n_graph);
        int32_t n_lend = 0;
        // the graph allocation in one part if any number of slots allows it (the graph allocator plans a split allocation less tightly);
        // LLAMA_MOE_CACHE_LEND_PARTS=N > 1 (experimental): else in up to N bank tails
        static const int max_parts = getenv("LLAMA_MOE_CACHE_LEND_PARTS") ? std::max(1, atoi(getenv("LLAMA_MOE_CACHE_LEND_PARTS"))) : 1;
        const int32_t n0 = n;
        // k15-pp (LLAMA_MOE_LM_LEND_GRAPH_PARTS=0 off): a graph-only lend tries one part per bank first. In one part the graph had to fit
        // the tail of a single bank, so the slots lent (and refilled after the group) were ~2.6x the graph: G4 Qwen3.8-Flash-Next 32 GB 8k,
        // a 1740 MiB graph took 2009 slots (4463 MiB); Qwen3.6 q4x local, 509 MiB took 740 slots (1341 MiB)
        static const bool graph_parts_env = getenv("LLAMA_MOE_LM_LEND_GRAPH_PARTS") == nullptr || atoi(getenv("LLAMA_MOE_LM_LEND_GRAPH_PARTS")) != 0;
        const bool parts_first = graph_only && graph_parts_env && n_graph > 1 && g.banks.size() > 1;
        const int  mp          = parts_first ? std::max<int>(max_parts, (int) g.banks.size()) : max_parts;
        const int  n_pass      = parts_first || (n_graph > 1 && max_parts > 1) ? 2 : 1;
        for (int pass = 0; pass < n_pass && n_lend == 0; ++pass) {
            const int multi = parts_first ? 1 - pass : pass;
            for (n = n0; n <= n_max; n = n < n_max ? std::min(n + step, n_max) : n_max + 1) {
                if (place(g, n, copies, n_copies, n_graph, multi ? std::min(n_graph, mp) : 1, graph_need, depth, guard, offsets, part_sizes)) {
                    n_lend = n;
                    break;
                }
            }
        }
        if (n_lend == 0) {
            if (!dry && g.lru.n_lent > 0) {
                unlend(g);
            }
            return;
        }
        // a graph that needs fewer slots than are lent keeps them (LLAMA_MOE_CACHE_LEND_SHRINK=1 gives them back): giving them back clears
        // and refills them, and the next larger graph of the prompt (the ubatches grow with the KV cache) evicts them again
        static const bool lend_shrink = getenv("LLAMA_MOE_CACHE_LEND_SHRINK") != nullptr && atoi(getenv("LLAMA_MOE_CACHE_LEND_SHRINK")) != 0;
        static const bool lend_keep_nested = getenv("LLAMA_MOE_CACHE_LEND_KEEP_NESTED") == nullptr || atoi(getenv("LLAMA_MOE_CACHE_LEND_KEEP_NESTED")) != 0;
        if (!dry && !lend_shrink && g.lru.n_lent > n_lend && g.lru.n_lent <= n_max) {
            std::vector<size_t> offsets_k(n_copies);
            std::vector<size_t> part_sizes_k(n_graph);
            if (place(g, g.lru.n_lent, copies, n_copies, n_graph, 1, graph_need, depth, guard, offsets_k, part_sizes_k)) {
                n_lend     = g.lru.n_lent;
                offsets    = std::move(offsets_k);
                part_sizes = std::move(part_sizes_k);
            } else if (lend_keep_nested) {
                // k19-fn32: place() is a best fit (the graph part goes to the smallest bank tail that holds it), so it can fail at more slots
                // where it fit at fewer: G4 Qwen3.8-Flash-Next 32 GB 30k, the last group (3 ubatches) fit in 2104 slots but not in the 2283
                // lent, and the cache gave all 2283 back, refilled 5.4 GB (111 ms with no kernel) and lent 2104 of them again. The placement
                // in n_lend slots lies in the last n_lend slots of every bank, inside the n_lent lent ones: keep them all with it
                // (LLAMA_MOE_CACHE_LEND_KEEP_NESTED=0: give them back as before)
                LLAMA_LOG_INFO("%s: %s: MoE cache keeps %d lent slots for a graph that fits in %d (nested placement)\n", __func__,
                    ggml_backend_name(backend), g.lru.n_lent, n_lend);
                n_lend = g.lru.n_lent;
            }
        }
        ggml_backend_buffer_t buf = devices[id].buf_bank.get();
        for (int i = 0; i < n_copies; ++i) {
            if (i < n_graph && part_sizes[i] == 0) {
                continue; // part not needed
            }
            copies[i].buffer = buf;
            copies[i].offset = offsets[i];
            if (i < n_graph) {
                copies[i].size = part_sizes[i];
            }
        }
        if (copies != all && graph_only) {
            // the graph entries lead the list; the staging copies stay in the scheduler's arena
            for (int i = 0; i < n_copies; ++i) {
                all[i] = copies[i];
            }
        } else if (copies != all) {
            for (int i = 0; i < n_copies; ++i) {
                all[n_all - n_copies + i].buffer = buf;
                all[n_all - n_copies + i].offset = offsets[i];
            }
        }
        if (dry) {
            if (save_rec) {
                devices[id].dry_save = std::max(devices[id].dry_save, need);
            }
            return;
        }
        // with device lookups the device state may be newer: the slots to evict are only known after it is copied back
        if (g.dev_lookup && g.dev_newer) {
            download(g);
        }
        // the copies of an earlier graph may have used the slots kept free of copies for the host lookups (see place): the lookups read a little
        // past the end of a slot, they find cleared memory there
        if (guard && g.lru.n_lent > 0) {
            ggml_backend_synchronize(devices[id].backend);
            for (ggml_tensor * bank : g.banks) {
                ggml_backend_tensor_memset(bank, 0, (size_t) (g.n_slots - n_lend)*bank->nb[2], bank->nb[2]);
                ggml_backend_tensor_memset(bank, 0, (size_t) g.n_slots*bank->nb[2], bank->nb[2]);
            }
        }
        // a seed is planned and committed within one graph (see llama_context::graph_compute), none is open here
        // closing it anyway keeps its rollback snapshot from outliving the slots it describes
        seed_end(g);
        if (g.lru.n_lent > n_lend) {
            unlend(g);
        }
        if (g.lru.n_lent != n_lend) {
            // the experts the lent slots hold (e.g. from the prefill at load) are uploaded again when the slots come back
            if (n_lend > g.lru.n_lent) {
                keep_evicted(g, n_lend);
                // the RAM tier keeps a copy of what the lent slots hold, the refill then reads it from RAM
                if (host && (size_t) (&g - groups.data()) < host_group.size() && host_group[&g - groups.data()]) {
                    evicted_tmp.clear();
                    for (int32_t s = g.n_slots - n_lend; s < g.n_slots - g.lru.n_lent; ++s) {
                        if (g.lru.key_of[s] >= 0) {
                            evicted_tmp.emplace_back(s, g.lru.key_of[s]);
                        }
                    }
                    shadow_slots(backend, g, evicted_tmp);
                }
            }
            g.lru.set_lent(n_lend);
            LLAMA_LOG_INFO("%s: %s: MoE cache lends %d of %d slots (%.2f MiB) to the staging copies of large batches (LLAMA_MOE_CACHE_LEND=0 to disable)\n",
                __func__, ggml_backend_name(backend), n_lend, g.n_slots, (double) n_lend*g.expert_bytes/1024.0/1024.0);
        }
        // k15-fn LLAMA_MOE_CACHE_LEND_POISON=1 (debug): the lent slots of a graph lend are filled with 0xff (NaN in f32 / f16 / bf16 and in
        // the scales of quantized blocks) before the graph, so a read of lent memory before the graph wrote it shows in the output
        static const bool lend_poison = getenv("LLAMA_MOE_CACHE_LEND_POISON") != nullptr && atoi(getenv("LLAMA_MOE_CACHE_LEND_POISON")) != 0;
        if (lend_poison && n_graph > 0) {
            ggml_backend_synchronize(backend);
            for (ggml_tensor * bank : g.banks) {
                ggml_backend_tensor_memset(bank, 0xff, (size_t) (g.n_slots - n_lend)*bank->nb[2], (size_t) n_lend*bank->nb[2]);
            }
            static int n_poison_log = 0;
            if (n_poison_log++ < 4) {
                LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_LEND_POISON: %d lent slots filled with 0xff before the graph\n", __func__, n_lend);
            }
        }
        n_lends++;
        lend_max = std::max(lend_max, n_lend);
    }

    // k15-fn: the most slots the cache could lend to the graph allocation of a merged layer-major graph on the device of backend (the n_max of
    // lend for a graph without host lookups: every slot but those of one layer), its slot count and bytes; 0 where it lends nothing to a graph
    // (lending off, the banks released, host lookups in its largest group, several devices without LLAMA_MOE_CACHE_LEND_MULTI)
    int32_t graph_lend_capacity(ggml_backend_t backend, int32_t * n_slots_out, size_t * expert_bytes_out) const {
        if (n_slots_out) { *n_slots_out = 0; }
        if (expert_bytes_out) { *expert_bytes_out = 0; }
        const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.backend == backend; });
        if (it_dev == devices.end() || !it_dev->buf_bank || it_dev->released || lend_mode == 0) {
            return 0;
        }
        if (!lend_multi && std::count_if(devices.begin(), devices.end(), [](const device & d) { return d.host_bytes > 0 && d.buf_bank; }) > 1) {
            return 0;
        }
        const int id = (int) (it_dev - devices.begin());
        const group * gl = nullptr;
        bool host_read = false;
        for (const group & g : groups) {
            if (g.id == id && g.n_slots > 0 && (gl == nullptr || (size_t) g.n_slots*g.expert_bytes > (size_t) gl->n_slots*gl->expert_bytes)) {
                gl = &g;
            }
            host_read = host_read || (g.id == id && g.host_read);
        }
        if (gl == nullptr || host_read) {
            return 0;
        }
        if (n_slots_out) { *n_slots_out = gl->n_slots; }
        if (expert_bytes_out) { *expert_bytes_out = gl->expert_bytes; }
        const int32_t n_keep = std::min<int32_t>(gl->n_slots / 2, gl->lru.n_expert + 1);
        return std::max<int32_t>(gl->n_slots / 2, gl->n_slots - n_keep);
    }

    bool lends_on(ggml_backend_t backend) const {
        const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.backend == backend; });
        if (it_dev == devices.end() || !it_dev->buf || lend_mode == 0) {
            return false;
        }
        return it_dev->lend_auto == 1;
    }

    // LLAMA_MOE_CACHE_LEND unset: the cache lends `need` bytes only if they are at least lend_min_frac of the cache of the device (a
    // small share frees little memory for the cache and costs the prompts the slots), or if the staging arena and the graph allocation
    // (`graph` more bytes, with the copies a seeding prompt holds there) do not fit in the free memory; decided at the first graph that
    // is allocated; the memory kept free for a later model or context (set_mem_reserve) is held against the copies and the graph
    // allocation without the copies of a seeding prompt (`seed` bytes of `graph`): a seed that does not fit is dropped by itself (see
    // llama_context::sched_reserve_seed), lending for it costs the prompts the slots
    // k15-fn: lending frees the staging arena (`need` bytes), and the fit gives those bytes to the cache (see llama_context::sched_reserve);
    // a prompt that seeds from its own ids then holds the copies of a split (`seed`) in the graph allocation, if the memory left holds
    // them. The cache lends where the arena is at least lend_save_min bytes larger than that split, whatever its share of the cache. The
    // fit and the load must decide the same, so the rule does not depend on the cache size or the free memory. G4 Qwen3.8-Flash-Next
    // UD-IQ4_XS + MTP at 32 GB: arena 1275 MiB (6.8% of the cache, lending was off), split 850 MiB -> on; Qwen3.6 UD-Q4_K_XL at 12 / 16 GB:
    // 354 / 210 MiB -> off (it lends there anyway because the copies do not fit, see below, and its fit is unchanged)
    // Discrete GPUs only: on an integrated GPU (gfx1151, unified memory) the rule is not measured, lending stays as it was there
    // (k19-igpu: kept so; the fit gives the lent arena's bytes to the cache from the device's free memory, which there is the host's
    // available memory that the RAM tier is sized from too)
    bool lend_saves(int32_t id, size_t need, size_t seed) const {
        if (!lend_save || need <= seed || need - seed < lend_save_min) {
            return false;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[id].backend);
        return dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
    }

    bool lend_decide(int32_t id, size_t need, size_t graph, size_t seed, bool dry) {
        device & d = devices[id];
        if (d.lend_auto >= 0) {
            return d.lend_auto != 0;
        }
        size_t cache = 0;
        for (const group & g : groups) {
            if (g.id == id) {
                cache += (size_t) g.n_slots*g.expert_bytes;
            }
        }
        const double frac = cache > 0 ? (double) need/cache : 1.0;
        const bool saves = frac < lend_min_frac && lend_saves(id, need, seed);
        bool on = frac >= lend_min_frac || saves;
        if (dry) {
            d.dry_need = std::max(d.dry_need, need);
            // a measurement (e.g. for --fit) counts the memory with lending: when the device is short of memory the cache lends anyway
            return true;
        }
        // headroom for the memory the backends allocate while computing (e.g. the CUDA pool); k16-fit: at least the memory --fit left
        // free (LLAMA_FIT_TARGET_MIB, set by common), which an unlent staging arena must not take
        const size_t margin = std::max<size_t>(512ull*1024*1024, getenv("LLAMA_FIT_TARGET_MIB") ? (size_t) atoll(getenv("LLAMA_FIT_TARGET_MIB"))*1024*1024 : 0);
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
        size_t free_rs = 0; // free memory less the memory kept free for a later model or context
        if (!on && dev != nullptr) {
            ggml_backend_dev_memory(dev, &free, &total);
            // the draft of speculative decoding is created after this context: the memory --fit gave it is not free for the copies
            // (LLAMA_DRAFT_MEM_RESERVE=0: the rule above alone)
            free_rs = free - std::min(free, d.mem_reserve);
        }
        const bool fit = on || dev == nullptr ||
            (free >= need + graph + margin && free_rs >= need + (graph - std::min(graph, seed)) + margin);
        if (!on && dev != nullptr && d.mem_reserve > 0) {
            free = free_rs; // the message reports what the copies could use
            if (fit) {
                LLAMA_LOG_INFO("%s: %s: %.2f MiB kept free for a later model or context leave %.2f MiB for the staging copies (%.2f MiB) "
                    "and %.2f MiB of graph allocation without the copies of a seeding prompt (%.2f MiB, dropped if they do not fit)\n",
                    __func__, ggml_backend_name(d.backend), d.mem_reserve/1024.0/1024.0, free_rs/1024.0/1024.0, need/1024.0/1024.0,
                    (graph - std::min(graph, seed))/1024.0/1024.0, seed/1024.0/1024.0);
            }
        }
        d.lend_auto = on || !fit ? 1 : 0;
        d.lend_saving = saves;
        if (saves) {
            LLAMA_LOG_INFO("%s: %s: MoE cache lending on: the staging copies of large batches (%.2f MiB, %.1f%% of the cache) free %.2f MiB beside the "
                "copies of a seeding prompt (%.2f MiB), at least %.2f MiB (LLAMA_MOE_CACHE_LEND_SAVE=1, LLAMA_MOE_CACHE_LEND_SAVE_MIB)\n",
                __func__, ggml_backend_name(d.backend), need/1024.0/1024.0, 100.0*frac, (need - seed)/1024.0/1024.0, seed/1024.0/1024.0,
                lend_save_min/1024.0/1024.0);
        } else if (on) {
            LLAMA_LOG_INFO("%s: %s: MoE cache lending on: the staging copies of large batches (%.2f MiB) are %.1f%% of the cache (%.2f MiB), at least %.1f%% (LLAMA_MOE_CACHE_LEND_MIN_FRAC)\n",
                __func__, ggml_backend_name(d.backend), need/1024.0/1024.0, 100.0*frac, cache/1024.0/1024.0, 100.0*lend_min_frac);
        } else if (!fit) {
            LLAMA_LOG_INFO("%s: %s: MoE cache lending on: the staging copies of large batches (%.2f MiB) are %.1f%% of the cache (%.2f MiB), below %.1f%%, "
                "but %.2f MiB free do not hold them next to %.2f MiB of graph allocation (with the copies of a seeding prompt)\n",
                __func__, ggml_backend_name(d.backend), need/1024.0/1024.0, 100.0*frac, cache/1024.0/1024.0, 100.0*lend_min_frac, free/1024.0/1024.0, graph/1024.0/1024.0);
        } else {
            LLAMA_LOG_INFO("%s: %s: MoE cache lending off: the staging copies of large batches (%.2f MiB) are %.1f%% of the cache (%.2f MiB), below %.1f%% "
                "(LLAMA_MOE_CACHE_LEND_MIN_FRAC, LLAMA_MOE_CACHE_LEND=1 to lend)\n",
                __func__, ggml_backend_name(d.backend), need/1024.0/1024.0, 100.0*frac, cache/1024.0/1024.0, 100.0*lend_min_frac);
        }
        return d.lend_auto != 0;
    }

    // k17-small (lazy KV step 2): the bank buffer of device d. With shrink_on it is a lazy buffer of the backend (the address range of
    // every bank is reserved whole, as the plain buffer would place them) and backed for the slots the groups have now (all of them at
    // load; after a shrink, restore backs only the slots left, the retired tails stay unbacked). A plain buffer otherwise, or if the
    // backend has no lazy buffers (then shrink gives nothing back)
    ggml_backend_buffer_t alloc_banks(device & d) {
        d.bank_lazy = false;
        if (shrink_on || ext_on) { // (k19-t4small: the decode extension backs only the slots in use)
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(d.buft);
            ggml_backend_reg_t reg = dev != nullptr && !ggml_backend_buft_is_host(d.buft) ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg != nullptr && lazy_alloc_fn == nullptr) {
                lazy_alloc_fn    = (ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_alloc_lazy");
                lazy_commit_fn   = (bool (*)(ggml_backend_buffer_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_commit");
                lazy_decommit_fn = (size_t (*)(ggml_backend_buffer_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_decommit");
            }
            const size_t sz = ggml_backend_alloc_ctx_tensors_from_buft_size(d.ctx_bank.get(), d.buft);
            ggml_backend_buffer_t buf = reg != nullptr && lazy_alloc_fn != nullptr && lazy_commit_fn != nullptr && lazy_decommit_fn != nullptr && sz > 0 ?
                lazy_alloc_fn(d.buft, sz) : nullptr;
            if (buf != nullptr) {
                // as ggml_backend_alloc_ctx_tensors: the banks first, then their views
                bool ok = true;
                ggml_tallocr ta = ggml_tallocr_new(buf);
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx_bank.get()); ok && t != nullptr; t = ggml_get_next_tensor(d.ctx_bank.get(), t)) {
                    if (t->data == nullptr && t->view_src == nullptr) {
                        ok = ggml_tallocr_alloc(&ta, t) == GGML_STATUS_SUCCESS;
                    }
                }
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx_bank.get()); ok && t != nullptr; t = ggml_get_next_tensor(d.ctx_bank.get(), t)) {
                    if (t->data == nullptr && t->view_src != nullptr) {
                        ok = ggml_backend_view_init(t) == GGML_STATUS_SUCCESS;
                    }
                }
                // back the slots of every group (and the spare slot after them)
                const char * base = (const char *) ggml_backend_buffer_get_base(buf);
                for (const group & g : groups) {
                    if (!ok || &devices[g.id] != &d || g.n_slots == 0) {
                        continue;
                    }
                    for (const ggml_tensor * bank : g.banks) {
                        // (k19-t4small: the decode extension backs a bank's rows as one allocation, and up to where ext_park keeps them)
                        ok = ok && (ext_on && lazy_commit_run_fn != nullptr ?
                            lazy_commit_run_fn(buf, (const char *) bank->data - base,
                                std::min(ggml_nbytes(bank), (size_t) (g.n_slots + 1)*bank->nb[2] + (size_t) 64*1024)) :
                            lazy_commit_fn(buf, (const char *) bank->data - base, (size_t) (g.n_slots + 1)*bank->nb[2]));
                    }
                }
                if (ok) {
                    d.bank_lazy = true;
                    return buf;
                }
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx_bank.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx_bank.get(), t)) {
                    t->data   = nullptr;
                    t->buffer = nullptr;
                }
                ggml_backend_buffer_free(buf);
                return nullptr; // no memory for the banks: as the plain allocation failing
            }
        }
        return ggml_backend_alloc_ctx_tensors_from_buft(d.ctx_bank.get(), d.buft);
    }

    // k17-small: the bank tensors the lookup ops of g read and write: the banks, or after a shrink their views over the slots left (they
    // follow the banks to the buffer they are in now, which restore replaces)
    std::vector<ggml_tensor *> op_banks(const group & g) const {
        if (g.banks_op.empty()) {
            return g.banks;
        }
        for (size_t i = 0; i < g.banks.size(); ++i) {
            g.banks_op[i]->data   = g.banks[i]->data;
            g.banks_op[i]->buffer = g.banks[i]->buffer;
        }
        return g.banks_op;
    }

    // k17-small: the index of the device of the cache on dev (-1: none)
    int32_t dev_id(ggml_backend_dev_t dev) const {
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].backend != nullptr && ggml_backend_get_device(devices[i].backend) == dev) {
                return (int32_t) i;
            }
        }
        return -1;
    }

    // k17-small: the bytes of the slots the groups of device id hold now
    size_t slot_bytes(int32_t id) const {
        size_t n = 0;
        for (const group & g : groups) {
            if (g.id == id) {
                n += (size_t) g.n_slots*g.expert_bytes;
            }
        }
        return n;
    }

    // k17-small: the bytes of the slots the groups of device id held at load
    size_t slot_bytes_load(int32_t id) const {
        size_t n = 0;
        for (const group & g : groups) {
            if (g.id == id) {
                n += (size_t) g.n_slots_load*g.expert_bytes;
            }
        }
        return n;
    }

    // k17-small: shrink keeps at least floor bytes of slots on device id, every group in proportion to its slots at load (at least the
    // experts of one token)
    void set_shrink_floor(int32_t id, size_t floor) {
        devices[id].shrink_floor = floor;
        size_t load = 0;
        for (const group & g : groups) {
            if (g.id == id) {
                load += (size_t) g.n_slots_load*g.expert_bytes;
            }
        }
        for (group & g : groups) {
            if (g.id != id || g.n_slots_load == 0) {
                continue;
            }
            const double  share = load > 0 ? std::min(1.0, (double) floor/(double) load) : 1.0;
            const int32_t n_min = (int32_t) std::ceil(share*g.n_slots_load);
            g.n_slots_min = std::min(g.n_slots_load, std::max<int32_t>(n_min, std::min<int32_t>(n_expert_used, g.n_slots_load)));
        }
    }

    // k17-small: retire the slots [n_new, n_slots) of g for good and give back the device memory of the bank pages wholly inside the rows
    // that are no longer used. The slot after the last one stays backed: it is the spare slot (probe uploads, CUDA MMQ reads past the
    // last expert into it). Every slot count the cache uses (the host LRU, lend's tail, the device lookup state, whose length the kernels
    // read) follows g.n_slots, so nothing can map an expert to a retired slot; the banks and their views keep their load size, so the
    // graphs keep their shapes. Returns the bytes given back
    size_t retire(group & g, int32_t n_new) {
        device & d = devices[g.id];
        ggml_backend_t backend = d.backend;
        const int32_t n_old = g.n_slots;
        // k19-ptres: never into the core, and a whole layer of slots stays above it (as core_place leaves)
        if (g.lru.n_core > 0) {
            n_new = std::max(n_new, std::min(n_old, g.lru.n_core + g.lru.n_expert));
        }
        if (n_new >= n_old) {
            return 0;
        }
        if (host && (size_t) (&g - groups.data()) < host_group.size() && host_group[&g - groups.data()]) {
            // the RAM tier keeps a copy of the experts the retired slots held, as for slots that are lent (see release)
            evicted_tmp.clear();
            for (int32_t s = n_new; s < n_old; ++s) {
                if (g.lru.key_of[s] >= 0) {
                    evicted_tmp.emplace_back(s, g.lru.key_of[s]);
                }
            }
            shadow_slots(backend, g, evicted_tmp);
        }
        g.lru.retire(n_new);
        g.n_slots = n_new;
        if (!g.ua_dec_key.empty()) {
            g.ua_dec_key.resize(n_new);
        }
        if (g.state != nullptr) {
            // the device lookup reads the slot count from the length of its state (n_hdr + 3*n_slots)
            g.state->ne[0] = n_hdr + 3*(int64_t) n_new;
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                g.state->nb[i] = g.state->nb[i - 1]*g.state->ne[i - 1];
            }
        }
        // the lookup ops require banks of n_slots + 1 slots: views of the banks over the slots left
        if (g.banks_op.empty()) {
            if (!ctx_banks_op) {
                size_t n_banks = 0;
                for (const group & gi : groups) {
                    n_banks += gi.banks.size();
                }
                ggml_init_params params = {
                    /*.mem_size   =*/ (n_banks + 1)*ggml_tensor_overhead(),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_banks_op.reset(ggml_init(params));
            }
            for (ggml_tensor * bank : g.banks) {
                ggml_tensor * v = ggml_view_3d(ctx_banks_op.get(), bank, bank->ne[0], bank->ne[1], n_new + 1, bank->nb[1], bank->nb[2], 0);
                ggml_format_name(v, "%s.op", bank->name);
                g.banks_op.push_back(v);
            }
        }
        for (ggml_tensor * v : g.banks_op) {
            v->ne[2] = n_new + 1;
            v->nb[3] = v->nb[2]*v->ne[2];
        }
        // the new spare slot held an expert: cleared as unlend clears the slots that come back
        for (ggml_tensor * bank : g.banks) {
            ggml_backend_tensor_memset(bank, 0, (size_t) n_new*bank->nb[2], bank->nb[2]);
        }
        if (g.dev_lookup && g.map != nullptr && g.state != nullptr) {
            upload(g);
        }
        size_t freed = 0;
        if (d.bank_lazy && d.buf_bank) {
            const char * base = (const char *) ggml_backend_buffer_get_base(d.buf_bank.get());
            for (const ggml_tensor * bank : g.banks) {
                // (lend places a padded copy at most an alignment past the spare slot: 64 KiB more stay backed)
                const size_t off  = (size_t) ((const char *) bank->data - base);
                const size_t keep = std::min(ggml_nbytes(bank), (size_t) (n_new + 1)*bank->nb[2] + (size_t) 64*1024);
                freed += lazy_decommit_fn(d.buf_bank.get(), off + keep, ggml_nbytes(bank) - keep);
            }
        }
        n_retired += n_old - n_new;
        graphs_stale = true;
        return freed;
    }

    // k19-t4small: the lookup state and the op views of g follow a slot count n (as retire sets them)
    void set_slot_count(group & g, int32_t n) {
        if (g.state != nullptr) {
            // the device lookup reads the slot count from the length of its state (n_hdr + 3*n_slots)
            g.state->ne[0] = n_hdr + 3*(int64_t) n;
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                g.state->nb[i] = g.state->nb[i - 1]*g.state->ne[i - 1];
            }
        }
        if (g.banks_op.empty()) {
            if (!ctx_banks_op) {
                size_t n_banks = 0;
                for (const group & gi : groups) {
                    n_banks += gi.banks.size();
                }
                ggml_init_params params = {
                    /*.mem_size   =*/ (n_banks + 1)*ggml_tensor_overhead(),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_banks_op.reset(ggml_init(params));
            }
            for (ggml_tensor * bank : g.banks) {
                ggml_tensor * v = ggml_view_3d(ctx_banks_op.get(), bank, bank->ne[0], bank->ne[1], n + 1, bank->nb[1], bank->nb[2], 0);
                ggml_format_name(v, "%s.op", bank->name);
                g.banks_op.push_back(v);
            }
        }
        for (ggml_tensor * v : g.banks_op) {
            v->ne[2] = n + 1;
            v->nb[3] = v->nb[2]*v->ne[2];
        }
    }

    // k19-t4small: the CUDA backend has lazy buffers (VMM) on the device of d: a probe reservation is made and freed
    bool ext_probe(device & d) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(d.buft);
        ggml_backend_reg_t reg = dev != nullptr && !ggml_backend_buft_is_host(d.buft) ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) {
            return false;
        }
        if (lazy_alloc_fn == nullptr) {
            lazy_alloc_fn    = (ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_alloc_lazy");
            lazy_commit_fn   = (bool (*)(ggml_backend_buffer_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_commit");
            lazy_decommit_fn = (size_t (*)(ggml_backend_buffer_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_decommit");
        }
        lazy_commit_run_fn = (bool (*)(ggml_backend_buffer_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_commit_run");
        if (lazy_alloc_fn == nullptr || lazy_commit_fn == nullptr || lazy_decommit_fn == nullptr || lazy_commit_run_fn == nullptr) {
            return false;
        }
        ggml_backend_buffer_t buf = lazy_alloc_fn(d.buft, (size_t) 2*1024*1024);
        if (buf == nullptr) {
            return false;
        }
        ggml_backend_buffer_free(buf);
        return true;
    }

    // k19-t4small (llama_moe_cache::ext_grow): back and add slots to the groups of device id, in proportion to their fitted slots, for
    // about bytes of device memory (each group at most to n_slots_cap). The new slots are empty, the next victims: the decode admits
    // its misses into them. Not while the banks are released or slots are lent. Returns the bytes of slots added
    size_t ext_grow(int32_t id, size_t bytes) {
        device & d = devices[id];
        if (!ext_on || bytes == 0 || !d.buf_bank || !d.bank_lazy || d.released || core_on) {
            return 0;
        }
        size_t fit = 0;
        for (const group & g : groups) {
            if (g.id == id && g.n_slots > 0 && g.n_slots_cap > g.n_slots && g.lru.n_lent == 0) {
                fit += (size_t) g.n_slots_fit*g.expert_bytes;
            }
        }
        if (fit == 0) {
            return 0;
        }
        const int64_t t0 = ggml_time_us();
        ggml_backend_t backend = d.backend;
        ua_release(id);
        ggml_backend_synchronize(backend);
        const double frac = (double) bytes/(double) fit;
        const char * base = (const char *) ggml_backend_buffer_get_base(d.buf_bank.get());
        size_t  grown   = 0;
        int32_t n_added = 0;
        bool    failed  = false;
        for (group & g : groups) {
            if (g.id != id || g.n_slots == 0 || g.n_slots_cap <= g.n_slots || g.lru.n_lent > 0 || failed) {
                continue;
            }
            const int32_t n_old = g.n_slots;
            // the rows backed stay a granularity chunk (2 MiB) short of the bank's end, so the allocation that backs them never takes a
            // chunk the next bank shares and ext_park gives it back whole
            size_t nb_min = SIZE_MAX;
            for (const ggml_tensor * bank : g.banks) {
                nb_min = std::min(nb_min, (size_t) bank->nb[2]);
            }
            const int32_t slack = (int32_t) ((((size_t) 2*1024*1024 + (size_t) 64*1024) + nb_min - 1)/nb_min) + 1;
            const int32_t n_new = std::min<int32_t>(g.n_slots_cap - slack, g.n_slots_fit + (int32_t) std::floor(frac*g.n_slots_fit));
            if (n_new <= n_old) {
                continue;
            }
            seed_end(g);
            if (g.dev_lookup && g.dev_newer) {
                download(g);
            }
            // back the rows of the new slots and the new spare slot (and the 64 KiB past it a lent copy may pad into, as retire keeps)
            bool ok = true;
            for (const ggml_tensor * bank : g.banks) {
                const size_t off  = (size_t) ((const char *) bank->data - base);
                const size_t want = std::min(ggml_nbytes(bank), (size_t) (n_new + 1)*bank->nb[2] + (size_t) 64*1024);
                ok = ok && lazy_commit_run_fn(d.buf_bank.get(), off, want);
            }
            if (!ok) {
                // no device memory for them: what was backed stays unused until the next park gives it back
                failed = true;
                ext_fail++;
                break;
            }
            // newly backed rows are cleared by the commit; the old spare slot and what follows it may hold bytes of a lent copy
            for (ggml_tensor * bank : g.banks) {
                ggml_backend_tensor_memset(bank, 0, (size_t) n_old*bank->nb[2], (size_t) (n_new + 1 - n_old)*bank->nb[2]);
            }
            g.lru.unretire(n_new);
            g.n_slots = n_new;
            if (!g.ua_dec_key.empty()) {
                g.ua_dec_key.resize(n_new, -1);
            }
            set_slot_count(g, n_new);
            if (g.dev_lookup && g.map != nullptr && g.state != nullptr) {
                upload(g);
            }
            grown   += (size_t) (n_new - n_old)*g.expert_bytes;
            n_added += n_new - n_old;
        }
        if (n_added > 0) {
            graphs_stale = true;
            n_ext_grows++;
        }
        ext_max_bytes = std::max(ext_max_bytes, ext_bytes(id));
        ext_us += ggml_time_us() - t0;
        if (n_ext_grows <= 3 && n_added > 0) {
            LLAMA_LOG_INFO("%s: %s: MoE cache k19-t4small decode extension: +%d slots (%.2f MiB) for %.2f MiB of free device memory%s, "
                "%.1f ms (LLAMA_MOE_DECODE_EXT=0 off)\n", __func__, ggml_backend_name(backend), n_added, grown/1048576.0, bytes/1048576.0,
                failed ? " (stopped: the device had no memory for more)" : "", (ggml_time_us() - t0)/1000.0);
        }
        return grown;
    }

    // k19-t4small (llama_moe_cache::ext_park): the groups of device id go back to their fitted slots, the rows above them are given
    // back to the device (retire). Returns the bytes given back
    size_t ext_park(int32_t id) {
        device & d = devices[id];
        if (!ext_on || !d.buf_bank || !d.bank_lazy || d.released) {
            return 0;
        }
        if (std::none_of(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && g.n_slots > g.n_slots_fit; })) {
            return 0;
        }
        const int64_t t0 = ggml_time_us();
        ggml_backend_t backend = d.backend;
        ua_release(id);
        ggml_backend_synchronize(backend);
        size_t freed = 0;
        for (group & g : groups) {
            if (g.id != id || g.n_slots <= g.n_slots_fit) {
                continue;
            }
            seed_end(g);
            if (g.dev_lookup && g.dev_newer) {
                download(g);
            }
            if (g.lru.n_lent > 0) {
                unlend(g);
            }
            freed += retire(g, g.n_slots_fit);
        }
        n_ext_parks++;
        ext_us += ggml_time_us() - t0;
        return freed;
    }

    // k19-t4small: the bytes of slots above the fitted ones on device id
    size_t ext_bytes(int32_t id) const {
        size_t n = 0;
        for (const group & g : groups) {
            if (g.id == id && g.n_slots > g.n_slots_fit) {
                n += (size_t) (g.n_slots - g.n_slots_fit)*g.expert_bytes;
            }
        }
        return n;
    }

    // k17-small (lazy KV step 2, llama_moe_cache::shrink): give back at least bytes of device memory of device id by retiring the tail
    // slots of its groups (spread over the groups in proportion to what each holds above its floor) down to the shrink floor. The work of
    // the slots stops first: the uploads ahead are dropped, a pending seed ends, the device state is read back and lent slots come back
    // (unlend; the next large graph lends again). Not while the banks are released (they hold no memory then). Returns the bytes freed
    size_t shrink(int32_t id, size_t bytes) {
        device & d = devices[id];
        if (!shrink_on || bytes == 0 || !d.buf_bank || !d.bank_lazy || d.released || d.shrink_floor == SIZE_MAX) {
            return 0;
        }
        // at the floor: nothing to give, and nothing is stopped
        if (std::none_of(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && g.n_slots > g.n_slots_min; })) {
            return 0;
        }
        ggml_backend_t backend = d.backend;
        ua_release(id);
        ggml_backend_synchronize(backend);
        // a graph built before reuses its allocation without a lend callback: after the lent slots come back it must be built again
        graphs_stale = true;
        for (group & g : groups) {
            if (g.id != id || g.n_slots == 0) {
                continue;
            }
            seed_end(g);
            if (g.dev_lookup && g.dev_newer) {
                download(g);
            }
            if (g.lru.n_lent > 0) {
                unlend(g);
            }
        }
        const size_t before = slot_bytes(id);
        size_t  freed     = 0;
        int32_t n_retired0 = (int32_t) n_retired;
        while (freed < bytes) {
            int32_t n_avail = 0;
            for (const group & g : groups) {
                n_avail += g.id == id && g.n_slots > g.n_slots_min ? 1 : 0;
            }
            if (n_avail == 0) {
                break;
            }
            for (group & g : groups) {
                if (g.id != id || g.n_slots <= g.n_slots_min || freed >= bytes) {
                    continue;
                }
                const size_t  want = (bytes - freed + n_avail - 1)/n_avail;
                const int32_t k    = (int32_t) std::min<int64_t>(g.n_slots - g.n_slots_min, std::max<int64_t>(1, (int64_t) ((want + g.expert_bytes - 1)/g.expert_bytes)));
                freed += retire(g, g.n_slots - k);
            }
        }
        if (freed > 0) {
            n_shrinks++;
            shrunk_bytes += freed;
            d.buf_size -= std::min(d.buf_size, freed);
        }
        LLAMA_LOG_INFO("%s: %s: MoE cache shrink %" PRId64 " for the KV cache: asked %.2f MiB, gave back %.2f MiB (%d slots retired), slots %.2f -> %.2f MiB "
            "(floor %.2f MiB); total %.2f MiB, %" PRId64 " slots (LLAMA_MOE_CACHE_SHRINK=0 to disable)\n", __func__, ggml_backend_name(backend), n_shrinks,
            bytes/1048576.0, freed/1048576.0, (int32_t) n_retired - n_retired0, before/1048576.0, slot_bytes(id)/1048576.0, d.shrink_floor/1048576.0,
            shrunk_bytes/1048576.0, n_retired);
        return freed;
    }

    // the lent slots come back empty: they are cleared once the work that used them is done, because kernels may read a little past
    // the end of the expert in the slot before (e.g. CUDA MMQ in the padding of the last row) and the copies left arbitrary bytes there
    // the experts that a seed could not admit while slots were lent are then uploaded from host memory (LLAMA_MOE_CACHE_REFILL=0 to disable)
    // k13-small (llama_moe_cache::release): the banks of device id are freed, every slot is lent to nothing first (the lookups and the
    // copies then find no expert cached, the lend callback lends nothing, see lend) and the experts they held wait in refill; restore
    // allocates the banks again and refills the slots as after a lend (unlend)
    bool release(int32_t id) {
        device & d = devices[id];
        if (d.released || !d.buf_bank || !d.ctx_bank || single_no_alloc) {
            return false;
        }
        // k19-ptres: released banks would empty the core, whose host copies are gone: the banks stay
        if (core_on) {
            if (core_refused++ == 0) {
                LLAMA_LOG_WARN("%s: MoE cache: k19-ptres core: the banks are not released (the core has no host copies to refill from)\n", __func__);
            }
            return false;
        }
        ggml_backend_t backend = d.backend;
        ua_release(id);
        ggml_backend_synchronize(backend);
        size_t n_held = 0;
        for (group & g : groups) {
            if (g.id != id || g.n_slots == 0) {
                continue;
            }
            seed_end(g);
            if (g.dev_lookup && g.dev_newer) {
                download(g);
            }
            // set_lent keeps one slot: it is emptied below
            const int32_t n = g.n_slots - 1;
            if (host && (size_t) (&g - groups.data()) < host_group.size() && host_group[&g - groups.data()]) {
                evicted_tmp.clear();
                for (int32_t s = 0; s < g.n_slots - g.lru.n_lent; ++s) {
                    if (g.lru.key_of[s] >= 0) {
                        evicted_tmp.emplace_back(s, g.lru.key_of[s]);
                    }
                }
                shadow_slots(backend, g, evicted_tmp);
            }
            for (int32_t s = 0; s < g.n_slots - g.lru.n_lent; ++s) {
                n_held += g.lru.key_of[s] >= 0 ? 1 : 0;
            }
            if (g.lru.n_lent < n) {
                keep_evicted(g, n);
                g.lru.set_lent(n);
            }
            const int32_t key = g.lru.key_of[0];
            if (key >= 0) {
                refill.resize(layers.size());
                std::vector<int32_t> & ids_il = refill[key / g.lru.n_expert];
                if (std::find(ids_il.begin(), ids_il.end(), key % g.lru.n_expert) == ids_il.end()) {
                    ids_il.push_back(key % g.lru.n_expert);
                }
                g.lru.evict(0);
            }
        }
        for (ggml_tensor * t = ggml_get_first_tensor(d.ctx_bank.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx_bank.get(), t)) {
            t->data   = nullptr;
            t->buffer = nullptr;
        }
        const size_t bytes = ggml_backend_buffer_get_size(d.buf_bank.get());
        d.buf_bank.reset();
        d.released = true;
        n_releases++;
        if (n_releases <= 2) {
            LLAMA_LOG_INFO("%s: %s: MoE cache banks released (%.2f MiB, %zu experts held) for the graphs of a prompt, refilled after it "
                "(LLAMA_MOE_LM_RELEASE=0 to disable)\n", __func__, ggml_backend_name(backend), bytes/1048576.0, n_held);
        }
        return true;
    }

    bool restore(int32_t id) {
        device & d = devices[id];
        if (!d.released) {
            return false;
        }
        // k19-fixG: a graph run while released may have placed a bank tensor it still referenced in its compute buffer; the allocation
        // below skips tensors with data, so every binding is cleared first
        for (ggml_tensor * t = ggml_get_first_tensor(d.ctx_bank.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx_bank.get(), t)) {
            t->data   = nullptr;
            t->buffer = nullptr;
        }
        d.buf_bank.reset(alloc_banks(d));
        if (!d.buf_bank && restore_soft) {
            // k19-place8 (see set_restore_soft): the banks stay released, the caller runs as during a prompt and tries again later
            if (n_restore_fail++ < 4) {
                LLAMA_LOG_WARN("%s: MoE cache: the released banks (%.2f MiB) could not be allocated again: they stay released, tried again "
                    "at the next small batch\n", __func__, d.buf_size/1048576.0);
            }
            return false;
        }
        if (!d.buf_bank) {
            GGML_ABORT("MoE cache: the released banks (%.2f MiB) could not be allocated again", d.buf_size/1048576.0);
        }
        ggml_backend_buffer_clear(d.buf_bank.get(), 0);
        ggml_backend_buffer_set_usage(d.buf_bank.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        d.released = false;
        for (group & g : groups) {
            if (g.id == id && g.n_slots > 0) {
                unlend(g);
            }
        }
        refill_routed_once = false;
        return true;
    }

    void unlend(group & g) {
        prompt_flush();
        ggml_backend_t backend = devices[g.id].backend;
        ggml_backend_synchronize(backend);
        seed_end(g);
        if (verify_enabled) {
            verify(g, "while lent");
        }
        const int32_t n = g.lru.n_lent;
        for (ggml_tensor * bank : g.banks) {
            ggml_backend_tensor_memset(bank, 0, (size_t) (g.n_slots - n)*bank->nb[2], (size_t) (n + 1)*bank->nb[2]);
        }
        g.lru.set_lent(0);

        // the prompt that just ended did not seed the cache: the slots that came back get the experts its last large ubatch used most (over every
        // layer of the group), then the experts they held before; the same number of uploads as without the routing, it only picks them
        bool routed = false;
        if ((refill_routed || refill_routed_once) && refill_enabled && !g.dev_newer && !refill_counts.empty()) {
            struct cand { int32_t count; int32_t il; int32_t e; };
            std::vector<cand> cands;
            for (int32_t il : g.layers) {
                if ((size_t) il >= refill_counts.size() || refill_counts[il].empty()) {
                    continue;
                }
                const std::vector<int32_t> & c = refill_counts[il];
                const int32_t * slots = g.lru.slot_map[il];
                for (int32_t e = 0; e < (int32_t) c.size() && e < g.lru.n_expert; ++e) {
                    if (c[e] > 0 && slots[e] < 0) {
                        cands.push_back({ c[e], il, e });
                    }
                }
            }
            if (!cands.empty()) {
                // ties go to the lower layer and expert, so the pick does not depend on the sort
                std::sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) {
                    return a.count != b.count ? a.count > b.count : a.il != b.il ? a.il < b.il : a.e < b.e;
                });
                if (cands.size() > (size_t) n) {
                    cands.resize(n);
                }
                std::vector<std::vector<int32_t>> pick(layers.size());
                for (const cand & c : cands) {
                    pick[c.il].push_back(c.e);
                }
                // the experts held before fill what the routing leaves, so the lists never exceed the slots that came back
                size_t rest = (size_t) n - cands.size();
                refill.resize(layers.size());
                for (int32_t il : g.layers) {
                    const int32_t * slots = g.lru.slot_map[il];
                    for (int32_t e : refill[il]) {
                        if (rest > 0 && slots[e] < 0 && std::find(pick[il].begin(), pick[il].end(), e) == pick[il].end()) {
                            pick[il].push_back(e);
                            rest--;
                        }
                    }
                    refill[il] = std::move(pick[il]);
                }
                routed = true;
            }
        }
        for (int32_t il : g.layers) {
            if ((size_t) il < refill_counts.size()) {
                refill_counts[il].clear();
            }
        }

        // k15-fn LLAMA_MOE_CACHE_LEND_SAVE_REFILL=0 (opt-in): slots lent for the memory lending frees (lend_saves) come back empty and
        // misses fill them, instead of uploading every expert they held before the prompt (G4 Qwen3.8-Flash-Next UD-IQ4_XS + MTP at
        // 32 GB: 994 slots, 2208 MiB after every 2k prompt). k15gn_g4: the 2k prompt loss of lending went from -5.3% to -2.0%, but the
        // next short prompt reads the experts back: chat TTFT 320 -> 342 ms (refilled: 318). Residency only, the outputs do not change
        if (devices[g.id].lend_saving && !lend_save_refill) {
            for (int32_t il : g.layers) {
                if ((size_t) il < refill.size()) {
                    refill[il].clear();
                }
            }
            return;
        }
        if (refill.empty() || g.dev_newer) {
            return;
        }
        n_refill_routed += routed ? 1 : 0;
        // at most the n slots that came back are filled, so the refill never evicts an expert that stayed cached
        size_t budget = (size_t) n;
        for (int32_t il : g.layers) {
            if ((size_t) il >= refill.size() || refill[il].empty()) {
                continue;
            }
            std::vector<int32_t> & ids_il = refill[il];
            const int32_t * slots = g.lru.slot_map[il];
            ids_il.erase(std::remove_if(ids_il.begin(), ids_il.end(), [&](int32_t e) { return slots[e] >= 0; }), ids_il.end());
            if (ids_il.size() > budget) {
                ids_il.resize(budget);
            }
            budget -= ids_il.size();
            size_t n_hit = 0;
            if (refill_enabled && !ids_il.empty() && g.lru.plan(il, ids_il.data(), ids_il.size(), fills, n_hit)) {
                // streamed experts: from the RAM tier, else the reads of the refill are issued together by the weight store (never inline page faults)
                if (host_layer(il)) {
                    host->begin_call();
                }
                stats_refill.bytes += upload_fills(backend, il, fills, nullptr);
                stats_refill.misses += fills.size();
            }
            ids_il.clear();
        }
        if (verify_enabled) {
            verify(g, "after refill");
        }
    }

    // the experts held by the slots that become lent (n_lent up to n) wait in refill for the slots to come back
    // a later seed of the layer replaces them with the experts it could not admit (see seed)
    void keep_evicted(group & g, int32_t n) {
        const int32_t n_expert = g.lru.n_expert;
        for (int32_t s = g.n_slots - n; s < g.n_slots - g.lru.n_lent; ++s) {
            const int32_t key = g.lru.key_of[s];
            if (key < 0) {
                continue;
            }
            refill.resize(layers.size());
            std::vector<int32_t> & ids_il = refill[key / n_expert];
            if (std::find(ids_il.begin(), ids_il.end(), key % n_expert) == ids_il.end()) {
                ids_il.push_back(key % n_expert);
            }
        }
    }

    // best-fit placement of the copies in the last n slots of every bank of g (and the extra slot after them)
    // a copy must not overlap the copies of the splits less than `depth` ordinals before it, which are live while it is written,
    // nor the graph allocation (live during the whole graph, placed first): the first n_graph entries are its parts, at most max_parts
    // and one per region (bank tail), part_sizes = their sizes (0 = unused); one part in the smallest region that holds it, else the
    // largest regions with slack for the space the graph allocator cannot use at the end of each part
    // guard: the group is read through host lookups, which may read a little past the end of a slot (e.g. CUDA MMQ in the padding of the last
    // row): the first lent slot and the slot after the last one stay free of copies (never with a graph allocation, see lend)
    bool place(const group & g, int32_t n, const ggml_backend_sched_lend_copy * copies, int n_copies, int n_graph, int max_parts, size_t graph_need, int depth, bool guard,
            std::vector<size_t> & offsets, std::vector<size_t> & part_sizes) {
        const device & d = devices[g.id];
        const uintptr_t base  = (uintptr_t) ggml_backend_buffer_get_base(d.buf_bank.get());
        const size_t    align = ggml_backend_buft_get_alignment(d.buft);

        struct region { size_t begin; size_t end; };
        std::vector<region> regions;
        for (const ggml_tensor * bank : g.banks) {
            const size_t begin = (size_t) ((uintptr_t) bank->data - base) + (size_t) (g.n_slots - n + (guard ? 1 : 0))*bank->nb[2];
            regions.push_back({ begin, begin + (size_t) (guard ? n - 1 : n + 1)*bank->nb[2] });
        }
        auto align_up = [&](size_t off) {
            return (size_t) (GGML_PAD(base + off, align) - base);
        };

        struct live { size_t begin; size_t end; int ordinal; };
        std::vector<live> lv;
        std::vector<live> in_region;

        // the graph allocation
        for (int i = 0; i < n_graph; ++i) {
            part_sizes[i] = 0;
            offsets[i]    = 0;
        }
        if (n_graph > 0 && graph_need > 0) {
            std::vector<int> order(regions.size());
            for (size_t r = 0; r < regions.size(); ++r) {
                order[r] = (int) r;
            }
            auto avail = [&](int r) { return regions[r].end - std::min(regions[r].end, align_up(regions[r].begin)); };
            std::sort(order.begin(), order.end(), [&](int x, int y) { return avail(x) > avail(y); });
            // smallest region that holds it in one part
            int one = -1;
            for (int r : order) {
                if (avail(r) >= graph_need) {
                    one = r;
                }
            }
            if (one >= 0) {
                offsets[0]    = align_up(regions[one].begin);
                part_sizes[0] = GGML_PAD(graph_need, align);
            } else if (max_parts <= 1) {
                return false;
            } else {
                // the tensors of a part never move to another one: the parts together need more than the allocation in one piece
                const size_t slack = graph_need/4 + (size_t) 64*1024*1024;
                size_t left = graph_need + slack;
                int k = 0;
                for (int r : order) {
                    if (left == 0 || k >= max_parts) {
                        break;
                    }
                    const size_t sz = std::min(avail(r), GGML_PAD(left, align)) / align * align;
                    if (sz == 0) {
                        continue;
                    }
                    offsets[k]    = align_up(regions[r].begin);
                    part_sizes[k] = sz;
                    left -= std::min(left, sz);
                    k++;
                }
                if (left > 0) {
                    return false;
                }
            }
            for (int i = 0; i < n_graph; ++i) {
                if (part_sizes[i] > 0) {
                    lv.push_back({ offsets[i], offsets[i] + part_sizes[i], -1 });
                }
            }
        }

        for (int i = n_graph; i < n_copies; ++i) {
            const size_t size = copies[i].size;
            lv.erase(std::remove_if(lv.begin(), lv.end(), [&](const live & l) { return l.ordinal >= 0 && l.ordinal <= copies[i].ordinal - depth; }), lv.end());

            size_t best_gap = SIZE_MAX;
            size_t best_off = 0;
            for (const region & r : regions) {
                in_region.clear();
                for (const live & l : lv) {
                    if (l.begin < r.end && l.end > r.begin) {
                        in_region.push_back(l);
                    }
                }
                std::sort(in_region.begin(), in_region.end(), [](const live & a, const live & b) { return a.begin < b.begin; });
                size_t cur = align_up(r.begin);
                auto consider = [&](size_t gap_end) {
                    if (gap_end > cur && gap_end - cur >= size && gap_end - cur < best_gap) {
                        best_gap = gap_end - cur;
                        best_off = cur;
                    }
                };
                for (const live & l : in_region) {
                    consider(l.begin);
                    cur = std::max(cur, align_up(l.end));
                }
                consider(r.end);
            }
            if (best_gap == SIZE_MAX) {
                return false;
            }
            offsets[i] = best_off;
            lv.push_back({ best_off, best_off + size, copies[i].ordinal });
        }
        return true;
    }

    // k15-lfm2: in-place prompts, see prompt_of. A group that does not hold every expert of its layers lends prompt_rings*M slots, M the
    // most experts a layer of the group misses (the slot maps do not change during such a graph: no seed, no admission). The copy of a
    // prompt map may be prefetched on the copy queue: each staging slot of the scheduler (told apart by where the copy of the map lands)
    // gets its own ring of M lent slots, so a ring is written again only after the scheduler retired the split that read it
    static constexpr int32_t prompt_rings = 4;
    bool prompt_group_full(const group & g) const {
        return (int64_t) g.n_slots >= (int64_t) g.layers.size()*g.lru.n_expert;
    }
    bool prompt_inplace_ok(int32_t il) const {
        if (!prompt_inplace_env || !prompt_ubatch_ok || prompt_inplace_blocked || cm || hybrid || lend_mode == 0 || host_layer(il)) {
            return false;
        }
        const group & g = groups[layers[il].ig];
        if (!g.dev_lookup || g.devseed || g.n_slots <= 0) {
            return false;
        }
        if (std::count_if(devices.begin(), devices.end(), [](const device & d) { return d.host_bytes > 0 && d.buf_bank; }) > 1) {
            return false;
        }
        // the rings of a group that misses experts hold a whole layer (a seed may evict any of its experts), in at most half its slots;
        // a smaller group keeps the staging copies (it could not admit a layer while lending the rings)
        return prompt_group_full(g) || g.n_slots/2 >= prompt_rings*g.lru.n_expert;
    }
    bool prompt_group_inplace(const group & g) const {
        return std::any_of(g.layers.begin(), g.layers.end(), [&](int32_t il) { return prompt_inplace_ok(il); });
    }
    bool prompt_ubatch_ok = false;  // set per ubatch by the context: the ubatch does not seed the cache
    bool prompt_lent = false;       // slots lent by in-place prompts (given back by set_graph)
    std::vector<int32_t> prompt_m;  // per group: the ring size of this graph (0: no ring)
    std::vector<std::pair<const void *, int32_t>> prompt_ring_of; // where a copy of a prompt map lands -> its ring
    bool prompt_bad = false;        // this graph's rings did not fit (see prompt_rings_set): prompt maps fall back to admitting
    // a ubatch that seeds from the routing of an earlier one (the context's seed_prev): the seed of every layer is planned before any of its
    // prompt maps (prompt_seed_plan), so the maps already read the post-seed slots; a seeded expert is read from its ring, then copied
    // from there into its slot once the split that read it retired (when the ring is written again, or at the end of the graph)
    std::vector<std::vector<moe_cache_lru::fill>> prompt_seeded; // per layer: expert -> its new slot, data not there yet
    struct ring_copy { int32_t ig; int32_t from; int32_t to; };
    std::vector<std::vector<ring_copy>> ring_pending; // per ring: the seed copies to make before it is written again
    ggml_backend_t ring_backend = nullptr;           // the backend that last wrote the rings

    int32_t prompt_misses_max(const group & g) const {
        int32_t m = 0;
        for (int32_t il : g.layers) {
            const int32_t * sl = g.lru.slot_map[il];
            int32_t c = 0;
            for (int32_t e = 0; e < g.lru.n_expert; ++e) {
                c += sl[e] < 0;
            }
            if ((size_t) il < prompt_seeded.size()) {
                c += (int32_t) prompt_seeded[il].size();
            }
            m = std::max(m, c);
        }
        return m;
    }

    // the seed of each layer of the groups that miss experts (as seed() plans it: each layer an equal share of the slots that are not lent,
    // its most used experts), from the routing the context stored (get_seed_counts); replanned after a change of the lent slots
    bool prompt_seed_counts = false; // set per ubatch by the context: seed from the stored routing
    void prompt_seed_plan() {
        prompt_seeded.assign(layers.size(), {});
        for (group & g : groups) {
            if (g.n_slots <= 0 || prompt_group_full(g) || !g.dev_lookup || g.dev_newer) {
                continue;
            }
            for (int32_t il : g.layers) {
                const layer & l = layers[il];
                if (!prompt_inplace_ok(il)) {
                    continue; // staged: seed() seeds it from the staging copy
                }
                const std::vector<int32_t> * cp = l.experts.empty() ? nullptr : get_seed_counts(l.experts[0]);
                if (cp == nullptr || (int32_t) cp->size() != g.lru.n_expert) {
                    continue;
                }
                const std::vector<int32_t> & counts = *cp;
                // (k19-ptres: the core slots take no seeds)
                const size_t n_avail = (size_t) (g.n_slots - g.lru.n_lent - g.lru.n_core);
                size_t n_seed = n_avail / g.layers.size();
                size_t n_full = (size_t) (g.n_slots - g.lru.n_core) / g.layers.size();
                if (n_full == 0) {
                    const size_t pos = std::find(g.layers.begin(), g.layers.end(), il) - g.layers.begin();
                    n_full = pos < (size_t) g.n_slots ? 1 : 0;
                    n_seed = pos < n_avail ? 1 : 0;
                }
                if (profile_loaded && profile_seed < 1.0) {
                    n_full = (size_t) (n_full*profile_seed);
                    n_seed = std::min(n_seed, n_full);
                    if (n_full == 0) {
                        continue;
                    }
                }
                seed_ids.clear();
                for (int32_t e = 0; e < g.lru.n_expert; ++e) {
                    if (counts[e] > 0) {
                        seed_ids.push_back(e);
                    }
                }
                auto by_count = [&](int32_t a, int32_t c) {
                    return counts[a] != counts[c] ? counts[a] > counts[c] : a < c;
                };
                if (n_full < seed_ids.size()) {
                    std::partial_sort(seed_ids.begin(), seed_ids.begin() + n_full, seed_ids.end(), by_count);
                    seed_ids.resize(n_full);
                } else if (n_seed < seed_ids.size()) {
                    std::sort(seed_ids.begin(), seed_ids.end(), by_count);
                }
                if (n_seed < seed_ids.size()) {
                    refill.resize(layers.size());
                    refill[il].assign(seed_ids.begin() + n_seed, seed_ids.end());
                    seed_ids.resize(n_seed);
                }
                if (seed_ids.empty()) {
                    continue;
                }
                size_t n_hit = 0;
                std::vector<moe_cache_lru::fill> fl;
                const bool ok = g.lru.plan(il, seed_ids.data(), seed_ids.size(), fl, n_hit, nullptr);
                GGML_ASSERT(ok);
                prompt_seeded[il] = std::move(fl);
                prompt_seeds++;
            }
        }
    }

    // LLAMA_MOE_CACHE_SEED_CHECK: the seeded slots are compared with the host experts after the ubatch
    void prompt_seed_rec(int32_t il, int32_t e, int32_t slot) {
        if (!seed_check) {
            return;
        }
        const group & g = groups[layers[il].ig];
        for (const ggml_tensor * w : layers[il].experts) {
            seed_committed.push_back({ w, g.banks[bindings.at(w).ip], e, slot });
        }
    }

    // the seeds whose slot was lent since the plan (their expert is no longer cached)
    void prompt_seed_filter(const group & g) {
        for (int32_t il : g.layers) {
            if ((size_t) il >= prompt_seeded.size()) {
                continue;
            }
            auto & v = prompt_seeded[il];
            v.erase(std::remove_if(v.begin(), v.end(), [&](const moe_cache_lru::fill & f) {
                return f.slot >= g.n_slots - g.lru.n_lent || g.lru.slot_map[il][f.expert] != f.slot;
            }), v.end());
        }
    }

    // the seed copies of a ring, from its slots into the slots of the seeded experts (stream order on backend)
    void prompt_ring_flush(int32_t ring, ggml_backend_t backend) {
        if (ring < 0 || ring >= (int32_t) ring_pending.size() || ring_pending[ring].empty()) {
            return;
        }
        seed_regions.clear();
        for (const ring_copy & c : ring_pending[ring]) {
            for (ggml_tensor * bank : groups[c.ig].banks) {
                const size_t nb = bank->nb[2];
                seed_regions.push_back({ bank, (size_t) c.from*nb, bank, (size_t) c.to*nb, nb });
            }
        }
        if (!ggml_backend_tensor_copy_regions_async(backend, seed_regions.data(), seed_regions.size())) {
            for (const auto & r : seed_regions) {
                ggml_tensor * bank = r.dst;
                const size_t nb = bank->nb[2];
                ggml_reset(ctx_pviews.get());
                ggml_tensor * src_v = ggml_view_3d(ctx_pviews.get(), bank, bank->ne[0], bank->ne[1], 1, bank->nb[1], nb, r.src_offset);
                ggml_tensor * dst_v = ggml_view_3d(ctx_pviews.get(), bank, bank->ne[0], bank->ne[1], 1, bank->nb[1], nb, r.dst_offset);
                ggml_backend_view_init(src_v);
                ggml_backend_view_init(dst_v);
                ggml_backend_tensor_copy_async(backend, backend, src_v, dst_v);
            }
        }
        ring_pending[ring].clear();
    }
    void prompt_ring_flush_all() {
        if (ring_backend == nullptr) {
            return;
        }
        // on the compute backend, after everything that read the rings
        ggml_backend_t b = nullptr;
        for (const device & d : devices) {
            if (d.backend != nullptr && d.buf_bank && ggml_backend_get_device(d.backend) == ggml_backend_get_device(ring_backend)) {
                b = d.backend;
            }
        }
        for (int32_t r = 0; r < (int32_t) ring_pending.size(); ++r) {
            prompt_ring_flush(r, b != nullptr ? b : ring_backend);
        }
        ring_backend = nullptr;
        prompt_seed_leftovers();
    }
    // planned seeds of layers whose prompt map was not built (the slot maps already name them): uploaded from the host weights
    void prompt_seed_leftovers() {
        for (int32_t il = 0; il < (int32_t) prompt_seeded.size(); ++il) {
            if (prompt_seeded[il].empty()) {
                continue;
            }
            group & g = groups[layers[il].ig];
            stats_seed.bytes += upload_fills(devices[g.id].backend, il, prompt_seeded[il], nullptr);
            stats_seed.misses += prompt_seeded[il].size();
            n_prompt_seed_left += prompt_seeded[il].size();
            for (const auto & f : prompt_seeded[il]) {
                prompt_seed_rec(il, f.expert, f.slot);
            }
            prompt_seeded[il].clear();
        }
    }

    // before a graph with in-place layers is allocated: the rings of every group that misses experts (synchronous)
    void prompt_rings_set() {
        for (device & d : devices) {
            if (d.backend != nullptr && d.buf_bank) {
                ggml_backend_synchronize(d.backend);
            }
        }
        prompt_ring_flush_all();
        prompt_seed_leftovers();
        prompt_ring_of.clear();
        prompt_bad = false;
        prompt_m.assign(groups.size(), 0);
        prompt_seeded.assign(layers.size(), {});
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots <= 0 || prompt_group_full(g) || !g.dev_lookup || !prompt_group_inplace(g)) {
                continue;
            }
            // lending evicts experts, which a layer then misses: grow until the rings hold the most misses of a layer
            int32_t m = prompt_misses_max(g);
            for (int it = 0; it < 4 && prompt_rings*m > g.lru.n_lent; ++it) {
                const int32_t n = std::min<int32_t>(g.lru.n_core > 0 ? std::min<int32_t>(g.n_slots/2, g.n_slots - g.lru.n_core - 2) : g.n_slots/2, prompt_rings*m);
                if (n <= g.lru.n_lent) {
                    break;
                }
                prompt_lend_now(g, n);
                m = prompt_misses_max(g);
            }
            prompt_m[ig] = m;
        }
        if (!prompt_seed_counts) {
            prompt_rings_check();
            return;
        }
        // the seeds are planned after the lending (they fill the slots that stay); a seeded expert, and one a seed evicts, is read from the
        // ring: grow the rings again if needed (the seeds of the slots lent then are dropped, unlend uploads those experts)
        prompt_seed_plan();
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots <= 0 || prompt_group_full(g) || !g.dev_lookup || !prompt_group_inplace(g)) {
                continue;
            }
            int32_t m = prompt_misses_max(g);
            for (int it = 0; it < 4 && prompt_rings*m > g.lru.n_lent; ++it) {
                const int32_t n = std::min<int32_t>(g.lru.n_core > 0 ? std::min<int32_t>(g.n_slots/2, g.n_slots - g.lru.n_core - 2) : g.n_slots/2, prompt_rings*m);
                if (n <= g.lru.n_lent) {
                    break;
                }
                prompt_lend_now(g, n);
                prompt_seed_filter(g);
                m = prompt_misses_max(g);
            }
            prompt_m[ig] = m;
        }
        prompt_rings_check();
    }

    void prompt_rings_check() {
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots <= 0 || prompt_group_full(g) || !g.dev_lookup || !prompt_group_inplace(g)) {
                continue;
            }
            const int32_t m = prompt_m[ig];
            if (prompt_rings*m > g.lru.n_lent) {
                prompt_bad = true;
                LLAMA_LOG_WARN("%s: MoE cache: in-place prompt rings of %d x %d slots do not fit in %d lent slots: this graph admits its misses\n",
                    __func__, prompt_rings, m, g.lru.n_lent);
            }
            prompt_m[ig] = std::max<int32_t>(m, 1);
        }
    }

    void prompt_lend_now(group & g, int32_t n) {
        if (g.lru.n_lent >= n) {
            return;
        }
        if (g.dev_lookup && g.dev_newer) {
            download(g);
        }
        seed_end(g);
        keep_evicted(g, n);
        g.lru.set_lent(n);
        prompt_lent = true;
        if (!prompt_log_done) {
            prompt_log_done = true;
            LLAMA_LOG_INFO("%s: %s: MoE cache: large batches read the cached experts in place, %d of %d slots (%.2f MiB) lent for the missing ones "
                "(LLAMA_MOE_PROMPT_INPLACE=0: staging copies)\n", __func__, ggml_backend_name(devices[g.id].backend), n, g.n_slots,
                (double) n*g.expert_bytes/1024.0/1024.0);
        }
    }

    // (kept for the lend() hook of a graph with in-place layers: its other copies stay in the scheduler's arena)
    void prompt_lend(ggml_backend_t, int32_t, bool) {}
    // the seed copies still pending (after the graph that read the rings, before the lent slots are written again)
    void prompt_flush() {
        if (ring_backend == nullptr) {
            prompt_seed_leftovers();
            return;
        }
        for (device & d : devices) {
            if (d.backend != nullptr && d.buf_bank) {
                ggml_backend_synchronize(d.backend);
            }
        }
        prompt_ring_flush_all();
    }

    int32_t prompt_layer(const ggml_tensor * t) const {
        const auto it = prompt_of.find(t);
        return it != prompt_of.end() ? it->second : -1;
    }

    // k15-lfm2: copy of the prompt map of layer il into dst (on the compute backend or the scheduler's copy queue): the slots of its cached
    // experts, its ring for the others, which are uploaded there
    bool prompt_copy(ggml_backend_t backend, int32_t il, ggml_tensor * dst, const std::vector<int32_t> *) {
        layer & l = layers[il];
        const int32_t ig = l.ig;
        group & g = groups[ig];
        GGML_ASSERT(ggml_backend_get_device(backend) == ggml_backend_get_device(devices[g.id].backend));
        const int32_t ne = g.lru.n_expert;
        int32_t * m = (int32_t *) l.prompt_map->data;
        fills.clear();
        size_t n_hit = 0;
        const int32_t * slots = g.lru.slot_map[il];
        const int32_t mg = ig < (int32_t) prompt_m.size() ? prompt_m[ig] : 0;
        if (!prompt_group_full(g) && mg > 0 && !prompt_bad && g.lru.n_lent >= prompt_rings*mg) {
            int32_t ring = -1;
            for (const auto & [ptr, r] : prompt_ring_of) {
                if (ptr == dst->data) {
                    ring = r;
                }
            }
            if (ring < 0) {
                if ((int32_t) prompt_ring_of.size() >= prompt_rings) {
                    // more staging slots than rings: wait for the device, every ring is free again
                    ggml_backend_synchronize(devices[g.id].backend);
                    for (int32_t r = 0; r < (int32_t) ring_pending.size(); ++r) {
                        prompt_ring_flush(r, backend);
                    }
                    prompt_ring_of.clear();
                    n_prompt_ring_waits++;
                }
                ring = (int32_t) prompt_ring_of.size();
                prompt_ring_of.emplace_back(dst->data, ring);
            }
            // the ring is written again: the scheduler retired the split that read it, its seed copies go first
            if ((int32_t) ring_pending.size() <= ring) {
                ring_pending.resize(ring + 1);
            }
            prompt_ring_flush(ring, backend);
            ring_backend = backend;
            const int32_t base = g.n_slots - g.lru.n_lent + ring*mg;
            std::vector<int32_t> seeded_slot(ne, -1);
            if ((size_t) il < prompt_seeded.size()) {
                for (const auto & f : prompt_seeded[il]) {
                    seeded_slot[f.expert] = f.slot;
                }
            }
            int32_t j = 0;
            for (int32_t e = 0; e < ne; ++e) {
                if (slots[e] >= 0 && seeded_slot[e] < 0) {
                    m[e] = slots[e];
                    n_hit++;
                } else {
                    GGML_ASSERT(j < mg && "MoE cache: an in-place prompt layer misses more experts than its ring holds");
                    m[e] = base + j;
                    fills.push_back({ e, base + j });
                    if (seeded_slot[e] >= 0) {
                        ring_pending[ring].push_back({ ig, base + j, seeded_slot[e] });
                        stats_seed.misses++;
                        stats_seed.bytes += g.expert_bytes;
                        prompt_seed_rec(il, e, seeded_slot[e]);
                    }
                    j++;
                }
            }
            if ((size_t) il < prompt_seeded.size()) {
                prompt_seeded[il].clear();
            }
            n_prompt_inplace++;
        } else {
            // a group that holds every expert: the missing experts are admitted into free slots (none is evicted). A group whose rings did
            // not fit admits them too, on the compute stream only (an admission evicts what a layer still being computed may read)
            if (!prompt_group_full(g)) {
                ggml_backend_synchronize(devices[g.id].backend);
                n_prompt_fallback++;
            }
            if ((size_t) il < prompt_seeded.size() && !prompt_seeded[il].empty()) {
                // planned seeds: the slot maps name them, their data is uploaded first
                stats_seed.bytes += upload_fills(backend, il, prompt_seeded[il], nullptr);
                stats_seed.misses += prompt_seeded[il].size();
                for (const auto & f : prompt_seeded[il]) {
                    prompt_seed_rec(il, f.expert, f.slot);
                }
                prompt_seeded[il].clear();
            }
            seed_ids.clear();
            for (int32_t e = 0; e < ne; ++e) {
                seed_ids.push_back(e);
            }
            if (!g.lru.plan(il, seed_ids.data(), seed_ids.size(), fills, n_hit, nullptr)) {
                GGML_ABORT("the MoE cache is too small for the experts of layer %d", il);
            }
            std::copy(g.lru.slot_map[il], g.lru.slot_map[il] + ne, m);
        }
        const size_t bytes = upload_fills(backend, il, fills, nullptr);
        prompt_hits   += n_hit;
        prompt_misses += fills.size();
        prompt_bytes  += bytes;
        ggml_backend_tensor_set_async(backend, dst, m, 0, (size_t) ne*sizeof(int32_t));
        return true;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, int64_t n_ubatch) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        // with device lookups, ubatches that do not use the device use the host state (see prepare), a ubatch that uses the device keeps the state there
        if (force_large || use_device(groups[layers[il].ig], n_ubatch)) {
            return nullptr;
        }
        const layer & l = layers[il];

        // k15-lfm2: a large batch reads its experts in place (see prompt_of)
        if (n_tokens > max_batch && prompt_inplace_ok(il)) {
            return l.prompt_map;
        }
        if (n_tokens > max_batch && prompt_inplace_env && prompt_ubatch_ok && !prompt_why_logged) {
            prompt_why_logged = true;
            const group & g = groups[l.ig];
            LLAMA_LOG_INFO("%s: MoE cache: layer %d of this large batch is staged (in-place prompts need: no CPU-miss rule %d, no hybrid gate %d, "
                "lending %d, no RAM tier %d, device lookups %d, no device seed %d, no layer-major prompts %d, slots %d of %d per layer)\n", __func__, il,
                !cm, !hybrid, lend_mode != 0, !host_layer(il), g.dev_lookup, !g.devseed, !prompt_inplace_blocked, g.n_slots, (int) g.lru.n_expert);
        }

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    // the graph about to be allocated: the groups whose banks it reads through host lookups (the get_rows of a slot map)
    // nullptr: a graph that is only measured or reserved, which never runs and reads nothing
    void set_graph(ggml_cgraph * gf) {
        prompt_flush();
        inplace_graph = false;
        for (int i = 0; gf != nullptr && i < ggml_graph_n_nodes(gf) && !inplace_graph; ++i) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            inplace_graph = node->op == GGML_OP_GET_ROWS && prompt_of.count(node->src[0]) > 0;
        }
        if (gf != nullptr && inplace_graph) {
            prompt_rings_set();
        }
        // the slots an in-place prompt lent come back for a graph without in-place layers (e.g. the decode after the prompt)
        if (gf != nullptr && !inplace_graph && prompt_lent) {
            prompt_lent = false;
            for (group & g : groups) {
                if (g.lru.n_lent > 0) {
                    unlend(g);
                }
            }
        }
        for (group & g : groups) {
            g.host_read = false;
            g.host_need = 0;
        }
        for (int i = 0; gf != nullptr && i < ggml_graph_n_nodes(gf); ++i) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_GET_ROWS) {
                continue;
            }
            const auto it = layer_of.find(node->src[0]);
            if (it != layer_of.end()) {
                group & g = groups[layers[it->second].ig];
                g.host_read = true;
                g.host_need = std::max<int64_t>(g.host_need, std::min<int64_t>(ggml_nelements(node->src[1]), node->src[0]->ne[1]));
            }
        }
    }

    ggml_tensor * get_resident(const ggml_tensor * t) const {
        // a trainable tensor (see llama_context::opt_init) changes after the copy was made and needs its gradient: use the original
        if (t != nullptr && (t->flags & GGML_TENSOR_FLAG_PARAM)) {
            return nullptr;
        }
        const auto it = t != nullptr ? resident.find(t) : resident.end();
        return it != resident.end() ? it->second : nullptr;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last, std::vector<ggml_backend_copy_region> * regions) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (ggml_backend_get_device(backend) != ggml_backend_get_device(devices[g.id].backend)) {
            return 0;
        }

        // k19-ptres: a core expert is always copied from its slot (its host copy is not registered and not in RAM): the core slots never
        // change, so the host map is right for them even where the device state is newer
        if (g.lru.n_core > 0) {
            const int32_t * cs = g.lru.slot_map[b.il];
            if (cs != nullptr && cs[e] >= 0 && cs[e] < g.lru.n_core) {
                int64_t n = 1;
                while (e + n <= last && cs[e + n] == cs[e] + n && cs[e + n] < g.lru.n_core) {
                    n++;
                }
                ggml_tensor * bank = g.banks[b.ip];
                GGML_ASSERT(bank->nb[2] == dst->nb[2]);
                if (regions != nullptr) {
                    regions->push_back({ bank, (size_t) cs[e]*bank->nb[2], dst, (size_t) e*dst->nb[2], (size_t) n*bank->nb[2] });
                    return n;
                }
                ggml_reset(ctx_views.get());
                ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], cs[e]*bank->nb[2]);
                ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
                ggml_backend_view_init(src_view);
                ggml_backend_view_init(dst_view);
                ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
                stats_copy.hits  += n;
                stats_copy.bytes += ggml_nbytes(src_view);
                stats_copy.calls += 1;
                return n;
            }
        }

        ua_dbg[0]++;
        // the device state is newer (e.g. after a warmup that selected every expert on the host): the host slots are stale, upload everything
        if (g.dev_newer) {
            ua_dbg[1]++;
            return 0;
        }

        // k15-pred: a layer with experts admitted ahead of its routing whose bank is not uploaded is read from the host
        if (!ua_readable(b.il, b.ip)) {
            ua_st_ub.blocked++;
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        // while a layer is seeded, its slots are filled after its copies, so the copies read the slots it had before
        const int32_t * slots = b.il == g.sd.il ? g.sd.old_slots.data() : g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            ua_dbg[slots == nullptr ? 2 : 3]++;
            return 0;
        }
        ua_dbg[4]++;
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];

        // batched: every cached run is copied, the copies are issued together with the uploads by the caller
        if (regions != nullptr) {
            GGML_ASSERT(bank->nb[2] == dst->nb[2]);
            regions->push_back({ bank, (size_t) slots[e]*bank->nb[2], dst, (size_t) e*dst->nb[2], (size_t) n*bank->nb[2] });
            return n;
        }

        // a short run is uploaded with its neighbors: many small uploads are slower than the bytes they save
        // (with the RAM tier the plan does not read the device-held experts ahead: every cached run is copied on the device)
        if (n*w->nb[2] < min_copy_bytes && !host_layer(b.il)) {
            return -n;
        }

        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);
        stats_copy.calls += 1;

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // k15-pred: a lookup reads the slots through the slot map: the experts admitted ahead get every bank first
        ua_lookup_guard(backend, g, il);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, unless the lookup is the first node of its split
        // (with GGML_SCHED_SPLIT_NO_INPUTS=0: only if the split already has inputs)
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        const bool tiered = host_layer(il);

        // reads ahead: issued while the device still computes the previous split, from the router input of the previous lookup of this graph
        const bool ahead = tiered && ahead_on();
        const ggml_tensor * xin = nullptr;
        bool xin_async = false;
        if (ahead) {
            ahead_reap();
            if (ahead_x_epoch == ahead_epoch && ahead_il >= 0 && il > ahead_il) {
                ahead_issue(il);
            }
            ahead_il = -1;
            xin = ahead_router_input(il, sel);
            if (xin != nullptr && (xin->type != GGML_TYPE_F32 || !ggml_is_contiguous(xin) || xin->ne[0] != ahead_n_embd || xin->ne[1] > ahead_max_tokens ||
                    ggml_nrows(xin) != xin->ne[1] || xin->buffer == nullptr || xin->data == nullptr)) {
                xin = nullptr;
            }
            if (xin != nullptr) {
                ahead_x.resize(ggml_nelements(xin));
                if (ggml_backend_buffer_get_type(xin->buffer) == ggml_backend_get_default_buffer_type(backend)) {
                    ggml_backend_tensor_get_async(backend, xin, ahead_x.data(), 0, ggml_nbytes(xin));
                    xin_async = true;
                }
            }
        }

        const bool prof = host_prof_mode() > 0 && ids.size() <= (size_t) 8*n_expert_used;
        int64_t t_prof = prof ? ggml_time_us() : 0;
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);
        if (prof) {
            const int64_t t = ggml_time_us();
            hprof.sync_us += t - t_prof;
            hprof.n++;
            t_prof = t;
        }

        // k19-mgpu2: the routing of a decode / verify window, for the next launch's resident device prefill (see rp_rec)
        if (rp_rec && tiered && ids.size() <= (size_t) 8*n_expert_used) {
            rp_count(il, ids);
        }

        if (tiered && hprof_prompted && !hprof_q.empty()) {
            hprof_wait_routed(il);
        }
        if (tiered && !fill_q.empty()) {
            fill_wait_routed(il);
        }
        if (ahead) {
            ahead_wait_routed(il);
            if (xin != nullptr && !xin_async) {
                // computed by another backend in an earlier split (done before this split's inputs are copied)
                ggml_backend_tensor_get(xin, ahead_x.data(), 0, ggml_nbytes(xin));
            }
            if (xin != nullptr) {
                ahead_il    = il;
                ahead_nt    = xin->ne[1];
                ahead_x_epoch = ahead_epoch;
            }
        }

        // k19-l4full: the route tail skip of a host lookup leaves its entries out of the plan (see tail_skip_host)
        const int32_t * pids  = ids.data();
        size_t          n_ids = ids.size();
        {
            const uint64_t n0 = tail_host_skipped;
            tail_filter_host(lookup, g, il, ids, pids, n_ids);
            if (prof && tail_host_skipped != n0) {
                hprof.n_skip_lookups++;
            }
        }

        // k19-mgpu2: deferred resident prompt fill: this layer's pending swaps go first, on the same stream, within the pass budget
        if (tiered && !rpf_pending.empty() && ids.size() <= (size_t) 8*n_expert_used && host->exclusive()) {
            resident_fill_lookup(g, backend, il);
        }

        // RAM tier: the experts routed by this call keep their RAM slots while it runs
        if (tiered) {
            host->begin_call();
            for (size_t i = 0; i < n_ids; ++i) {
                const int32_t e  = pids[i];
                const int32_t sl = e >= 0 && e < (int32_t) g.lru.n_expert ? host->find(il, e, false) : -1;
                if (sl >= 0) {
                    host->pin(sl);
                }
            }
        }

        size_t n_hit = 0;
        if (!g.lru.plan(il, pids, n_ids, fills, n_hit, tiered && host->exclusive() ? &evicted_tmp : nullptr)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (prof) {
            const int64_t t = ggml_time_us();
            hprof.plan_us += t - t_prof;
            t_prof = t;
        }

        // upload the missing experts (from the RAM tier, the model files or the mapping)
        hprof_on = prof;
        const size_t bytes = upload_fills(backend, il, fills, tiered && host->exclusive() ? &evicted_tmp : nullptr);
        hprof_on = false;
        if (prof) {
            const int64_t t = ggml_time_us();
            hprof.up_us += t - t_prof;
            if (host_prof_mode() >= 2) {
                ggml_backend_synchronize(backend);
                hprof.copy_us += ggml_time_us() - t;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    // statistics of a batch of copies of the experts of w issued by the caller: copies from the banks and uploads
    void count_batch(const ggml_tensor * w, const std::vector<ggml_backend_copy_region> & regions) {
        if (bindings.find(w) == bindings.end()) {
            return;
        }
        n_batches++;
        for (const auto & r : regions) {
            stats & st = r.src == w ? stats_upload : stats_copy;
            if (r.src == w) {
                st.misses += r.size/w->nb[2];
            } else {
                st.hits   += r.size/w->nb[2];
            }
            st.bytes += r.size;
            st.calls += 1;
        }
    }

    void count_upload(const ggml_tensor * w, size_t n_experts, size_t bytes) {
        if (bindings.find(w) != bindings.end()) {
            stats_upload.misses += n_experts;
            stats_upload.bytes  += bytes;
            stats_upload.calls  += 1;
        }
    }

    static int32_t ggml_get_seed_layer(const ggml_tensor * plan) {
        int il = -1, pos = 0, n_seed_max = 0, n_seed = 0;
        GGML_ASSERT(ggml_moe_cache_seed_info(plan, &il, &pos, &n_seed_max, &n_seed));
        return il;
    }

    // every slot of the group empty (host and device), as after a failed graph that seeded it in part
    void devseed_clear(group & g) {
        ggml_backend_synchronize(devices[g.id].backend);
        seed_end(g);
        for (int32_t il : g.layers) {
            if ((size_t) il < refill.size()) {
                refill[il].clear();
            }
        }
        if (g.lru.n_lent > 0) {
            unlend(g);
        }
        for (int32_t il : g.layers) {
            std::fill(g.lru.slot_map[il], g.lru.slot_map[il] + g.lru.n_expert, -1);
        }
        g.lru.clear();
        for (ggml_tensor * bank : g.banks) {
            // (k17-small: the slots left and the spare slot, not the retired rows; k19-ptres: the core keeps its experts)
            ggml_backend_tensor_memset(bank, 0, (size_t) g.lru.n_core*bank->nb[2], (size_t) (g.n_slots + 1 - g.lru.n_core)*bank->nb[2]);
        }
        g.dev_newer = false;
        g.dev_gen   = UINT64_MAX;
        LLAMA_LOG_WARN("%s: %s: a graph that seeds the MoE cache on the device failed, the cache of %zu layers is cleared\n",
            __func__, ggml_backend_name(devices[g.id].backend), g.layers.size());
    }

    // device seed (LLAMA_MOE_CACHE_DEVSEED): the groups whose device state can be seeded by the graph
    bool devseed_init(int64_t min_tokens) {
        devseed_min = std::max<int64_t>(min_tokens, max_batch + 1);
        bool any = false;
        for (group & g : groups) {
            g.devseed = false;
            if (g.n_slots == 0 || !g.dev_lookup || g.map == nullptr || g.state == nullptr || g.layers.empty()) {
                continue;
            }
            ggml_init_params params = { 8*ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr ctx { ggml_init(params) };
            ggml_tensor * ids  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, 2048);
            ggml_tensor * plan = ggml_moe_cache_seed_plan(ctx.get(), ids, g.map, g.state, g.scores, devseed_max(g), g.layers[0], 0);
            g.devseed = ggml_backend_supports_op(devices[g.id].backend, plan);
            any = any || g.devseed;
        }
        if (!any) {
            devseed_min = INT64_MAX;
        }
        return any;
    }

    // the most experts a layer of the group admits: an equal share of the slots (one for each of the first layers if there are fewer slots than layers)
    static int32_t devseed_max(const group & g) {
        return std::max<int32_t>(1, std::min<int32_t>(g.lru.n_expert, g.n_slots / (int32_t) g.layers.size()));
    }

    // k19-fixG: none in a graph built as a large ubatch (set_force_large: the tail and output members of a layer-major group that
    // releases the banks), which must not read or bind the banks
    ggml_tensor * devseed_plan(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_tokens) const {
        if (force_large || n_tokens < devseed_min || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const group & g = groups[layers[il].ig];
        if (!g.devseed) {
            return nullptr;
        }
        const int32_t pos = (int32_t) (std::find(g.layers.begin(), g.layers.end(), il) - g.layers.begin());
        ggml_tensor * plan = ggml_moe_cache_seed_plan(ctx, ids, g.map, g.state, g.scores, devseed_max(g), il, pos);
        if (g.lru.n_core > 0) {
            ggml_moe_cache_set_core(plan, g.lru.n_core);
        }
        return plan;
    }

    // the bank of the cached expert tensor w, and its layer
    int32_t layer_of_weight(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it == bindings.end() ? -1 : it->second.il;
    }

    ggml_tensor * devseed_bank(const ggml_tensor * w, int32_t & il) const {
        if (force_large) {
            return nullptr;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return nullptr;
        }
        const group & g = groups[layers[it->second.il].ig];
        if (!g.devseed) {
            return nullptr;
        }
        il = it->second.il;
        return g.banks[it->second.ip];
    }

    // before the graph runs (after its allocation, which places the lent slots): enable or disable its seed plans; the device state of the
    // groups it seeds is brought up to date with the host first
    void devseed_prepare(ggml_cgraph * gf, bool enable) {
        std::vector<bool> used(groups.size(), false);
        // a plan runs only if every bank of its group gets exactly one fill after it, from an expert product the scheduler left on the
        // group device that reads a device copy of the experts (else, e.g. without op offload, it is disabled and its fills copy nothing),
        // and the plans of a group come in layer order (a plan may evict the slots of earlier layers only)
        std::map<const ggml_tensor *, std::vector<int>> fills; // plan -> fills per bank
        std::map<const ggml_tensor *, bool> fill_ok;
        std::vector<int32_t> last_il(groups.size(), -1);
        std::map<const ggml_tensor *, bool> order_ok;
        if (enable) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * node = ggml_graph_node(gf, i);
                int il = -1, pos = 0, n_seed_max = 0, n_cur = 0;
                if (ggml_moe_cache_seed_info(node, &il, &pos, &n_seed_max, &n_cur)) {
                    const int32_t ig = layers[il].ig;
                    order_ok[node] = ig >= 0 && il > last_il[ig];
                    if (ig >= 0) {
                        last_il[ig] = std::max(last_il[ig], il);
                    }
                    fills[node].assign(groups[ig].banks.size(), 0);
                    fill_ok[node] = true;
                    continue;
                }
                if (node->op != GGML_OP_MOE_CACHE_LOOKUP || node->view_src == nullptr || node->src[0] == nullptr) {
                    continue;
                }
                const auto it = fills.find(node->src[0]);
                if (it == fills.end()) {
                    continue;
                }
                const group & g = groups[layers[ggml_get_seed_layer(node->src[0])].ig];
                const ggml_tensor * mm = node->src[1];
                const ggml_tensor * w  = mm != nullptr && mm->op == GGML_OP_MUL_MAT_ID ? mm->src[0] : nullptr;
                const ggml_backend_buffer_t buf = w != nullptr ? (w->view_src ? w->view_src->buffer : w->buffer) : nullptr;
                const bool dev_ok = buf != nullptr && w->data != nullptr && !ggml_backend_buffer_is_host(buf) &&
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf)) == ggml_backend_get_device(devices[g.id].backend);
                const auto ib = std::find(g.banks.begin(), g.banks.end(), node->src[2]);
                if (!dev_ok || ib == g.banks.end()) {
                    fill_ok[node->src[0]] = false;
                    continue;
                }
                it->second[ib - g.banks.begin()]++;
            }
        }
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            int il = -1, pos = 0, n_seed_max = 0, n_cur = 0;
            if (!ggml_moe_cache_seed_info(node, &il, &pos, &n_seed_max, &n_cur)) {
                continue;
            }
            const int32_t ig = layers[il].ig;
            group & g = groups[ig];
            bool ok = enable && order_ok[node] && fill_ok[node];
            for (int n : fills[node]) {
                ok = ok && n == 1;
            }
            if (!ok) {
                if (enable) {
                    devseed_disabled++;
                }
                ggml_moe_cache_seed_set(node, 0, 0);
                continue;
            }
            if (!used[ig]) {
                used[ig] = true;
                if (g.dev_newer) {
                    download(g);
                }
                if (g.dev_gen != g.lru.gen) {
                    upload(g);
                }
            }
            // the lent slots (the last n_lent) hold copies of this graph
            const int32_t n_avail = g.n_slots - g.lru.n_lent;
            const int32_t n_layer = (int32_t) g.layers.size();
            int32_t n_seed = n_avail / n_layer;
            if (n_seed == 0) {
                n_seed = pos < n_avail ? 1 : 0;
            }
            n_seed = std::min<int32_t>(n_seed, n_seed_max);
            ggml_moe_cache_seed_set(node, std::max(0, n_seed), std::max(0, n_avail));
        }
    }

    // after the graph ran: the seeded groups come back to the host, so the copies of the next large batch read the new slots
    void devseed_end(ggml_cgraph * gf, bool ok) {
        std::vector<bool> used(groups.size(), false);
        bool any = false;
        std::vector<int32_t> n_fill(1);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            int il = -1, pos = 0, n_seed_max = 0, n_seed = 0;
            if (!ggml_moe_cache_seed_info(node, &il, &pos, &n_seed_max, &n_seed) || n_seed == 0) {
                continue;
            }
            const int32_t ig = layers[il].ig;
            group & g = groups[ig];
            if (!used[ig]) {
                ggml_backend_synchronize(devices[g.id].backend);
            }
            used[ig] = true;
            any = true;
            ggml_moe_cache_seed_set(node, 0, 0);
            if (!ok) {
                continue;
            }
            ggml_backend_tensor_get(node, n_fill.data(), 0, sizeof(int32_t));
            devseed_admitted += n_fill[0];
            devseed_bytes    += (size_t) n_fill[0]*g.expert_bytes;
        }
        if (!any) {
            return;
        }
        if (!ok) {
            // some plans may have run without their fills: the device map may name slots that do not hold their experts
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                if (used[ig]) {
                    devseed_clear(groups[ig]);
                }
            }
            return;
        }
        devseed_graphs++;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (used[ig]) {
                group & g = groups[ig];
                download(g);
                if (seed_check && !verify_enabled) {
                    verify(g, "after the device seed");
                }
            }
        }
    }

    // k11 (LLAMA_MOE_LAYER_MAJOR_SEED_REPLAY): the device seed that a prefetched large ubatch run by itself does (devseed_prepare / the plans of
    // its graph / devseed_end), for a ubatch that ran inside a merged layer-major graph without it: after that graph, from its routing (rec:
    // per layer, in layer order, the routing outputs that graph left on the device). The plans run in layer order on the device state that the ubatch's
    // own graph would have started from (the host state uploaded as devseed_prepare does), with the same seed shares; the experts they admit
    // are uploaded from host memory (the graph's copies of them are gone) instead of filled from the expert products
    bool devseed_replay(const std::vector<std::pair<int32_t, ggml_tensor *>> & rec, int64_t n_tokens, size_t & bytes_out, double & ms_out) {
        bytes_out = 0;
        ms_out    = 0.0;
        if (n_tokens < devseed_min || rec.empty()) {
            return false;
        }
        const int64_t t0 = ggml_time_us();
        ggml_init_params params = { (2*rec.size() + 4)*ggml_tensor_overhead() + ggml_graph_overhead_custom(4*rec.size() + 16, false), nullptr, true };
        ggml_context_ptr ctx { ggml_init(params) };
        struct item { int32_t il; ggml_tensor * plan; };
        std::vector<item> items;
        int32_t last_il = -1;
        ggml_backend_t backend = nullptr;
        for (const auto & [il, ids] : rec) {
            // the plans of a group come in layer order; the routing is read where the graph left it, on the device of the group
            if (il <= last_il || ids == nullptr || ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids) || ids->ne[1] != n_tokens ||
                    ids->data == nullptr || ids->buffer == nullptr || ggml_backend_buffer_is_host(ids->buffer)) {
                return false;
            }
            last_il = il;
            // a leaf over the routing output (the plan graph does not compute it again)
            ggml_tensor * leaf = ggml_new_tensor(ctx.get(), ids->type, GGML_MAX_DIMS, ids->ne);
            if (ggml_backend_tensor_alloc(ids->buffer, leaf, ids->data) != GGML_STATUS_SUCCESS) {
                return false;
            }
            ggml_tensor * plan = devseed_plan(ctx.get(), il, leaf, n_tokens);
            if (plan == nullptr) {
                continue;
            }
            const ggml_backend_t b = devices[groups[layers[il].ig].id].backend;
            if ((backend != nullptr && b != backend) ||
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(ids->buffer)) != ggml_backend_get_device(b)) {
                return false;
            }
            backend = b;
            items.push_back({ il, plan });
        }
        if (items.empty()) {
            return false;
        }
        // the plan results in a buffer kept for the next replay
        const size_t align = ggml_backend_get_alignment(backend);
        size_t need = 0;
        for (const auto & it : items) {
            need += GGML_PAD(ggml_nbytes(it.plan), align);
        }
        if (!replay_buf || ggml_backend_buffer_get_size(replay_buf.get()) < need ||
                ggml_backend_buffer_get_type(replay_buf.get()) != ggml_backend_get_default_buffer_type(backend)) {
            replay_buf.reset(ggml_backend_alloc_buffer(backend, need));
            if (!replay_buf) {
                return false;
            }
        }
        // all plan results, read back at once
        ggml_tensor * all = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t) (need/sizeof(int32_t)));
        {
            uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(replay_buf.get());
            if (ggml_backend_tensor_alloc(replay_buf.get(), all, base) != GGML_STATUS_SUCCESS) {
                return false;
            }
            size_t off = 0;
            for (const auto & it : items) {
                if (ggml_backend_tensor_alloc(replay_buf.get(), it.plan, base + off) != GGML_STATUS_SUCCESS) {
                    return false;
                }
                off += GGML_PAD(ggml_nbytes(it.plan), align);
            }
        }
        const int64_t ta = ggml_time_us();
        std::vector<bool> used(groups.size(), false);
        for (const auto & it : items) {
            const int32_t ig = layers[it.il].ig;
            group & g = groups[ig];
            if (!used[ig]) {
                used[ig] = true;
                if (g.dev_newer) {
                    download(g);
                }
                if (g.dev_gen != g.lru.gen) {
                    upload(g);
                }
            }
            int il = -1, pos = 0, n_seed_max = 0, n_cur = 0;
            GGML_ASSERT(ggml_moe_cache_seed_info(it.plan, &il, &pos, &n_seed_max, &n_cur));
            const int32_t n_avail = g.n_slots - g.lru.n_lent;
            const int32_t n_layer = (int32_t) g.layers.size();
            int32_t n_seed = n_avail / n_layer;
            if (n_seed == 0) {
                n_seed = pos < n_avail ? 1 : 0;
            }
            n_seed = std::min<int32_t>(n_seed, n_seed_max);
            ggml_moe_cache_seed_set(it.plan, std::max(0, n_seed), std::max(0, n_avail));
        }
        const int64_t tb = ggml_time_us();
        // one graph, the plans in layer order (as in the graph of the ubatch: they read and write one device state)
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 4*rec.size() + 16, false);
        for (const auto & it : items) {
            ggml_build_forward_expand(gf, it.plan);
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                if (used[ig]) {
                    devseed_clear(groups[ig]);
                }
            }
            return false;
        }
        // the admitted experts and their slots: one read of all plan results
        std::vector<int32_t> out(need/sizeof(int32_t));
        ggml_backend_tensor_get(all, out.data(), 0, need);
        const int64_t t1 = ggml_time_us();
        // into their slots, in layer order (a later layer may evict a slot an earlier one filled: its upload comes later)
        std::vector<moe_cache_lru::fill> fl;
        size_t bytes = 0;
        size_t off = 0;
        for (const auto & it : items) {
            const group & g = groups[layers[it.il].ig];
            const int32_t * r = out.data() + off/sizeof(int32_t);
            off += GGML_PAD(ggml_nbytes(it.plan), align);
            const int32_t n_seed_max = (int32_t) (ggml_nelements(it.plan) - 1)/2;
            const int32_t n = std::min(r[0], n_seed_max);
            fl.clear();
            for (int32_t j = 0; j < n; ++j) {
                fl.push_back({ r[1 + j], r[1 + n_seed_max + j] });
            }
            std::sort(fl.begin(), fl.end(), [](const moe_cache_lru::fill & a, const moe_cache_lru::fill & b) { return a.expert < b.expert; });
            if (!fl.empty()) {
                bytes += upload_fills(backend, it.il, fl, nullptr);
            }
            devseed_admitted += n;
            devseed_bytes    += (size_t) n*g.expert_bytes;
        }
        ggml_backend_synchronize(backend);
        const int64_t t2 = ggml_time_us();
        devseed_graphs++;
        devseed_replays++;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (used[ig]) {
                group & g = groups[ig];
                download(g);
                if (seed_check && !verify_enabled) {
                    verify(g, "after the replayed device seed");
                }
            }
        }
        const int64_t t3 = ggml_time_us();
        if (getenv("LLAMA_MOE_LAYER_MAJOR_LOG") != nullptr) {
            LLAMA_LOG_INFO("%s: replay: setup %.2f ms, state upload %.2f ms, plans %.2f ms, expert uploads %.2f ms, download %.2f ms\n", __func__,
                (ta - t0)/1000.0, (tb - ta)/1000.0, (t1 - tb)/1000.0, (t2 - t1)/1000.0, (t3 - t2)/1000.0);
        }
        bytes_out = bytes;
        ms_out    = (t3 - t0)/1000.0;
        devseed_replay_us += t3 - t0;
        devseed_replay_up_us += t2 - t1;
        devseed_replay_bytes += bytes;
        return true;
    }
    ggml_backend_buffer_ptr replay_buf; // the plan results of devseed_replay

    void seed(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * dst, const std::vector<int32_t> & counts, const std::vector<bool> & staged) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return;
        }
        const binding & b = it->second;
        group & g = groups[layers[b.il].ig];
        if (ggml_backend_get_device(backend) != ggml_backend_get_device(devices[g.id].backend) || g.dev_newer) {
            return;
        }

        // the first expert tensor of the layer plans the slots of its most used experts, each layer gets an equal share
        seed_state & sd = g.sd;
        if (b.il != sd.il) {
            seed_end(g);

            const int32_t n_expert = g.lru.n_expert;
            // while slots are lent, the experts that the lent slots would hold are uploaded when they come back (see unlend)
            // (k19-ptres: the core slots take no seeds)
            const size_t  n_avail  = (size_t) (g.n_slots - g.lru.n_lent - g.lru.n_core);
            size_t        n_seed   = n_avail / g.layers.size();
            size_t        n_full   = (size_t) (g.n_slots - g.lru.n_core) / g.layers.size();
            if (n_full == 0) {
                // fewer slots than layers: one expert for each of the first layers of the group, instead of none
                const size_t pos = std::find(g.layers.begin(), g.layers.end(), b.il) - g.layers.begin();
                n_full = pos < (size_t) g.n_slots ? 1 : 0;
                n_seed = pos < n_avail ? 1 : 0;
            }
            if (profile_loaded && profile_seed < 1.0) {
                // a warm start (LLAMA_MOE_CACHE_PROFILE): the prompt seeds only this share of each layer, the profile keeps the rest
                n_full = (size_t) (n_full*profile_seed);
                n_seed = std::min(n_seed, n_full);
                if (n_full == 0) {
                    return;
                }
            }
            GGML_ASSERT((int32_t) counts.size() == n_expert && (int32_t) staged.size() == n_expert);
            seed_ids.clear();
            for (int32_t e = 0; e < n_expert; ++e) {
                if (counts[e] > 0 && staged[e]) {
                    seed_ids.push_back(e);
                }
            }
            auto by_count = [&](int32_t a, int32_t c) {
                return counts[a] != counts[c] ? counts[a] > counts[c] : a < c;
            };
            if (n_full < seed_ids.size()) {
                std::partial_sort(seed_ids.begin(), seed_ids.begin() + n_full, seed_ids.end(), by_count);
                seed_ids.resize(n_full);
            } else if (n_seed < seed_ids.size()) {
                std::sort(seed_ids.begin(), seed_ids.end(), by_count);
            }
            if (n_seed < seed_ids.size()) {
                refill.resize(layers.size());
                refill[b.il].assign(seed_ids.begin() + n_seed, seed_ids.end());
                seed_ids.resize(n_seed);
            }
            if (seed_ids.empty()) {
                return;
            }

            const int32_t * slots = g.lru.slot_map[b.il];
            sd.il = b.il;
            sd.old_slots.assign(slots, slots + n_expert);
            sd.done = 0;
            sd.old_key_of = g.lru.key_of;
            sd.old_prev   = g.lru.prev;
            sd.old_next   = g.lru.next;
            sd.old_head   = g.lru.head;
            sd.old_tail   = g.lru.tail;
            if (g.lru.grow_q16 > 0) {
                sd.old_scores = g.lru.scores;
                sd.old_w      = g.lru.w;
            }
            size_t n_hit = 0;
            const bool tiered = host_layer(b.il) && host->exclusive();
            const bool ok = g.lru.plan(b.il, seed_ids.data(), seed_ids.size(), sd.fills, n_hit, tiered ? &evicted_tmp : nullptr);
            GGML_ASSERT(ok);
            sd.gen = g.lru.gen;
            if (tiered) {
                // exclusive: the experts the seed evicts move to RAM before any of their device slots is overwritten
                // k19-ramevict: a large batch seeds layer b.il after its experts were staged, so the RAM slots they need make room in the
                // layers this pass has finished first (see llama_moe_host_cache::set_stream_layer); decode never seeds
                host->set_stream_layer(b.il);
                shadow_slots(backend, g, evicted_tmp, true);
                host->set_stream_layer(-1);
            }
        }

        // fill the new slots from the staged copy of the experts, after it was filled on the same backend
        ggml_tensor * bank = g.banks[b.ip];
        ggml_tensor * src  = const_cast<ggml_tensor *>(dst);
        const auto & fills = sd.fills;

        // an expert tensor whose copy lacks a planned expert (e.g. only routed experts uploaded) must not fill the slots, seed_end then drops the layer
        GGML_ASSERT(staged.size() == (size_t) w->ne[2]);
        for (const auto & f : fills) {
            if (!staged[f.expert]) {
                seed_unstaged++;
                return;
            }
        }
        if (seed_check) {
            for (const auto & f : fills) {
                sd.pending.push_back({ w, bank, f.expert, f.slot });
            }
        }

        // batched: one call for the whole tensor
        seed_regions.clear();
        for (size_t i = 0; i < fills.size();) {
            size_t n = 1;
            while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                n++;
            }
            seed_regions.push_back({ src, (size_t) fills[i].expert*src->nb[2], bank, (size_t) fills[i].slot*bank->nb[2], n*src->nb[2] });
            i += n;
        }
        if (!seed_regions.empty() && ggml_backend_tensor_copy_regions_async(backend, seed_regions.data(), seed_regions.size())) {
            for (const auto & r : seed_regions) {
                stats_seed.bytes += r.size;
            }
            sd.done |= 1u << b.ip;
            return;
        }

        for (size_t i = 0; i < fills.size();) {
            size_t n = 1;
            while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                n++;
            }
            ggml_reset(ctx_views.get());
            ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), src,  src->ne[0],  src->ne[1],  n, src->nb[1],  src->nb[2],  fills[i].expert*src->nb[2]);
            ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], fills[i].slot*bank->nb[2]);
            ggml_backend_view_init(src_view);
            ggml_backend_view_init(dst_view);
            ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
            stats_seed.bytes += ggml_nbytes(src_view);
            i += n;
        }
        sd.done |= 1u << b.ip;
    }

    void seed_end(group & g) {
        seed_state & sd = g.sd;
        if (sd.il < 0) {
            return;
        }
        const layer & l = layers[sd.il];
        if (sd.done == (1u << l.experts.size()) - 1) {
            stats_seed.misses += sd.fills.size();
            if (host_layer(sd.il) && host->exclusive()) {
                // exclusive: promoted to the device, the RAM copies go (spare room: they stay as the first to go); the uploads that read
                // them are guarded by their events
                for (const auto & f : sd.fills) {
                    if (host_spare) {
                        const int32_t h = host->held(sd.il, f.expert);
                        if (h >= 0) {
                            host->set_dup(h, true);
                            hst.promoted++;
                        }
                    } else if (host->contains(sd.il, f.expert) && !host->is_async(host->slot_of_key(sd.il, f.expert))) {
                        // (k19-ramprof2: a slot being read in the background is left to its read: published, it is a duplicate)
                        host->release_key(sd.il, f.expert);
                        hst.promoted++;
                        sss.prom_seed++;
                    }
                }
            }
            seed_committed.insert(seed_committed.end(), sd.pending.begin(), sd.pending.end());
        } else {
            // some expert tensors were copied without filling their slots, drop the experts admitted to the layer
            // a host lookup of another layer may have reassigned a slot since the plan, its new owner stays
            const int32_t key0 = sd.il*g.lru.n_expert;
            for (const auto & f : sd.fills) {
                if (g.lru.slot_map[sd.il][f.expert] == f.slot) {
                    g.lru.slot_map[sd.il][f.expert] = -1;
                }
                if (g.lru.key_of[f.slot] == key0 + f.expert) {
                    g.lru.key_of[f.slot] = -1;
                }
            }
            // the LRU did not change since the plan: restore its order and, if no slot was written, the experts it evicted
            if (g.lru.gen == sd.gen) {
                if (sd.done == 0) {
                    for (const auto & f : sd.fills) {
                        const int32_t k = sd.old_key_of[f.slot];
                        g.lru.key_of[f.slot] = k;
                        if (k >= 0) {
                            g.lru.slot_map[k / g.lru.n_expert][k % g.lru.n_expert] = f.slot;
                        }
                    }
                }
                g.lru.prev = sd.old_prev;
                g.lru.next = sd.old_next;
                g.lru.head = sd.old_head;
                g.lru.tail = sd.old_tail;
                if (g.lru.grow_q16 > 0) {
                    g.lru.scores = sd.old_scores;
                    g.lru.w      = sd.old_w;
                }
            }
            g.lru.gen++; // the device copy of the state is refreshed before the next device lookup
        }
        sd.il = -1;
        sd.fills.clear();
        sd.pending.clear();
    }

    void seed_end() {
        for (group & g : groups) {
            seed_end(g);
        }
    }

    // LLAMA_MOE_CACHE_SEED_CHECK: the device work of the seeding ubatch must be complete
    void seed_verify() {
        if (seed_committed.empty()) {
            return;
        }
        std::vector<uint8_t> buf;
        size_t n_bad     = 0;
        size_t n_checked = 0;
        for (const auto & r : seed_committed) {
            // a later lookup may have evicted the seeded expert (or the device state replaced the host slots): nothing to compare
            const binding & b = bindings.at(r.w);
            const group   & g = groups[layers[b.il].ig];
            if (g.dev_newer || g.lru.key_of[r.slot] != b.il*g.lru.n_expert + r.expert) {
                continue;
            }
            n_checked++;
            const size_t size = r.w->nb[2];
            GGML_ASSERT(r.bank->nb[2] == size && r.w->data != nullptr);
            buf.resize(size);
            ggml_backend_tensor_get(r.bank, buf.data(), (size_t) r.slot*r.bank->nb[2], size);
            if (memcmp(buf.data(), (const uint8_t *) r.w->data + (size_t) r.expert*size, size) != 0) {
                n_bad++;
            }
        }
        LLAMA_LOG_INFO("llama_moe_cache: seed check: %zu expert copies verified, %zu mismatches, %zu evicted since\n",
            n_checked, n_bad, seed_committed.size() - n_checked);
        seed_checked    += n_checked;
        seed_mismatches += n_bad;
        seed_committed.clear();
    }

    void set_seed_counts(const ggml_tensor * w, const std::vector<int32_t> & counts) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return;
        }
        seed_counts.resize(layers.size());
        seed_counts[it->second.il] = counts;
    }

    const std::vector<int32_t> * get_seed_counts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        if (it == bindings.end() || (size_t) it->second.il >= seed_counts.size()) {
            return nullptr;
        }
        const auto & c = seed_counts[it->second.il];
        return (int64_t) c.size() == w->ne[2] ? &c : nullptr;
    }

    void clear_seed_counts() {
        seed_counts.clear();
    }

    void set_refill_counts(const ggml_tensor * w, const std::vector<int32_t> & counts) {
        const auto it = bindings.find(w);
        if (it == bindings.end() || (!refill_routed && !refill_routed_once)) {
            return;
        }
        refill_counts.resize(layers.size());
        refill_counts[it->second.il] = counts;
    }

    bool has_seed_counts() const {
        return std::any_of(seed_counts.begin(), seed_counts.end(), [](const std::vector<int32_t> & c) { return !c.empty(); });
    }

    // k19-prewexact: see llama_moe_cache::prewarm_begin. The state a decode moves, as it was before the server's verify-graph prewarm
    struct pw_group {
        moe_cache_lru lru;
        std::vector<std::vector<int32_t>> slot_maps; // [layers of the group] the host slot maps
        std::vector<uint8_t> dev[4];                 // the device map, state, counts, scores (raw)
        std::vector<int32_t> key;                    // the expert each slot holds (the device's key_of if it is newer, else the host's)
        std::vector<int32_t> ua_dec_key;
        seed_state sd;
        uint64_t dev_gen   = 0;
        bool     dev_newer = false;
        int32_t  n_slots   = 0;
    };
    bool pw_active = false; // a snapshot is held, the rules are quiet (pw_quiet)
    bool pw_quiet  = false; // no CPU-miss window, timing trial, overlap rule / trial, prompt trial or probe moves
    bool pw_cm_paused = false;
    std::vector<pw_group> pw_groups;
    std::vector<std::vector<int32_t>> pw_seed_counts;
    std::vector<std::vector<int32_t>> pw_refill;
    std::vector<std::vector<int32_t>> pw_refill_counts;
    std::vector<int32_t> pw_seed_ids;
    stats   pw_stats[6];
    size_t  pw_n_large = 0;
    size_t  pw_n_batches = 0;
    size_t  pw_n_refill_routed = 0;
    bool    pw_refill_routed_once = false;
    bool    pw_st_after_large = false;
    bool    pw_ua_skip = false;
    int64_t pw_ua_prev_tokens = 0;
    bool    pw_flags[3] = { false, false, false }; // seed_own, prefill_deferred, prefilled_dev

    static void pw_get(const ggml_tensor * t, std::vector<uint8_t> & out) {
        out.clear();
        if (t != nullptr && t->buffer != nullptr) {
            out.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, out.data(), 0, out.size());
        }
    }

    std::vector<int32_t> pw_key(const group & g) const {
        if (g.dev_newer && g.dev_lookup && g.state != nullptr && g.state->buffer != nullptr) {
            std::vector<int32_t> k(g.n_slots);
            ggml_backend_tensor_get(g.state, k.data(), n_hdr*sizeof(int32_t), k.size()*sizeof(int32_t));
            return k;
        }
        return g.lru.key_of;
    }

    // why the state cannot be taken back exactly (nullptr: it can)
    const char * pw_refused() const {
        if (host) {
            return "a RAM tier is active";
        }
        for (const device & d : devices) {
            if (d.released) {
                return "the banks are released";
            }
        }
        return nullptr;
    }

    bool prewarm_begin() {
        if (pw_active) {
            return true;
        }
        if (const char * why = pw_refused()) {
            LLAMA_LOG_WARN("llama_moe_cache%s: k19-prewexact: no snapshot before the verify-graph prewarm (%s): the prewarm moves the cache\n", tag, why);
            return false;
        }
        ua_drain();
        for (const device & d : devices) {
            if (d.backend != nullptr) {
                ggml_backend_synchronize(d.backend);
            }
        }
        pw_groups.assign(groups.size(), pw_group());
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            pw_group & s = pw_groups[ig];
            s.lru = g.lru;
            s.slot_maps.resize(g.layers.size());
            for (size_t j = 0; j < g.layers.size(); ++j) {
                const int32_t * p = g.lru.slot_map[g.layers[j]];
                s.slot_maps[j].assign(p, p + (p != nullptr ? g.lru.n_expert : 0));
            }
            pw_get(g.map,    s.dev[0]);
            pw_get(g.state,  s.dev[1]);
            pw_get(g.counts, s.dev[2]);
            pw_get(g.scores, s.dev[3]);
            s.key        = pw_key(g);
            s.ua_dec_key = g.ua_dec_key;
            s.sd         = g.sd;
            s.dev_gen    = g.dev_gen;
            s.dev_newer  = g.dev_newer;
            s.n_slots    = g.n_slots;
        }
        pw_seed_counts   = seed_counts;
        pw_refill        = refill;
        pw_refill_counts = refill_counts;
        pw_seed_ids      = seed_ids;
        pw_stats[0] = stats_small; pw_stats[1] = stats_large; pw_stats[2] = stats_copy;
        pw_stats[3] = stats_upload; pw_stats[4] = stats_seed; pw_stats[5] = stats_refill;
        pw_n_large            = n_large_ubatches;
        pw_n_batches          = n_batches;
        pw_n_refill_routed    = n_refill_routed;
        pw_refill_routed_once = refill_routed_once;
        pw_st_after_large     = st_after_large;
        pw_ua_skip            = ua_skip;
        pw_ua_prev_tokens     = ua_prev_tokens;
        pw_flags[0] = seed_own; pw_flags[1] = prefill_deferred; pw_flags[2] = prefilled_dev;
        pw_cm_paused = cm_paused;
        cm_paused    = true; // cm_tick moves nothing (not cpu_miss_pause: its end starts a fresh window)
        pw_quiet     = true;
        pw_active    = true;
        return true;
    }

    // the experts of the slots that changed come back into the banks, then every saved value; false: refused (the state stays as the
    // prewarm left it, the rules resume as after the k19-tokcost pause)
    bool prewarm_end() {
        if (!pw_active) {
            return false;
        }
        pw_active = false;
        pw_quiet  = false;
        const char * why = pw_refused();
        for (size_t ig = 0; ig < groups.size() && why == nullptr; ++ig) {
            const group & g = groups[ig];
            // (the lent slots may differ: the snapshot's lent slots hold no expert and the next large batch lends them as it would
            // have, with the same refill list)
            if (g.n_slots != pw_groups[ig].n_slots || g.lru.n_slots != pw_groups[ig].lru.n_slots) {
                why = "the slot count changed";
            }
        }
        if (why != nullptr) {
            LLAMA_LOG_WARN("llama_moe_cache%s: k19-prewexact: the cache is not taken back after the verify-graph prewarm (%s)\n", tag, why);
            cm_paused = pw_cm_paused;
            cm_have0  = false;
            cm_nb     = 0;
            pw_groups.clear();
            return false;
        }
        ua_drain();
        for (const device & d : devices) {
            if (d.backend != nullptr) {
                ggml_backend_synchronize(d.backend);
            }
        }
        size_t n_copied = 0;
        size_t bytes    = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            pw_group & s = pw_groups[ig];
            ggml_backend_t backend = devices[g.id].backend;
            const std::vector<int32_t> cur = pw_key(g);
            const int32_t n_expert = g.lru.n_expert;
            std::vector<llama_weight_store::region> regions;
            for (int32_t sl = 0; sl < g.n_slots - s.lru.n_lent; ++sl) {
                const int32_t k = s.key[sl];
                if (k < 0 || cur[sl] == k) {
                    continue;
                }
                const int32_t il = k / n_expert;
                const int32_t e  = k % n_expert;
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size() && ip < g.banks.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t nb = w->nb[2];
                    if (wstore != nullptr && wstore->covers(w)) {
                        regions.push_back({ w, (size_t) e*nb, g.banks[ip], (size_t) sl*nb, nb });
                    } else {
                        ggml_backend_tensor_set_async(backend, g.banks[ip], (const uint8_t *) w->data + (size_t) e*nb, (size_t) sl*nb, nb);
                    }
                    bytes += nb;
                }
                n_copied++;
            }
            if (!regions.empty() && !wstore->upload_batch(backend, regions.data(), regions.size())) {
                for (const auto & r : regions) {
                    ggml_backend_tensor_set_async(backend, r.dst, (const uint8_t *) r.src->data + r.src_off, r.dst_off, r.size);
                }
            }
            g.lru = s.lru;
            for (size_t j = 0; j < g.layers.size(); ++j) {
                int32_t * p = g.lru.slot_map[g.layers[j]];
                if (p != nullptr && !s.slot_maps[j].empty()) {
                    memcpy(p, s.slot_maps[j].data(), s.slot_maps[j].size()*sizeof(int32_t));
                }
            }
            ggml_tensor * ts[4] = { g.map, g.state, g.counts, g.scores };
            for (int k = 0; k < 4; ++k) {
                if (ts[k] != nullptr && ts[k]->buffer != nullptr && s.dev[k].size() == ggml_nbytes(ts[k])) {
                    ggml_backend_tensor_set_async(backend, ts[k], s.dev[k].data(), 0, s.dev[k].size());
                }
            }
            g.ua_dec_key = s.ua_dec_key;
            g.sd         = s.sd;
            g.dev_gen    = s.dev_gen;
            g.dev_newer  = s.dev_newer;
            ggml_backend_synchronize(backend);
        }
        seed_counts   = pw_seed_counts;
        refill        = pw_refill;
        refill_counts = pw_refill_counts;
        seed_ids      = pw_seed_ids;
        stats_small = pw_stats[0]; stats_large = pw_stats[1]; stats_copy = pw_stats[2];
        stats_upload = pw_stats[3]; stats_seed = pw_stats[4]; stats_refill = pw_stats[5];
        n_large_ubatches   = pw_n_large;
        n_batches          = pw_n_batches;
        n_refill_routed    = pw_n_refill_routed;
        refill_routed_once = pw_refill_routed_once;
        st_after_large     = pw_st_after_large;
        ua_skip            = pw_ua_skip;
        ua_prev_tokens     = pw_ua_prev_tokens;
        seed_own = pw_flags[0]; prefill_deferred = pw_flags[1]; prefilled_dev = pw_flags[2];
        cm_paused          = pw_cm_paused; // the window goes on where it was (its counters are back too)
        pw_groups.clear();
        LLAMA_LOG_INFO("llama_moe_cache%s: k19-prewexact: the cache is back as before the verify-graph prewarm: %zu slots got their expert "
            "back (%.1f MiB), slot state, counters and rule state restored (LLAMA_SPEC_PREWARM_EXACT=0: off)\n", tag, n_copied, bytes/1048576.0);
        return true;
    }

    // LLAMA_MOE_CACHE_FP_LOG=1 (diagnostic): hashes of the device state, the host LRU, the rule state and the bookkeeping
    void fingerprint(const char * when) {
        auto h0 = []() { return (uint64_t) 1469598103934665603ULL; };
        auto mixb = [](uint64_t & h, const void * p, size_t n) {
            const uint8_t * b = (const uint8_t *) p;
            for (size_t i = 0; i < n; ++i) {
                h = (h ^ b[i])*1099511628211ULL;
            }
        };
        auto mixv = [&](uint64_t & h, auto v) { mixb(h, &v, sizeof(v)); };
        uint64_t hd = h0(), hl = h0(), hr = h0(), ha = h0();
        for (const device & d : devices) {
            if (d.backend != nullptr) {
                ggml_backend_synchronize(d.backend);
            }
        }
        std::vector<uint8_t> buf;
        for (const group & g : groups) {
            const ggml_tensor * ts[4] = { g.map, g.state, g.counts, g.scores };
            for (int k = 0; k < 4; ++k) {
                pw_get(ts[k], buf);
                mixb(hd, buf.data(), buf.size());
            }
            mixb(hl, g.lru.key_of.data(), g.lru.key_of.size()*sizeof(int32_t));
            mixb(hl, g.lru.prev.data(),   g.lru.prev.size()*sizeof(int32_t));
            mixb(hl, g.lru.next.data(),   g.lru.next.size()*sizeof(int32_t));
            mixb(hl, g.lru.scores.data(), g.lru.scores.size()*sizeof(int32_t));
            mixv(hl, g.lru.head); mixv(hl, g.lru.tail); mixv(hl, g.lru.w); mixv(hl, g.lru.n_lent); mixv(hl, g.lru.n_core);
            for (int32_t il : g.layers) {
                if (g.lru.slot_map[il] != nullptr) {
                    mixb(hl, g.lru.slot_map[il], g.lru.n_expert*sizeof(int32_t));
                }
            }
            mixv(ha, g.lru.gen); mixv(ha, g.dev_gen); mixv(ha, (int) g.dev_newer); mixv(ha, g.n_slots);
        }
        mixv(hr, (int) cm); mixv(hr, (int) hybrid); mixb(hr, cm_on.data(), cm_on.size()); mixv(hr, cm_rate); mixv(hr, (int) cm_measured);
        mixv(hr, (int) cm_have0); mixv(hr, cm_nb); mixv(hr, cm_win); mixv(hr, cm_seen); mixv(hr, cm_nwin); mixv(hr, cm_gen);
        mixv(hr, cmt_wait); mixv(hr, cmt_done); mixv(hr, cmt_n); mixv(hr, (int) cmw_on); mixv(hr, cmw_n); mixv(hr, cm_n_prompt);
        mixv(hr, cm_n_small); mixb(hr, cm_c0, sizeof(cm_c0)); mixv(hr, (int) ov_live); mixv(hr, ov_seen); mixv(hr, (int) ov_have0);
        mixv(hr, ov_n1); mixv(hr, ov_done); mixv(hr, ov_wait); mixv(hr, win_done); mixv(hr, win_wait); mixv(hr, (int) hybrid_probe);
        mixv(hr, cm_recal_wait); mixv(hr, (int) cm_paused);
        const stats * st[6] = { &stats_small, &stats_large, &stats_copy, &stats_upload, &stats_seed, &stats_refill };
        for (const stats * s : st) {
            mixv(ha, s->hits); mixv(ha, s->misses); mixv(ha, s->bytes); mixv(ha, s->calls);
        }
        mixv(ha, n_large_ubatches); mixv(ha, n_batches); mixv(ha, ua_prev_tokens);
        mixv(ha, (int) seed_own); mixv(ha, (int) prefill_deferred); mixv(ha, (int) prefilled_dev);
        for (const auto & v : seed_counts)   { mixb(ha, v.data(), v.size()*sizeof(int32_t)); mixv(ha, v.size()); }
        for (const auto & v : refill)        { mixb(ha, v.data(), v.size()*sizeof(int32_t)); mixv(ha, v.size()); }
        for (const auto & v : refill_counts) { mixb(ha, v.data(), v.size()*sizeof(int32_t)); mixv(ha, v.size()); }
        LLAMA_LOG_INFO("llama_moe_cache%s: k19-prewexact fp %s: device %016llx lru %016llx rules %016llx aux %016llx\n", tag, when,
            (unsigned long long) hd, (unsigned long long) hl, (unsigned long long) hr, (unsigned long long) ha);
    }

    void log_stats() {
        if (pin_layers_want > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: pinned expert ranges: %zu of %zu layers, %.2f MiB\n", tag, pin_layers, pin_layers_want, pin_bytes/1048576.0);
        }
        auto log = [this](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache%s: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                tag, name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (ua_st.layers > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: uploads ahead: %" PRIu64 " layers, %" PRIu64 " admitted, %" PRIu64 " right (%.1f%%), %" PRIu64 " wrong, %" PRIu64
                " hits lost, %" PRIu64 " misses left, %.1f MiB ahead, %.1f MiB wasted, %" PRIu64 " fallbacks\n", tag, ua_st.layers, ua_st.admitted, ua_st.hit,
                ua_st.admitted > 0 ? 100.0*ua_st.hit/ua_st.admitted : 0.0, ua_st.waste, ua_st.lost, ua_st.miss_rest, ua_st.bytes/1048576.0,
                ua_st.waste_bytes/1048576.0, ua_st.fallback);
        }
        if (device_lookup) {
            stats st;
            for (const group & g : groups) {
                if (!g.dev_lookup || g.n_slots == 0 || g.state == nullptr || g.state->data == nullptr) {
                    continue;
                }
                uint64_t cnt[2] = { 0, 0 };
                ggml_backend_tensor_get(g.state, cnt, 2*sizeof(int32_t), sizeof(cnt));
                st.hits   += cnt[0];
                st.misses += cnt[1];
                st.bytes  += cnt[1]*g.expert_bytes;
            }
            log("device lookups (ubatch <= 32)", st);
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-q4xcache: %llu layer graphs of 2-%d token windows with the host path off overlapped their "
                "copies, %llu of them 5+ tokens (k19-bigverify, LLAMA_MOE_OVERLAP_PLAIN_MAX) (LLAMA_MOE_OVERLAP_SMALL_PLAIN=%d)\n", tag,
                (unsigned long long) n_ov_plain, (int) GGML_MOE_MAILBOX_MAX_TOKENS, (unsigned long long) n_ov_plain_wide, overlap_plain_on() ? 1 : 0);
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-g4pass: %llu overlap queries of 5-%d token windows on the single-graph mailbox took the copy "
                "overlap (LLAMA_MOE_G4PASS=%d)\n", tag, (unsigned long long) n_ov_wide_host, (int) GGML_MOE_MAILBOX_MAX_TOKENS, g4pass_on() ? 1 : 0);
            LLAMA_LOG_INFO("llama_moe_cache%s: k19-g4mid: %llu overlap queries of 5-%d token mailbox windows took the copy overlap in the "
                "%.1f%% cached band (LLAMA_MOE_G4MID=%d)\n", tag, (unsigned long long) n_ov_wide_mid, (int) GGML_MOE_MAILBOX_MAX_TOKENS,
                100.0*cache_frac, g4mid_on() ? 1 : 0);
            if (ext_on) {
                LLAMA_LOG_INFO("llama_moe_cache%s: k19-t4small decode extension: %lld grows, %lld parks, at most %.2f MiB above the fitted slots, "
                    "%.1f ms in all%s (LLAMA_MOE_DECODE_EXT=0 off)\n", tag, (long long) n_ext_grows, (long long) n_ext_parks,
                    ext_max_bytes/1048576.0, ext_us/1000.0, ext_fail > 0 ? ", some grows stopped short (no device memory)" : "");
            }
            if (getenv("LLAMA_MOE_CACHE_ADMIT_MARGIN") != nullptr) {
                LLAMA_LOG_INFO("llama_moe_cache%s: admission margin %s (+%s): %llu missing experts left to the host instead of replacing a resident\n", tag,
                    getenv("LLAMA_MOE_CACHE_ADMIT_MARGIN"), getenv("LLAMA_MOE_CACHE_ADMIT_ADD") ? getenv("LLAMA_MOE_CACHE_ADMIT_ADD") : "0",
                    (unsigned long long) admit_rejected());
            }
            if (tail_rank > 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: route tail skip (rank %d%s): %llu missing experts skipped in windows of at most %" PRId64 " tokens\n",
                    tag, tail_rank, tail_dflt ? ", default" : "", (unsigned long long) tail_skipped(), tail_max_tokens);
                if (tail_host_on()) {
                    LLAMA_LOG_INFO("llama_moe_cache%s: route tail skip in host lookups: %llu entries, %llu experts left out of the plan "
                        "(LLAMA_MOE_ROUTE_TAIL_SKIP_HOST=0 off)\n", tag, (unsigned long long) tail_host_skipped, (unsigned long long) tail_host_experts);
                }
            }
            if (host_prof_mode() > 0 && hprof.n > 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: host lookups (LLAMA_MOE_HOST_PROF=%d): %llu small lookups, sync %.1f ms, plan %.1f ms, uploads %.1f ms "
                    "(copies back issued %.1f, RAM uploads issued %.1f, commit %.1f, file reads + uploads %.1f for %llu experts), copies waited %.1f ms, "
                    "%llu lookups with tail skips\n", tag, host_prof_mode(), (unsigned long long) hprof.n, hprof.sync_us/1000.0, hprof.plan_us/1000.0,
                    hprof.up_us/1000.0, hprof.demote_us/1000.0, hprof.ram_us/1000.0, hprof.commit_us/1000.0, hprof.disk_us/1000.0,
                    (unsigned long long) hprof.n_disk, hprof.copy_us/1000.0, (unsigned long long) hprof.n_skip_lookups);
            }
            if (cm && cm_cal) {
                uint64_t c[3];
                cm_counters(c);
                const uint64_t n = c[0] + c[1] + c[2];
                int64_t n_on = 0;
                for (int64_t i = 1; i <= cm_max; ++i) {
                    n_on += cm_on[i] != 0;
                }
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: %llu missing experts computed on the host (%.2f MiB not fetched), hit rate with them = %.2f%%, "
                    "host path on for %" PRId64 " batch sizes now, %d decisions, last miss rate %.1f%%, ubatches on the host path: %zu small, %zu above %" PRId64 " tokens; "
                    "%d trials, small batches %s\n",
                    tag, (unsigned long long) c[2], (double) c[2]*(groups.empty() ? 0.0 : (double) groups[0].expert_bytes)/1048576.0, n > 0 ? 100.0*c[0]/n : 0.0,
                    n_on, (int) cm_gen, 100.0*cm_rate, cm_n_small, cm_n_prompt, max_batch, cmt_n, cm_small_ok ? "on" : "off");
            }
            if (hybrid_share < 1.0f) {
                uint64_t n_host = 0;
                for (const group & g : groups) {
                    if (g.counts == nullptr || g.counts->data == nullptr || !g.dev_lookup) {
                        continue;
                    }
                    uint64_t cnt = 0;
                    ggml_backend_tensor_get(g.counts, &cnt, (size_t) g.counts->ne[0]*(g.counts->ne[1] - 1)*sizeof(int32_t), sizeof(cnt));
                    n_host += cnt;
                }
                const size_t n = st.hits + st.misses + n_host;
                LLAMA_LOG_INFO("llama_moe_cache%s: hybrid: fetch share = %.3f, %llu missing experts computed on the host (not fetched), hit rate with them = %.2f%%\n", tag, hybrid_share, (unsigned long long) n_host, n > 0 ? 100.0*st.hits/n : 0.0);
            }
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu experts copied from the cache, %.2f MiB\n", tag, stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (stats_upload.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu experts uploaded, %.2f MiB\n", tag, stats_upload.misses, stats_upload.bytes/1024.0/1024.0);
        }
        if (n_large_ubatches > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu ubatches, copies per ubatch: %.1f from the cache, %.1f uploads, in %.1f batched calls\n", tag, n_large_ubatches, (double) stats_copy.calls/n_large_ubatches, (double) stats_upload.calls/n_large_ubatches, (double) n_batches/n_large_ubatches);
        }
        if (stats_seed.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu experts admitted to the cache, %.2f MiB\n", tag, stats_seed.misses, stats_seed.bytes/1024.0/1024.0);
        }
        if (n_prompt_inplace + n_prompt_fallback > 0 || prompt_hits + prompt_misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches in place: %zu layer copies (%zu admitting), %zu experts read from the slots, %zu uploaded "
                "(%.2f MiB), %zu ring waits, %zu seeded layers, %zu seeds uploaded late\n", tag, n_prompt_inplace, n_prompt_fallback, prompt_hits, prompt_misses, prompt_bytes/1024.0/1024.0, n_prompt_ring_waits, prompt_seeds, n_prompt_seed_left);
        }
        if (devseed_disabled > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu device seed plans not run (expert products not on the cache device)\n", tag, devseed_disabled);
        }
        if (devseed_graphs > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu ubatches seeded on the device, %zu experts admitted, %.2f MiB\n", tag,
                devseed_graphs, devseed_admitted, devseed_bytes/1024.0/1024.0);
        }
        if (devseed_replays > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: %zu device seeds replayed after merged layer-major graphs: %.2f MiB uploaded again, %.2f ms (uploads %.2f ms)\n", tag,
                devseed_replays, devseed_replay_bytes/1024.0/1024.0, devseed_replay_us/1000.0, devseed_replay_up_us/1000.0);
        }
        if (seed_unstaged > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu expert tensor seeds skipped (planned expert not staged)\n", tag, seed_unstaged);
        }
        if (seed_skipped > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu prompt ends not seeded (the seeding graph does not fit in the compute buffer)\n", tag, seed_skipped);
        }
        if (n_lends > 0) {
            LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu graphs used lent slots (at most %d slots lent), %zu experts refilled after them, %.2f MiB (%zu batched fill calls)\n", tag, n_lends, lend_max, stats_refill.misses, stats_refill.bytes/1024.0/1024.0, n_fill_batches);
            if (n_refill_routed > 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: large batches: %zu refills took the experts most used by the end of a prompt (LLAMA_MOE_CACHE_REFILL_ROUTED=1)\n", tag,
                    n_refill_routed);
            }
        }
        if (seed_check) {
            LLAMA_LOG_INFO("llama_moe_cache%s: seed check total: %zu expert copies verified, %zu mismatches\n", tag, seed_checked, seed_mismatches);
        }
        if (host) {
            const auto & hs = host->st;
            const uint64_t n = hst.hits + hst.misses;
            LLAMA_LOG_INFO("llama_moe_cache: host tier: %s, device misses: %llu RAM hits, %llu RAM misses, RAM hit rate = %.2f%%, uploaded from RAM = %.2f MiB, "
                    "read into RAM = %llu (%.2f MiB), demoted = %llu (%.2f MiB, %llu without a slot), promoted = %llu, admitted from prompts = %llu, %llu evictions kept in RAM\n",
                    host->exclusive() ? "exclusive" : "inclusive", (unsigned long long) hst.hits, (unsigned long long) hst.misses, n ? 100.0*hst.hits/n : 0.0,
                    hst.up_bytes/1048576.0, (unsigned long long) hst.read, hst.read_bytes/1048576.0, (unsigned long long) hst.demoted, hst.demote_bytes/1048576.0,
                    (unsigned long long) hst.demote_skip, (unsigned long long) hst.promoted, (unsigned long long) hst.admitted, (unsigned long long) hst.demote_kept);
            if (d2hs.calls > 0 || !d2h_on) {
                LLAMA_LOG_INFO("llama_moe_cache: copies back on a side stream: %llu host lookups, %llu chunks, %llu uploads from RAM of which %llu "
                        "issued before the last copy back finished (LLAMA_MOE_D2H_STREAM=%d, %d chunks)\n", (unsigned long long) d2hs.calls,
                        (unsigned long long) d2hs.chunks, (unsigned long long) d2hs.uploads, (unsigned long long) d2hs.early, d2h_on ? 1 : 0, d2h_nchunk);
            }
            LLAMA_LOG_INFO("llama_moe_cache: copy back of evicted experts: %llu done (%.2f MiB), %llu skipped as RAM held them (%.2f MiB), "
                    "%llu forced over the RAM copy (%.2f MiB) (LLAMA_MOE_COPYBACK_SKIP=%d)\n",
                    (unsigned long long) hst.demoted, hst.demote_bytes/1048576.0, (unsigned long long) hst.demote_kept, hst.demote_kept_bytes/1048576.0,
                    (unsigned long long) hst.demote_forced, hst.demote_forced_bytes/1048576.0, copyback_skip ? 1 : 0);
            LLAMA_LOG_INFO("llama_moe_cache: k19-l4held: %llu seed evictions copied back asynchronously (LLAMA_MOE_SHADOW_ASYNC=%d); RAM tier layer quota %s: "
                    "%llu refused, %llu same-layer, %llu ranked, %llu slack; %zu device-prefill RAM copies marked as duplicates\n",
                    (unsigned long long) n_shadow_async, shadow_async() ? 1 : 0, host->layer_quota_on() ? "on" : "off",
                    (unsigned long long) host->quota_refused, (unsigned long long) host->quota_same, (unsigned long long) host->quota_ranked,
                    (unsigned long long) host->quota_slack, n_prefill_dup);
            if (host->exclusive()) {
                LLAMA_LOG_INFO("llama_moe_cache: k19-l4fill: refill of freed RAM slots %s (LLAMA_MOE_HOST_FILL=%d, %d speculative reads at once): %llu prompts "
                        "counted, %llu rounds (%llu at the server's idle point), %llu read (%.1f MiB), %llu done (%llu duplicates), %llu cancelled, %llu failed, %llu waited for (%.1f ms), "
                        "%llu unmarked RAM copies of device experts marked as duplicates\n",
                        fill_mode != 0 ? "on" : "off", fill_mode, fill_parallel, (unsigned long long) fls.prompts, (unsigned long long) fls.rounds,
                        (unsigned long long) fls.idle_rounds,
                        (unsigned long long) fls.issued, fls.issued_bytes/1048576.0, (unsigned long long) fls.done, (unsigned long long) fls.dup,
                        (unsigned long long) fls.cancelled, (unsigned long long) fls.failed, (unsigned long long) fls.waited, fls.waited_us/1000.0,
                        (unsigned long long) fls.dup_marked);
                fill_diag("exit");
            }
            LLAMA_LOG_INFO("llama_moe_cache: host tier: %d slots, peak %zu held, %llu evicted, %llu without a slot, %llu waits, pinned = %.2f MiB of a %.2f MiB budget, "
                    "registered expert buffers = %.2f MiB, MemAvailable %.1f MiB at start, %.1f MiB now\n",
                    host->n_slots(), hs.peak_held, (unsigned long long) hs.evicted, (unsigned long long) hs.no_slot, (unsigned long long) hs.waits,
                    host->pinned_bytes()/1048576.0, host->budget_bytes()/1048576.0, hst.registered/1048576.0, hst.avail_init/1048576.0, meminfo_avail()/1048576.0);
            LLAMA_LOG_INFO("llama_moe_cache: host tier: prompt seeds evicted %llu RAM experts: %llu by the finished-layer rule, %llu of layers the pass had not reached "
                    "(LLAMA_MOE_HOST_EVICT_DONE=%d)\n", (unsigned long long) hs.stream_evicted, (unsigned long long) hs.done_evicted,
                    (unsigned long long) hs.ahead_evicted, host->evict_done_on() ? 1 : 0);
            LLAMA_LOG_INFO("llama_moe_cache: host tier: %.1f ms pinning slot memory, %llu allocation failures, %.2f MiB pageable (%llu chunks past the pin cap or a failed pin)\n",
                    hs.alloc_us/1000.0, (unsigned long long) hs.alloc_fail, host->pageable_bytes()/1048576.0, (unsigned long long) hs.pin_fallbacks);
            if (!hprof_ticket.empty()) {
                hprof_request_log("at exit");
                LLAMA_LOG_INFO("llama_moe_cache: host profile (k19-ramprof2): %llu dropped for requests' reads, %llu copied at the first batch (%.1f MiB), %llu background reads issued (%.1f MiB) in %llu rounds, "
                        "%llu done, %llu waited for by a lookup (%.1f ms), %llu cancelled by a large batch, %llu without a free slot, %llu failed, %llu duplicates of "
                        "device experts; guesses: %llu marked, %llu used, %llu replaced unused, %zu still unused\n",
                        (unsigned long long) hps.dropped, (unsigned long long) hps.load_copied, hps.load_copied_bytes/1048576.0, (unsigned long long) hps.issued, hps.issued_bytes/1048576.0,
                        (unsigned long long) hps.rounds, (unsigned long long) hps.done, (unsigned long long) hps.waited, hps.waited_us/1000.0,
                        (unsigned long long) hps.cancelled, (unsigned long long) hps.no_slot, (unsigned long long) hps.failed, (unsigned long long) hps.dup,
                        (unsigned long long) hs.spec_set, (unsigned long long) hs.spec_used, (unsigned long long) hs.spec_taken, host->n_spec_slots());
            }
            if (ahead_depth > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: host tier reads ahead (%d layers): %llu layer predictions, %llu experts predicted, %llu already held, %llu read ahead = %.2f MiB, "
                        "%llu used, %llu waited for (%.1f ms), %llu without a slot, recall %.2f%% (%llu of %llu routed), predicting %.1f ms, %llu cancelled, %llu over the in-flight cap\n", ahead_depth,
                        (unsigned long long) ast.layers, (unsigned long long) ast.predicted, (unsigned long long) ast.held, (unsigned long long) ast.issued,
                        ast.bytes/1048576.0, (unsigned long long) ast.used, (unsigned long long) ast.late, ast.waited_us/1000.0, (unsigned long long) ast.no_slot,
                        ast.routed ? 100.0*ast.recalled/ast.routed : 0.0, (unsigned long long) ast.recalled, (unsigned long long) ast.routed, ast.predict_us/1000.0,
                        (unsigned long long) ast.cancelled, (unsigned long long) ast.capped);
            }
            if (host_verify) {
                LLAMA_LOG_INFO("llama_moe_cache: host tier VERIFY %llu regions, %.1f MiB, %llu mismatches\n", (unsigned long long) hst.verify_regions,
                        hst.verify_bytes/1048576.0, (unsigned long long) hst.verify_mismatch);
            }
        } else if (hst.registered > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: registered expert buffers = %.2f MiB\n", hst.registered/1048576.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        layer_set set, llama_weight_store * ws, int64_t host_mib) :
    pimpl(new impl(model, backends, bufts, size, set, ws, host_mib)) {
}

bool llama_moe_cache::mtp_enabled() {
    const char * env = getenv("LLAMA_MOE_CACHE_MTP");
    return env == nullptr || atoi(env) != 0;
}

llama_moe_cache::layer_set llama_moe_cache::select_layers(const llama_model & model, bool ctx_mtp) {
    const auto & hp = model.hparams;
    // as the KV cache filters of the NextN layers: not for a router layer the trunk attends or a model made only of NextN layers
    if (!mtp_enabled() || hp.n_layer_nextn == 0 || hp.n_layer() == 0 || hp.router_layer >= 0) {
        return layer_set::all;
    }
    return ctx_mtp ? layer_set::nextn : layer_set::trunk;
}

size_t llama_moe_cache::budget(const llama_model & model, size_t size, layer_set set) {
    if (set == layer_set::all) {
        // no layer the cache can hold (host experts on a CUDA / ROCm / MUSA device, llama_moe_cache_layer_cacheable): Metal, Vulkan,
        // CPU only, or every expert on the devices. No cache, as for the trunk / NextN sets below, instead of a constructor that throws
        // and fails the context ("requires a CUDA / ROCm / MUSA GPU backend")
        if (size > 0 && llama_moe_cache_set_bytes(model, layer_set::all) == 0) {
            LLAMA_LOG_WARN("%s: a MoE cache of %.2f MiB has no layer to hold (it caches host experts of layers on CUDA / ROCm / MUSA "
                "devices only): running without it\n", __func__, size/1048576.0);
            return 0;
        }
        return size;
    }
    const size_t bytes_trunk = llama_moe_cache_set_bytes(model, layer_set::trunk);
    const size_t bytes_nextn = llama_moe_cache_set_bytes(model, layer_set::nextn);
    if (bytes_nextn == 0 || bytes_trunk == 0) {
        // one of the caches has nothing to hold, the other gets the whole budget
        return (set == layer_set::nextn ? bytes_nextn : bytes_trunk) > 0 ? size : 0;
    }

    // the memory to hold all experts of a set, with room for the small per-expert tensors, the lookup state and the alignment
    auto need = [](size_t bytes) { return bytes + bytes/32 + (size_t) 64*1024*1024; };

    // the MTP cache gets the share of its experts in the budget (the share its layers had in a single cache), or what the main cache
    // leaves once it holds all of its experts, at most what holds all of the MTP experts
    size_t mtp = (size_t) ((double) size*bytes_nextn/(bytes_trunk + bytes_nextn));
    if (size > need(bytes_trunk)) {
        mtp = std::max(mtp, size - need(bytes_trunk));
    }
    mtp = std::min({ mtp, need(bytes_nextn), size });
    if (const char * env = getenv("LLAMA_MOE_CACHE_MTP_MIB")) {
        mtp = std::min(size, (size_t) std::max(0, atoi(env))*1024*1024);
    }
    return set == layer_set::nextn ? mtp : size - mtp;
}

double llama_moe_cache::trial_frac_single() {
    static const double v = [] {
        const char * e = getenv("LLAMA_MOE_CPU_MISS_TRIAL_FRAC_SINGLE");
        return e != nullptr ? std::max(0.0, atof(e)) : 0.50;
    }();
    return v;
}

size_t llama_moe_cache::host_bytes(const llama_model & model) {
    return llama_moe_cache_set_bytes(model, layer_set::trunk);
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

int llama_moe_cache::hybrid_mode() const {
    return pimpl->hybrid_mode();
}

ggml_tensor * llama_moe_cache::hybrid_assign(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_ubatch, bool & host) const {
    return pimpl->hybrid_assign(ctx, il, ids, n_ubatch, host);
}

ggml_tensor * llama_moe_cache::hybrid_fetch_experts(ggml_context * ctx, int32_t il, ggml_tensor * res, bool overlap) const {
    return pimpl->hybrid_fetch_experts(ctx, il, res, overlap);
}

bool llama_moe_cache::prefetch_ok(int32_t il_prev, int32_t il, int64_t n_ubatch) const {
    return pimpl->prefetch_ok(il_prev, il, n_ubatch);
}

ggml_tensor * llama_moe_cache::prefetch(ggml_context * ctx, int32_t il, ggml_tensor * logits, ggml_tensor * dep, int64_t n_ubatch) const {
    return pimpl->prefetch(ctx, il, logits, dep, n_ubatch);
}

bool llama_moe_cache::overlap(int64_t n_ubatch) const {
    return pimpl->overlap_batch(n_ubatch);
}

bool llama_moe_cache::overlap_small(int64_t n_ubatch) const {
    return pimpl->overlap_small_batch(n_ubatch);
}

bool llama_moe_cache::overlap_plain(int32_t il, int64_t n_ubatch) const {
    bool plain = false;
    return pimpl->hybrid_layer(il, n_ubatch, &plain) && plain;
}

bool llama_moe_cache::fetch_defer(int64_t n_ubatch) const {
    return pimpl->fetch_defer(n_ubatch);
}

bool llama_moe_cache::hybrid_active() const {
    return pimpl->hybrid_active();
}

void llama_moe_cache::set_prompt_batch(bool prompt, int64_t n_tokens_all) {
    pimpl->cm_prompt_now = prompt;
    pimpl->cm_prompt_all = prompt ? n_tokens_all : 0;
}

bool llama_moe_cache::host_computes(int64_t n_tokens) const {
    if (pimpl->force_large) {
        return false;
    }
    if (pimpl->cm) {
        return pimpl->cm_share_of(n_tokens) < 1.0f;
    }
    return !pimpl->force_large && n_tokens <= n_small_max && pimpl->hybrid_active();
}

bool llama_moe_cache::cpu_miss_enabled() const {
    return pimpl->cm;
}

bool llama_moe_cache::host_never() const {
    return !pimpl->hybrid && !pimpl->cm;
}

bool llama_moe_cache::host_split_frozen() const {
    return pimpl->split_frozen();
}

void llama_moe_cache::tail_skip(ggml_tensor * op, int32_t il, int64_t n_tokens) const {
    pimpl->tail_skip(op, il, n_tokens);
}

void llama_moe_cache::tail_skip_host(ggml_tensor * get_rows, int32_t il, int64_t n_tokens) const {
    pimpl->tail_skip_host(get_rows, il, n_tokens);
}

bool llama_moe_cache::tail_skip_off_exact() const {
    static const bool keep = [] {
        const char * e = getenv("LLAMA_MOE_ROUTE_TAIL_SKIP_EXACT");
        return e != nullptr && atoi(e) != 0;
    }();
    return pimpl->tail_dflt && !keep;
}

bool llama_moe_cache::freeze_small() {
    return pimpl->cm_freeze();
}

bool llama_moe_cache::pin_small() {
    return pimpl->cm_freeze(true);
}

bool llama_moe_cache::pin_small_set(bool on) {
    return pimpl->cm_pin_set(on);
}

bool llama_moe_cache::request_begin() {
    pimpl->stats_request_snapshot();
    return pimpl->cm_request_begin();
}

bool llama_moe_cache::request_resets() const {
    return pimpl->cm_request_resets();
}

bool llama_moe_cache::cpu_miss_drop_slow() {
    return pimpl->cm_drop_slow();
}

int llama_moe_cache::host_mode() const {
    return pimpl->cm ? 2 : pimpl->hybrid ? 1 : 0;
}

bool llama_moe_cache::cpu_miss_tick_sync(int64_t n_tokens) const {
    return pimpl->cm_tick_reads(n_tokens);
}

void llama_moe_cache::cpu_miss_tick(int64_t n_tokens) {
    // k15-fv diagnostic LLAMA_MOE_ROUND_LOG=1 (off by default; it reads the device counters before every ubatch of at most 16 tokens,
    // i.e. it waits for the previous one): hits / fetched / host experts and uploaded MiB of the previous such ubatch
    static const bool round_log = getenv("LLAMA_MOE_ROUND_LOG") != nullptr && atoi(getenv("LLAMA_MOE_ROUND_LOG")) != 0;
    if (round_log && n_tokens <= 16) {
        static int64_t  prev_n = -1;
        static uint64_t prev_c[3] = { 0, 0, 0 };
        uint64_t c[3];
        pimpl->cm_counters(c);
        size_t eb = 0;
        for (const auto & g : pimpl->groups) {
            eb = std::max(eb, g.expert_bytes);
        }
        if (prev_n > 0) {
            LLAMA_LOG_INFO("moe_round: n %" PRId64 " hits %" PRIu64 " fetched %" PRIu64 " host %" PRIu64 " h2d %.1f MiB\n", prev_n,
                c[0] - prev_c[0], c[1] - prev_c[1], c[2] - prev_c[2], (double) (c[1] - prev_c[1])*eb/1048576.0);
        }
        prev_n = n_tokens;
        memcpy(prev_c, c, sizeof(c));
    }
    pimpl->cm_tick(n_tokens);
}

void llama_moe_cache::cpu_miss_pause(bool on) {
    if (pimpl->cm_paused == on) {
        return;
    }
    pimpl->cm_paused = on;
    if (!on) {
        // the next counted batch starts a fresh window: nothing the paused batches moved is in it
        pimpl->cm_have0 = false;
        pimpl->cm_nb    = 0;
    }
}

bool llama_moe_cache::small_batch(int64_t n_tokens) const {
    return !pimpl->force_large && (n_tokens <= n_small_max || pimpl->cm_serves(n_tokens));
}

void llama_moe_cache::set_force_large(bool on) {
    pimpl->force_large = on;
}

bool llama_moe_cache::force_large() const {
    return pimpl->force_large;
}

bool llama_moe_cache::recal_pending() const {
    return pimpl->cm && pimpl->cm_cal && pimpl->cm_recal_pending && !pimpl->pw_quiet;
}

bool llama_moe_cache::recal(ggml_backend_t backend_cpu, bool (*abort)(void *), void * abort_data) {
    pimpl->cm_recal_pending  = false;
    pimpl->cm_cal_abort      = abort;
    pimpl->cm_cal_abort_data = abort_data;
    pimpl->cm_calibrate(backend_cpu);
    pimpl->cm_cal_abort      = nullptr;
    pimpl->cm_cal_abort_data = nullptr;
    if (pimpl->cm_cal_aborted) {
        pimpl->cm_cal_aborted   = false;
        pimpl->cm_recal_pending = true;
        return false;
    }
    return true;
}

bool llama_moe_cache::hybrid_probe_pending() const {
    return pimpl->hybrid && pimpl->hybrid_probe && !pimpl->pw_quiet;
}

void llama_moe_cache::hybrid_probe(ggml_backend_t backend_cpu) {
    pimpl->probe(backend_cpu);
}

void llama_moe_cache::hybrid_set_threads(int32_t n_threads, bool quiet) {
    pimpl->single_threads = n_threads;
    pimpl->set_threads(n_threads, quiet);
    pimpl->cm_cal_log_file(quiet);
}

ggml_tensor * llama_moe_cache::single_publish(ggml_context * ctx, int32_t il, ggml_tensor * host_ids, ggml_tensor * x,
        ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down) const {
    if (!pimpl->single_usable(host_ids->ne[1]) || host_ids->type != GGML_TYPE_I32 || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) {
        return nullptr;
    }
    auto * mb = pimpl->single_mailbox(il, (int) host_ids->ne[0], (int) x->ne[0], up, gate, down);
    if (mb == nullptr) {
        return nullptr;
    }
    return ggml_moe_mailbox_publish(ctx, host_ids, x, mb->mdev, mb->ctl, il);
}

void llama_moe_cache::single_note_form(int32_t il, bool ok) const {
    pimpl->single_note_form(il, ok);
}

ggml_tensor * llama_moe_cache::single_merge(ggml_context * ctx, ggml_tensor * dev, ggml_tensor * pub) const {
    // no timeout: the merge waits for the host (see single_timeout_ms). LLAMA_MOE_SINGLE_MERGE_TIMEOUT_MS=N (diagnostic): a merge whose
    // batch the host thread neither computes nor picked up within N ms gives up (that ubatch keeps its device rows only, so its output is
    // not exact) and the next graph turns the path off (single_kick): a hang turns into a logged failure
    static const int merge_timeout_ms = [] {
        const char * e = getenv("LLAMA_MOE_SINGLE_MERGE_TIMEOUT_MS");
        return e != nullptr ? std::max(0, atoi(e)) : 0;
    }();
    return ggml_moe_mailbox_merge(ctx, dev, pub, merge_timeout_ms);
}

void llama_moe_cache::single_kick(int64_t n_tokens) {
    pimpl->single_kick(n_tokens);
}

bool llama_moe_cache::hybrid_fixed() const {
    return pimpl->hybrid && pimpl->hybrid_fixed;
}

double llama_moe_cache::cache_frac() const {
    return pimpl->cache_frac;
}

double llama_moe_cache::host_total_bytes() const {
    return pimpl->host_total;
}

double llama_moe_cache::release_frac() const {
    size_t load  = 0;
    size_t floor = 0;
    bool   any   = false;
    for (size_t id = 0; id < pimpl->devices.size(); ++id) {
        const size_t l = pimpl->slot_bytes_load((int32_t) id);
        const size_t f = pimpl->devices[id].shrink_floor;
        load  += l;
        floor += f == SIZE_MAX ? l : std::min(f, l);
        any    = any || f != SIZE_MAX;
    }
    if (!any || load == 0 || pimpl->cache_frac < 0.0) {
        return pimpl->cache_frac;
    }
    return pimpl->cache_frac*(double) floor/(double) load;
}

void llama_moe_cache::hybrid_reset() {
    pimpl->hybrid_reset();
}

bool llama_moe_cache::core_wanted() const {
    return pimpl->core_mode != 0;
}

void llama_moe_cache::core_request() {
    pimpl->core_request();
}

size_t llama_moe_cache::core_dropped_bytes() const {
    return pimpl->core_dropped;
}

bool llama_moe_cache::overlap_trial_pending() const {
    return pimpl->ov_pending() && !pimpl->pw_quiet;
}

bool llama_moe_cache::overlap_tick_sync() const {
    return pimpl->ov_rule ? pimpl->ov_rule_reads() : (pimpl->ov_done >= 0 || pimpl->ov_wait == 1);
}

void llama_moe_cache::overlap_tick(int64_t n_tokens) {
    pimpl->ov_tick(n_tokens);
}

bool llama_moe_cache::hybrid_window_pending() const {
    if (pimpl->pw_quiet) {
        return false;
    }
    if (pimpl->cm) {
        return pimpl->cmt_pending();
    }
    return pimpl->hybrid && (pimpl->win_done >= 0 || pimpl->win_wait >= 0);
}

bool llama_moe_cache::hybrid_tick_sync() const {
    return !pimpl->cm; // CPU misses: each timed ubatch is synchronized at its end (see hybrid_tick_end)
}

bool llama_moe_cache::prompt_auto_pending() const {
    return pimpl->pt_pending() && !pimpl->pw_quiet;
}

void llama_moe_cache::prompt_auto_tick(int64_t n_tokens) {
    pimpl->pt_tick(n_tokens);
}

bool llama_moe_cache::hybrid_tick_timed() const {
    return pimpl->cm && (pimpl->cmt_timed() || pimpl->pt_timed());
}

void llama_moe_cache::hybrid_tick_end() {
    pimpl->cmt_end();
    pimpl->pt_end();
}

void llama_moe_cache::hybrid_tick(int64_t n_tokens) {
    if (pimpl->cm) {
        pimpl->cmt_tick(n_tokens);
        return;
    }
    pimpl->tick(n_tokens);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, int64_t n_ubatch) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used, n_ubatch);
}

ggml_tensor * llama_moe_cache::lookup(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_ubatch) const {
    return pimpl->lookup(ctx, il, ids, n_ubatch);
}

void llama_moe_cache::upload_ahead_begin(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * ids, const std::vector<int32_t> & counts) {
    pimpl->ua_begin(backend, w, ids, counts);
}

bool llama_moe_cache::upload_ahead_pending(const ggml_tensor * w) const {
    return pimpl->ua_pending(w);
}

void llama_moe_cache::upload_ahead_mark(ggml_backend_t backend, const ggml_tensor * w) {
    pimpl->ua_mark(backend, w);
}

void llama_moe_cache::upload_ahead_wait(ggml_backend_t backend, const ggml_tensor * w) {
    pimpl->ua_wait(backend, w);
}

void llama_moe_cache::upload_ahead_end(ggml_backend_t backend, const ggml_tensor * w) {
    pimpl->ua_end(backend, w);
}

void llama_moe_cache::prepare(int64_t n_tokens) {
    pimpl->prepare(n_tokens);
}

ggml_tensor * llama_moe_cache::get_resident(const ggml_tensor * t) const {
    return pimpl->get_resident(t);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::devseed_init(int64_t min_tokens) {
    return pimpl->devseed_init(min_tokens);
}

ggml_tensor * llama_moe_cache::devseed_plan(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_tokens) const {
    return pimpl->devseed_plan(ctx, il, ids, n_tokens);
}

ggml_tensor * llama_moe_cache::devseed_bank(const ggml_tensor * w, int32_t & il) const {
    return pimpl->devseed_bank(w, il);
}

void llama_moe_cache::devseed_prepare(ggml_cgraph * gf, bool enable) {
    pimpl->devseed_prepare(gf, enable);
}

void llama_moe_cache::devseed_end(ggml_cgraph * gf, bool ok) {
    pimpl->devseed_end(gf, ok);
}

bool llama_moe_cache::devseed_replay(const std::vector<std::pair<int32_t, ggml_tensor *>> & rec, int64_t n_tokens, size_t & bytes, double & ms) {
    return pimpl->devseed_replay(rec, n_tokens, bytes, ms);
}

int llama_moe_cache::seed_mode() {
    // 3 (a lagged seed of each layer of a prefetched ubatch) was removed: its readiness tracking was not sound
    const char * env = getenv("LLAMA_MOE_CACHE_SEED");
    const int mode = env != nullptr ? atoi(env) : 2;
    return mode == 3 ? 2 : mode;
}

void llama_moe_cache::set_graph(ggml_cgraph * gf) {
    pimpl->set_graph(gf);
}

bool llama_moe_cache::is_slot_map(const ggml_tensor * t) const {
    return pimpl->layer_of.find(t) != pimpl->layer_of.end();
}

void llama_moe_cache::set_weight_store(llama_weight_store * ws) {
    // the RAM tier reads through the store: it moves with it (and is dropped without one); its reads ahead complete first
    if (pimpl->host && pimpl->wstore != nullptr && pimpl->wstore != ws) {
        pimpl->hprof_stop();
        pimpl->fill_stop();
        pimpl->ahead_drain();
        pimpl->ahead_il = -1;
        pimpl->wstore->set_tier(nullptr);
    }
    pimpl->wstore = ws;
    if (pimpl->host && ws != nullptr) {
        ws->set_tier(&pimpl->tier);
    } else if (pimpl->host) {
        pimpl->host.reset();
        pimpl->host_group.assign(pimpl->groups.size(), false);
    }
}

int32_t llama_moe_cache::prompt_layer(const ggml_tensor * t) const {
    return pimpl->prompt_layer(t);
}

std::vector<ggml_tensor *> llama_moe_cache::prompt_experts(int32_t il) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() ? pimpl->layers[il].experts : std::vector<ggml_tensor *>();
}

bool llama_moe_cache::prompt_copy(ggml_backend_t backend, int32_t il, ggml_tensor * dst, const std::vector<int32_t> * counts) {
    return pimpl->prompt_copy(backend, il, dst, counts);
}

void llama_moe_cache::prompt_flush() {
    pimpl->prompt_flush();
}

void llama_moe_cache::prompt_inplace_block(bool blocked) {
    pimpl->prompt_inplace_blocked = blocked;
}

void llama_moe_cache::prompt_ubatch(bool ok, bool seed_prev) {
    pimpl->rank_clear(); // k19-l4held: the routing of the previous ubatch no longer ranks the offers
    // LLAMA_MOE_PROMPT_INPLACE_SEED=0: a ubatch that seeds from an earlier routing keeps the staging copies
    static const bool seed_on = [] { const char * e = getenv("LLAMA_MOE_PROMPT_INPLACE_SEED"); return e == nullptr || atoi(e) != 0; }();
    pimpl->prompt_ubatch_ok   = ok && (!seed_prev || seed_on);
    pimpl->prompt_seed_counts = ok && seed_prev && seed_on;
}

bool llama_moe_cache::prompt_ubatch_on() const {
    return pimpl->prompt_ubatch_ok;
}

bool llama_moe_cache::prompt_inplace_on() const {
    return pimpl->prompt_inplace_env;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

void llama_moe_cache::prompt_layer_read(ggml_backend_t backend, const ggml_tensor * w, const std::vector<bool> & used) {
    pimpl->prompt_layer_read(backend, w, used);
}

void llama_moe_cache::prompt_diag_time(int64_t sync_us, int64_t up_us) {
    pimpl->prompt_diag_time(sync_us, up_us);
}

void llama_moe_cache::prompt_diag_extra(int k, int64_t us) {
    pimpl->prompt_diag_extra(k, us);
}

bool llama_moe_cache::prompt_rank_copy(const ggml_tensor * w, std::vector<bool> & used) {
    return pimpl->prompt_rank_copy(w, used);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last, std::vector<ggml_backend_copy_region> * regions) {
    return pimpl->copy_experts(backend, w, dst, e, last, regions);
}

void llama_moe_cache::count_batch(const ggml_tensor * w, const std::vector<ggml_backend_copy_region> & regions) {
    pimpl->count_batch(w, regions);
}

void llama_moe_cache::count_large_ubatch() {
    pimpl->n_large_ubatches++;
}

void llama_moe_cache::count_upload(const ggml_tensor * w, size_t n_experts, size_t bytes) {
    pimpl->count_upload(w, n_experts, bytes);
}

void llama_moe_cache::seed(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * dst, const std::vector<int32_t> & counts, const std::vector<bool> & staged) {
    pimpl->seed(backend, w, dst, counts, staged);
}

void llama_moe_cache::lend(ggml_backend_t backend, ggml_backend_sched_lend_copy * copies, int n_copies, int depth, bool dry, int64_t n_tokens) {
    pimpl->lend(backend, copies, n_copies, depth, dry, n_tokens);
}

void llama_moe_cache::seed_unavailable() {
    if (pimpl->prefill_deferred) {
        pimpl->prefill_deferred = false;
        pimpl->prefill();
    }
}

int32_t llama_moe_cache::layer_of_weight(const ggml_tensor * w) const {
    return pimpl->layer_of_weight(w);
}

bool llama_moe_cache::resident_fill_wanted() const {
    return pimpl->resident_fill_wanted();
}

void llama_moe_cache::resident_fill() {
    pimpl->resident_fill();
}

void llama_moe_cache::set_resident_counts(int32_t il, const std::vector<int32_t> & counts) {
    pimpl->set_resident_counts(il, counts);
}

bool llama_moe_cache::has_resident_counts() const {
    return pimpl->has_resident_counts();
}

bool llama_moe_cache::resident_union() const {
    // resident mode alone (>= LLAMA_MOE_RESIDENT_MIN_FRAC of the experts in RAM + device slots): with the host reserve the tier can
    // hold the complement without spare room (G4 fndl 48 GiB, full 6 GiB reserve: 15045 + 9253 slots for 24576 experts), and prompt
    // seeds there demoted 151 GB and read 12 GB from disk per 8k prompt
    return pimpl->host && pimpl->host->exclusive() && pimpl->resident_mode;
}

void llama_moe_cache::seeds_off() {
    pimpl->seed_own = false;
}

void llama_moe_cache::prompts_do_not_seed() {
    pimpl->seed_own = false;
    seed_unavailable();
}

void llama_moe_cache::count_seed_skipped() {
    pimpl->seed_skipped++;
}

bool llama_moe_cache::lends() const {
    return pimpl->lend_mode != 0;
}

bool llama_moe_cache::lends_on(ggml_backend_t backend) const {
    return pimpl->lends_on(backend);
}

int32_t llama_moe_cache::graph_lend_capacity(ggml_backend_t backend, int32_t * n_slots, size_t * expert_bytes) const {
    return pimpl->graph_lend_capacity(backend, n_slots, expert_bytes);
}

size_t llama_moe_cache::lend_capacity_bytes(ggml_backend_t backend) const {
    size_t eb = 0;
    const int32_t n = pimpl->graph_lend_capacity(backend, nullptr, &eb);
    return (size_t) n*eb;
}

bool llama_moe_cache::release() {
    bool any = false;
    for (size_t id = 0; id < pimpl->devices.size(); ++id) {
        any = pimpl->release((int32_t) id) || any;
    }
    return any;
}

void llama_moe_cache::restore() {
    for (size_t id = 0; id < pimpl->devices.size(); ++id) {
        pimpl->restore((int32_t) id);
    }
}

void llama_moe_cache::set_restore_soft(bool on) {
    pimpl->restore_soft = on;
}

size_t llama_moe_cache::shrink(ggml_backend_dev_t dev, size_t bytes) {
    const int32_t id = pimpl->dev_id(dev);
    return id >= 0 ? pimpl->shrink(id, bytes) : 0;
}

bool llama_moe_cache::ext_on() const {
    return pimpl->ext_on;
}

size_t llama_moe_cache::ext_grow(ggml_backend_dev_t dev, size_t bytes) {
    const int32_t id = pimpl->dev_id(dev);
    return id >= 0 ? pimpl->ext_grow(id, bytes) : 0;
}

size_t llama_moe_cache::ext_park() {
    size_t n = 0;
    for (size_t id = 0; id < pimpl->devices.size(); ++id) {
        n += pimpl->ext_park((int32_t) id);
    }
    return n;
}

size_t llama_moe_cache::ext_bytes() const {
    size_t n = 0;
    for (size_t id = 0; id < pimpl->devices.size(); ++id) {
        n += pimpl->ext_bytes((int32_t) id);
    }
    return n;
}

bool llama_moe_cache::can_shrink(ggml_backend_dev_t dev) const {
    const int32_t id = pimpl->dev_id(dev);
    return id >= 0 && pimpl->shrink_on && pimpl->devices[id].bank_lazy && pimpl->devices[id].buf_bank;
}

void llama_moe_cache::set_shrink_floor(ggml_backend_dev_t dev, size_t bytes) {
    const int32_t id = pimpl->dev_id(dev);
    if (id >= 0) {
        pimpl->set_shrink_floor(id, bytes);
    }
}

size_t llama_moe_cache::slot_bytes(ggml_backend_dev_t dev) const {
    const int32_t id = pimpl->dev_id(dev);
    return id >= 0 ? pimpl->slot_bytes(id) : 0;
}

bool llama_moe_cache::take_graphs_stale() {
    const bool res = pimpl->graphs_stale;
    pimpl->graphs_stale = false;
    return res;
}

void llama_moe_cache::set_refill_routed_once(bool on) {
    pimpl->refill_routed_once = on;
}

bool llama_moe_cache::released() const {
    for (const auto & d : pimpl->devices) {
        if (d.released) {
            return true;
        }
    }
    return false;
}

void llama_moe_cache::set_graph_lend(bool on) {
    pimpl->graph_lend_scope = on;
}

bool llama_moe_cache::lends_graph_any() const {
    static const bool graph_only_env = getenv("LLAMA_MOE_LM_LEND_GRAPH_ONLY") == nullptr || atoi(getenv("LLAMA_MOE_LM_LEND_GRAPH_ONLY")) != 0;
    if (lends_on_any()) {
        return true;
    }
    if (!graph_only_env || !pimpl->lend_auto) {
        return false;
    }
    for (const auto & d : pimpl->devices) {
        if (d.buf_bank && !d.released && pimpl->lend_mode != 0 && d.lend_auto == 0) {
            return true;
        }
    }
    return false;
}

bool llama_moe_cache::lends_on_any() const {
    for (const auto & d : pimpl->devices) {
        if (d.buf_bank && pimpl->lend_mode != 0 && d.lend_auto == 1) {
            return true;
        }
    }
    return false;
}

bool llama_moe_cache::lends_graph() const {
    return pimpl->lend_mode == 2;
}

bool llama_moe_cache::lends_graph_to(int64_t n_tokens) const {
    if (pimpl->lend_mode != 2) {
        return false;
    }
    // the largest group of each device lends (see impl::lend): a batch its device lookup serves does not borrow
    for (size_t id = 0; id < pimpl->devices.size(); ++id) {
        const impl::group * gl = nullptr;
        for (const auto & g : pimpl->groups) {
            if (g.id == (int32_t) id && g.n_slots > 0 && (gl == nullptr || (size_t) g.n_slots*g.expert_bytes > (size_t) gl->n_slots*gl->expert_bytes)) {
                gl = &g;
            }
        }
        if (gl != nullptr && pimpl->devices[id].buf_bank && !pimpl->use_device(*gl, n_tokens)) {
            return true;
        }
    }
    return false;
}

size_t llama_moe_cache::lend_save_need(ggml_backend_t backend) const {
    for (const auto & d : pimpl->devices) {
        if (d.backend == backend) {
            return d.dry_save;
        }
    }
    return 0;
}

size_t llama_moe_cache::lend_need_max() const {
    size_t need = 0;
    for (const auto & d : pimpl->devices) {
        need = std::max(need, d.dry_need);
    }
    return need;
}

void llama_moe_cache::set_mem_reserve(ggml_backend_t backend, size_t bytes) {
    for (auto & d : pimpl->devices) {
        if (d.backend == backend) {
            d.mem_reserve = bytes;
        }
    }
}

double llama_moe_cache::cached_frac() const {
    return pimpl->cache_frac;
}

double llama_moe_cache::host_uncached_bytes() const {
    return pimpl->host_uncached;
}

int64_t llama_moe_cache::n_tokens_cached() const {
    int64_t n = n_small_max;
    for (const auto & g : pimpl->groups) {
        if (g.n_slots == 0) {
            continue;
        }
        while (n > 0 && std::min<int64_t>(n*pimpl->n_expert_used, g.ref[0]->ne[2]) > g.n_slots) {
            n--;
        }
    }
    return n;
}

bool llama_moe_cache::seed_check() const {
    return pimpl->seed_check;
}

void llama_moe_cache::seed_verify() {
    pimpl->seed_verify();
}

void llama_moe_cache::seed_end() {
    pimpl->seed_end();
}

void llama_moe_cache::set_seed_counts(const ggml_tensor * w, const std::vector<int32_t> & counts) {
    pimpl->set_seed_counts(w, counts);
}

bool llama_moe_cache::refills_routed() const {
    return pimpl->refill_routed;
}

bool llama_moe_cache::fill_wants_counts() const {
    return pimpl->fill_active();
}

bool llama_moe_cache::autoprof_wants_counts() const {
    return pimpl->ap_rec;
}

void llama_moe_cache::autoprof_counts_begin() {
    pimpl->ap_counts_begin();
}

void llama_moe_cache::autoprof_counts(const ggml_tensor * w, const std::vector<int32_t> & counts) {
    pimpl->ap_counts(w, counts);
}

void llama_moe_cache::fill_stop() {
    pimpl->fill_stop();
}

void llama_moe_cache::fill_idle() {
    pimpl->fill_idle();
}

void llama_moe_cache::fill_counts_begin() {
    pimpl->fill_counts_begin();
}

void llama_moe_cache::fill_counts(const ggml_tensor * w, const std::vector<int32_t> & counts) {
    pimpl->fill_counts(w, counts);
}

void llama_moe_cache::set_refill_counts(const ggml_tensor * w, const std::vector<int32_t> & counts) {
    pimpl->set_refill_counts(w, counts);
}

const std::vector<int32_t> * llama_moe_cache::get_seed_counts(const ggml_tensor * w) const {
    return pimpl->get_seed_counts(w);
}

void llama_moe_cache::clear_seed_counts() {
    pimpl->clear_seed_counts();
}

bool llama_moe_cache::has_seed_counts() const {
    return pimpl->has_seed_counts();
}

bool llama_moe_cache::prewarm_begin() {
    return pimpl->prewarm_begin();
}

bool llama_moe_cache::prewarm_end() {
    return pimpl->prewarm_end();
}

void llama_moe_cache::fingerprint(const char * when) {
    pimpl->fingerprint(when);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf_bank) {
            res[ggml_backend_buffer_get_type(d.buf_bank.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

//
// k19-ramprof: the routing profile file
//

bool llama_moe_profile_load(const std::string & path, const std::vector<int32_t> & n_expert, std::vector<llama_moe_profile_entry> & out) {
    out.clear();
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    int  ver     = 0;
    long n_layer = 0;
    bool ok = fscanf(f, "moe-cache-profile %d %ld", &ver, &n_layer) == 2 && ver == 1 && n_layer == (long) n_expert.size();
    std::set<int64_t> seen;
    llama_moe_profile_entry p;
    int r = 0;
    while (ok && (r = fscanf(f, "%d %d %f", &p.il, &p.e, &p.heat)) == 3) {
        if (p.il < 0 || p.il >= (int32_t) n_expert.size() || p.e < 0 || p.e >= n_expert[p.il] || !std::isfinite(p.heat) ||
                !seen.insert((int64_t) p.il << 32 | p.e).second) {
            ok = false;
            break;
        }
        out.push_back(p);
    }
    // a line cut short (a partial last line) is a truncated file, not the end of the list
    ok = ok && r == EOF && !ferror(f);
    fclose(f);
    if (!ok || out.empty()) {
        out.clear();
        return false;
    }
    return true;
}

bool llama_moe_profile_write(const std::string & path, size_t n_layer, const std::vector<llama_moe_profile_entry> & entries) {
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "w");
    if (f == nullptr) {
        return false;
    }
    bool ok = fprintf(f, "moe-cache-profile 1 %zu\n", n_layer) > 0;
    for (const llama_moe_profile_entry & p : entries) {
        ok = ok && fprintf(f, "%d %d %.6g\n", p.il, p.e, p.heat) > 0;
    }
    ok = fclose(f) == 0 && ok;
    ok = ok && rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) {
        remove(tmp.c_str());
    }
    return ok;
}

std::vector<llama_moe_profile_entry> llama_moe_profile_ram_order(const std::vector<llama_moe_profile_entry> & entries) {
    std::vector<llama_moe_profile_entry> v = entries;
    std::stable_sort(v.begin(), v.end(), [](const llama_moe_profile_entry & a, const llama_moe_profile_entry & b) { return a.heat > b.heat; });
    return v;
}

std::vector<std::vector<int32_t>> llama_moe_profile_pick(const std::vector<llama_moe_profile_entry> & entries, const std::vector<int32_t> & layers,
        int32_t n_expert, int32_t n_avail, size_t * n_prof, size_t * n_pad) {
    const size_t n_layer = layers.size();
    std::vector<std::vector<uint8_t>> take(n_layer, std::vector<uint8_t>(std::max<int32_t>(0, n_expert), 0));
    std::map<int32_t, size_t> pos;
    for (size_t i = 0; i < n_layer; ++i) {
        pos.emplace(layers[i], i);
    }
    size_t np = 0;
    size_t nd = 0;
    int32_t n_take = 0;
    for (const llama_moe_profile_entry & p : llama_moe_profile_ram_order(entries)) {
        if (n_take >= n_avail) {
            break;
        }
        const auto it = pos.find(p.il);
        if (it == pos.end() || p.e < 0 || p.e >= n_expert || !(p.heat > 0.0f) || take[it->second][p.e]) {
            continue;
        }
        take[it->second][p.e] = 1;
        n_take++;
        np++;
    }
    std::vector<int32_t> next(n_layer, 0);
    for (bool any = true; n_take < n_avail && any;) {
        any = false;
        for (size_t i = 0; i < n_layer && n_take < n_avail; ++i) {
            int32_t & e = next[i];
            while (e < n_expert && take[i][e]) {
                e++;
            }
            if (e < n_expert) {
                take[i][e] = 1;
                e++;
                n_take++;
                nd++;
                any = true;
            }
        }
    }
    std::vector<std::vector<int32_t>> out(n_layer);
    for (size_t i = 0; i < n_layer; ++i) {
        for (int32_t e = 0; e < n_expert; ++e) {
            if (take[i][e]) {
                out[i].push_back(e);
            }
        }
    }
    if (n_prof != nullptr) {
        *n_prof = np;
    }
    if (n_pad != nullptr) {
        *n_pad = nd;
    }
    return out;
}

size_t llama_moe_profile_reserve(llama_moe_host_cache & host, const std::vector<llama_moe_profile_entry> & order,
        const std::function<bool(int32_t il, int32_t e)> & want, const std::function<void(size_t i, int32_t s)> & take) {
    size_t n = 0;
    std::set<int32_t> full; // layers whose layout has no free slot left (a reservation there failed)
    for (size_t i = 0; i < order.size(); ++i) {
        const int32_t il = order[i].il;
        const int32_t e  = order[i].e;
        if (!host.caches(il) || full.count(il) > 0 || host.contains(il, e) || !want(il, e)) {
            continue;
        }
        const int32_t s = host.reserve(il, e, false, 0, false);
        if (s < 0) {
            full.insert(il);
            continue;
        }
        take(i, s);
        n++;
    }
    return n;
}
