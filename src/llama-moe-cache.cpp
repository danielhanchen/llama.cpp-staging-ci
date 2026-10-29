// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "llama-moe-cache.h"
#include "llama-sysmem.h"

#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-weight-store.h"
#include "llama-moe-host-cache.h"

#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-moe-mailbox.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cinttypes>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

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
        std::fill(key_of.begin(), key_of.end(), -1);
        std::fill(scores.begin(), scores.end(), 0);
        w = w_init;
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = n_slots > 0 ? 0 : -1;
        tail = n_slots - 1;
        gen++;
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

    // empty slot s (it stays in the list, k13-small release)
    void evict(int32_t s) {
        if (key_of[s] >= 0) {
            slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            key_of[s] = -1;
            gen++;
        }
    }

    // lend the last n slots (their experts are evicted), the slots no longer lent come back empty as the least recently used
    void set_lent(int32_t n) {
        GGML_ASSERT(n >= 0 && n < n_slots);
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

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
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
        if (s == head) {
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
        for (int pass = 0; pass < 2; ++pass) {
            for (int32_t s = 0; s < n_slots - n_lent; ++s) {
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

// the experts of layer il that a cache can hold: all of them in host memory, the layer assigned to a GPU
static bool llama_moe_cache_layer_cacheable(const llama_model & model, size_t il, std::vector<ggml_tensor *> & experts) {
    experts = llama_moe_cache_layer_experts(model.layers[il]);
    if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
        return false;
    }
    ggml_backend_dev_t dev = model.dev_layer((int) il);
    if (dev == nullptr) {
        return false;
    }
    const auto type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
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
    std::vector<llama_weight_store::region> ws_regions;

    // RAM tier (see llama_moe_host_cache): experts of the groups streamed from the model files, behind the device slots
    // --moe-host-cache-mib (-1 auto), LLAMA_MOE_HOST_CACHE=0 off, LLAMA_MOE_HOST_CACHE_POLICY=exclusive|inclusive
    std::unique_ptr<llama_moe_host_cache> host;
    std::vector<bool> host_group; // [group] served by the RAM tier (host lookups, never mapped for the device)
    // exclusive with spare room (budget + device slots >= streamed experts): promoted experts keep their RAM copy as a duplicate (the
    // first to go) and RAM misses fill free slots; without spare room promotions free the RAM slot and misses go to the device only
    // (V16 / R29 and the T4 at R31: keeping duplicates there cost the prompts their RAM admissions, decode -53% / -32%)
    bool host_spare = false;
    bool host_verify = false;     // LLAMA_WEIGHT_STREAM_VERIFY=1: every upload from the tier is compared with the mapping
    struct host_stats {
        uint64_t hits = 0, misses = 0;        // device misses of small batches (and refills) found / not found in RAM
        uint64_t read = 0, read_bytes = 0;    // RAM misses read into a slot (inclusive)
        uint64_t demoted = 0, demote_bytes = 0, demote_skip = 0; // device evictions copied back into RAM (exclusive)
        uint64_t demote_kept = 0;             // device evictions RAM still held a copy of (no copy back)
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
    bool     cm_recal_pending = false; // (timed only between requests: 2 x ~40 ms inside a reply cost chat decode 2% on G4)
    int      cm_nb    = 0;     // batches in the window
    int      cm_win   = cm_win1;
    bool     cm_have0 = false;
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
    struct prof_entry {
        int32_t il;
        int32_t e;
        float   heat;
    };
    std::string profile_file;      // load
    std::string profile_save_file; // save at exit
    std::vector<prof_entry> profile;
    bool   profile_loaded = false;
    double profile_seed   = 1.0;
    size_t profile_filled = 0;

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
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
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
            // LLAMA_MOE_CPU_MISS (unset or 1 = on, 0 = off): with LLAMA_MOE_CACHE_HYBRID and LLAMA_MOE_CACHE_FETCH unset, the host computes
            // the missing experts per batch size where a calibration at the first small batch says it is faster (see cm_calibrate, cm_rule)
            {
                const char * env_cm = getenv("LLAMA_MOE_CPU_MISS");
                cm = hybrid && env == nullptr && env_f == nullptr && (env_cm == nullptr || atoi(env_cm) != 0);
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
                ggml_tensor * bank = ggml_new_tensor_3d(ctx_bank, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
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
                    ggml_tensor * cached = ggml_view_3d(ctx_bank, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
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
                g.state = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8 + 3*(int64_t) g.n_slots);
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
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf_bank.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx_bank.get(), d.buft));
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
            if (!profile_file.empty() && profile_read() && prefill_profile()) {
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
            const int32_t n_slots  = g.n_slots - g.lru.n_lent;
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
            }
        }
        if (n_fill > 0) {
            prefilled_dev = true;
            LLAMA_LOG_INFO("%s: MoE cache prefilled with %zu experts, %.2f MiB (LLAMA_MOE_CACHE_PREFILL=0 to disable)\n", __func__, n_fill, n_bytes/1024.0/1024.0);
        }
    }

    // the ranked (layer, expert) list of profile_file, see prof_entry; false (and a warning) if it is missing or does not match
    bool profile_read() {
        FILE * f = fopen(profile_file.c_str(), "r");
        if (f == nullptr) {
            LLAMA_LOG_WARN("llama_moe_cache%s: LLAMA_MOE_CACHE_PROFILE: no profile %s, the cache starts empty\n", tag, profile_file.c_str());
            return false;
        }
        int ver = 0;
        long n_layer = 0;
        bool ok = fscanf(f, "moe-cache-profile %d %ld", &ver, &n_layer) == 2 && ver == 1 && n_layer == (long) layers.size();
        profile.clear();
        std::set<int64_t> seen;
        prof_entry p;
        while (ok && fscanf(f, "%d %d %f", &p.il, &p.e, &p.heat) == 3) {
            if (p.il < 0 || p.il >= (int32_t) layers.size() || layers[p.il].ig < 0 || p.e < 0 || p.e >= groups[layers[p.il].ig].lru.n_expert ||
                    !seen.insert((int64_t) p.il << 32 | p.e).second) {
                ok = false;
                break;
            }
            profile.push_back(p);
        }
        fclose(f);
        if (!ok || profile.empty()) {
            LLAMA_LOG_WARN("llama_moe_cache%s: LLAMA_MOE_CACHE_PROFILE: %s does not match this model's cache, ignored\n", tag, profile_file.c_str());
            profile.clear();
            return false;
        }
        return true;
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
    void profile_save() {
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
        const std::string tmp = profile_save_file + ".tmp";
        FILE * f = fopen(tmp.c_str(), "w");
        if (f == nullptr) {
            LLAMA_LOG_WARN("llama_moe_cache%s: LLAMA_MOE_CACHE_PROFILE_SAVE: cannot write %s\n", tag, tmp.c_str());
            return;
        }
        fprintf(f, "moe-cache-profile 1 %zu\n", layers.size());
        for (const auto * v : { &out_res, &out_rest }) {
            for (const prof_entry & p : *v) {
                fprintf(f, "%d %d %.6g\n", p.il, p.e, p.heat);
            }
        }
        const bool ok = fclose(f) == 0 && rename(tmp.c_str(), profile_save_file.c_str()) == 0;
        LLAMA_LOG_INFO("llama_moe_cache%s: MoE cache profile %s %s: %zu resident + %zu other used experts\n", tag, ok ? "saved to" : "NOT saved to",
                profile_save_file.c_str(), out_res.size(), out_rest.size());
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
        std::mutex           reg_mtx; // the owners; held by the thread while it serves them
        std::vector<impl *>  owners;
        std::thread          thr;
        std::atomic<bool>    stop{false};
        ggml_backend_ptr     cpu;
        int                  threads = 1;
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
            single_form_state = -1;
        } else if (single_form_state == 0) {
            ggml_backend_dev_t dev = il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0 ? ggml_backend_get_device(backend(il)) : nullptr;
            ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
            const bool alloc = reg != nullptr && ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_mailbox_alloc") != nullptr &&
                (!single_auto || strcmp(ggml_backend_reg_name(reg), "CUDA") == 0);
            single_form_state = alloc ? 1 : -1;
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
    const int single_max_tok = [] {
        const char * e = getenv("LLAMA_MOE_SINGLE_MAX_TOKENS");
        return std::max(1, std::min<int>(GGML_MOE_MAILBOX_MAX_TOKENS, e != nullptr ? atoi(e) : 4));
    }();

    bool single_usable(int64_t n_tokens) const {
        return single_req && !single_off && !single_no_alloc && n_tokens >= 1 && n_tokens <= single_max_tok;
    }

    bool single_register() {
        if (single_reg) {
            return true;
        }
        single_hub_t & h = single_hub();
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
            if (set_n != nullptr) {
                set_n(h.cpu.get(), h.threads);
            }
            const char * ef = getenv("LLAMA_MOE_SINGLE_FUSED");
            if (ef == nullptr || atoi(ef) != 0) {
                h.fused       = (decltype(h.fused))       ggml_backend_reg_get_proc_address(creg, "ggml_cpu_moe_host_ffn");
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
            "(one host thread for every cache of the process), slow-batch warning above %d ms (merges never time out), %s\n", tag, single_max_tok, h.threads,
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
            const uint32_t req = __atomic_load_n(&hdr->request, __ATOMIC_ACQUIRE);
            if (req == mb.done || (stall_after >= 0 && (int64_t) single_batches >= stall_after)) {
                continue;
            }
            // LLAMA_MOE_SINGLE_DEBUG_DELAY_PHASE=1 (test): the delay before the pickup, as if the host thread were not scheduled
            single_debug_delay(false);
            work = true;
            bool ok = false;
            const int n_tokens = hdr->n_tokens;
            single_plan * p = nullptr;
            // picked up: the merge on the device waits for this batch without a timeout from now on (see moe_mailbox_merge)
            __atomic_store_n(&hdr->started, req, __ATOMIC_RELEASE);
            if (!single_failed.load(std::memory_order_relaxed) && n_tokens >= 1 && n_tokens <= GGML_MOE_MAILBOX_MAX_TOKENS &&
                    single_plan_get(mb, n_tokens, p)) {
                const int64_t t0 = single_now_us();
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
                single_us += dt;
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
                __atomic_store_n(&hdr->error, 2u, __ATOMIC_RELEASE);
            }
            single_batches++;
            mb.done = req;
            __atomic_store_n(&hdr->completed, req, __ATOMIC_RELEASE);
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
                    __atomic_store_n(&((ggml_moe_mailbox_hdr *) mbp->host)->busy, on ? 1u : 0u, __ATOMIC_RELEASE);
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
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#else
                std::this_thread::yield();
#endif
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
            failed = failed || (mb && __atomic_load_n(&((ggml_moe_mailbox_hdr *) mb->host)->error, __ATOMIC_ACQUIRE) != 0);
        }
        if (failed) {
            LLAMA_LOG_ERROR("llama_moe_cache%s: single-graph host misses failed (a merge waited %d ms or the host thread failed): "
                "the split path from now on\n", tag, single_timeout_ms);
            single_off = true;
            cm_gen++;
            return;
        }
        if (n_tokens < 1 || n_tokens > single_max_tok) {
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
                h.cpu.reset();
            }
            single_reg = false;
            if (single_batches > 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: single-graph host misses: %" PRIu64 " host batches, %" PRIu64 " expert rows, %.1f us per batch "
                    "(%.2f us per row), %" PRIu64 " above 1 ms, at most %.1f ms\n", tag, single_batches, single_rows, (double) single_us/single_batches,
                    single_rows > 0 ? (double) single_us/single_rows : 0.0, single_slow, single_max_us/1e3);
                LLAMA_LOG_INFO("llama_moe_cache%s: single-graph host misses: %" PRIu64 " of the process's host batches in one region "
                    "(LLAMA_MOE_SINGLE_FUSED)\n", tag, single_hub().fused_batches);
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
        ua_shutdown();
        log_stats();
        ua_free();
        for (const auto & [unmap, buf] : mapped) {
            unmap(buf);
        }
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
        // k18-lowram: the tier never plans into a host reserve kept for the rest of the machine (a Studio user's desktop, browser and the
        // backend live there): LLAMA_MOE_HOST_RESERVE_MIB (default 6144, 0 = the auto budget as before). Only this path reaches it: the tier
        // exists where the weight store streams the experts, so every host that pins them loads as before. G4 Flash-Next UD-IQ4_XS 32 GB
        // at 48 GiB RAM without it: 39.7 GB of slots and MemAvailable down to 2.5 GiB during the run
        size_t reserve = (size_t) 6144 << 20;
        if (const char * e = getenv("LLAMA_MOE_HOST_RESERVE_MIB")) {
            reserve = (size_t) std::max<long long>(0, atoll(e)) << 20;
        }
        const size_t avail_plan = avail > reserve ? avail - reserve : 0;
        llama_moe_host_plan plan = llama_moe_host_decide(e_bytes, v_bytes, mapped, avail_plan, token_bytes, host_mib, policy_env, force);
        // where the whole reserve would cost the union of RAM and the device slots (the experts the device does not hold no longer fit),
        // but the budget without it holds them: the budget is that complement + 1 GiB, so resident mode stays and the reserve gets the
        // rest. G4 Flash-Next 32 GB at 48 GiB: the full 6 GiB reserve left 35.4 GB for 35.4 GB of complement, resident mode was lost and
        // the 8k / 16k cells read 12.0 / 18.3 GB from disk (TTFT 85 / 147 s, decode 25 / 34 tok/s against 63 / 96 s, 55 / 59 tok/s)
        if (reserve > 0 && host_mib < 0 && e_bytes > v_bytes) {
            const size_t need = e_bytes - v_bytes + ((size_t) 1 << 30);
            const llama_moe_host_plan full = llama_moe_host_decide(e_bytes, v_bytes, mapped, avail, token_bytes, host_mib, policy_env, force);
            if (full.on && full.budget >= need && (!plan.on || plan.budget < need)) {
                plan = full;
                plan.budget = need;
                plan.spare = need + v_bytes >= e_bytes;
                LLAMA_LOG_INFO("%s: MoE host cache: the %.1f MiB host reserve would leave the experts the device slots do not hold outside RAM: "
                        "the budget is that complement + 1 GiB (%.1f MiB), the reserve gets the rest\n", __func__, reserve/1048576.0, need/1048576.0);
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
                    prefill();
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
        prefill_resident();
        LLAMA_LOG_INFO("%s: MoE host cache on: %s, budget %.1f MiB (%s) for %.1f MiB of streamed experts (of %.1f MiB mapped), %.1f MiB in device slots, "
                "the experts of %.1f tokens, MemAvailable %.1f MiB, pinned on first use (%s)%s (LLAMA_MOE_HOST_CACHE=0 off)\n", __func__,
                policy == LLAMA_MOE_HOST_EXCLUSIVE ? (host_spare ? "exclusive (spare room: duplicates kept)" : "exclusive") : "inclusive", budget/1048576.0, plan.why,
                e_bytes/1048576.0, mapped/1048576.0, v_bytes/1048576.0, plan.tokens, avail/1048576.0,
                host->describe().c_str(), host_verify ? ", VERIFY" : "");
    }

    // the experts whose pages are in the page cache (e.g. read while loading) are copied into free slots: pinning the slots would
    // otherwise evict those pages and the first prompt would read them again (LLAMA_MOE_HOST_CACHE_PREFILL=0 off)
    void prefill_resident() {
#if defined(__linux__) || defined(__APPLE__)
        const char * env_pf = getenv("LLAMA_MOE_HOST_CACHE_PREFILL");
        if (env_pf != nullptr && atoi(env_pf) == 0) {
            return;
        }
        // LLAMA_MOE_HOST_CACHE_PREFILL=2: also read the experts that are not in the page cache, layer by layer from the model files
        // (readahead of each layer's expert tensors first), skipping the experts the device slots hold: with the device cache prefilled
        // (LLAMA_MOE_CACHE_PREFILL=1) the first prompt then reads nothing from the files when RAM + VRAM hold every expert (low-RAM boxes)
        const bool from_file = (env_pf != nullptr && atoi(env_pf) >= 2) || resident_mode;
        const int64_t t0 = ggml_time_us();
        size_t n_file = 0;
        const size_t pg = (size_t) std::max(4096L, sysconf(_SC_PAGESIZE));
        std::vector<unsigned char> vec;
        std::vector<std::vector<bool>> res;
        size_t n_fill = 0;
        size_t n_bytes = 0;
        bool full = false;
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
                    const int32_t s = host->reserve(il, e, false);
                    if (s < 0) {
                        // the slots of this layer's expert layout are used up; the file fill goes on with the layers of the other
                        // layouts (Flash-Next IQ2_XS has three: stopping at the first full one left 3137 RAM slots empty)
                        full = !from_file;
                        break;
                    }
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const ggml_tensor * w = l.experts[ip];
                        host->set_lead(s, (int) ip, 0);
                        memcpy(host->area(s, (int) ip), (const uint8_t *) w->data + (size_t) e*w->nb[2], w->nb[2]);
                        n_bytes += w->nb[2];
                    }
                    host->publish(s);
                    n_fill++;
                    n_file += file ? 1 : 0;
                }
            }
        }
        hst.prefilled = n_fill;
        LLAMA_LOG_INFO("%s: MoE host cache: %zu experts (%.1f MiB) copied from the page cache into RAM slots in %.1f ms, %zu of them read from the files"
                " (LLAMA_MOE_HOST_CACHE_PREFILL=0 off, 2 also reads the experts that are not in the page cache)\n",
                __func__, n_fill, n_bytes/1048576.0, (ggml_time_us() - t0)/1000.0, n_file);
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
            const int32_t sl = e >= src->ne[2] ? -1 : lease ? host->held(il, e) : host->find(il, e, false);
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
                sl = host->reserve(il, e, false);
            }
            if (sl >= 0 && host->filling(sl) && host->fill_piece(sl, ip, a, p + (o - off), e1 - o)) {
                hst.admitted++;
            }
            o = e1;
        }
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
                const int32_t h = host->held(il, e);
                if (h >= 0) {
                    host->set_dup(h, false);
                    hst.demote_kept++;
                }
                continue;
            }
            const int32_t r = host->reserve(il, e);
            if (r < 0) {
                hst.demote_skip++;
                continue;
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
            if (ev != nullptr && host->exclusive()) {
                demote(backend, g, *ev);
            }
            for (const auto & f : fl) {
                const int32_t s = host->find(il, f.expert);
                if (s >= 0) {
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
            host->commit(backend);
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
        if (ws && !wstore->upload_batch(backend, ws_regions.data(), ws_regions.size())) {
            for (const auto & r : ws_regions) {
                ggml_backend_tensor_set_async(backend, r.dst, (const uint8_t *) r.src->data + r.src_off, r.dst_off, r.size);
            }
        }
        flush_fill_regions(backend);
        return bytes;
    }

    // the device slots that are lent (or overwritten by a seed) lose their experts: the RAM tier keeps a copy (synchronous, once per prompt)
    void shadow_slots(ggml_backend_t backend, group & g, const std::vector<std::pair<int32_t, int32_t>> & ev) {
        if (!host || ev.empty()) {
            return;
        }
        // the copies into the device slots may be on the compute backend or the copy queue of the device: both done
        ggml_backend_synchronize(backend);
        if (devices[g.id].backend != backend) {
            ggml_backend_synchronize(devices[g.id].backend);
        }
        const int32_t n_expert = g.lru.n_expert;
        for (const auto & [s, key] : ev) {
            const int32_t il = key / n_expert;
            const int32_t e  = key % n_expert;
            if (!host_layer(il)) {
                continue;
            }
            if (host->contains(il, e)) {
                const int32_t h = host->held(il, e);
                if (h >= 0) {
                    host->set_dup(h, false);
                    hst.demote_kept++;
                }
                continue;
            }
            const int32_t r = host->reserve(il, e);
            if (r < 0) {
                hst.demote_skip++;
                continue;
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
        std::vector<ggml_tensor *> banks = g.banks;
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
                    if (it == dev_base.end()) {
                        void * base = ggml_backend_buffer_get_base(t->buffer);
                        bool registered = false;
                        void * dev_ptr = host_map(dev, base, ggml_backend_buffer_get_size(t->buffer), &registered);
                        if (registered) {
                            mapped.emplace_back(host_unmap, base);
                            added.emplace_back(dev, t->buffer);
                            hst.registered += ggml_backend_buffer_get_size(t->buffer);
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
    }

    bool use_device(const group & g, int64_t n_tokens) const {
        return !force_large && g.dev_lookup && g.n_slots > 0 && n_tokens > 0 && (n_tokens <= max_batch || cm_serves(n_tokens)) &&
            std::min<int64_t>(n_tokens*n_expert_used, g.ref[0]->ne[2]) <= g.n_slots;
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
        for (int64_t n = 1; n <= cm_max; ++n) {
            // batches above the frozen sizes (short prompts) keep the plain cache: their calibrated host path is timing-driven, and
            // a frozen cache gives the same text in every run
            cm_on[n] = n <= std::min<int64_t>({ cm_frozen_max, max_batch }) ? 1 : 0;
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
    bool cm_request_begin() {
        static const int on = [] { const char * e = getenv("LLAMA_MOE_CPU_MISS_PER_REQUEST"); return e != nullptr ? atoi(e) : 0; }();
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
    bool cm_prompt_off() const {
        static const bool pin_dev = [] {
            const char * e = getenv("LLAMA_MOE_PIN_PROMPT_DEV");
            return e == nullptr || atoi(e) != 0;
        }();
        static const bool all_dev = [] {
            const char * e = getenv("LLAMA_MOE_CPU_MISS_PROMPT_DEV");
            return e != nullptr && atoi(e) != 0;
        }();
        return cm_prompt_now && cm && ((cm_frozen && cm_frozen_fetch && pin_dev) || all_dev);
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

    // the host computes the misses of a batch of n_tokens: the fetch share, 1 = every missing expert is fetched
    float cm_share_of(int64_t n_tokens) const {
        if (cm_prompt_off()) {
            return 1.0f;
        }
        if (cm_frozen && !cm_frozen_fetch && cm && cm_cal && n_tokens >= 1 && n_tokens <= std::min<int64_t>({ cm_frozen_max, max_batch, cm_max })) {
            return 0.0f;
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
        if (!use_device(g, n_ubatch) || ids->ne[1] > n_ubatch || ids->ne[0] > n_expert_used || l.host_dev.size() != g.banks.size()) {
            return nullptr;
        }
        std::vector<ggml_tensor *> banks = g.banks;
        std::vector<const void *>  host  = l.host_dev;
        ggml_tensor * op = ggml_moe_cache_lookup(ctx, ids, g.map, g.state, banks.data(), host.data(), (int) banks.size(), il);
        if (g.scores != nullptr) {
            ggml_moe_cache_set_scores(op, g.scores, g.lru.grow_q16, il == g.layers[0]);
        }
        return op;
    }

    // the layer can use the hybrid lookup for a ubatch of this size
    bool hybrid_layer(int32_t il, int64_t n_ubatch) const {
        if (force_large || !(hybrid || overlap_batch(n_ubatch)) || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return false;
        }
        // CPU misses off for this batch size: the graph of k9 (plain lookup, or the fetch overlap of single tokens)
        if (cm && cm_share_of(n_ubatch) >= 1.0f && !overlap_batch(n_ubatch)) {
            return false;
        }
        const group & g = groups[layers[il].ig];
        return g.counts != nullptr && use_device(g, n_ubatch) && layers[il].host_dev.size() == g.banks.size();
    }

    // while the hybrid path is off but may be used (probe, trials), the lookup fetches every missing expert (share 1, same results
    // as ggml_moe_cache_lookup) and keeps the use counters up to date
    ggml_tensor * hybrid_assign(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_ubatch, bool & host) const {
        host = false;
        if (!hybrid_layer(il, n_ubatch) || ids->ne[1] > n_ubatch || ids->ne[0] > n_expert_used) {
            return nullptr;
        }
        float share = 1.0f;
        if (cm) {
            share = cm_share_of(n_ubatch);
            host  = share < 1.0f;
        } else {
            host  = hybrid_active();
            share = host ? hybrid_share : 1.0f;
        }
        const group & g = groups[layers[il].ig];
        ggml_tensor * op = ggml_moe_cache_assign(ctx, ids, g.map, g.state, g.counts, share, il);
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
    // nor computed; its contribution is zero and the other experts keep their router weights. On by default (7) where Strata has it on by
    // default: CUDA (and HIP RDNA4, decided by the backend), unless every expert is on the device (no cache, nothing to skip), 90% or
    // more of the experts are on the device (resident layers + cache slots, LLAMA_MOE_ROUTE_TAIL_SKIP_MAX_CACHED=<percent>), or the
    // experts are NVFP4. The graph also leaves it off for windows of several sequences (Strata: off with --batch slots), prompt ubatches
    // (not every row an output), more than 8 tokens (LLAMA_MOE_ROUTE_TAIL_SKIP_MAX_TOKENS; Strata's verify window) and experts with
    // biases (the bias would still be added). LLAMA_MOE_ROUTE_TAIL_SKIP=0: off (the exact path); =N: rank N, also where the default is off.
    // Changes the output (Strata: KL 0.003-0.006, top-1 ~98%, docs/DETAILS.md)
    void tail_skip_init(const llama_model & model, layer_set set) {
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
            tail_rank = rank;
            tail_dflt = true;
            LLAMA_LOG_INFO("%s: route tail skip: rank %d by default (%.1f%% of the experts on the device): a missing expert that every token "
                "of a decode / verify window of at most %" PRId64 " tokens routes at rank %d or lower is neither fetched nor computed "
                "(changes the output; LLAMA_MOE_ROUTE_TAIL_SKIP=0 off)\n", __func__, rank, 100.0*frac, tail_max_tokens, rank);
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

    uint64_t tail_skipped() const {
        uint64_t n = 0;
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

    ggml_tensor * hybrid_fetch_experts(ggml_context * ctx, int32_t il, ggml_tensor * res, bool ov) const {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        std::vector<ggml_tensor *> banks = g.banks;
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
            for (ggml_tensor * bank : g.banks) {
                ggml_backend_tensor_memset(bank, 0, 0, ggml_nbytes(bank));
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
            return 3 + (ov_live ? 4 : 0) + 8*(int) (cm_gen & 0xfffff) + (cm_prompt_off() ? (1 << 28) : 0) + (short_live() ? (1 << 29) : 0);
        }
        // a single-graph failure (single_off) rebuilds the graphs without their mailboxes: a reused graph would keep merging from a
        // mailbox whose control block says failed (wrong rows in every later ubatch with a pinned fetch share, LLAMA_MOE_CACHE_FETCH)
        return (!hybrid ? 0 : hybrid_active() ? 2 : 1) + (ov_live ? 4 : 0) + (single_off ? 8 : 0);
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
                v[i] = (int32_t) ((next + i) % n_expert);
            }
            next += n;
            for (int r = 1; r < rows; ++r) {
                std::copy(v.begin(), v.begin() + n, v.begin() + (size_t) r*n);
            }
            ggml_backend_tensor_set(ids, v.data(), 0, v.size()*sizeof(int32_t));
        };
        auto median = [](std::vector<int64_t> v) {
            std::sort(v.begin(), v.end());
            return (double) v[v.size()/2];
        };
        auto timed = [&](ggml_cgraph * gf) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_graph_compute(backend_cpu, gf);
            return ggml_time_us() - t0;
        };

        // a busy host gives inconsistent times (more experts not slower, an implausible bandwidth): measured again, at most 3 times
        double d_a = 0.0, d_b = 0.0, d_t = 0.0, slope = 0.0;
        bool ok = false;
        int attempt = 0;
        for (; attempt < 3 && !ok; ++attempt) {
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
            d_a = median(ta);
            d_b = median(tb);
            d_t = median(tt);
            slope = kb > ka ? (d_b - d_a)/(kb - ka) : d_a/ka;
            const double gbs = slope > 0.0 ? g.expert_bytes/slope/1e3 : 0.0;
            ok = slope > 0.0 && gbs >= 0.2 && gbs <= 400.0 && d_t >= 0.9*d_a;
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
                const int64_t e = next++ % n_expert;
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
                cm_dt = t1_new < cm_t1 ? dt_new : cm_dt;
                cm_t1 = std::min(cm_t1, t1_new);
            } else if (rel <= cm_recal_tol) {
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: re-calibration %.1f us per expert agrees with %.1f (within %.0f%%)\n", tag,
                    t1_new, cm_t1, 100.0*cm_recal_tol);
                return;
            } else {
                const double t1_old = cm_t1;
                cm_t1 = std::min(cm_t1, t1_new);
                cm_dt = t1_new < t1_old ? dt_new : cm_dt;
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses: re-calibration %.1f us per expert vs %.1f at start-up (%.0f%% apart): host %.1f us "
                    "per expert kept\n", tag, t1_new, t1_old, 100.0*rel, cm_t1);
            }
        } else {
            cm_t1    = std::max(slope, 0.05);
            cm_dt    = std::max(0.0, (d_t - d_a)/((double) ka*(T - 1)));
            cm_tup   = std::max(median(tu)/ka, 0.05);
            cm_split = std::max(0.0, d_a - ka*cm_t1) + 2.0*median(trt);
        }
        {
            int64_t n_embd = 0;
            for (const ggml_tensor * t : g.ref) {
                n_embd = std::max<int64_t>(n_embd, t->ne[0]);
            }
            cm_act = 2.0*n_embd*sizeof(float)/std::max(1e-3, g.expert_bytes/cm_tup);
        }
        GGML_UNUSED(attempt);

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
            const double half = n <= max_batch ? cm_small_half : 0.354;
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
        cm_cal = true;
        cm_recal_wait = cm_recal_after;
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
        return n_tokens >= 1 && n_tokens <= max_batch && n_tokens <= cm_max && cm_on[n_tokens] != 0;
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
        // the host path much slower than without it ends the trial early
        if (cmt_t[0].size() >= 4 && cmt_t[1].size() >= 4 &&
                *std::min_element(cmt_t[1].begin(), cmt_t[1].end()) > (1.0 + 8*cmt_margin)*median(cmt_t[0])) {
            cmt_done = 5*cmt_block;
        }
        if (cmt_done >= 5*cmt_block) {
            cmt_done = -1;
            const double t_off = median(cmt_t[0]);
            const double t_on  = median(cmt_t[1]);
            const bool   on    = cmt_t[0].size() >= 3 && cmt_t[1].size() >= 3 && t_on < (1.0 - cmt_margin)*t_off;
            cmt_n++;
            if (cmt_n <= 4 || on != cm_small_ok || cm_dbg) {
                LLAMA_LOG_INFO("llama_moe_cache%s: CPU misses trial %d: %.3f ms per small batch without the host path, %.3f ms with it (medians of %zu and %zu) -> %s for small batches%s\n",
                    tag, cmt_n, t_off/1e3, t_on/1e3, cmt_t[0].size(), cmt_t[1].size(), on ? "on" : "off",
                    cmt_host_slow(t_on, t_off) ? ", off for short prompts (the host is slower than calibrated)" : "");
            }
            cmt_t[0].clear();
            cmt_t[1].clear();
            cm_gen++; // the small batches and the short prompts (cm_prompt_ok) may change
            cm_small_ok   = on;
            cm_small_live = on;
            const bool host_ok = !cmt_host_slow(t_on, t_off);
            if (host_ok != cmt_host_ok) {
                cm_gen++;
            }
            cmt_host_ok = host_ok;
            cmt_wait = cmt_gap;
            // a clear loss is checked again much later: each trial runs 2 blocks on the slower path, and their fewer admissions leave
            // the cache colder for a while (G4 Qwen3.6 UD-Q4_K_XL 12 GB, gpt-oss-20b 8 GB: -4..-8% decode with the path off but re-tried)
            cmt_gap  = std::min(on || t_on < 1.05*t_off ? 2*cmt_gap : std::max(4*cmt_gap, 512), cmt_gap_max);
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
        return cm && cm_cal && !cm_force && cm_counted(n_tokens) && cm_seen >= cm_skip && (!cm_have0 || cm_nb + 1 >= cm_win);
    }

    // before the graph of every ubatch (backends synchronized when cm_tick_reads)
    void cm_tick(int64_t n_tokens) {
        if (!cm || !cm_cal || !cm_counted(n_tokens)) {
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
                const double need = (n <= max_batch ? cm_margin*cm_split_scale*cm_split*(mb_cheap ? single_split_frac : 1.0) :
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
                v[i] = (int32_t) (next++ % n_expert);
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
                const int64_t e = next++ % n_expert;
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
        const int32_t n_max  = guard       ? (int32_t) std::min<int64_t>(g.n_slots / 2, g.n_slots - g.host_need - 1)
                             : n_graph > 0 ? std::max<int32_t>(g.n_slots / 2, g.n_slots - n_keep)
                             :               g.n_slots / 2;
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
        if (!dry && !lend_shrink && g.lru.n_lent > n_lend && g.lru.n_lent <= n_max) {
            std::vector<size_t> offsets_k(n_copies);
            std::vector<size_t> part_sizes_k(n_graph);
            if (place(g, g.lru.n_lent, copies, n_copies, n_graph, 1, graph_need, depth, guard, offsets_k, part_sizes_k)) {
                n_lend     = g.lru.n_lent;
                offsets    = std::move(offsets_k);
                part_sizes = std::move(part_sizes_k);
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
        d.buf_bank.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx_bank.get(), d.buft));
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
                const size_t n_avail = (size_t) (g.n_slots - g.lru.n_lent);
                size_t n_seed = n_avail / g.layers.size();
                size_t n_full = g.n_slots / g.layers.size();
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
                const int32_t n = std::min<int32_t>(g.n_slots/2, prompt_rings*m);
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
                const int32_t n = std::min<int32_t>(g.n_slots/2, prompt_rings*m);
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
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
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

        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

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

        // RAM tier: the experts routed by this call keep their RAM slots while it runs
        if (tiered) {
            host->begin_call();
            for (int32_t e : ids) {
                const int32_t sl = e >= 0 && e < (int32_t) g.lru.n_expert ? host->find(il, e, false) : -1;
                if (sl >= 0) {
                    host->pin(sl);
                }
            }
        }

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit, tiered && host->exclusive() ? &evicted_tmp : nullptr)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts (from the RAM tier, the model files or the mapping)
        const size_t bytes = upload_fills(backend, il, fills, tiered && host->exclusive() ? &evicted_tmp : nullptr);

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
            ggml_backend_tensor_memset(bank, 0, 0, ggml_nbytes(bank));
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

    ggml_tensor * devseed_plan(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_tokens) const {
        if (n_tokens < devseed_min || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const group & g = groups[layers[il].ig];
        if (!g.devseed) {
            return nullptr;
        }
        const int32_t pos = (int32_t) (std::find(g.layers.begin(), g.layers.end(), il) - g.layers.begin());
        return ggml_moe_cache_seed_plan(ctx, ids, g.map, g.state, g.scores, devseed_max(g), il, pos);
    }

    // the bank of the cached expert tensor w, and its layer
    ggml_tensor * devseed_bank(const ggml_tensor * w, int32_t & il) const {
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
            const size_t  n_avail  = (size_t) (g.n_slots - g.lru.n_lent);
            size_t        n_seed   = n_avail / g.layers.size();
            size_t        n_full   = g.n_slots / g.layers.size();
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
                shadow_slots(backend, g, evicted_tmp);
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
                    } else if (host->contains(sd.il, f.expert)) {
                        host->release_key(sd.il, f.expert);
                        hst.promoted++;
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
            if (tail_rank > 0) {
                LLAMA_LOG_INFO("llama_moe_cache%s: route tail skip (rank %d%s): %llu missing experts skipped in windows of at most %" PRId64 " tokens\n",
                    tag, tail_rank, tail_dflt ? ", default" : "", (unsigned long long) tail_skipped(), tail_max_tokens);
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
            LLAMA_LOG_INFO("llama_moe_cache: host tier: %d slots, peak %zu held, %llu evicted, %llu without a slot, %llu waits, pinned = %.2f MiB of a %.2f MiB budget, "
                    "registered expert buffers = %.2f MiB, MemAvailable %.1f MiB at start, %.1f MiB now\n",
                    host->n_slots(), hs.peak_held, (unsigned long long) hs.evicted, (unsigned long long) hs.no_slot, (unsigned long long) hs.waits,
                    host->pinned_bytes()/1048576.0, host->budget_bytes()/1048576.0, hst.registered/1048576.0, hst.avail_init/1048576.0, meminfo_avail()/1048576.0);
            LLAMA_LOG_INFO("llama_moe_cache: host tier: %.1f ms pinning slot memory, %llu allocation failures, %.2f MiB pageable (%llu chunks past the pin cap or a failed pin)\n",
                    hs.alloc_us/1000.0, (unsigned long long) hs.alloc_fail, host->pageable_bytes()/1048576.0, (unsigned long long) hs.pin_fallbacks);
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

bool llama_moe_cache::overlap(int64_t n_ubatch) const {
    return pimpl->overlap_batch(n_ubatch);
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
    return pimpl->cm && pimpl->cm_cal && pimpl->cm_recal_pending;
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
    return pimpl->hybrid && pimpl->hybrid_probe;
}

void llama_moe_cache::hybrid_probe(ggml_backend_t backend_cpu) {
    pimpl->probe(backend_cpu);
}

void llama_moe_cache::hybrid_set_threads(int32_t n_threads, bool quiet) {
    pimpl->single_threads = n_threads;
    pimpl->set_threads(n_threads, quiet);
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

void llama_moe_cache::hybrid_reset() {
    pimpl->hybrid_reset();
}

bool llama_moe_cache::overlap_trial_pending() const {
    return pimpl->ov_pending();
}

bool llama_moe_cache::overlap_tick_sync() const {
    return pimpl->ov_rule ? pimpl->ov_rule_reads() : (pimpl->ov_done >= 0 || pimpl->ov_wait == 1);
}

void llama_moe_cache::overlap_tick(int64_t n_tokens) {
    pimpl->ov_tick(n_tokens);
}

bool llama_moe_cache::hybrid_window_pending() const {
    if (pimpl->cm) {
        return pimpl->cmt_pending();
    }
    return pimpl->hybrid && (pimpl->win_done >= 0 || pimpl->win_wait >= 0);
}

bool llama_moe_cache::hybrid_tick_sync() const {
    return !pimpl->cm; // CPU misses: each timed ubatch is synchronized at its end (see hybrid_tick_end)
}

bool llama_moe_cache::prompt_auto_pending() const {
    return pimpl->pt_pending();
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

bool llama_moe_cache::resident_union() const {
    return pimpl->host && pimpl->host->exclusive() && pimpl->host_spare && pimpl->resident_mode;
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
