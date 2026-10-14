#include "llama-moe-cache.h"

#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
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

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
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

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
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
    void load(const std::vector<int32_t> & layers, const int32_t * dev_key_of, const int32_t * dev_ref) {
        GGML_ASSERT(n_lent == 0);
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
        for (int pass = 0; pass < 2; ++pass) {
            for (int32_t s = 0; s < n_slots; ++s) {
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

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        int lend_auto = -1; // automatic lending (LLAMA_MOE_CACHE_LEND unset): -1 not decided yet, 0 off, 1 on
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

        // device lookup: slot maps of all layers [n_expert, n_layer] and eviction state, see ggml_moe_cache_lookup
        ggml_tensor * map   = nullptr;
        ggml_tensor * state = nullptr;
        uint64_t dev_gen   = UINT64_MAX; // generation of the host state last copied to the device
        bool     dev_newer = false;      // the device state changed since it was last copied to the host
        size_t   expert_bytes = 0;       // all the expert tensors of one expert
        bool     dev_lookup   = false;   // small batches are looked up on the device (per group, so mixed devices work)
        ggml_tensor * counts  = nullptr; // hybrid: use counters of the experts [n_expert, n_layer + 1], see ggml_moe_cache_assign
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
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
    bool  verify_enabled = false; // LLAMA_MOE_CACHE_VERIFY=1 (debug): compare the cached experts with host memory
    stats stats_refill;
    std::vector<std::vector<int32_t>> seed_counts; // [n_layer] routing of an earlier ubatch of the prompt, see set_seed_counts
    std::vector<ggml_backend_copy_region> seed_regions;
    size_t seed_unstaged = 0; // seeds of a layer dropped because a planned expert was not staged in the copy of an expert tensor
    size_t seed_skipped  = 0; // prompt ends that did not seed because the graph of the seed did not fit (see seed_unavailable)
    bool   prefill_deferred = false; // the default seeding skipped the prefill at load, it runs if prompts cannot seed

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

    // lending (see lend): LLAMA_MOE_CACHE_LEND=0 off, 1 on with one cache device, 2 also with several (not validated)
    int    lend_mode    = 1;     // LLAMA_MOE_CACHE_LEND: 0 off, 1 staging copies, 2 also the graph allocator (experimental)
    bool   lend_auto    = true;  // LLAMA_MOE_CACHE_LEND unset: lend only if the staging copies are a large share of the cache, see lend
    bool   seed_own     = false; // prompts seed from their own ids by default (LLAMA_MOE_CACHE_SEED unset or 2), see lend
    double lend_min_frac = 0.08; // LLAMA_MOE_CACHE_LEND_MIN_FRAC: that share
    bool   lend_multi   = false; // LLAMA_MOE_CACHE_LEND_MULTI=1: also with the cache on several devices (not validated)
    bool   lend_off_log = false;
    size_t n_lends      = 0; // graphs whose staging copies used lent slots
    int32_t lend_max    = 0; // most slots lent at once

    // device lookup for small batches: the selected experts stay on the device and the misses are read from mapped host memory
    bool device_lookup = false; // requested, see LLAMA_MOE_CACHE_DEVICE; each group then checks its device
    std::vector<std::pair<void (*)(void *), void *>> mapped; // host buffers mapped by the cache, with the function that unmaps them

    // hybrid small batches (device lookups only): the host computes the missing experts that are not fetched
    bool  hybrid       = false; // LLAMA_MOE_CACHE_HYBRID / LLAMA_MOE_CACHE_FETCH (default off)
    bool  hybrid_probe = false; // the fetch share is measured before the first small batch (LLAMA_MOE_CACHE_FETCH not set)
    bool  hybrid_live  = false; // the graphs of small batches use the hybrid path
    float hybrid_share = 1.0f;  // share of the missing experts that are fetched and admitted, 1 = all (no hybrid path)
    bool  overlap      = false; // LLAMA_MOE_CACHE_OVERLAP: small batches compute the hits while the missing experts are copied

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

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
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
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_SEED_CHECK");
            seed_check = env != nullptr && atoi(env) != 0;
        }
        // LLAMA_MOE_CACHE_HYBRID: 0 or unset = off, 1 = auto (trials, kept only if faster), 2 = on with the measured share;
        // LLAMA_MOE_CACHE_FETCH=0..1 turns it on with that share (the same in every run)
        // off by default: the measured share and the trials depend on timing, so two runs could give different results
        {
            const char * env   = getenv("LLAMA_MOE_CACHE_HYBRID");
            const char * env_f = getenv("LLAMA_MOE_CACHE_FETCH");
            const int    mode  = env != nullptr ? atoi(env) : 0;
            hybrid = device_lookup && (mode > 0 || env_f != nullptr) && model.hparams.n_expert >= 4;
            if (hybrid && env_f != nullptr) {
                hybrid_share = std::min(1.0f, std::max(0.0f, (float) atof(env_f)));
                hybrid       = hybrid_share < 1.0f;
                hybrid_live  = hybrid;
            } else {
                hybrid_probe = hybrid;
                hybrid_auto  = mode == 1;
            }
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_LEND");
            lend_mode = env == nullptr ? 1 : atoi(env);
            lend_auto = env == nullptr;
            env = getenv("LLAMA_MOE_CACHE_SEED");
            seed_own = env == nullptr || atoi(env) == 2;
            env = getenv("LLAMA_MOE_CACHE_LEND_MIN_FRAC");
            if (env != nullptr) {
                lend_min_frac = std::max(0.0, atof(env));
            }
            env = getenv("LLAMA_MOE_CACHE_LEND_MULTI");
            lend_multi = env != nullptr && atoi(env) != 0;
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_REFILL");
            refill_enabled = env == nullptr || atoi(env) != 0;
        }
        {
            const char * env = getenv("LLAMA_MOE_CACHE_VERIFY");
            verify_enabled = env != nullptr && atoi(env) != 0;
        }

        {
            const char * env = getenv("LLAMA_MOE_CACHE_OVERLAP");
            overlap = device_lookup && env != nullptr && atoi(env) != 0 && model.hparams.n_expert >= 4;
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
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
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;

            // the budget also holds the device copies of the small per-expert tensors and the state of the device lookup
            const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
            size_t fixed = 0;
            for (int32_t il : g.layers) {
                for (const ggml_tensor * t : llama_moe_cache_layer_small(model.layers[il])) {
                    fixed += GGML_PAD(ggml_nbytes(t), alignment);
                }
            }
            if (device_lookup) {
                fixed += GGML_PAD((size_t) n_expert*model.layers.size()*sizeof(int32_t), alignment) + GGML_PAD(8*sizeof(int32_t), alignment);
            }
            auto total_size = [&](int32_t n_slots) {
                return alloc_size(g, n_slots) + fixed + (device_lookup ? 3*sizeof(int32_t)*(size_t) n_slots : 0);
            };
            while (g.n_slots < max_slots && total_size(g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size()) + (device_lookup ? 2 : 0) + (hybrid || overlap ? 1 : 0);
            for (int32_t il : g.layers) {
                n_tensors[g.id] += llama_moe_cache_layer_small(model.layers[il]).size();
            }
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
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
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
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
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
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
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
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
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
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
            const char * seed = getenv("LLAMA_MOE_CACHE_SEED");
            if (env != nullptr ? atoi(env) != 0 : seed != nullptr && atoi(seed) == 1) {
                prefill();
            }
            // the default seeding may not fit in the compute buffer, the cache is then prefilled (see seed_unavailable)
            prefill_deferred = env == nullptr && (seed == nullptr || atoi(seed) == 2);
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
                hybrid       = false;
                hybrid_probe = false;
                hybrid_live  = false;
                hybrid_share = 1.0f; // no hybrid line in the stats
            } else if (!model.hparams.no_alloc) {
                if (overlap) {
                    LLAMA_LOG_INFO("%s: MoE cache: small batches compute the cached experts while the missing ones are copied (LLAMA_MOE_CACHE_OVERLAP)\n", __func__);
                }
                if (hybrid && hybrid_probe) {
                    LLAMA_LOG_INFO("%s: hybrid MoE cache: the host computes the missing experts that are not fetched, fetch share measured at the first small batch, %s (LLAMA_MOE_CACHE_FETCH=0..1 to set the share)\n",
                        __func__, hybrid_auto ? "auto: kept only if trials of small batches are faster (LLAMA_MOE_CACHE_HYBRID=1)" : "on (LLAMA_MOE_CACHE_HYBRID=2)");
                } else if (hybrid) {
                    LLAMA_LOG_INFO("%s: hybrid MoE cache: the host computes the missing experts that are not fetched, fetch share = %.3f (LLAMA_MOE_CACHE_FETCH)\n", __func__, hybrid_share);
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // fill every slot at load, an equal share of the experts of each layer, so the first prompt reads them from the cache
    // like the experts a split keeps on the device (an empty cache uploads every expert of a large batch)
    void prefill() {
        size_t n_fill  = 0;
        size_t n_bytes = 0;
        std::vector<int32_t> fill_ids;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t n_expert = g.lru.n_expert;
            const size_t  n_layer  = g.layers.size();
            // a late prefill (see seed_unavailable) leaves the lent slots to the staging copies
            const int32_t n_slots  = g.n_slots - g.lru.n_lent;
            for (size_t i = 0; i < n_layer; ++i) {
                const int32_t il = g.layers[i];
                const int32_t n  = std::min<int32_t>(n_expert, n_slots/n_layer + (i < n_slots % n_layer ? 1 : 0));
                fill_ids.resize(n);
                for (int32_t e = 0; e < n; ++e) {
                    fill_ids[e] = e;
                }
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
            LLAMA_LOG_INFO("%s: MoE cache prefilled with %zu experts, %.2f MiB (LLAMA_MOE_CACHE_PREFILL=0 to disable)\n", __func__, n_fill, n_bytes/1024.0/1024.0);
        }
    }

    ~impl() {
        log_stats();
        for (const auto & [unmap, buf] : mapped) {
            unmap(buf);
        }
    }

    using host_map_t   = void * (*)(ggml_backend_dev_t, void *, size_t, bool *);
    using host_unmap_t = void (*)(void *);

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
        return ggml_backend_supports_op(backend, op);
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
            auto host_map   = (host_map_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_map");
            auto host_unmap = (host_unmap_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_unmap");

            // the mappings this group adds are appended to mapped, the ones shared with earlier groups stay untouched
            const size_t n_mapped = mapped.size();
            std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_t>> added;

            bool ok = true;
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
                LLAMA_LOG_WARN("%s: %s cannot map the host experts (pageable or memory-mapped), the MoE cache uses host lookups for %zu layers\n",
                    __func__, ggml_backend_name(backend), g.layers.size());
                continue;
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
        return g.dev_lookup && g.n_slots > 0 && n_tokens > 0 && n_tokens <= max_batch &&
            std::min<int64_t>(n_tokens*n_expert_used, g.ref[0]->ne[2]) <= g.n_slots;
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
        return ggml_moe_cache_lookup(ctx, ids, g.map, g.state, banks.data(), host.data(), (int) banks.size(), il);
    }

    // the layer can use the hybrid lookup for a ubatch of this size
    bool hybrid_layer(int32_t il, int64_t n_ubatch) const {
        if (!(hybrid || overlap) || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
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
        host = hybrid_active();
        const group & g = groups[layers[il].ig];
        return ggml_moe_cache_assign(ctx, ids, g.map, g.state, g.counts, host ? hybrid_share : 1.0f, il);
    }

    ggml_tensor * hybrid_fetch_experts(ggml_context * ctx, int32_t il, ggml_tensor * res, bool ov) const {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        std::vector<ggml_tensor *> banks = g.banks;
        std::vector<const void *>  host  = l.host_dev;
        return ggml_moe_cache_fetch(ctx, res, banks.data(), host.data(), (int) banks.size(), ov);
    }

    bool hybrid_active() const {
        return hybrid && hybrid_live && hybrid_share < 1.0f;
    }

    int hybrid_mode() const {
        return !(hybrid || overlap) ? 0 : hybrid_active() ? 2 : 1;
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

    // fetch share: the host and the device finish at the same time when share*t_upload = (1 - share)*t_host per missing expert
    // t_host: one token through the expert tensors of a cached layer on the host backend, t_upload: copy of an expert from host memory
    void probe(ggml_backend_t backend_cpu) {
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
        ggml_backend_tensor_set(g.state, buf.data(), n_hdr*sizeof(int32_t), buf.size()*sizeof(int32_t));

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
        g.lru.load(g.layers, buf.data(), buf.data() + g.n_slots);

        g.dev_gen   = g.lru.gen;
        g.dev_newer = false;
        if (verify_enabled) {
            verify(g, "after device lookups");
        }
    }

    void prepare(int64_t n_tokens) {
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
    }

    // the scheduler places the prefetched copies of a large batch: they go to the last slots of the largest group of the device, which
    // are evicted while lent, so the device does not keep a staging arena that only large batches use; n_copies = 0 takes the slots back
    void lend(ggml_backend_t backend, ggml_backend_sched_lend_copy * copies, int n_copies, int depth, bool dry, int64_t n_tokens) {
        const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.backend == backend; });
        if (it_dev == devices.end() || !it_dev->buf) {
            return;
        }
        const int32_t id = (int32_t) (it_dev - devices.begin());

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
        if (!lend_multi && std::count_if(devices.begin(), devices.end(), [](const device & d) { return d.host_bytes > 0 && d.buf; }) > 1) {
            if (!lend_off_log) {
                LLAMA_LOG_INFO("%s: MoE cache on several devices: the cache does not lend slots to the staging copies (LLAMA_MOE_CACHE_LEND_MULTI=1 to force)\n", __func__);
                lend_off_log = true;
            }
            return;
        }

        // the largest group of the device
        group * gl = nullptr;
        for (group & g : groups) {
            if (g.id == id && g.n_slots > 0 && (gl == nullptr || (size_t) g.n_slots*g.expert_bytes > (size_t) gl->n_slots*gl->expert_bytes)) {
                gl = &g;
            }
        }
        // small batches looked up on the device may use every slot of the group
        if (gl == nullptr || use_device(*gl, n_tokens)) {
            return;
        }
        group & g = *gl;

        // the graph allocator borrows only with LLAMA_MOE_CACHE_LEND=2 (experimental: outputs were seen to differ with the CUDA
        // copy-queue kernel, see plans/impl_moe_vram_lend.md), by default it keeps its own buffer
        std::vector<ggml_backend_sched_lend_copy> only;
        ggml_backend_sched_lend_copy * all = copies;
        const int n_all = n_copies;
        if (lend_mode != 2 && n_copies > 0 && copies[0].ordinal < 0) {
            only.assign(copies + 1, copies + n_copies);
            copies   = only.data();
            n_copies = (int) only.size();
            if (n_copies == 0) {
                return;
            }
        }

        // the copies of `depth` consecutive splits are live together, the ones with a negative ordinal (graph allocator) all the time
        size_t need  = 0;
        size_t perm  = 0;
        size_t split = 0; // copies of the largest split
        for (int i = 0; i < n_copies; ++i) {
            if (copies[i].ordinal < 0) {
                perm += copies[i].size;
                continue;
            }
            size_t live = 0;
            size_t own  = 0;
            for (int j = i; j < n_copies && copies[j].ordinal < copies[i].ordinal + depth; ++j) {
                live += copies[j].size;
                own  += copies[j].ordinal == copies[i].ordinal ? copies[j].size : 0;
            }
            need  = std::max(need, live);
            split = std::max(split, own);
        }
        need += perm;

        // a prompt that seeds from its own ids holds the copies of a split in the graph allocation (see llama_context::sched_reserve_seed)
        const size_t graph = (n_all > 0 && all[0].ordinal < 0 ? all[0].size : 0) + (seed_own ? split : 0);
        if (lend_auto && !lend_decide(id, need, graph, dry)) {
            return;
        }

        // the smallest number of slots that holds every copy, at most half of the slots (large batches also read and seed the cache)
        const int32_t n_max = g.n_slots / 2;
        int32_t n = (int32_t) std::min<size_t>((need + g.expert_bytes - 1) / g.expert_bytes, (size_t) n_max + 1);
        const int32_t step = std::max<int32_t>(1, n / 16);
        std::vector<size_t> offsets(n_copies);
        int32_t n_lend = 0;
        for (; n <= n_max; n = n < n_max ? std::min(n + step, n_max) : n_max + 1) {
            if (place(g, n, copies, n_copies, depth, offsets)) {
                n_lend = n;
                break;
            }
        }
        if (n_lend == 0) {
            if (!dry && g.lru.n_lent > 0) {
                unlend(g);
            }
            return;
        }
        ggml_backend_buffer_t buf = devices[id].buf.get();
        for (int i = 0; i < n_copies; ++i) {
            copies[i].buffer = buf;
            copies[i].offset = offsets[i];
        }
        if (copies != all) {
            for (int i = 0; i < n_copies; ++i) {
                all[n_all - n_copies + i].buffer = buf;
                all[n_all - n_copies + i].offset = offsets[i];
            }
        }
        if (dry) {
            return;
        }
        // with device lookups the device state may be newer: the slots to evict are only known after it is copied back
        if (g.dev_lookup && g.dev_newer) {
            download(g);
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
            }
            g.lru.set_lent(n_lend);
            LLAMA_LOG_INFO("%s: %s: MoE cache lends %d of %d slots (%.2f MiB) to the staging copies of large batches (LLAMA_MOE_CACHE_LEND=0 to disable)\n",
                __func__, ggml_backend_name(backend), n_lend, g.n_slots, (double) n_lend*g.expert_bytes/1024.0/1024.0);
        }
        n_lends++;
        lend_max = std::max(lend_max, n_lend);
    }

    // LLAMA_MOE_CACHE_LEND unset: the cache lends `need` bytes only if they are at least lend_min_frac of the cache of the device (a
    // small share frees little memory for the cache and costs the prompts the slots), or if the staging arena and the graph allocation
    // (`graph` more bytes, with the copies a seeding prompt holds there) do not fit in the free memory; decided at the first graph that
    // is allocated
    bool lend_decide(int32_t id, size_t need, size_t graph, bool dry) {
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
        bool on = frac >= lend_min_frac;
        if (dry) {
            // a measurement (e.g. for --fit) counts the memory with lending: when the device is short of memory the cache lends anyway
            return true;
        }
        // headroom for the memory the backends allocate while computing (e.g. the CUDA pool)
        constexpr size_t margin = 512ull*1024*1024;
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
        if (!on && dev != nullptr) {
            ggml_backend_dev_memory(dev, &free, &total);
        }
        const bool fit = on || dev == nullptr || free >= need + graph + margin;
        d.lend_auto = on || !fit ? 1 : 0;
        if (on) {
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
    void unlend(group & g) {
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

        if (refill.empty() || g.dev_newer) {
            return;
        }
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
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w    = l.experts[ip];
                    ggml_tensor       * bank = g.banks[ip];
                    const size_t expert_size = w->nb[2];
                    for (size_t i = 0; i < fills.size();) {
                        size_t m = 1;
                        while (i + m < fills.size() && fills[i + m].expert == fills[i].expert + (int32_t) m && fills[i + m].slot == fills[i].slot + (int32_t) m) {
                            m++;
                        }
                        ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, m*expert_size);
                        stats_refill.bytes += m*expert_size;
                        i += m;
                    }
                }
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
    // nor a copy with a negative ordinal (live during the whole graph, placed first)
    bool place(const group & g, int32_t n, const ggml_backend_sched_lend_copy * copies, int n_copies, int depth, std::vector<size_t> & offsets) {
        const device & d = devices[g.id];
        const uintptr_t base  = (uintptr_t) ggml_backend_buffer_get_base(d.buf.get());
        const size_t    align = ggml_backend_buft_get_alignment(d.buft);

        struct region { size_t begin; size_t end; };
        std::vector<region> regions;
        for (const ggml_tensor * bank : g.banks) {
            const size_t begin = (size_t) ((uintptr_t) bank->data - base) + (size_t) (g.n_slots - n)*bank->nb[2];
            regions.push_back({ begin, begin + (size_t) (n + 1)*bank->nb[2] });
        }
        auto align_up = [&](size_t off) {
            return (size_t) (GGML_PAD(base + off, align) - base);
        };

        struct live { size_t begin; size_t end; int ordinal; };
        std::vector<live> lv;
        std::vector<live> in_region;
        for (int i = 0; i < n_copies; ++i) {
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

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, int64_t n_ubatch) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        // with device lookups, ubatches that do not use the device use the host state (see prepare), a ubatch that uses the device keeps the state there
        if (use_device(groups[layers[il].ig], n_ubatch)) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_resident(const ggml_tensor * t) const {
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

        // the device state is newer (e.g. after a warmup that selected every expert on the host): the host slots are stale, upload everything
        if (g.dev_newer) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        // while a layer is seeded, its slots are filled after its copies, so the copies read the slots it had before
        const int32_t * slots = b.il == g.sd.il ? g.sd.old_slots.data() : g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
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
        if (n*w->nb[2] < min_copy_bytes) {
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
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
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
            const size_t  n_seed   = (g.n_slots - g.lru.n_lent) / g.layers.size();
            const size_t  n_full   = g.n_slots / g.layers.size();
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
            size_t n_hit = 0;
            const bool ok = g.lru.plan(b.il, seed_ids.data(), seed_ids.size(), sd.fills, n_hit);
            GGML_ASSERT(ok);
            sd.gen = g.lru.gen;
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
            seed_committed.insert(seed_committed.end(), sd.pending.begin(), sd.pending.end());
        } else {
            // some expert tensors were copied without filling their slots, drop the experts admitted to the layer
            for (const auto & f : sd.fills) {
                g.lru.slot_map[sd.il][f.expert] = -1;
                g.lru.key_of[f.slot] = -1;
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
        size_t n_bad = 0;
        for (const auto & r : seed_committed) {
            const size_t size = r.w->nb[2];
            GGML_ASSERT(r.bank->nb[2] == size && r.w->data != nullptr);
            buf.resize(size);
            ggml_backend_tensor_get(r.bank, buf.data(), (size_t) r.slot*r.bank->nb[2], size);
            if (memcmp(buf.data(), (const uint8_t *) r.w->data + (size_t) r.expert*size, size) != 0) {
                n_bad++;
            }
        }
        LLAMA_LOG_INFO("llama_moe_cache: seed check: %zu expert copies verified, %zu mismatches\n", seed_committed.size(), n_bad);
        seed_checked    += seed_committed.size();
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

    bool has_seed_counts() const {
        return std::any_of(seed_counts.begin(), seed_counts.end(), [](const std::vector<int32_t> & c) { return !c.empty(); });
    }

    void log_stats() {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
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
                LLAMA_LOG_INFO("llama_moe_cache: hybrid: fetch share = %.3f, %llu missing experts computed on the host (not fetched), hit rate with them = %.2f%%\n",
                    hybrid_share, (unsigned long long) n_host, n > 0 ? 100.0*st.hits/n : 0.0);
            }
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (stats_upload.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts uploaded, %.2f MiB\n", stats_upload.misses, stats_upload.bytes/1024.0/1024.0);
        }
        if (n_large_ubatches > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu ubatches, copies per ubatch: %.1f from the cache, %.1f uploads, in %.1f batched calls\n",
                n_large_ubatches, (double) stats_copy.calls/n_large_ubatches, (double) stats_upload.calls/n_large_ubatches, (double) n_batches/n_large_ubatches);
        }
        if (stats_seed.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts admitted to the cache, %.2f MiB\n", stats_seed.misses, stats_seed.bytes/1024.0/1024.0);
        }
        if (seed_unstaged > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu expert tensor seeds skipped (planned expert not staged)\n", seed_unstaged);
        }
        if (seed_skipped > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu prompt ends not seeded (the seeding graph does not fit in the compute buffer)\n", seed_skipped);
        }
        if (n_lends > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu graphs used lent slots (at most %d slots lent), %zu experts refilled after them, %.2f MiB\n",
                n_lends, lend_max, stats_refill.misses, stats_refill.bytes/1024.0/1024.0);
        }
        if (seed_check) {
            LLAMA_LOG_INFO("llama_moe_cache: seed check total: %zu expert copies verified, %zu mismatches\n", seed_checked, seed_mismatches);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, backends, bufts, size)) {
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

bool llama_moe_cache::overlap() const {
    return pimpl->overlap;
}

bool llama_moe_cache::hybrid_active() const {
    return pimpl->hybrid_active();
}

bool llama_moe_cache::hybrid_probe_pending() const {
    return pimpl->hybrid && pimpl->hybrid_probe;
}

void llama_moe_cache::hybrid_probe(ggml_backend_t backend_cpu) {
    pimpl->probe(backend_cpu);
}

bool llama_moe_cache::hybrid_window_pending() const {
    return pimpl->hybrid && (pimpl->win_done >= 0 || pimpl->win_wait >= 0);
}

void llama_moe_cache::hybrid_tick(int64_t n_tokens) {
    pimpl->tick(n_tokens);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, int64_t n_ubatch) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used, n_ubatch);
}

ggml_tensor * llama_moe_cache::lookup(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_ubatch) const {
    return pimpl->lookup(ctx, il, ids, n_ubatch);
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

bool llama_moe_cache::is_slot_map(const ggml_tensor * t) const {
    return pimpl->layer_of.find(t) != pimpl->layer_of.end();
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

void llama_moe_cache::count_seed_skipped() {
    pimpl->seed_skipped++;
}

bool llama_moe_cache::lends() const {
    return pimpl->lend_mode != 0;
}

bool llama_moe_cache::lends_graph() const {
    return pimpl->lend_mode == 2;
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
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
