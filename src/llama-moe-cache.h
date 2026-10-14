#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // the cache serves the ubatches of up to this many tokens, the larger ones (prompts) copy the experts they use
    static constexpr int64_t n_small_max = 32;

    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // n_ubatch is the size of the ubatch (n_tokens the rows of the layer, fewer for an output-only last layer)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, int64_t n_ubatch) const;

    // for small batches: the slots of the experts selected by ids [n_expert_used, n_tokens] in layer il, looked up on the device (which also copies the misses)
    // n_ubatch is the ubatch size, the same for all layers; nullptr if the device lookup is not used, then use get_slot_map
    ggml_tensor * lookup(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_ubatch) const;

    // before computing a ubatch: moves the state of the cache between the host and the devices if the other side uses it
    void prepare(int64_t n_tokens);

    // hybrid small batches [LLAMA_MOE_CACHE_HYBRID, LLAMA_MOE_CACHE_FETCH]: the GPU computes the cached experts, the host computes the
    // missing ones it is left; only a share of the missing experts is fetched and admitted
    // the hybrid lookup of layer il (see ggml_moe_cache_assign), nullptr if the layer does not use it, and the copies of the experts it
    // admits (see ggml_moe_cache_fetch); host = false: every missing expert is fetched (the use counters are kept while the hybrid path is off)
    ggml_tensor * hybrid_assign(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_ubatch, bool & host) const;
    ggml_tensor * hybrid_fetch_experts(ggml_context * ctx, int32_t il, ggml_tensor * res) const;

    // true if the graphs of small batches may use the hybrid path
    bool hybrid_active() const;
    // graph key: 0 = plain lookups, 1 = lookups that keep the use counters, 2 = hybrid
    int  hybrid_mode() const;

    // the fetch share is measured once, before the first small batch: the host computes the experts of a layer with the decode threads
    bool hybrid_probe_pending() const;
    void hybrid_probe(ggml_backend_t backend_cpu);

    // auto mode: before the graph of each ubatch while the trial runs, after the previous ubatch finished (backends synchronized);
    // switches between the plain and the hybrid path, then keeps the faster one (see hybrid_active)
    bool hybrid_window_pending() const;
    void hybrid_tick(int64_t n_tokens);

    // ggml_backend_sched lend callback for a ubatch of n_tokens: places the prefetched copies of a large batch and its graph allocation
    // in slots taken from the cache while they are used (never for small batches, which may use every slot)
    void lend(ggml_backend_t backend, ggml_backend_sched_lend_copy * copies, int n_copies, int depth, bool dry, int64_t n_tokens);

    // the cache may lend slots (LLAMA_MOE_CACHE_LEND)
    bool lends() const;
    bool lends_graph() const; // the graph allocator may borrow too (LLAMA_MOE_CACHE_LEND=2)

    // the largest ubatch (at most n_small_max tokens) whose experts every cached layer reads from the cache without copying them
    int64_t n_tokens_cached() const;

    // device copy of a small per-expert host tensor (scale or bias) of a cached layer, nullptr if there is none
    ggml_tensor * get_resident(const ggml_tensor * t) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // true if t is a slot map (host memory that changes between graph computations)
    bool is_slot_map(const ggml_tensor * t) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache, -n if a run of n cached experts is too short to copy
    // with regions != nullptr, every cached run is appended to regions instead of being copied (and is not counted)
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last, std::vector<ggml_backend_copy_region> * regions = nullptr);

    // for large batches: count the copies of a batch of the experts of w issued by the caller (copies from the cache and uploads)
    void count_batch(const ggml_tensor * w, const std::vector<ggml_backend_copy_region> & regions);

    // for large batches: count a ubatch of more than 32 tokens (for the copy statistics)
    void count_large_ubatch();

    // for large batches: count the experts of w uploaded from host memory (one copy call)
    void count_upload(const ggml_tensor * w, size_t n_experts, size_t bytes);

    // for large batches: admit the most used experts of the layer of w, counts[e] is the number of tokens routed to expert e
    // must be called for each expert tensor of the layer after dst is filled, the slots are filled from dst
    // staged[e]: expert e was copied to dst for this ubatch (only those can fill a slot)
    void seed(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * dst, const std::vector<int32_t> & counts, const std::vector<bool> & staged);

    // after the graph is computed: drop the admissions of layers that were not seeded for all of their expert tensors
    void seed_end();

    // the graph of a prefetched ubatch that seeds from its own ids does not fit: prefill the cache now if the default seeding skipped it
    void seed_unavailable();
    // a prompt end that does not seed for that reason (statistics)
    void count_seed_skipped();

    // LLAMA_MOE_CACHE_SEED_CHECK=1 (debug): after the seeding ubatch is synchronized, compare the seeded slots with the host experts
    bool seed_check() const;
    void seed_verify();

    // routing of an earlier large ubatch of the prompt, used to seed when the seeding ubatch copies all experts without reading its ids (counts[e] = tokens routed to e)
    void set_seed_counts(const ggml_tensor * w, const std::vector<int32_t> & counts);
    const std::vector<int32_t> * get_seed_counts(const ggml_tensor * w) const; // nullptr if there are none for the layer of w
    void clear_seed_counts();
    bool has_seed_counts() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
