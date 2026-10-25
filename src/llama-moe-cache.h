#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
class llama_weight_store;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
#include <stdexcept>

// thrown by the constructor when the budget cannot hold the experts of one token in any layer (a fit probe tries such sizes)
struct llama_moe_cache_too_small : std::runtime_error {
    llama_moe_cache_too_small() : std::runtime_error("MoE cache is too small to hold the experts of one token") {}
};

class llama_moe_cache {
public:
    // the cache serves the ubatches of up to this many tokens, the larger ones (prompts) copy the experts they use
    static constexpr int64_t n_small_max = 32;

    // the layers a context caches: a main context never runs the NextN (MTP) layers, an MTP draft context runs only those
    enum class layer_set {
        all,   // every layer (models without NextN layers, or LLAMA_MOE_CACHE_MTP=0)
        trunk, // the layers of the main graph
        nextn, // the NextN (MTP) layers, run by the MTP draft context
    };

    // LLAMA_MOE_CACHE_MTP=0 restores the single cache of the main context over all layers, with no cache in the MTP context
    static bool mtp_enabled();

    // the layers that a context of type ctx_type caches
    static layer_set select_layers(const llama_model & model, bool ctx_mtp);

    // the share of the total budget size (--moe-cache-mib) that the cache of the given layers gets: the main and the MTP cache split the
    // budget by the host bytes of their experts, the MTP cache also gets what the main cache cannot use once it holds all of its experts
    // (LLAMA_MOE_CACHE_MTP_MIB=N sets the MTP share); the same for both contexts, so --fit measures the same caches the contexts create
    static size_t budget(const llama_model & model, size_t size, layer_set set);

    // k15-pp3: host bytes of the experts a cache of the trunk layers could hold (0: every expert is on a device)
    static size_t host_bytes(const llama_model & model);

    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // size is already the share of the layer set (see budget)
    // ws: the weight store that reads the misses of host lookups from the model files (see set_weight_store), nullptr if none
    // host_mib: MiB of the RAM tier of the experts the weight store streams (pinned slots behind the device slots), -1 = auto, 0 = off
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            layer_set set = layer_set::all, llama_weight_store * ws = nullptr, int64_t host_mib = -1);
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
    ggml_tensor * hybrid_fetch_experts(ggml_context * ctx, int32_t il, ggml_tensor * res, bool overlap) const;

    // single token ubatches, LLAMA_MOE_CACHE_FETCH_OVERLAP (unset/1 while enough missing expert bytes per layer and token are copied,
    // see ggml_backend_moe_fetch_overlap_min_bytes, 2 on, 3 timing trials, 4 device hint only, 0 off): the copies of
    // the missing experts run beside the experts already cached (see ggml_moe_cache_fetch); true if the graph of a ubatch of n_ubatch
    // tokens uses it now
    bool overlap(int64_t n_ubatch) const;

    // CPU misses [LLAMA_MOE_CPU_MISS]: the rule reads the device counters before the graph of some ubatches (synchronize the backends
    // first when cpu_miss_tick_sync), the tick goes before the graph of every ubatch
    bool cpu_miss_enabled() const; // LLAMA_MOE_CPU_MISS is on for this cache (until a failed calibration turns it off)
    // k13: the host never computes an expert of this cache (no hybrid path, no CPU misses): every expert runs on the device from a slot
    bool host_never() const;
    // k13: frozen small batches (freeze_small): the host computes every miss of a small batch, nothing is fetched or admitted between
    // prompts (the assign is frozen, see ggml_moe_cache_assign_freeze), so which expert of a token the host computes depends on the
    // token's own ids and the cache as the prompts left it, the same in a verify window as in a single token decode
    bool host_split_frozen() const;
    // k13: CPU misses: small batches (up to 8 tokens) fetch nothing and the host computes every miss, without trials; true if frozen
    bool freeze_small();
    // k13: CPU misses: small batches (up to 8 tokens) take the host path without trials or miss-rate windows, but still fetch their
    // share and admit (the cache follows the decode; not exact under drafts); true if pinned
    bool pin_small();
    // k15-lfm2: see llama_moe_cpu_miss_drop_slow
    bool cpu_miss_drop_slow();
    // k13: who computes the missing experts of small batches: 0 = never the host (every miss fetched), 1 = the hybrid path with a fixed
    // share (the default gate of small caches, or LLAMA_MOE_CACHE_FETCH), 2 = the CPU-miss rule
    int host_mode() const;
    bool cpu_miss_tick_sync(int64_t n_tokens) const;
    void cpu_miss_tick(int64_t n_tokens);
    // the ubatch of n_tokens is looked up on the device like a small batch: at most n_small_max tokens, or a short prompt whose missing
    // experts the host computes (the large batch path, its copies and its seed do not apply)
    bool small_batch(int64_t n_tokens) const;
    // k15-pp2: while set, every ubatch is treated as a large one (no device lookup, slot map, host path or CPU misses): its MUL_MAT_ID
    // read the layer's host experts as the large ubatches do (the last ubatch of a prompt as a member of a layer-major group)
    void set_force_large(bool on);
    bool force_large() const;

    // true if the graphs of small batches may use the hybrid path
    bool hybrid_active() const;
    // the host computes some missing experts of a ubatch of n_tokens now (CPU misses on for that size, or the hybrid path)
    bool host_computes(int64_t n_tokens) const;
    // single-graph host misses [LLAMA_MOE_SINGLE=1, opt-in]: the host experts of a small batch (up to GGML_MOE_MAILBOX_MAX_TOKENS tokens)
    // go through a mapped mailbox instead of a graph split (see ggml_moe_mailbox_publish); up, gate, down: the host expert tensors of
    // layer il (separate gate and up, SiLU, no biases or scales); nullptr if not used, then build the split path
    ggml_tensor * single_publish(ggml_context * ctx, int32_t il, ggml_tensor * host_ids, ggml_tensor * x,
            ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down) const;
    // the device rows dev [n_embd, n_expert_used, n_tokens] plus the host rows of the publish pub
    ggml_tensor * single_merge(ggml_context * ctx, ggml_tensor * dev, ggml_tensor * pub) const;
    // the graph builder tells whether layer il has the expert form the host plans compute (see single_weigh)
    void single_note_form(int32_t il, bool ok) const;
    // k15: the context marks each decode: a prompt batch (tokens without outputs) keeps its ubatches off the CPU-miss host path while
    // the small batches are pinned (see cm_prompt_off)
    void set_prompt_batch(bool prompt);
    // before the graph of every ubatch of n_tokens: wakes the host thread (batches that may publish), turns the path off after a failure
    void single_kick(int64_t n_tokens);

    // graph key: 0 = plain lookups, 1 = lookups that keep the use counters, 2 = hybrid, + 4 with the fetch overlap
    int  hybrid_mode() const;

    // the fetch share is measured once, before the first small batch: the host computes the experts of a layer with the decode threads
    bool hybrid_probe_pending() const;
    void hybrid_probe(ggml_backend_t backend_cpu);
    // k13-fn: one re-calibration of the CPU-miss host timings after warm-up, run between requests (see cm_recal_wait)
    bool recal_pending() const;
    void recal(ggml_backend_t backend_cpu);

    // default and forced hybrid path (LLAMA_MOE_CACHE_HYBRID unset or 2): the fetch share follows from the decode threads, call once after
    // the constructor; the same inputs then give the same output in every run (see hybrid_fixed)
    void hybrid_set_threads(int32_t n_threads, bool quiet);

    // fixed share (LLAMA_MOE_CACHE_FETCH, or from the threads): the cache, the use counters and the residual of the share are reset before a batch whose
    // sequences are all empty and after a state is read, so the same request gives the same output as in a new process
    bool hybrid_fixed() const;
    // k13-small: the cached share of the host experts' bytes (-1 before the slots are sized)
    double cache_frac() const;
    void hybrid_reset();

    // auto mode: before the graph of each ubatch while the trial runs, after the previous ubatch finished (backends synchronized);
    // switches between the plain and the hybrid path, then keeps the faster one (see hybrid_active)
    bool hybrid_window_pending() const;
    // the next tick times the ubatches (synchronize the backends before it); CPU misses: only while a trial runs or is about to start
    bool hybrid_tick_sync() const;
    // CPU misses trial: the ubatch after the tick is timed; synchronize after its graph is computed, then hybrid_tick_end
    bool hybrid_tick_timed() const;
    void hybrid_tick_end();

    // fetch overlap rule (default) or auto (LLAMA_MOE_CACHE_FETCH_OVERLAP=3): true while a window or trial runs or waits; the tick goes before the graph of
    // every ubatch then, after the previous ubatch finished when overlap_tick_sync() (the backends are synchronized)
    bool overlap_trial_pending() const;
    bool overlap_tick_sync() const;
    void overlap_tick(int64_t n_tokens);
    void hybrid_tick(int64_t n_tokens);

    // ggml_backend_sched lend callback for a ubatch of n_tokens: places the prefetched copies of a large batch and its graph allocation
    // in slots taken from the cache while they are used (never for small batches, which may use every slot)
    void lend(ggml_backend_t backend, ggml_backend_sched_lend_copy * copies, int n_copies, int depth, bool dry, int64_t n_tokens);

    // device seed (LLAMA_MOE_CACHE_DEVSEED, see ggml_moe_cache_seed_plan): a prefetched large ubatch seeds the cache on the device from its
    // own routing, without reading it on the host; devseed_init: whether any group can (device lookups and a backend that runs the plan)
    // min_tokens: the graphs of ubatches of at least that many tokens get the seed plans (the prefetch threshold), INT64_MAX: none
    bool devseed_init(int64_t min_tokens);
    // the seed plan of layer il for the routing ids of a large batch of n_tokens, nullptr if its group is not seeded on the device
    ggml_tensor * devseed_plan(ggml_context * ctx, int32_t il, ggml_tensor * ids, int64_t n_tokens) const;
    // the bank that the device seed fills for the cached expert tensor w and its layer, nullptr if none
    ggml_tensor * devseed_bank(const ggml_tensor * w, int32_t & il) const;
    // the graph gf (allocated) is about to run: enable or disable its seed plans; after it ran: the seeded groups come back to the host
    void devseed_prepare(ggml_cgraph * gf, bool enable);
    // ok = false (the graph failed): the groups it may have seeded in part are cleared
    void devseed_end(ggml_cgraph * gf, bool ok);
    // k11 (LLAMA_MOE_LAYER_MAJOR_SEED_REPLAY): the device seed of a prefetched large ubatch that ran inside a merged layer-major graph, after
    // that graph, from its routing (per layer in layer order: the routing outputs of the ubatch, still on the device where that graph left
    // them); the admitted experts are uploaded from host memory; bytes / ms: what it uploaded and its time; false if nothing is seeded
    bool devseed_replay(const std::vector<std::pair<int32_t, ggml_tensor *>> & rec, int64_t n_tokens, size_t & bytes, double & ms);

    // LLAMA_MOE_CACHE_SEED: 0 off, 1 only ubatches whose copies read their ids, 2 (default) also prefetched ubatches
    static int seed_mode();

    // the graph that is allocated next, before its lend callbacks (nullptr for a reserved graph, which does not run)
    void set_graph(ggml_cgraph * gf);

    // the cache may lend slots (LLAMA_MOE_CACHE_LEND)
    bool lends() const;
    // k11 (LLAMA_MOE_LM_LEND): while on, the cache also lends its slots to the graph allocation of the graphs allocated (a merged layer-major
    // group whose compute buffers would grow past the free memory), not only to their staging copies
    void set_graph_lend(bool on);
    // k13-small: the banks are freed for the graphs of a prompt (every slot emptied, its expert kept for the refill), restore allocates
    // them again and refills the slots; nothing may look up or upload experts in between (the layer-major group does neither)
    bool release();
    void restore();
    bool released() const;
    // the next restore refills the slots with the experts the recorded routing used most (set_refill_counts), as LLAMA_MOE_CACHE_REFILL_ROUTED=1
    void set_refill_routed_once(bool on);
    // the cache lends slots on some device (decided by the first large graph)
    bool lends_on_any() const;
    // lends_on_any, or a device whose lending is off for the staging copies that still lends to a merged layer-major graph (k15-pp)
    bool lends_graph_any() const;
    // the cache lends slots on the device of backend (decided by the first large graph, see lend): the staging arena is not allocated there
    bool lends_on(ggml_backend_t backend) const;
    bool lends_graph() const; // the graph allocator may borrow too (LLAMA_MOE_CACHE_LEND=2)
    // whether a graph of n_tokens may get its allocation lent (LLAMA_MOE_CACHE_LEND=2 and not a batch the device lookup serves)
    bool lends_graph_to(int64_t n_tokens) const;

    // the largest staging memory (the copies of large batches) a measurement asked the cache to lend, on any device (0 if none)
    size_t lend_need_max() const;

    // device memory of backend kept free for a model or context created later (e.g. a draft): the automatic lending decision
    // counts it as used (see lend_decide)
    void set_mem_reserve(ggml_backend_t backend, size_t bytes);

    // the largest ubatch (at most n_small_max tokens) whose experts every cached layer reads from the cache without copying them
    int64_t n_tokens_cached() const;
    // share of the host experts' bytes the cache holds (-1 before the slots are sized)
    double cached_frac() const;
    // bytes of the host experts the cache cannot hold (0 before the slots are sized)
    double host_uncached_bytes() const;

    // device copy of a small per-expert host tensor (scale or bias) of a cached layer, nullptr if there is none
    ggml_tensor * get_resident(const ggml_tensor * t) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // true if t is a slot map (host memory that changes between graph computations)
    bool is_slot_map(const ggml_tensor * t) const;

    // the misses of the host lookups are read from the model files by the weight store (nullptr: copied from the mapping)
    void set_weight_store(llama_weight_store * ws);

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

    // the context runs as with LLAMA_MOE_CACHE_SEED=1 (see llama_context, LLAMA_MOE_CACHE_SEED_AUTO): prefill the cache now if the
    // default seeding skipped it, and do not count a seed split in the lent memory
    void prompts_do_not_seed();

    // LLAMA_MOE_CACHE_SEED_CHECK=1 (debug): after the seeding ubatch is synchronized, compare the seeded slots with the host experts
    bool seed_check() const;
    void seed_verify();

    // routing of an earlier large ubatch of the prompt, used to seed when the seeding ubatch copies all experts without reading its ids (counts[e] = tokens routed to e)
    void set_seed_counts(const ggml_tensor * w, const std::vector<int32_t> & counts);
    // the routing of the last large ubatch of a prompt that did not seed: the lent slots that come back get its most used experts
    void set_refill_counts(const ggml_tensor * w, const std::vector<int32_t> & counts);
    bool refills_routed() const; // LLAMA_MOE_CACHE_REFILL_ROUTED=1 (opt-in)
    const std::vector<int32_t> * get_seed_counts(const ggml_tensor * w) const; // nullptr if there are none for the layer of w
    void clear_seed_counts();
    bool has_seed_counts() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
