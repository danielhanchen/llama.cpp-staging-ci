#pragma once

// RAM tier of the MoE cache: a byte-budgeted set of slots in pinned host memory behind the device slots of llama_moe_cache, each slot
// holds one expert of one layer for every expert tensor of the layer (a bundle, e.g. gate + up + down)
// the copies are issued by the caller; this keeps the slots, their index, the eviction order and the events that guard the memory
// exact: a slot only ever holds bytes read from the model files (verified with LLAMA_WEIGHT_STREAM_VERIFY=1) or copied back from a
// device slot that holds them

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

enum llama_moe_host_policy {
    LLAMA_MOE_HOST_INCLUSIVE = 0, // the RAM tier keeps what it reads, whether the device caches it or not
    LLAMA_MOE_HOST_EXCLUSIVE = 1, // the RAM tier keeps the experts the device does not: device evictions move into it, promotions leave it
};

// whether and how big. Auto (host_mib < 0, not forced) needs the device slots to hold the experts of at least
// LLAMA_MOE_HOST_MIN_TOKENS tokens (token_bytes: what one token routes over the streamed layers, n_expert_used experts each). Below that
// nearly every routed expert misses the device, and with the exclusive policy every miss also evicts a device slot that is copied back
// into RAM: the pinned slots then mostly replace the page cache the run had and the copies cost what the RAM hits save (L4, the experts of
// 1.4 tokens on the device: 38% device hits, 84% RAM hits, SSD bytes -3..-5%, 28k copies back, decode after 8k -14%; the experts of 6.9
// tokens: 71% device hits, SSD bytes -25%, 8k prompt -27%, decode +22..93%; 10.7 and 12.4 tokens: 8k prompt -48% / -40%)
// k19-disktier: that loss is the exclusive policy's copies back. Below the minimum, where the streamed experts do not fit the RAM the tier may
// plan into (mapped > avail: the disk tier, the page cache cannot hold them either), the tier goes on with the INCLUSIVE policy instead (no
// copies back; misses are read into RAM slots, the prompt's uploads only fill free slots). Without it the decode reads its misses through
// the page cache, whose LRU churns on a working set larger than itself: Colab L4 12 GB, Flash-Next UD-IQ4_XS 94 GB, 52 GB RAM (3.3 / 3.9
// tokens, tier off): MTP n2 chat 80.4 / 8k 1.95 tok/s, DFlash2 chat 4.0 / 8k 41.8, 31-87 GB read per request; the same VM with the tier
// on (5.2 tokens, plain decode): 39.2 / 32.9. Only where the budget holds at least LLAMA_MOE_HOST_DISK_MIN_FRAC of the streamed
// experts: with the tier on its misses are read unbuffered, so it replaces the page cache instead of adding to it, and a small tier
// loses to it (Kaggle T4, Qwen3.6-35B-A3B UD-Q4_K_XL under a RAM cap, 1.1-1.2 tokens: inclusive at 23% / 33% of the experts = the page
// cache within -8..+9%; at 42% / 52%: +38% / +31% chat, +22% / +39% 2k, same md5). disk_inclusive = false
// (LLAMA_MOE_HOST_DISK_INCLUSIVE=0): off below the minimum, as before
#define LLAMA_MOE_HOST_MIN_TOKENS 4
#define LLAMA_MOE_HOST_DISK_MIN_FRAC 0.4

struct llama_moe_host_plan {
    bool        on       = false;
    size_t      budget   = 0;
    int         policy   = 1;    // llama_moe_host_policy
    double      tokens   = 0.0;  // device slots / token_bytes
    bool        spare    = false; // exclusive: the budget and the device slots together exceed the streamed experts (see set_dup)
    const char * why     = "";
};

// e_bytes: host bytes of the streamed experts, v_bytes: their device slots, mapped: every mapped host weight the page cache serves,
// avail: MemAvailable, token_bytes: bytes one token routes over the streamed layers, host_mib: --moe-host-cache-mib (-1 auto, 0 off),
// policy_env: -1 or a forced llama_moe_host_policy, force: LLAMA_MOE_HOST_CACHE=1 (the auto budget even below the device minimum),
// disk_inclusive: below the device minimum on a disk tier, on with the inclusive policy (see LLAMA_MOE_HOST_MIN_TOKENS), with a budget of at
// least disk_min_frac of e_bytes
llama_moe_host_plan llama_moe_host_decide(size_t e_bytes, size_t v_bytes, size_t mapped, size_t avail, size_t token_bytes, int64_t host_mib,
        int policy_env, bool force, bool disk_inclusive = true, double disk_min_frac = LLAMA_MOE_HOST_DISK_MIN_FRAC);

class llama_moe_host_cache {
public:
    // layer_sizes[il]: bytes of one expert of each expert tensor of layer il (nb[2]), empty if the layer is not cached
    // buft: host buffer type of the slots (pinned memory of the device, plain host memory in tests)
    // the budget is split among the layouts like the experts, the slot memory is allocated in chunks when it is first needed
    // pin guard: at most pin_cap bytes come from buft; past it, or once buft cannot pin (it returns another type), the chunks come from
    // buft_pageable (plain host memory: slower copies, but RAM hits instead of reads), nullptr = the tier stops growing there
    llama_moe_host_cache(ggml_backend_buffer_type_t buft, size_t budget, int policy, int32_t n_expert, const std::vector<std::vector<size_t>> & layer_sizes,
            size_t chunk_size = 0, ggml_backend_buffer_type_t buft_pageable = nullptr, size_t pin_cap = SIZE_MAX);
    ~llama_moe_host_cache();

    llama_moe_host_cache(const llama_moe_host_cache &) = delete;
    llama_moe_host_cache & operator=(const llama_moe_host_cache &) = delete;

    bool    exclusive() const { return policy == LLAMA_MOE_HOST_EXCLUSIVE; }
    int32_t n_slots() const;   // slots of the budget (allocated or not)
    size_t  pinned_bytes() const { return bytes_alloc - bytes_pageable; }
    size_t  pageable_bytes() const { return bytes_pageable; }
    size_t  budget_bytes() const { return budget; }
    bool    caches(int32_t il) const { return il >= 0 && (size_t) il < layer_class.size() && layer_class[il] >= 0; }
    // the expert layout (class) of layer il, -1 if not cached
    int32_t class_of(int32_t il) const { return caches(il) ? layer_class[il] : -1; }

    // the slot of expert e of layer il if it holds the expert (a copy back from the device is waited for), -1 otherwise
    // counts a use (decayed count) when touch
    int32_t find(int32_t il, int32_t e, bool touch = true);
    bool    contains(int32_t il, int32_t e) const; // also while it is being filled
    int32_t slot_of_key(int32_t il, int32_t e) const { return caches(il) ? slot_of[il][e] : -1; }
    // the slot if it holds the expert or a copy back into it is committed (not waited for), -1 otherwise
    int32_t held(int32_t il, int32_t e) const;

    // a slot for expert e of layer il (not held), being filled: evicts the expert with the lowest decayed count (CLOCK) that is not
    // pinned, leased or being filled if no slot is free, and waits for the copies that still read the slot; -1 if none can be taken
    // evict = false: only a free slot (or memory still in the budget); min_age > 0: never evicts a slot read or written by the last
    // min_age commits (the reservation then never waits for the device)
    // take_spec: a slot set_spec marked (filled from a routing profile, not used since) counts as free (see set_spec)
    // keep (k19-l4held, a layer at its quota): a non-evicting reservation may also take a slot of the same layer whose expert keep does
    // not mark (e.g. not routed by the prompt that offers e), lowest count first
    int32_t reserve(int32_t il, int32_t e, bool evict = true, uint32_t min_age = 0, bool take_spec = true, const std::vector<bool> * keep = nullptr);
    void    publish(int32_t s);  // the slot holds its expert
    void    abort(int32_t s);    // the slot is free again
    void    release(int32_t s);  // drops the expert (exclusive promotion), the slot is reused once the copies that read it are done
    void    release_key(int32_t il, int32_t e);

    // memory of expert tensor ip of the slot: the area starts 4 KiB aligned, holds the expert at area + lead (lead < 4096) and has room
    // for an unbuffered read of the expert at any lead
    uint8_t * area(int32_t s, int ip) const;
    size_t    area_size(int32_t s, int ip) const;
    void      set_lead(int32_t s, int ip, size_t lead);
    uint8_t * data(int32_t s, int ip) const; // area + lead
    size_t    expert_size(int32_t s, int ip) const;
    int       n_tensors(int32_t s) const;
    int32_t   key(int32_t s) const { return slots[s].key; }
    int32_t   n_expert_of() const { return n_expert; } // keys are il*n_expert + e

    // experts routed by the current call: never evicted until the next begin_call
    void begin_call();
    void pin(int32_t s);

    // the held experts the reads ahead of the current plan skip: never evicted until the next new_plan, which also drops the bundles
    // that were being filled from the uploads of the plan and did not complete
    void new_plan();
    void lease(int32_t s);

    // filling a slot from pieces (uploads of a prompt): bytes [off, off + n) of the expert of tensor ip; the slot is published once
    // every tensor is complete; pieces must arrive in order per tensor, others are ignored; returns true when the slot completes
    bool fill_piece(int32_t s, int ip, size_t off, const uint8_t * p, size_t n);
    bool filling(int32_t s) const;

    // a reserved slot filled by an asynchronous read (e.g. a read ahead of the routing): fill_piece and new_plan leave it alone until
    // it is published or aborted; count: its decayed count (0: the first to go if it is not used)
    void set_async(int32_t s, uint8_t count);
    // exclusive: a held slot whose expert the device also holds (a duplicate) is the first a reservation takes, before the CLOCK;
    // set_dup(s, false) once RAM holds the only copy again (the device evicted it)
    void set_dup(int32_t s, bool dup);
    // k19-ramevict: the layer a prompt pass is streaming (-1: none, e.g. decode). While set, an evicting reservation that finds no
    // duplicate takes its victim first from the experts of the layers the pass has finished (layer <= il: not needed again until the
    // next pass), by the CLOCK over them, and only then from any layer (the plain CLOCK). Placement only: what a slot holds is unchanged.
    // Idea from Straydog (MIT). Off by default, LLAMA_MOE_HOST_EVICT_DONE=1 on (the counters below count either way). Kaggle T4, Qwen3.6-35B-A3B
    // UD-Q4_K_XL, tier 8 under a 15 GiB RAM cap (exclusive tier, disk misses), 3 launches each: same md5, the upcoming-layer evictions of the
    // prompt seeds 1057-1122 -> 11-13 per launch, disk bytes per request 8k -2.1%, 16k -3.2%, but TTFT 8k / 16k within the A/A spread and
    // decode after 8k -3% (the finished layers' experts are what the decode reads next)
    void set_stream_layer(int32_t il) { stream_il = il; }
    bool evict_done_on() const { return evict_done; }
    bool is_dup(int32_t s) const { return slots[s].dup != 0; }
    bool is_async(int32_t s) const { return slots[s].async != 0; }

    // k19-ramprof2: a held slot filled from a routing profile (a guess, LLAMA_MOE_HOST_PROFILE) is marked until it is used (a touching
    // find, a read by a copy, a write) or replaced. A reservation (take_spec) takes a marked slot when no slot is free and no duplicate
    // can go, before any eviction by the CLOCK, so a full tier of guesses still keeps what the prompts and misses read (as free slots
    // would); the layers a prompt pass has finished go first (see set_stream_layer), then the layers before the reserving one, coldest
    // of a layer first (marked in heat order). Placement only. Without marked slots nothing changes.
    void   set_spec(int32_t s);
    bool   is_spec(int32_t s) const { return slots[s].spec != 0; }
    size_t n_spec_slots() const { return n_spec; }
    // k19-l4held: per-layer share of a layout's slots (LLAMA_MOE_HOST_LAYER_QUOTA, set by the MoE cache: exclusive tier by default).
    // Every reservation fills the free slots of its layout, so the experts of the layers that come first in a fill (the page-cache
    // fill at load, the uploads of the first prompt, both in layer order) took a layout's slots and the layers after them kept only
    // their device slots: Colab L4 full card, Qwen3.8-Flash-Next UD-IQ4_XS from the VM disk, 42-token chat after a 640-token prompt:
    // layers 0-3 missed 55-96 of their ~120 routed experts, 30 of 48 layers none, 760 misses of 5741 where 8% of the experts held
    // nowhere at random would give ~460. With the quota a layer holds at most ceil(slots of its layout / its layers) slots: a
    // reservation of a layer at its quota takes a free slot only when the layout keeps enough free slots for the layers below theirs,
    // else a slot of the same layer (a duplicate of a device-held expert, an unused guess, or, when it may evict, the lowest count),
    // else nothing. Placement only: what a slot holds is the expert's bytes; the tier still fills completely.
    void    set_layer_quota(bool on);
    bool    layer_quota_on() const { return quota_on; }
    bool    at_quota(int32_t il) const { return quota_on && caches(il) && n_key_layer[il] >= quota[il]; }
    int32_t quota_of(int32_t il) const { return caches(il) ? quota[il] : 0; }
    int32_t n_keys_of(int32_t il) const { return caches(il) ? n_key_layer[il] : 0; }
    uint64_t quota_refused = 0, quota_same = 0, quota_slack = 0, quota_ranked = 0;

    // slots without an expert (free, or not allocated yet within the budget)
    int32_t n_free() const;
    // k19-l4fill: a non-evicting reservation of layer il would find a slot whose memory is already allocated: a free one or a duplicate
    // of a device-held expert that no current call / plan / uncommitted copy uses (never a guess, never memory still to be pinned)
    bool    can_take_alloc(int32_t il) const;
    // ... the number of such duplicates over every layout (the diagnostics)
    int32_t n_dup_slots() const;
    // slots being filled (reserved, not published yet)
    int32_t n_filling() const;
    // ... of the layout of layer il, and its slots
    int32_t n_free_of(int32_t il) const;
    int32_t n_slots_of(int32_t il) const;
    // releases up to n unused guesses (the coldest of each layer first, the layers in turn; never one in use by this call / plan or read
    // by uncommitted copies): a prompt's plan leases every held slot of its layers, so only free slots take what it reads; returns the
    // number released
    size_t drop_spec(size_t n);

    // events: the copies that read slots (mark_read) or write them (mark_write, e.g. device -> host) since the last commit are recorded
    // on backend; a slot is reused or read by the host only after them
    void mark_read(int32_t s);
    void mark_write(int32_t s);
    void commit(ggml_backend_t backend);

    // generation of the held set, changes with every publish, release and eviction
    uint64_t generation() const { return gen; }

    struct stats {
        uint64_t hits       = 0; // find() of a held expert
        uint64_t misses     = 0; // find() of an expert not held
        uint64_t reserved   = 0;
        uint64_t evicted    = 0;
        uint64_t no_slot    = 0; // reserve() failures
        uint64_t published  = 0;
        uint64_t aborted    = 0;
        uint64_t released   = 0;
        uint64_t waits      = 0; // event waits before a slot was reused or read
        uint64_t alloc_fail = 0;
        uint64_t alloc_us   = 0; // time spent allocating (pinning) slot memory
        uint64_t pin_fallbacks = 0; // chunks allocated pageable (pin cap reached or pinning failed)
        uint64_t dup_taken  = 0; // reservations that took a duplicate
        uint64_t stream_evicted = 0; // evictions while a prompt layer was set (see set_stream_layer)
        uint64_t done_evicted   = 0; // ... taken by the finished-layer rule
        uint64_t ahead_evicted  = 0; // ... of an expert of a layer the pass has not reached yet (the plain CLOCK; the rule's fallback)
        uint64_t spec_set   = 0; // k19-ramprof2: slots marked by set_spec
        uint64_t spec_used  = 0; // ... used while marked (the guess was right)
        uint64_t spec_taken = 0; // ... taken by a reservation while still unused
        uint64_t spec_dropped = 0; // ... released by drop_spec while still unused
        size_t   peak_held  = 0; // most slots holding an expert at once
    };
    stats st;

    std::string describe() const;

private:
    enum slot_state : uint8_t { SLOT_FREE = 0, SLOT_FILLING = 1, SLOT_VALID = 2, SLOT_WRITING = 3 };

    struct slot_t {
        int32_t  key      = -1; // il*n_expert + e
        uint8_t  state    = SLOT_FREE;
        uint8_t  count    = 0;  // decayed use count (CLOCK)
        uint8_t  done     = 0;  // fill_piece: mask of the complete tensors
        uint8_t  pieces   = 0;  // the slot is filled from pieces
        uint8_t  async    = 0;  // filled by an asynchronous read (see set_async)
        uint8_t  dup      = 0;  // the device also holds the expert (see set_dup)
        uint8_t  spec     = 0;  // filled from a routing profile and not used since (see set_spec)
        uint32_t pin      = 0;  // call epoch that pinned it
        uint32_t lease_id = 0;  // plan that leased it
        uint64_t read_seq  = 0; // last commit that reads it
        uint64_t write_seq = 0; // last commit that writes it
    };

    struct klass {
        std::vector<size_t> sizes;  // per tensor: bytes of one expert
        std::vector<size_t> offs;   // per tensor: offset of its area in a slot
        std::vector<size_t> caps;   // per tensor: area size
        size_t  slot_bytes = 0;
        int32_t first   = 0; // first slot id
        int32_t n       = 0; // slots of the budget
        int32_t n_alloc = 0; // slots with memory
        int32_t per_chunk = 0;
        int32_t hand    = 0; // CLOCK hand (relative)
        std::vector<ggml_backend_buffer_t> chunks;
        std::vector<int32_t> free_list;
        std::vector<int32_t> dups;      // slots marked dup (lazily pruned)
    };

    struct inflight {
        uint64_t seq;
        ggml_backend_event_t ev;
        ggml_backend_dev_t   dev;
    };

    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_type_t buft_pageable;
    size_t  pin_cap;
    bool    pin_failed = false;
    size_t  bytes_pageable = 0;
    size_t  budget;
    int     policy;
    int32_t n_expert;
    size_t  chunk_size;

    std::vector<int32_t> layer_class;              // [n_layer] class or -1
    std::vector<std::vector<int32_t>> slot_of;     // [n_layer][n_expert] slot or -1
    std::vector<klass>   classes;
    std::vector<slot_t>  slots;
    std::vector<int32_t> slot_class;
    std::vector<uint16_t> leads;   // [slot][max_tensors]
    std::vector<uint32_t> got;     // [slot][max_tensors] fill_piece progress
    int max_tensors = 0;

    std::vector<std::vector<int32_t>> spec_of; // [n_layer] slots marked by set_spec (lazily pruned), coldest last
    size_t   n_spec     = 0;

    std::vector<int32_t> n_key_layer; // [n_layer] slots whose key is of the layer (any state but free)
    std::vector<int32_t> quota;       // [n_layer] k19-l4held layer quota
    bool     quota_on   = false;
    int32_t  quota_victim(int32_t il, bool allow_evict, uint32_t min_age, bool take_spec, const std::vector<bool> * keep);

    int32_t  stream_il  = -1;
    bool     evict_done = true;

    uint32_t call_epoch = 1;
    uint32_t plan_id    = 1;
    uint64_t gen        = 0;
    size_t   n_held     = 0;
    size_t   bytes_alloc = 0;

    uint64_t seq_next = 1;      // seq of the next commit
    bool     marks    = false;  // marks since the last commit
    std::deque<inflight> pending;
    std::vector<int32_t> deferred; // released slots read by copies that are not committed yet
    std::unordered_map<ggml_backend_dev_t, std::vector<ggml_backend_event_t>> ev_pool;
    std::vector<ggml_backend_event_t> ev_all;
    uint64_t seq_done = 0;      // every commit up to it is complete

    bool alloc_chunk(klass & c);
    void wait_seq(uint64_t seq);
    void evict(int32_t s);
    void set_key(int32_t s, int32_t key);
    void spec_use(slot_t & sl);
    int32_t spec_victim(int32_t il, uint32_t min_age);
};
