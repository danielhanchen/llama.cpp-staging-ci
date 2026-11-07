#include "llama-moe-host-cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

static constexpr size_t k_page = 4096;

static size_t round_up_page(size_t x) {
    return (x + k_page - 1) / k_page * k_page;
}

llama_moe_host_plan llama_moe_host_decide(size_t e_bytes, size_t v_bytes, size_t mapped, size_t avail, size_t token_bytes, int64_t host_mib,
        int policy_env, bool force, bool disk_inclusive, double disk_min_frac) {
    llama_moe_host_plan p;
    p.tokens = token_bytes > 0 ? (double) v_bytes/token_bytes : 0.0;
    if (host_mib == 0 || e_bytes == 0) {
        p.why = "off";
        return p;
    }
    // the device slots already hold every streamed expert (e.g. --fit keeps the other layers on the CPU): nothing for RAM to hold
    if (e_bytes <= v_bytes && host_mib < 0) {
        p.why = "the device slots hold every streamed expert";
        return p;
    }
    // auto budget: what is available minus room for the page cache, the host compute buffers and the prefetch ring of the weight store;
    // the other mapped host weights (e.g. the experts of layers the CPU computes) are read through the page cache, the tier takes the
    // share of the streamed experts among them
    mapped = std::max(e_bytes, mapped);
    const size_t room = std::max<size_t>((size_t) 4 << 30, (size_t) (0.15*avail)) + std::min<size_t>((size_t) 4 << 30, avail/16);
    size_t budget = host_mib > 0 ? (size_t) host_mib << 20 : (avail > room ? (size_t) ((double) (avail - room)*e_bytes/mapped) : 0);
    p.policy = budget >= e_bytes ? 0 : 1;
    if (policy_env >= 0) {
        p.policy = policy_env;
    }
    p.spare = budget + v_bytes >= e_bytes;
    // no more than the experts it can hold; exclusive with spare room too: the copies of experts the device also holds are kept as the
    // first to go (a capacity of only the experts the device does not hold left 17 GiB of RAM unused at V32 / R29: the prompt seeds
    // the device, the RAM copies left, decode refilled few: 23% RAM hits, more disk bytes per token than without the tier); without
    // spare room the budget is below that cap anyway
    const size_t slack = (size_t) 64 << 20;
    budget = std::min(budget, e_bytes + slack);
    p.budget = budget;
    if (budget == 0) {
        p.why = "no budget";
        return p;
    }
    if (host_mib < 0 && !force && p.tokens < LLAMA_MOE_HOST_MIN_TOKENS) {
        // k19-disktier: the mapped host weights exceed what the tier may plan into (the page cache cannot hold the streamed experts
        // either): inclusive, so no miss costs a copy back (an explicitly forced exclusive policy keeps the old rule), where the budget
        // holds enough of the experts to beat the page cache it replaces
        if (disk_inclusive && token_bytes > 0 && mapped > avail && policy_env != LLAMA_MOE_HOST_EXCLUSIVE &&
                (double) budget >= disk_min_frac*(double) e_bytes) {
            p.policy = LLAMA_MOE_HOST_INCLUSIVE;
            p.spare  = false;
            p.on     = true;
            p.why    = "auto, disk tier below the device minimum: inclusive (LLAMA_MOE_HOST_DISK_INCLUSIVE=0 off)";
            return p;
        }
        p.why = "the device slots hold the experts of too few tokens, nearly every routed expert misses them";
        return p;
    }
    p.on  = true;
    p.why = host_mib > 0 ? "--moe-host-cache-mib" : (force ? "forced (LLAMA_MOE_HOST_CACHE=1)" : "auto");
    return p;
}

llama_moe_host_cache::llama_moe_host_cache(ggml_backend_buffer_type_t buft, size_t budget, int policy, int32_t n_expert,
        const std::vector<std::vector<size_t>> & layer_sizes, size_t chunk_size, ggml_backend_buffer_type_t buft_pageable, size_t pin_cap) :
        buft(buft), buft_pageable(buft_pageable), pin_cap(pin_cap), budget(budget), policy(policy), n_expert(n_expert), chunk_size(chunk_size ? chunk_size : (size_t) 256 << 20) {
    layer_class.assign(layer_sizes.size(), -1);
    slot_of.resize(layer_sizes.size());
    std::vector<int32_t> n_layers;
    for (size_t il = 0; il < layer_sizes.size(); ++il) {
        const auto & sz = layer_sizes[il];
        if (sz.empty()) {
            continue;
        }
        GGML_ASSERT(sz.size() <= 8);
        int32_t c = -1;
        for (size_t i = 0; i < classes.size(); ++i) {
            if (classes[i].sizes == sz) {
                c = (int32_t) i;
            }
        }
        if (c < 0) {
            klass k;
            k.sizes = sz;
            for (size_t s : sz) {
                // an unbuffered read starts at the 4 KiB boundary before the expert and ends at the one after it
                const size_t cap = round_up_page(s) + 2*k_page;
                k.offs.push_back(k.slot_bytes);
                k.caps.push_back(cap);
                k.slot_bytes += cap;
            }
            classes.push_back(k);
            n_layers.push_back(0);
            c = (int32_t) classes.size() - 1;
        }
        n_layers[c]++;
        layer_class[il] = c;
        slot_of[il].assign(n_expert, -1);
        max_tensors = std::max(max_tensors, (int) sz.size());
    }
    // each layout keeps the same share of its experts
    double total = 0.0;
    for (size_t c = 0; c < classes.size(); ++c) {
        total += (double) n_layers[c]*n_expert*classes[c].slot_bytes;
    }
    int32_t first = 0;
    for (size_t c = 0; c < classes.size(); ++c) {
        klass & k = classes[c];
        const double w   = (double) n_layers[c]*n_expert*k.slot_bytes;
        const double share = total > 0.0 ? (double) budget*w/total : 0.0;
        k.n = (int32_t) std::min<double>((double) n_layers[c]*n_expert, share/(double) k.slot_bytes);
        k.first = first;
        k.per_chunk = (int32_t) std::max<size_t>(1, this->chunk_size/k.slot_bytes);
        first += k.n;
    }
    slots.resize(first);
    slot_class.resize(first);
    for (size_t c = 0; c < classes.size(); ++c) {
        for (int32_t i = 0; i < classes[c].n; ++i) {
            slot_class[classes[c].first + i] = (int32_t) c;
        }
    }
    leads.assign((size_t) first*std::max(1, max_tensors), 0);
    got.assign((size_t) first*std::max(1, max_tensors), 0);
}

llama_moe_host_cache::~llama_moe_host_cache() {
    for (const auto & p : pending) {
        ggml_backend_event_synchronize(p.ev);
    }
    for (auto * ev : ev_all) {
        ggml_backend_event_free(ev);
    }
    for (auto & c : classes) {
        for (auto * b : c.chunks) {
            ggml_backend_buffer_free(b);
        }
    }
}

int32_t llama_moe_host_cache::n_slots() const {
    return (int32_t) slots.size();
}

bool llama_moe_host_cache::alloc_chunk(klass & c) {
    if (c.n_alloc >= c.n) {
        return false;
    }
    const int32_t n = std::min(c.per_chunk, c.n - c.n_alloc);
    const size_t  size = (size_t) n*c.slot_bytes + k_page;
    const int64_t t0 = ggml_time_us();
    ggml_backend_buffer_t b = nullptr;
    bool pageable = false;
    if (!pin_failed && pinned_bytes() + size <= pin_cap) {
        b = ggml_backend_buft_alloc_buffer(buft, size);
        if (b != nullptr && ggml_backend_buffer_get_type(b) != buft) {
            // e.g. the CUDA host buffer type falls back to pageable memory when pinning fails: no more pinning tries
            pin_failed = true;
            if (buft_pageable != nullptr) {
                pageable = true; // plain host memory: keep it
            } else {
                ggml_backend_buffer_free(b); // slow copies: the tier stops growing
                b = nullptr;
            }
        } else if (b == nullptr) {
            pin_failed = true;
        }
    }
    if (b == nullptr && buft_pageable != nullptr && (pin_failed || pinned_bytes() + size > pin_cap)) {
        b = ggml_backend_buft_alloc_buffer(buft_pageable, size);
        pageable = b != nullptr;
    }
    if (pageable) {
        st.pin_fallbacks++;
        bytes_pageable += ggml_backend_buffer_get_size(b);
    }
    st.alloc_us += ggml_time_us() - t0;
    if (b == nullptr) {
        // no more (pinned) memory: the tier keeps what it has
        st.alloc_fail++;
        c.n = c.n_alloc;
        return false;
    }
    c.chunks.push_back(b);
    bytes_alloc += ggml_backend_buffer_get_size(b);
    for (int32_t i = n - 1; i >= 0; --i) {
        c.free_list.push_back(c.first + c.n_alloc + i);
    }
    c.n_alloc += n;
    return true;
}

uint8_t * llama_moe_host_cache::area(int32_t s, int ip) const {
    const klass & c = classes[slot_class[s]];
    const int32_t r = s - c.first;
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(c.chunks[r / c.per_chunk]);
    base = (uint8_t *) (((uintptr_t) base + k_page - 1) / k_page * k_page);
    return base + (size_t) (r % c.per_chunk)*c.slot_bytes + c.offs[ip];
}

size_t llama_moe_host_cache::area_size(int32_t s, int ip) const {
    return classes[slot_class[s]].caps[ip];
}

size_t llama_moe_host_cache::expert_size(int32_t s, int ip) const {
    return classes[slot_class[s]].sizes[ip];
}

int llama_moe_host_cache::n_tensors(int32_t s) const {
    return (int) classes[slot_class[s]].sizes.size();
}

void llama_moe_host_cache::set_lead(int32_t s, int ip, size_t lead) {
    GGML_ASSERT(lead < k_page);
    leads[(size_t) s*max_tensors + ip] = (uint16_t) lead;
}

uint8_t * llama_moe_host_cache::data(int32_t s, int ip) const {
    return area(s, ip) + leads[(size_t) s*max_tensors + ip];
}

void llama_moe_host_cache::set_key(int32_t s, int32_t key) {
    slot_t & sl = slots[s];
    if (sl.key >= 0) {
        slot_of[sl.key / n_expert][sl.key % n_expert] = -1;
    }
    sl.key = key;
    if (key >= 0) {
        slot_of[key / n_expert][key % n_expert] = s;
    }
}

void llama_moe_host_cache::wait_seq(uint64_t seq) {
    if (seq == 0 || seq <= seq_done) {
        return;
    }
    GGML_ASSERT(seq < seq_next && "moe host cache: copies marked on a slot were not committed");
    while (!pending.empty() && pending.front().seq <= seq) {
        inflight f = pending.front();
        pending.pop_front();
        ggml_backend_event_synchronize(f.ev);
        ev_pool[f.dev].push_back(f.ev);
        seq_done = f.seq;
        st.waits++;
    }
    seq_done = std::max(seq_done, seq);
}

int32_t llama_moe_host_cache::find(int32_t il, int32_t e, bool touch) {
    if (!caches(il)) {
        return -1;
    }
    const int32_t s = slot_of[il][e];
    if (s < 0 || slots[s].state == SLOT_FILLING) {
        st.misses += touch;
        return -1;
    }
    slot_t & sl = slots[s];
    if (sl.state == SLOT_WRITING) {
        wait_seq(sl.write_seq);
        sl.state = SLOT_VALID;
    }
    if (touch) {
        st.hits++;
        sl.count = (uint8_t) std::min<int>(sl.count + 1, 7);
    }
    return s;
}

int32_t llama_moe_host_cache::held(int32_t il, int32_t e) const {
    if (!caches(il)) {
        return -1;
    }
    const int32_t s = slot_of[il][e];
    return s >= 0 && (slots[s].state == SLOT_VALID || slots[s].state == SLOT_WRITING) ? s : -1;
}

bool llama_moe_host_cache::contains(int32_t il, int32_t e) const {
    return caches(il) && slot_of[il][e] >= 0;
}

void llama_moe_host_cache::evict(int32_t s) {
    slot_t & sl = slots[s];
    if (sl.state == SLOT_VALID || sl.state == SLOT_WRITING) {
        n_held--;
    }
    set_key(s, -1);
    sl.state = SLOT_FREE;
    sl.dup   = 0;
    gen++;
}

void llama_moe_host_cache::set_dup(int32_t s, bool dup) {
    slot_t & sl = slots[s];
    if (dup && !sl.dup && (sl.state == SLOT_VALID || sl.state == SLOT_WRITING)) {
        classes[slot_class[s]].dups.push_back(s);
    }
    sl.dup = dup && (sl.state == SLOT_VALID || sl.state == SLOT_WRITING) ? 1 : 0;
}

int32_t llama_moe_host_cache::reserve(int32_t il, int32_t e, bool allow_evict, uint32_t min_age) {
    if (!caches(il) || slot_of[il][e] >= 0) {
        return -1;
    }
    klass & c = classes[layer_class[il]];
    if (c.n == 0) {
        st.no_slot++;
        return -1;
    }
    int32_t s = -1;
    if (!c.free_list.empty() || alloc_chunk(c)) {
        s = c.free_list.back();
        c.free_list.pop_back();
    }
    // a duplicate of an expert the device holds goes first, it is as good as a free slot (newest first; stale entries are pruned)
    for (size_t i = c.dups.size(); s < 0 && i-- > 0; ) {
        const int32_t cand = c.dups[i];
        slot_t & sl = slots[cand];
        if (!sl.dup || (sl.state != SLOT_VALID && sl.state != SLOT_WRITING)) {
            c.dups.erase(c.dups.begin() + i);
            continue;
        }
        if (sl.pin == call_epoch || sl.lease_id == plan_id || sl.read_seq == seq_next || sl.write_seq == seq_next ||
                (min_age > 0 && std::max(sl.read_seq, sl.write_seq) + min_age >= seq_next && std::max(sl.read_seq, sl.write_seq) > seq_done)) {
            continue;
        }
        c.dups.erase(c.dups.begin() + i);
        s = cand;
        evict(s);
        st.evicted++;
        st.dup_taken++;
    }
    if (s >= 0) {
        // taken from the duplicates
    } else if (allow_evict && c.n_alloc > 0) {
        // CLOCK over the decayed counts: the hand lowers the count of each slot it passes, the first at zero is evicted
        const int64_t max_steps = (int64_t) c.n_alloc*9;
        for (int64_t step = 0; step < max_steps; ++step) {
            const int32_t cand = c.first + c.hand;
            c.hand = (c.hand + 1) % c.n_alloc;
            slot_t & sl = slots[cand];
            // copies back from the device count once committed (reserve waits for them before the slot is written)
            if ((sl.state != SLOT_VALID && sl.state != SLOT_WRITING) || sl.pin == call_epoch || sl.lease_id == plan_id ||
                    sl.read_seq == seq_next || sl.write_seq == seq_next) {
                continue;
            }
            if (min_age > 0 && std::max(sl.read_seq, sl.write_seq) + min_age >= seq_next && std::max(sl.read_seq, sl.write_seq) > seq_done) {
                continue;
            }
            if (sl.count > 0) {
                sl.count--;
                continue;
            }
            s = cand;
            evict(s);
            st.evicted++;
            break;
        }
    }
    if (s < 0) {
        st.no_slot++;
        return -1;
    }
    slot_t & sl = slots[s];
    // the copies that still read the old bundle (or a released one) must be done before the slot is written
    wait_seq(sl.read_seq);
    wait_seq(sl.write_seq);
    set_key(s, il*n_expert + e);
    sl.state  = SLOT_FILLING;
    sl.count  = 1;
    sl.done   = 0;
    sl.pieces = 0;
    sl.async  = 0;
    sl.dup    = 0;
    for (int ip = 0; ip < max_tensors; ++ip) {
        leads[(size_t) s*max_tensors + ip] = 0;
        got[(size_t) s*max_tensors + ip]   = 0;
    }
    st.reserved++;
    return s;
}

void llama_moe_host_cache::publish(int32_t s) {
    slot_t & sl = slots[s];
    GGML_ASSERT(sl.state == SLOT_FILLING || sl.state == SLOT_WRITING);
    if (sl.state == SLOT_FILLING) {
        n_held++;
    }
    sl.state = sl.state == SLOT_WRITING ? SLOT_WRITING : SLOT_VALID;
    sl.pieces = 0;
    sl.async  = 0;
    st.published++;
    st.peak_held = std::max(st.peak_held, n_held);
    gen++;
}

void llama_moe_host_cache::abort(int32_t s) {
    slot_t & sl = slots[s];
    GGML_ASSERT(sl.state == SLOT_FILLING);
    set_key(s, -1);
    sl.state  = SLOT_FREE;
    sl.pieces = 0;
    sl.async  = 0;
    classes[slot_class[s]].free_list.push_back(s);
    st.aborted++;
}

void llama_moe_host_cache::release(int32_t s) {
    slot_t & sl = slots[s];
    if (sl.state == SLOT_FREE) {
        return;
    }
    if (sl.state == SLOT_FILLING) {
        GGML_ASSERT(!sl.async && "moe host cache: a slot being read asynchronously is released");
        abort(s);
        return;
    }
    evict(s);
    sl.lease_id = 0;
    if (sl.read_seq == seq_next || sl.write_seq == seq_next) {
        deferred.push_back(s); // read by copies not committed yet: free after the commit
    } else {
        classes[slot_class[s]].free_list.push_back(s);
    }
    st.released++;
}

void llama_moe_host_cache::release_key(int32_t il, int32_t e) {
    if (!caches(il) || slot_of[il][e] < 0) {
        return;
    }
    release(slot_of[il][e]);
}

void llama_moe_host_cache::begin_call() {
    call_epoch++;
}

void llama_moe_host_cache::pin(int32_t s) {
    slots[s].pin = call_epoch;
}

void llama_moe_host_cache::new_plan() {
    plan_id++;
    for (size_t s = 0; s < slots.size(); ++s) {
        if (slots[s].state == SLOT_FILLING && slots[s].pieces && !slots[s].async) {
            abort((int32_t) s);
        }
    }
}

void llama_moe_host_cache::lease(int32_t s) {
    slots[s].lease_id = plan_id;
}

bool llama_moe_host_cache::filling(int32_t s) const {
    return slots[s].state == SLOT_FILLING;
}

void llama_moe_host_cache::set_async(int32_t s, uint8_t count) {
    slot_t & sl = slots[s];
    GGML_ASSERT(sl.state == SLOT_FILLING && !sl.pieces);
    sl.async = 1;
    sl.count = count;
}

bool llama_moe_host_cache::fill_piece(int32_t s, int ip, size_t off, const uint8_t * p, size_t n) {
    slot_t & sl = slots[s];
    if (sl.state != SLOT_FILLING || sl.async) {
        return false;
    }
    sl.pieces = 1;
    const size_t size = expert_size(s, ip);
    uint32_t & g = got[(size_t) s*max_tensors + ip];
    if (off != g || off + n > size) {
        return false; // out of order or a repeated piece: ignored, the slot stays incomplete
    }
    memcpy(area(s, ip) + off, p, n);
    g += (uint32_t) n;
    if (g == size) {
        sl.done |= (uint8_t) (1u << ip);
    }
    if (sl.done == (uint8_t) ((1u << n_tensors(s)) - 1)) {
        publish(s);
        return true;
    }
    return false;
}

void llama_moe_host_cache::mark_read(int32_t s) {
    slots[s].read_seq = seq_next;
    marks = true;
}

void llama_moe_host_cache::mark_write(int32_t s) {
    slot_t & sl = slots[s];
    sl.write_seq = seq_next;
    if (sl.state == SLOT_FILLING) {
        n_held++;
        st.published++;
        st.peak_held = std::max(st.peak_held, n_held);
        gen++;
    }
    sl.state = SLOT_WRITING;
    marks = true;
}

void llama_moe_host_cache::commit(ggml_backend_t backend) {
    if (!marks) {
        return;
    }
    marks = false;
    for (int32_t s : deferred) {
        classes[slot_class[s]].free_list.push_back(s);
    }
    deferred.clear();
    const uint64_t seq = seq_next++;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_event_t ev = nullptr;
    auto & pool = ev_pool[dev];
    if (!pool.empty()) {
        ev = pool.back();
        pool.pop_back();
    } else if (dev != nullptr) {
        ev = ggml_backend_event_new(dev);
        if (ev != nullptr) {
            ev_all.push_back(ev);
        }
    }
    if (ev == nullptr) {
        // no events (e.g. the CPU backend): the copies are done once the backend is synchronized
        ggml_backend_synchronize(backend);
        while (!pending.empty()) {
            wait_seq(pending.back().seq);
        }
        seq_done = seq;
        return;
    }
    ggml_backend_event_record(ev, backend);
    pending.push_back({ seq, ev, dev });
    // the old ones are long done, waiting keeps the list short
    while (pending.size() > 256) {
        wait_seq(pending.front().seq);
    }
}

std::string llama_moe_host_cache::describe() const {
    char buf[256];
    std::string res;
    for (const auto & c : classes) {
        size_t sum = 0;
        for (size_t s : c.sizes) {
            sum += s;
        }
        snprintf(buf, sizeof(buf), "%s%d slots of %.2f MiB (%zu tensors, %.2f MiB of expert)", res.empty() ? "" : ", ", c.n, c.slot_bytes/1048576.0, c.sizes.size(),
                sum/1048576.0);
        res += buf;
    }
    return res;
}
