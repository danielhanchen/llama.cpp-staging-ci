// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
//
// k19-kvwin: the recent-window ring of a KV cache. Every token is stored in the cache (the "body", typically q4_0) and, while it is
// among the last W positions of its sequence or among its first S positions, in a ring slot of a wider type (typically q8_0) written
// from the same rotated K / V. Query row r (position p) reads a visible cell from the ring iff the cell holds a live slot and
// (pos > p - W or pos < S); otherwise from the body. The ring holds n_seq_per_stream*(W + S) + n_ubatch slots per stream, so every
// cell in a row's window has one in normal operation (appends, rollback of rejected drafts). See plans/k19_kvwin.md.

#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

static ggml_type kv_win_type_from_name(const char * name, ggml_type def) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (tn != nullptr && strcmp(tn, name) == 0) {
            return (ggml_type) t;
        }
    }
    LLAMA_LOG_WARN("%s: k19-kvwin: unknown type %s, using %s\n", __func__, name, ggml_type_name(def));
    return def;
}

llama_kv_win_params llama_kv_win_params_from(const llama_context_params & params) {
    llama_kv_win_params p;
    p.n_win   = params.kv_win;
    p.n_sinks = params.kv_win_sinks;
    p.type_k  = params.type_k_win;
    p.type_v  = params.type_v_win;
    if (const char * e = getenv("LLAMA_KV_WIN")) {
        p.n_win = atoi(e);
    }
    if (const char * e = getenv("LLAMA_KV_WIN_SINKS")) {
        p.n_sinks = std::max(0, atoi(e));
        // sinks alone: an empty recent window
        if (p.n_win < 0) {
            p.n_win = 0;
        }
    }
    if (const char * e = getenv("LLAMA_KV_WIN_TYPE")) {
        p.type_k = p.type_v = kv_win_type_from_name(e, p.type_k);
    }
    if (const char * e = getenv("LLAMA_KV_WIN_TYPE_K")) {
        p.type_k = kv_win_type_from_name(e, p.type_k);
    }
    if (const char * e = getenv("LLAMA_KV_WIN_TYPE_V")) {
        p.type_v = kv_win_type_from_name(e, p.type_v);
    }
    if (p.n_win < -1) {
        p.n_win = -1;
    }
    return p;
}

static thread_local const llama_kv_win_params * g_kv_win_creating = nullptr;

const llama_kv_win_params * llama_kv_win_creating() {
    return g_kv_win_creating;
}

llama_kv_win_creating_scope::llama_kv_win_creating_scope(const llama_kv_win_params * p) {
    g_kv_win_creating = p != nullptr && p->n_win >= 0 ? p : nullptr;
}

llama_kv_win_creating_scope::~llama_kv_win_creating_scope() {
    g_kv_win_creating = nullptr;
}

ggml_type llama_kv_win_window_type(ggml_type t, bool is_k) {
    const llama_kv_win_params * w = g_kv_win_creating;
    // (the control arm, W 0 and S 0, keeps every cache as without the mode)
    if (w == nullptr || !ggml_is_quantized(t) || (w->n_win <= 0 && w->n_sinks <= 0)) {
        return t;
    }
    return is_k ? w->type_k : w->type_v;
}

bool llama_kv_cache::win_init(const llama_kv_win_params & p) {
    if (p.n_win < 0) {
        return false;
    }
    GGML_ASSERT(!win_on() && "win_init called twice");

    const char * why = nullptr;
    if (other) {
        why = "the cache shares the cells of another context";
    } else if (v_trans) {
        why = "flash attention is off (the V cache is transposed)";
    } else if (layers.empty()) {
        why = "no attention layer";
    } else if (!ggml_is_quantized(layers[0].k->type) && !(layers[0].v != nullptr && ggml_is_quantized(layers[0].v->type))) {
        why = "neither K nor V is quantized";
    }
    if (why != nullptr) {
        LLAMA_LOG_WARN("%s: k19-kvwin: recent-window ring off: %s\n", __func__, why);
        return false;
    }

    const uint32_t kv_size     = get_size();
    const uint32_t n_seq_strm  = n_stream == 1 ? n_seq_max : 1;
    const uint64_t n_want      = (uint64_t) n_seq_strm*((uint64_t) p.n_win + (uint64_t) std::max(0, p.n_sinks)) + p.n_ubatch;
    // a ring larger than the cache never fills: cap it at the cells
    uint64_t r = std::min<uint64_t>(n_want, kv_size);
    r = GGML_PAD(std::max<uint64_t>(r, 1), 256);

    const ggml_type tk = p.type_k;
    // (MLA: no V cache, V is a view of K: the ring likewise)
    const ggml_type tv_cache = layers[0].v != nullptr ? layers[0].v->type : layers[0].k->type;
    const ggml_type tv = layers[0].v == nullptr ? tk : ggml_is_quantized(tv_cache) ? p.type_v : tv_cache;

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    for (auto & layer : layers) {
        GGML_ASSERT(layer.k != nullptr && layer.k->buffer != nullptr);
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(layer.k->buffer);
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*layers.size()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            it = ctx_map.emplace(buft, ggml_context_ptr(ggml_init(params))).first;
        }
        ggml_context * ctx = it->second.get();
        layer.k_win = ggml_new_tensor_3d(ctx, tk, layer.k->ne[0], r + 1, n_stream);
        ggml_format_name(layer.k_win, "%s_win", layer.k->name);
        if (layer.v != nullptr) {
            layer.v_win = ggml_new_tensor_3d(ctx, tv, layer.v->ne[0], r + 1, n_stream);
            ggml_format_name(layer.v_win, "%s_win", layer.v->name);
        }
    }

    size_t total = 0;
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            // a measurement: a dummy buffer (memory_breakdown counts the tensors), as the cache does
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
            total += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
            if (buf == nullptr) {
                throw std::runtime_error("failed to allocate the k19-kvwin ring buffer");
            }
            ggml_backend_buffer_clear(buf, 0);
            total += ggml_backend_buffer_get_size(buf);
        }
        win_ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    win = p;
    win.n_sinks = std::max(0, p.n_sinks);
    win.type_k = tk;
    win.type_v = tv;
    win_r = (uint32_t) r;

    // the two-source flash attention where every ring sits on a device that has it (LLAMA_KV_WIN_FUSED=0: the concat stopgap)
    {
        const char * e = getenv("LLAMA_KV_WIN_FUSED");
        win_fused_ = e == nullptr || atoi(e) != 0;
        for (const auto & layer : layers) {
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(layer.k->buffer);
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            // the plain CPU buffer type has no device: the CPU backend reads it
            if (dev == nullptr && buft == ggml_backend_cpu_buffer_type()) {
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            }
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg == nullptr || ggml_backend_reg_get_proc_address(reg, "ggml_backend_fa_src2") == nullptr) {
                win_fused_ = false;
            }
        }
    }

    v_rings.resize(n_stream);
    for (auto & rg : v_rings) {
        rg.reset(win_r, kv_size);
    }

    LLAMA_LOG_INFO("%s: k19-kvwin: ring of %u slots x %u streams (last %d + first %d positions, %u seqs per stream, ubatch %u) "
            "K %s / V %s beside the cache K %s / V %s: %.2f MiB, attention %s\n", __func__, win_r, n_stream, win.n_win, win.n_sinks, n_seq_strm,
            p.n_ubatch, ggml_type_name(tk), ggml_type_name(tv), ggml_type_name(layers[0].k->type), ggml_type_name(tv_cache), total/1048576.0,
            win_fused_ ? "two-source flash attention" : "concat stopgap (no two-source flash attention on the device)");

    return true;
}

bool llama_kv_cache::win_slot_live(uint32_t strm, int32_t j) const {
    const auto & rg = v_rings[strm];
    const int32_t c = rg.slot_cell[j];
    return c >= 0 && rg.cell_slot[c] == j && !v_cells[strm].is_empty(c);
}

void llama_kv_cache::win_release_cell(uint32_t strm, uint32_t c) {
    auto & rg = v_rings[strm];
    const int32_t j = rg.cell_slot[c];
    if (j >= 0 && rg.slot_cell[j] == (int32_t) c) {
        rg.slot_cell[j] = -1;
        rg.free_slots.insert(j);
    }
    rg.cell_slot[c] = -1;
}

int32_t llama_kv_cache::win_take(uint32_t strm, const llama_pos * lo) {
    auto & rg = v_rings[strm];
    if (rg.free_slots.empty()) {
        // reclaim every slot whose cell is empty, or out of the window of every sequence it belongs to (and not a sink)
        const auto & cells = v_cells[strm];
        for (uint32_t j = 0; j < win_r; ++j) {
            const int32_t c = rg.slot_cell[j];
            if (c < 0) {
                continue; // already free
            }
            bool keep = rg.cell_slot[c] == (int32_t) j && !cells.is_empty(c);
            if (keep) {
                const llama_pos pos = cells.pos_get(c);
                keep = pos < win.n_sinks;
                const auto & bits = cells.seq_get_all(c);
                for (uint32_t q = 0; !keep && q < n_seq_max && q < (uint32_t) bits.size(); ++q) {
                    keep = bits.test(q) && (int64_t) pos > (int64_t) lo[q] - win.n_win;
                }
            }
            if (!keep) {
                if (rg.cell_slot[c] == (int32_t) j) {
                    rg.cell_slot[c] = -1;
                }
                rg.slot_cell[j] = -1;
                rg.free_slots.insert((int32_t) j);
            }
        }
    }
    if (rg.free_slots.empty()) {
        return -1;
    }
    const int32_t j = *rg.free_slots.begin();
    rg.free_slots.erase(rg.free_slots.begin());
    return j;
}

void llama_kv_cache::win_assign(const slot_info & sinfo, const llama_ubatch & ubatch) {
    if (!win_on() || win_sim || other) {
        return;
    }

    // the lowest position a later row of each sequence can have: its lowest position in this ubatch, else the one after its last
    llama_pos lo[LLAMA_MAX_SEQ];
    for (uint32_t q = 0; q < LLAMA_MAX_SEQ; ++q) {
        lo[q] = std::numeric_limits<llama_pos>::max();
    }
    for (uint32_t q = 0; q < n_seq_max && q < LLAMA_MAX_SEQ; ++q) {
        lo[q] = v_cells[seq_to_stream[q]].seq_pos_max(q) + 1;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id q = ubatch.seq_id[i][k];
            lo[q] = std::min(lo[q], ubatch.pos[i]);
        }
    }

    static std::atomic<int> n_full_log { 0 };
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const uint32_t strm = sinfo.strm[s];
        auto & rg = v_rings[strm];
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;
            const uint32_t c = sinfo.idxs[s][ii];
            const llama_pos p = ubatch.pos[i];
            const bool need = !win_no_assign && (win.n_win > 0 || p < win.n_sinks);
            if (!need) {
                win_release_cell(strm, c);
                continue;
            }
            const int32_t j0 = rg.cell_slot[c];
            if (j0 >= 0 && rg.slot_cell[j0] == (int32_t) c) {
                continue; // the cell keeps its slot (the new token overwrites the row)
            }
            rg.cell_slot[c] = -1;
            const int32_t j = win_take(strm, lo);
            if (j < 0) {
                if (n_full_log.fetch_add(1) < 4) {
                    LLAMA_LOG_WARN("%s: k19-kvwin: ring full (%u slots, stream %u): the cell of position %d reads the cache\n",
                            __func__, win_r, strm, p);
                }
                continue;
            }
            rg.slot_cell[j] = (int32_t) c;
            rg.cell_slot[c] = j;
        }
    }
}

std::vector<std::pair<uint32_t, uint32_t>> llama_kv_cache::win_slots(uint32_t strm) const {
    std::vector<std::pair<uint32_t, uint32_t>> res;
    if (!win_on()) {
        return res;
    }
    for (uint32_t j = 0; j < win_r; ++j) {
        if (win_slot_live(strm, (int32_t) j)) {
            res.emplace_back((uint32_t) v_rings[strm].slot_cell[j], j);
        }
    }
    return res;
}

ggml_tensor * llama_kv_cache::build_input_win_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    if (!win_on()) {
        return nullptr;
    }
    ggml_tensor * res = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, ubatch.n_tokens);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_win_idxs");
    return res;
}

ggml_tensor * llama_kv_cache::build_input_win_mask(ggml_context * ctx, int64_t n_tps, int64_t n_stream_ub) const {
    if (!win_on()) {
        return nullptr;
    }
    ggml_tensor * res = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, win_r, n_tps, 1, n_stream_ub);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_win_mask");
    return res;
}

void llama_kv_cache::set_input_win(ggml_tensor * idxs, ggml_tensor * mask, ggml_tensor * kq_mask, const llama_ubatch * ubatch,
        const slot_info & sinfo, ggml_tensor * cells_in) const {
    GGML_ASSERT(win_on());
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (uint32_t) (sinfo.size()*sinfo.n_stream()));

    if (idxs != nullptr && idxs->buffer != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(idxs->buffer));
        int64_t * data = (int64_t *) idxs->data;
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const uint32_t strm = sinfo.strm[s];
            const auto & rg = v_rings[strm];
            const int64_t offs = (int64_t) strm*(win_r + 1);
            for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
                const uint32_t c = sinfo.idxs[s][ii];
                const int32_t  j = rg.cell_slot[c];
                const bool ok = j >= 0 && rg.slot_cell[j] == (int32_t) c;
                data[s*sinfo.size() + ii] = offs + (ok ? j : (int64_t) win_r);
            }
        }
    }

    // the cell of every live slot, per token (a sparse attention gathers its selection at them)
    if (cells_in != nullptr && cells_in->buffer != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cells_in->buffer) && cells_in->ne[0] == (int64_t) win_r && cells_in->ne[1] == (int64_t) n_tokens);
        GGML_ASSERT(sinfo.n_stream() == 1);
        const uint32_t strm = sinfo.strm[0];
        const auto & rg = v_rings[strm];
        int32_t * data = (int32_t *) cells_in->data;
        for (uint32_t j = 0; j < win_r; ++j) {
            data[j] = win_slot_live(strm, (int32_t) j) ? rg.slot_cell[j] : 0;
        }
        for (uint32_t i = 1; i < n_tokens; ++i) {
            memcpy(data + (size_t) i*win_r, data, win_r*sizeof(int32_t));
        }
    }

    if (mask == nullptr || mask->buffer == nullptr) {
        return;
    }
    // LLAMA_KV_WIN_PROF=1: the host time of the ring masks, logged every 256 calls
    static const bool prof = getenv("LLAMA_KV_WIN_PROF") != nullptr && atoi(getenv("LLAMA_KV_WIN_PROF")) != 0;
    const int64_t t0 = prof ? ggml_time_us() : 0;
    struct prof_guard {
        int64_t t0;
        uint32_t n;
        ~prof_guard() {
            if (t0 == 0) {
                return;
            }
            static std::atomic<int64_t> us { 0 };
            static std::atomic<int64_t> rows { 0 };
            static std::atomic<int64_t> calls { 0 };
            us += ggml_time_us() - t0;
            rows += n;
            if (++calls % 256 == 0 || n >= 512) {
                LLAMA_LOG_INFO("k19-kvwin prof: %lld mask builds, %lld rows, %.3f ms total, %.2f us per row\n", (long long) calls.load(),
                        (long long) rows.load(), us.load()/1e3, (double) us.load()/std::max<int64_t>(1, rows.load()));
            }
        }
    } pg { t0, n_tokens };
    GGML_ASSERT(ggml_backend_buffer_is_host(mask->buffer));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && mask->ne[0] == (int64_t) win_r);

    const int64_t n_stream_ub = mask->ne[3];
    const int64_t n_tps       = mask->ne[1];
    GGML_ASSERT(n_tps*n_stream_ub == (int64_t) n_tokens);

    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);
    ggml_fp16_t * G = (ggml_fp16_t *) mask->data;

    // no body mask (a graph that stores K / V without attending): nothing reads the ring mask
    if (kq_mask == nullptr || kq_mask->buffer == nullptr) {
        std::fill(G, G + ggml_nelements(mask), f16_ninf);
        return;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer) && kq_mask->type == GGML_TYPE_F16);
    GGML_ASSERT(kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream_ub);
    const int64_t n_kv = kq_mask->ne[0];
    ggml_fp16_t * B = (ggml_fp16_t *) kq_mask->data;

    struct entry {
        llama_pos pos;
        int32_t   slot;
        int32_t   cell;
    };
    std::vector<entry> ents;
    uint64_t n_moved = 0;

    for (int64_t s = 0; s < n_stream_ub; ++s) {
        const uint32_t strm = seq_to_stream[ubatch->seq_id[s*n_tps][0]];
        const auto & rg    = v_rings[strm];
        const auto & cells = v_cells[strm];

        ents.clear();
        for (uint32_t j = 0; j < win_r; ++j) {
            const int32_t c = rg.slot_cell[j];
            if (c >= 0 && c < n_kv && rg.cell_slot[c] == (int32_t) j && !cells.is_empty(c)) {
                ents.push_back({ cells.pos_get(c), (int32_t) j, c });
            }
        }
        std::sort(ents.begin(), ents.end(), [](const entry & a, const entry & b) { return a.pos < b.pos || (a.pos == b.pos && a.slot < b.slot); });

        // the sinks: a prefix of the sorted entries
        const auto it_s = std::lower_bound(ents.begin(), ents.end(), (llama_pos) win.n_sinks, [](const entry & e, llama_pos v) { return e.pos < v; });

        // rows [ii0, ii1) of this stream; each row writes only its own ring and body rows
        const auto rows = [&](int64_t ii0, int64_t ii1, uint64_t & moved) {
            for (int64_t ii = ii0; ii < ii1; ++ii) {
                const int64_t   i = s*n_tps + ii;
                const llama_pos p = ubatch->pos[i];
                ggml_fp16_t * Bi = B + i*n_kv;
                ggml_fp16_t * Gi = G + i*win_r;
                std::fill(Gi, Gi + win_r, f16_ninf);

                const auto move = [&](const entry & e) {
                    const ggml_fp16_t v = Bi[e.cell];
                    if (v == f16_ninf) {
                        return;
                    }
                    Gi[e.slot] = v;
                    Bi[e.cell] = f16_ninf;
                    ++moved;
                };
                for (auto it = ents.begin(); it != it_s; ++it) {
                    move(*it);
                }
                // the window (p - W, p]
                const llama_pos lo_w = (llama_pos) std::max<int64_t>((int64_t) p - win.n_win + 1, win.n_sinks);
                auto it = std::lower_bound(it_s, ents.end(), lo_w, [](const entry & e, llama_pos v) { return e.pos < v; });
                for (; it != ents.end() && it->pos <= p; ++it) {
                    move(*it);
                }
            }
        };
        // prompt ubatches on up to 8 threads (LLAMA_KV_WIN_THREADS)
        static const int max_threads = [] {
            const char * e = getenv("LLAMA_KV_WIN_THREADS");
            return e != nullptr ? std::max(1, atoi(e)) : (int) std::max(1u, std::min(8u, std::thread::hardware_concurrency()/2));
        }();
        const int64_t n_thr = n_tps >= 256 ? std::min<int64_t>(max_threads, n_tps/128) : 1;
        if (n_thr <= 1) {
            rows(0, n_tps, n_moved);
        } else {
            std::vector<std::thread> workers;
            std::vector<uint64_t> moved(n_thr, 0);
            const int64_t chunk = (n_tps + n_thr - 1)/n_thr;
            for (int64_t t = 1; t < n_thr; ++t) {
                const int64_t ii0 = std::min(n_tps, t*chunk);
                const int64_t ii1 = std::min(n_tps, ii0 + chunk);
                workers.emplace_back([&rows, &moved, ii0, ii1, t]() { rows(ii0, ii1, moved[t]); });
            }
            rows(0, std::min(n_tps, chunk), moved[0]);
            for (auto & w : workers) {
                w.join();
            }
            for (uint64_t m : moved) {
                n_moved += m;
            }
        }
    }

    static std::atomic<int> n_log { 0 };
    if (debug > 0 || n_log.fetch_add(1) < 2) {
        LLAMA_LOG_INFO("%s: k19-kvwin: %llu (row, cell) pairs read from the ring (%u tokens, %lld cells, %u slots)\n", __func__,
                (unsigned long long) n_moved, n_tokens, (long long) n_kv, win_r);
    }
}

ggml_tensor * llama_kv_cache::get_k_win(ggml_context * ctx, int32_t il, const slot_info & sinfo) const {
    const auto & layer = layers[map_layer_ids.at(il)];
    ggml_tensor * w = layer.k_win;
    if (w == nullptr) {
        return nullptr;
    }
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    const int64_t  d  = hparams.n_embd_head_k(il);
    return ggml_view_4d(ctx, w,
            d, hparams.n_head_kv(il), win_r, ns,
            ggml_row_size(w->type, d),
            w->nb[1],
            w->nb[2],
            w->nb[2]*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v_win(ggml_context * ctx, int32_t il, const slot_info & sinfo) const {
    const auto & layer = layers[map_layer_ids.at(il)];
    ggml_tensor * w = layer.v_win;
    if (w == nullptr) {
        return nullptr;
    }
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    const int64_t  d  = hparams.n_embd_head_v(il);
    return ggml_view_4d(ctx, w,
            d, hparams.n_head_kv(il), win_r, ns,
            ggml_row_size(w->type, d),
            w->nb[1],
            w->nb[2],
            w->nb[2]*sinfo.s0);
}

static ggml_tensor * kv_win_store(ggml_context * ctx, ggml_tensor * dst, ggml_tensor * cur, ggml_tensor * idxs) {
    const int64_t n_embd_gqa = cur->ne[0]*cur->ne[1];
    const int64_t n_tokens   = cur->ne[2];
    GGML_ASSERT(ggml_row_size(cur->type, cur->ne[0]) == cur->nb[1]);
    GGML_ASSERT(n_embd_gqa == dst->ne[0]);
    cur = ggml_view_2d(ctx, cur, n_embd_gqa, n_tokens, cur->nb[2], 0);
    // the streams are merged: the idxs are global (strm*(win_r + 1) + slot)
    ggml_tensor * d2 = ggml_reshape_2d(ctx, dst, dst->ne[0], dst->ne[1]*dst->ne[2]);
    return ggml_set_rows(ctx, d2, cur, idxs);
}

ggml_tensor * llama_kv_cache::cpy_k_win(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il) const {
    ggml_tensor * w = layers[map_layer_ids.at(il)].k_win;
    return w != nullptr && idxs != nullptr ? kv_win_store(ctx, w, k_cur, idxs) : nullptr;
}

ggml_tensor * llama_kv_cache::cpy_v_win(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il) const {
    ggml_tensor * w = layers[map_layer_ids.at(il)].v_win;
    return w != nullptr && idxs != nullptr && v_cur != nullptr ? kv_win_store(ctx, w, v_cur, idxs) : nullptr;
}

void llama_kv_cache::set_input_win_dev(ggml_tensor * rcells, ggml_tensor * flags, int64_t n_kv, int64_t n_stream, const llama_ubatch * ubatch, bool used) const {
    GGML_ASSERT(win_on());
    // (an input the graph does not read has no buffer: nothing to set)
    const bool r_ok = rcells != nullptr && rcells->buffer != nullptr;
    const bool f_ok = flags  != nullptr && flags->buffer  != nullptr;
    GGML_ASSERT(!r_ok || (rcells->ne[0] == 5 && rcells->ne[1] == (int64_t) win_r && rcells->ne[2] == n_stream));
    GGML_ASSERT(!f_ok || (flags->ne[0] == n_kv && flags->ne[1] == n_stream));
    const int64_t n_tps = (int64_t) ubatch->n_tokens/n_stream;
    const bool is_2d = ubatch->is_pos_2d();

    // written in place where the inputs are host memory, else staged and uploaded (as set_input_kq_mask does)
    const bool r_host = r_ok && ggml_backend_buffer_is_host(rcells->buffer);
    const bool f_host = f_ok && ggml_backend_buffer_is_host(flags->buffer);
    std::vector<int32_t> r_stage(r_host ? 0 : 5*(size_t) win_r*n_stream), f_stage(f_host ? 0 : (size_t) n_kv*n_stream);
    int32_t * rc = r_host ? (int32_t *) rcells->data : r_stage.data();
    int32_t * fl = f_host ? (int32_t *) flags->data  : f_stage.data();

    for (int64_t s = 0; s < n_stream; ++s) {
        const uint32_t strm  = seq_to_stream[ubatch->seq_id[s*n_tps][0]];
        const auto &   rg    = v_rings[strm];
        const auto &   cells = v_cells[strm];
        int32_t * fs = fl + s*n_kv;
        std::fill(fs, fs + n_kv, 0);
        for (uint32_t j = 0; j < win_r; ++j) {
            int32_t * cj = rc + 5*((int64_t) s*win_r + j);
            const int32_t c = rg.slot_cell[j];
            if (!win_slot_live(strm, (int32_t) j) || c >= n_kv) {
                cj[0] = INT32_MIN; cj[1] = 0; cj[2] = 0; cj[3] = 0; cj[4] = 0;
                continue;
            }
            // the encoding of llama_kv_cache::set_input_kq_mask (device path): pos, sequence bits 0..31 / 32..63, 2D position
            uint32_t lo = 0, hi = 0;
            const auto & bits = cells.seq_get_all(c);
            for (uint32_t q = 0; q < 64 && q < n_seq_max && q < (uint32_t) bits.size(); ++q) {
                if (bits.test(q)) {
                    (q < 32 ? lo : hi) |= 1u << (q & 31);
                }
            }
            cj[0] = cells.pos_get(c); cj[1] = (int32_t) lo; cj[2] = (int32_t) hi;
            if (is_2d) {
                const auto & ext = cells.ext_get(c);
                cj[3] = ext.x; cj[4] = ext.y;
            } else {
                cj[3] = 0; cj[4] = 0;
            }
            if (used) {
                fs[c] = 1;
            }
        }
    }
    if (r_ok && !r_host) {
        ggml_backend_tensor_set(rcells, r_stage.data(), 0, ggml_nbytes(rcells));
    }
    if (f_ok && !f_host) {
        ggml_backend_tensor_set(flags, f_stage.data(), 0, ggml_nbytes(flags));
    }
    // with no attention reading the ring the ring cells stay as they are: the ring mask is then unused (flags 0)
}

void llama_kv_cache::set_input_win_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int32_t * data = (int32_t *) dst->data;
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t j = 0; j <= win_r; ++j) {
            int32_t sh = 0;
            if (j < win_r && win_slot_live(s, (int32_t) j)) {
                sh = cells.get_shift(v_rings[s].slot_cell[j]);
            }
            data[s*(win_r + 1) + j] = sh;
        }
    }
}

//
// llama_kv_cache_context
//

ggml_tensor * llama_kv_cache_context::build_input_win_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_win_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_win_mask(ggml_context * ctx, const llama_ubatch & ubatch, const llama_cparams & cparams) const {
    const int64_t n_stream_ub = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
    return kv->build_input_win_mask(ctx, ubatch.n_tokens/n_stream_ub, n_stream_ub);
}

void llama_kv_cache_context::set_input_win(ggml_tensor * idxs, ggml_tensor * mask, ggml_tensor * kq_mask, const llama_ubatch * ubatch,
        ggml_tensor * cells) const {
    kv->set_input_win(idxs, mask, kq_mask, ubatch, sinfos[i_cur], cells);
}

void llama_kv_cache_context::set_input_win_dev(ggml_tensor * rcells, ggml_tensor * flags, int64_t n_kv, int64_t n_stream, const llama_ubatch * ubatch, bool used) const {
    kv->set_input_win_dev(rcells, flags, n_kv, n_stream, ubatch, used);
}

ggml_tensor * llama_kv_cache_context::get_k_win(ggml_context * ctx, int32_t il) const {
    return kv->get_k_win(ctx, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v_win(ggml_context * ctx, int32_t il) const {
    return kv->get_v_win(ctx, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_win(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il) const {
    return kv->cpy_k_win(ctx, k_cur, idxs, il);
}

ggml_tensor * llama_kv_cache_context::cpy_v_win(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il) const {
    return kv->cpy_v_win(ctx, v_cur, idxs, il);
}

//
// state: after the K / V rows of a stream, one byte per saved cell (1: its ring row follows), the ring types, then per layer the K
// ring rows of those cells and the V ring rows. A blob without them (written without the ring) leaves the restored cells without a
// slot: they read their body values (the only deviation of a restore from the run that wrote it)
//

void llama_kv_cache::state_write_win(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const uint32_t strm = cr.strm;
    const auto & rg = v_rings[strm];

    std::vector<uint8_t> flags;
    std::vector<int32_t> slots;
    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const int32_t j = rg.cell_slot[i];
            const bool live = j >= 0 && win_slot_live(strm, j);
            flags.push_back(live ? 1 : 0);
            if (live) {
                slots.push_back(j);
            }
        }
    }
    const uint32_t n_flags = flags.size();
    io.write(&n_flags, sizeof(n_flags));
    io.write(flags.data(), flags.size());

    const int32_t tk = (int32_t) win.type_k;
    const int32_t tv = (int32_t) win.type_v;
    io.write(&tk, sizeof(tk));
    io.write(&tv, sizeof(tv));

    for (int kv = 0; kv < 2; ++kv) {
        for (const auto & layer : layers) {
            ggml_tensor * w = kv == 0 ? layer.k_win : layer.v_win;
            if (w == nullptr) {
                continue;
            }
            const uint64_t row = w->nb[1];
            io.write(&row, sizeof(row));
            for (const int32_t j : slots) {
                io.write_tensor(w, ((size_t) strm*(win_r + 1) + j)*row, row);
            }
        }
    }
}

bool llama_kv_cache::state_read_win(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    uint32_t n_flags = 0;
    io.read(&n_flags, sizeof(n_flags));
    if (n_flags != cell_count) {
        LLAMA_LOG_ERROR("%s: k19-kvwin: ring flags for %u cells, %u restored\n", __func__, n_flags, cell_count);
        return false;
    }
    std::vector<uint8_t> flags(n_flags);
    io.read(flags.data(), flags.size());
    int32_t tk = 0;
    int32_t tv = 0;
    io.read(&tk, sizeof(tk));
    io.read(&tv, sizeof(tv));

    const bool use = win_on() && tk == (int32_t) win.type_k && tv == (int32_t) win.type_v && !sinfo.empty();
    static std::atomic<int> n_log { 0 };
    if (!use && win_on() && n_log.fetch_add(1) < 2) {
        LLAMA_LOG_WARN("%s: k19-kvwin: the blob's ring is %s / %s, this cache's %s / %s: the restored cells read the cache\n", __func__,
                ggml_type_name((ggml_type) tk), ggml_type_name((ggml_type) tv), ggml_type_name(win.type_k), ggml_type_name(win.type_v));
    }

    // slots for the flagged cells
    std::vector<int32_t> slots;
    if (use) {
        llama_pos lo[LLAMA_MAX_SEQ];
        for (uint32_t q = 0; q < LLAMA_MAX_SEQ; ++q) {
            lo[q] = std::numeric_limits<llama_pos>::max();
        }
        for (uint32_t q = 0; q < n_seq_max && q < LLAMA_MAX_SEQ; ++q) {
            lo[q] = v_cells[seq_to_stream[q]].seq_pos_max(q) + 1;
        }
        auto & rg = v_rings[strm];
        for (uint32_t i = 0; i < n_flags; ++i) {
            const uint32_t c = sinfo.idxs[0][i];
            win_release_cell(strm, c);
            if (!flags[i]) {
                continue;
            }
            const int32_t j = win_take(strm, lo);
            slots.push_back(j);
            if (j >= 0) {
                rg.slot_cell[j] = (int32_t) c;
                rg.cell_slot[c] = j;
            }
        }
    } else {
        for (uint32_t i = 0; i < n_flags; ++i) {
            if (flags[i]) {
                slots.push_back(-1);
            }
        }
    }

    std::vector<uint8_t> scratch;
    for (int kv = 0; kv < 2; ++kv) {
        for (const auto & layer : layers) {
            ggml_tensor * w = use ? (kv == 0 ? layer.k_win : layer.v_win) : nullptr;
            // a cache without a ring (or with other ring types) has the same layers: the blob has a section per K / V that exists
            const bool present = kv == 0 || layer.v != nullptr;
            if (!present) {
                continue;
            }
            uint64_t row = 0;
            io.read(&row, sizeof(row));
            if (w != nullptr && row != w->nb[1]) {
                LLAMA_LOG_ERROR("%s: k19-kvwin: mismatched ring row size (%zu != %zu)\n", __func__, (size_t) row, (size_t) w->nb[1]);
                return false;
            }
            for (const int32_t j : slots) {
                if (w != nullptr && j >= 0) {
                    io.read_tensor(w, ((size_t) strm*(win_r + 1) + j)*row, row);
                } else {
                    scratch.resize(row);
                    io.read(scratch.data(), row);
                }
            }
        }
    }
    return true;
}
