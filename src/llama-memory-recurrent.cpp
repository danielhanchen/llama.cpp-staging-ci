// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "llama-memory-recurrent.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);

    // k16-draft: commit by replay for the delta-net models (see llama_memory_recurrent::replay). LLAMA_RS_REPLAY unset / auto: only
    // when layer weights spill to host memory while others are on a device (a MoE cache / host experts: the memory the snapshot rows
    // would take goes to the cache there); with everything resident (or nothing offloaded) the snapshot rows stay, since the commit's
    // second delta-net pass costs decode (B200, full card: fn1 -1.5%, Qwen3.6-35B-A3B -2.8%). =1 forces replay, =0 the snapshot rows.
    // The placement is read from the weights' buffers, which a no_alloc (fit probe) model also sets, so a probe sizes the same rows.
    {
        const char * e = getenv("LLAMA_RS_REPLAY");
        const int mode = e == nullptr || strcmp(e, "auto") == 0 ? -1 : (atoi(e) != 0 ? 1 : 0);
        const bool arch_ok = model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE || model.arch == LLM_ARCH_QWEN4EXP;
        const bool can = n_rs_seq > 0 && arch_ok && hparams.n_embd_head_kda == 0 &&
                 hparams.ssm_dt_rank > 0 && hparams.ssm_d_state > 0 && hparams.ssm_d_inner % hparams.ssm_dt_rank == 0;
        bool want = mode == 1;
        if (can && mode < 0) {
            bool on_host = false, on_dev = false;
            for (const auto & it : model.tensors_by_name) {
                const ggml_tensor * t = it.second;
                if (t == nullptr || t->buffer == nullptr || it.first.compare(0, 4, "blk.") != 0) {
                    continue;
                }
                (ggml_backend_buffer_is_host(t->buffer) ? on_host : on_dev) = true;
            }
            want = on_host && on_dev;
            LLAMA_LOG_INFO("%s: delta-net rollback: %s (LLAMA_RS_REPLAY auto: layer weights %s; =1 / =0 forces replay / snapshot rows)\n",
                    __func__, want ? "replay" : "snapshot rows", on_host && on_dev ? "spill to host memory" :
                    on_dev ? "all on a device" : "all in host memory");
        }
        replay = can && want;
        const char * es = getenv("LLAMA_RS_REPLAY_SCOPED");
        replay_scoped = es == nullptr || atoi(es) != 0;
        pend_n.assign(mem_size, 0);
        pend_rb.assign(mem_size, 0);
        par.assign(mem_size, 0);
        conv_rb.assign(mem_size, 0);
    }

    cells.clear();
    cells.resize(mem_size);

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // r and s per layer, plus the separate PLE conv row where the model has one
                /*.mem_size   =*/ size_t((hparams.ple_conv_state() > 0 ? 3u : 2u)*n_layer*ggml_tensor_overhead() + 4u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);
    p_l.resize(n_layer);
    rk_l.assign(n_layer, nullptr);
    rv_l.assign(n_layer, nullptr);
    rg_l.assign(n_layer, nullptr);
    rb_l.assign(n_layer, nullptr); // unused: gate and beta share rg_l
    rk_h.assign(n_layer, 0);

    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        // the recurrent states are small and every token reads and writes all of them, so they stay with their layer
        // even when the KV cache is kept in host memory (-nkvo); LLAMA_KV_PREFETCH=0: they follow -nkvo as before
        static const bool keep_on_device = getenv("LLAMA_KV_PREFETCH") == nullptr || atoi(getenv("LLAMA_KV_PREFETCH")) != 0;

        if (offload || keep_on_device) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows = mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), replay ? mem_size : n_rows);
        if (replay) {
            const int64_t n_win = 1 + n_rs_seq;
            const int64_t H_v   = hparams.ssm_dt_rank;
            const int64_t S_v   = hparams.ssm_d_inner / H_v;
            const int64_t S_k   = hparams.ssm_d_state;
            // two halves of n_win token slots per cell: a fused window (ggml_gated_delta_net_replay) reads the stored window from one
            // half and writes its own into the other (par); gate and beta interleaved in one store
            rk_l[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_k, H_v, 2*n_win, mem_size);
            rv_l[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H_v, 2*n_win, mem_size);
            rg_l[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2,   H_v, 2*n_win, mem_size);
            ggml_format_name(rk_l[i], "replay_k_l%d", i);
            ggml_format_name(rv_l[i], "replay_v_l%d", i);
            ggml_format_name(rg_l[i], "replay_gb_l%d", i);
        }
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        r_l[i] = r;
        s_l[i] = s;

        // the PLE history needs its own row: Meta must mirror it while the delta-net conv state next door stays split
        if (hparams.ple_conv_state() > 0 && hparams.is_ple(i)) {
            ggml_tensor * p = ggml_new_tensor_2d(ctx, type_r, hparams.ple_conv_state(), n_rows);
            ggml_format_name(p, "cache_ple_r_l%d", i);
            p_l[i] = p;
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    // k16-draft: a memory probe (no_alloc: the --fit measurements) allocated the states for real, so it both counted them (context) and
    // saw them gone from the free memory it fits into: the fit gave them twice. G4 Qwen3.8-Flash-Next 16 GB, Studio argv (-np 4, MTP
    // draft 2: 3 rows per sequence): the main fit saw 14630 instead of 15982 MiB free (1352 MiB, the states at n_rs_seq 2), 15530 without
    // a draft (n_rs_seq 0, 452 MiB), so a draft was charged its rollback rows (900 MiB) twice.
    // The probe now sizes the tensors from a dummy buffer (as the KV cache does) and holds real memory only for the rows a context without
    // rollback snapshots has (mem_size): that copy is the slack every fit had so far, which covers the fit's compute estimate being short
    // of the real buffer (local B200 fn1 16 GB: fit 1091 MiB, sched_reserve 1459 MiB; without the slack the server ended at 515-600 MiB
    // free under the 1024 MiB target). Only the rollback rows (n_rs_seq) are no longer counted twice.
    // k16-fit: the compute shortfall is the seed graph, which the probe now prices (llama_context::sched_reserve, LLAMA_FIT_SEED_PRICE),
    // so by default (=2) the probe holds no real memory at all and every state is counted once.
    // LLAMA_RS_PROBE_DUMMY=0: allocate as before (all rows twice); =1: real memory for the base rows only (their slack kept)
    static const int probe_dummy = getenv("LLAMA_RS_PROBE_DUMMY") == nullptr ? 2 : atoi(getenv("LLAMA_RS_PROBE_DUMMY"));
    probe_sized = hparams.no_alloc && probe_dummy != 0;
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (probe_sized) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer, sized by memory_breakdown
            // the base rows: one row per cell of every state (the replay stores and the snapshot rows excluded)
            size_t rows_bytes = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                const bool is_s = strncmp(t->name, "cache_s_l", 9) == 0;
                if (strncmp(t->name, "replay_", 7) == 0) {
                    continue;
                }
                rows_bytes += (is_s && replay) ? ggml_nbytes(t) : ggml_nbytes(t) / (1 + n_rs_seq);
            }
            if (probe_dummy == 1 && rows_bytes > 0) {
                ggml_backend_buffer_t base = ggml_backend_buft_alloc_buffer(buft, rows_bytes);
                if (base != nullptr) {
                    probe_base.emplace_back(base);
                }
            }
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // so that the backend scheduler does not try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        if (!probe_sized) {
            ggml_backend_buffer_clear(buf, 0);
        }
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    // k16-draft: the commit folded into the next window (ggml_gated_delta_net_replay), on the backends that implement it (CUDA, CPU);
    // LLAMA_RS_REPLAY_FUSED=0: every commit through the commit graph (llama_context::rs_replay_commit)
    if (replay) {
        const char * e = getenv("LLAMA_RS_REPLAY_FUSED");
        replay_fused_ok = e == nullptr || atoi(e) != 0;
        for (const ggml_tensor * t : s_l) {
            if (t == nullptr || t->buffer == nullptr) {
                continue;
            }
            const char * nm = ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer));
            if (strncmp(nm, "CUDA", 4) != 0 && strncmp(nm, "ROCm", 4) != 0 && strncmp(nm, "MUSA", 4) != 0 && strncmp(nm, "CPU", 3) != 0) {
                replay_fused_ok = false;
            }
        }
        LLAMA_LOG_INFO("%s: replay commit %s (LLAMA_RS_REPLAY_FUSED=0: a commit graph before each window)\n", __func__,
                replay_fused_ok ? "folded into the next window" : "by a commit graph");
    }

    if (is_empty()) {
        replay = false;
        if (n_rs_seq > 0) {
            n_rs_seq = 0;
            LLAMA_LOG_INFO("%s: disabling rollback snapshots because the memory module is empty\n", __func__);
        }
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();
        const size_t memory_size_p = size_p_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB, P (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s + memory_size_p) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f),
                ggml_type_name(type_r), (float)memory_size_p / (1024.0f * 1024.0f));
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
    }
    // k19-npmtp: no window outlives its sequence (a commit no longer runs over every cell before the next ubatch)
    std::fill(pend_n.begin(),  pend_n.end(),  0);
    std::fill(pend_rb.begin(), pend_rb.end(), 0);
    std::fill(conv_rb.begin(), conv_rb.end(), 0);

    head = 0;
    used = 0;

    if (data && !probe_sized) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id >= this->n_seq_max) {
        LLAMA_LOG_ERROR("%s: invalid seq_id (%d) - larger than n_seq_max (%d)\n", __func__, seq_id, this->n_seq_max);
        return false;
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        set_rs_idx(seq_id, 0);
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    if (seq_id >= (int64_t) size) {
        // could be fatal
        return false;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                // a cell shared by several sequences cannot move back for only one of them
                if (cell.seq_id.size() > 1) {
                    return false;
                }
                // the filter kept no layer (e.g. an MTP draft context), so only the position moves back
                if (is_empty()) {
                    cell.pos = p0 - 1;
                    return true;
                }
                const llama_pos rollback = cell.pos - (p0 - 1);
                // pending rollback is single-use
                const bool pending = rs_idx[seq_id] != 0;
                if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq && replay) {
                    // k16-draft: the window is still uncommitted; the commit replays only its kept tokens
                    const uint32_t c = (uint32_t) tail_id;
                    if (pend_n[c] == 0 || (uint32_t) rollback > pend_n[c] || pend_rb[c] != 0) {
                        return false;
                    }
                    pend_rb[c] = (uint32_t) rollback;
                    cell.pos = p0 - 1;
                    return true;
                }
                if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                    // k15-rsr: the snapshot is copied onto the live row now, so the next decode reads its state from the same row as
                    // every other decode: its graph and CUDA graph are reused instead of rebuilt and re-captured twice (once for the
                    // snapshot row, once back); same bytes, same output. Else (LLAMA_RS_ROLLBACK_COPY=0) the next decode reads the snapshot row
                    if (!rollback_copy((uint32_t) tail_id, (uint32_t) rollback)) {
                        set_rs_idx(seq_id, (uint32_t) rollback);
                    }
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            //printf("[DEBUG] inside `llama_memory_recurrent::seq_rm`: `seq_id` is negative, so returning false\n");
            return false;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                pend_n[i]  = 0;
                pend_rb[i] = 0;
                conv_rb[i] = 0;
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    replay_commit_if_pending(); // k16-draft: the rows are current before they are read or copied

    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        auto & tail_src = cells[seq_id_src];
        auto & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            // clear destination seq_id if it wasn't empty
            auto & cell_dst = cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            auto & cell_src = cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
    }
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    replay_commit_if_pending(); // k16-draft: the rows are current before they are read or copied

    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

bool llama_memory_recurrent::has_pending() const {
    if (!replay) {
        return false;
    }
    for (uint32_t c = 0; c < size; ++c) {
        if (pend_n[c] != 0 || conv_rb[c] != 0) {
            return true;
        }
    }
    return false;
}

// k16-draft: after find_slot, whether this ubatch commits the stored windows of its cells itself (ggml_gated_delta_net_replay): a
// window of every sequence in its own cell. The conv snapshot of a rollback is still copied by the commit graph (conv_rb)
bool llama_memory_recurrent::replay_fuse(const llama_ubatch & ubatch) {
    rp_fused = false;
    if (!replay || !replay_fused_ok || ubatch.n_seqs == 0 || ubatch.n_seq_tokens > 1 + n_rs_seq || head + ubatch.n_seqs > size) {
        return false;
    }
    for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
        const uint32_t c = head + s;
        if (cells[c].src0 != (int32_t) c) {
            return false;
        }
    }
    rp.assign(2*ubatch.n_seqs, 0);
    for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
        const uint32_t c = head + s;
        rp[2*s + 0] = (int32_t) (pend_n[c] > pend_rb[c] ? pend_n[c] - pend_rb[c] : 0);
        rp[2*s + 1] = par[c];
        conv_rb[c]  = pend_rb[c];
        pend_n[c]   = 0;
        pend_rb[c]  = 0;
    }
    rp_fused = true;
    return true;
}

void llama_memory_recurrent::replay_commit_if_pending(bool sync) const {
    if (commit_cb && has_pending()) {
        commit_cb(sync);
    }
}

// k19-npmtp: commit the pending windows of the cells in `which` only. The others are held out of the commit graph (its builder and its
// graph key read pend_n / pend_rb / conv_rb) and put back after it, so they can still be rolled back
void llama_memory_recurrent::replay_commit_cells(const std::vector<uint8_t> & which, bool sync) {
    if (!commit_cb || !has_pending()) {
        return;
    }
    if (!replay_scoped) {
        commit_cb(sync);
        return;
    }
    struct held { uint32_t c, n, rb, crb; };
    std::vector<held> hold;
    bool any = false;
    for (uint32_t c = 0; c < size; ++c) {
        if (pend_n[c] == 0 && conv_rb[c] == 0) {
            continue;
        }
        if (c < which.size() && which[c]) {
            any = true;
            continue;
        }
        hold.push_back({ c, pend_n[c], pend_rb[c], conv_rb[c] });
        pend_n[c]  = 0;
        pend_rb[c] = 0;
        conv_rb[c] = 0;
    }
    if (any) {
        commit_cb(sync);
    }
    for (const held & h : hold) {
        pend_n[h.c]  = h.n;
        pend_rb[h.c] = h.rb;
        conv_rb[h.c] = h.crb;
    }
}

// k19-npmtp: find_slot is about to move or share the rows of cell c (a cell swap, a shared tail copied): its window first
void llama_memory_recurrent::replay_commit_cell(int32_t c) {
    if (!replay || in_prepare || c < 0 || (uint32_t) c >= size || (pend_n[c] == 0 && conv_rb[c] == 0)) {
        return;
    }
    std::vector<uint8_t> which(size, 0);
    which[c] = 1;
    replay_commit_cells(which, /*sync =*/ false);
}

// k19-npmtp: before a ubatch, the windows of the cells it reads or copies: placed (after find_slot) the cells [head, head + n) and
// their copy sources, else the tails of its sequences and every cell a copy is staged from or to (find_slot commits what it moves)
void llama_memory_recurrent::replay_commit_ubatch(const llama_ubatch & ubatch, bool placed) {
    if (!commit_cb || !has_pending()) {
        return;
    }
    std::vector<uint8_t> which(size, 0);
    auto mark = [&](int32_t c) {
        if (c >= 0 && (uint32_t) c < size) {
            which[c] = 1;
        }
    };
    if (placed) {
        // its own cells, and the other cells of [head, head + n) whose rows the graph copies from another cell (an identity copy
        // leaves a row as it is, so the window of such a cell stays pending)
        for (uint32_t c = head; c < head + n && c < size; ++c) {
            if (c < head + ubatch.n_seqs || (cells[c].src0 >= 0 && cells[c].src0 != (int32_t) c)) {
                mark((int32_t) c);
                mark(cells[c].src0);
            }
        }
    } else {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
                const llama_seq_id sid = ubatch.seq_id[i][j];
                if (sid >= 0 && (uint32_t) sid < size) {
                    mark(cells[sid].tail);
                }
            }
        }
    }
    for (uint32_t c = 0; c < size; ++c) {
        if (cells[c].src >= 0 && cells[c].src != (int32_t) c) {
            mark((int32_t) c);
            mark(cells[c].src);
        }
    }
    replay_commit_cells(which, /*sync =*/ false);
}

// k16-draft: after find_slot placed the ubatch's sequences in cells [head, head + n_seqs): a window (at most 1 + n_rs_seq tokens per
// sequence) is left uncommitted by the graph (build_recurrent_attn), a larger ubatch writes its final state
void llama_memory_recurrent::replay_mark(const llama_ubatch & ubatch) {
    if (!replay) {
        return;
    }
    const uint32_t n_win = 1 + n_rs_seq;
    const uint32_t n_tok = ubatch.n_seq_tokens <= n_win ? ubatch.n_seq_tokens : 0;
    for (uint32_t s = 0; s < ubatch.n_seqs && head + s < size; ++s) {
        pend_n[head + s]  = n_tok;
        pend_rb[head + s] = 0;
        // a fused window wrote its inputs into the other half; the graph of an unfused one into half 0
        par[head + s] = rp_fused ? (uint8_t) (1 - par[head + s]) : 0;
    }
}

bool llama_memory_recurrent::rollback_copy(uint32_t cell, uint32_t group) {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_RS_ROLLBACK_COPY");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!enabled || group == 0 || group > n_rs_seq || cell >= size) {
        return false;
    }
    // only small states (LFM2.5's short convolutions: 16 KiB per layer): a copy synchronises once per tensor, so for large ones (the
    // delta-net states of Flash-Next fn4, MiBs per layer) the copies cost more than the graph rebuild they save (G4 fn4 32 GB chat -0.35%)
    static const size_t max_bytes = [] {
        const char * e = getenv("LLAMA_RS_ROLLBACK_COPY_MAX_KB");
        return (size_t) (e != nullptr ? atof(e) : 1024.0)*1024;
    }();
    size_t bytes = 0;
    for (size_t il = 0; il < r_l.size(); ++il) {
        for (ggml_tensor * t : { r_l[il], s_l[il], il < p_l.size() ? p_l[il] : nullptr }) {
            if (t != nullptr && t->ne[1] > 0) {
                bytes += t->nb[1];
            }
        }
    }
    if (bytes > max_bytes) {
        return false;
    }
    // views of the two rows of every state tensor, in a context of views only (no allocation)
    const size_t n_views = 2*3*(size_t) r_l.size() + 8;
    ggml_init_params ip = { n_views*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }
    bool ok = true;
    auto copy_row = [&](ggml_tensor * t) {
        if (!ok || t == nullptr || t->buffer == nullptr || t->ne[0] == 0 || t->ne[1] < (int64_t) ((group + 1)*size)) {
            return;
        }
        ggml_tensor * src = ggml_view_1d(ctx, t, t->ne[0], (size_t) (group*size + cell)*t->nb[1]);
        ggml_tensor * dst = ggml_view_1d(ctx, t, t->ne[0], (size_t) cell*t->nb[1]);
        if (ggml_backend_view_init(src) != GGML_STATUS_SUCCESS || ggml_backend_view_init(dst) != GGML_STATUS_SUCCESS) {
            ok = false;
            return;
        }
        ggml_backend_tensor_copy(src, dst);
    };
    for (size_t il = 0; il < r_l.size() && ok; ++il) {
        copy_row(r_l[il]);
        copy_row(s_l[il]);
        if (il < p_l.size()) {
            copy_row(p_l[il]);
        }
    }
    ggml_free(ctx);
    if (ok && n_rollback_copy++ == 0) {
        LLAMA_LOG_INFO("%s: speculative rollbacks copy the state snapshot onto the live row (decode graphs kept; LLAMA_RS_ROLLBACK_COPY=0: off)\n", __func__);
    }
    return ok;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0) {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
        return;
    }

    assert(n_seq_max == rs_idx.size());

    GGML_ASSERT((uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(idx <= n_rs_seq);

    rs_idx[seq_id] = idx;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        ret[buft] += probe_sized ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft) : ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    in_prepare = true;
    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }
    in_prepare = false;

    // restore the original state
    cells = std::move(org_cells);
    used = org_used;
    head = org_head;

    return success;
}

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    replay_commit_cell(seq.tail);
                    auto & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                replay_commit_cell(seq_meta.tail);
                auto & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id); // will be overwritten
                GGML_ASSERT(!orig_cell.is_empty()); // has at least one remaining seq_id
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            replay_commit_cell(dst_id);
            replay_commit_cell(src_id);
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap tails
            for (uint32_t j = 0; j < size; ++j) {
                int32_t & tail = cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            // A hybrid model whose filter drops every recurrent layer (e.g. the MTP context of a hybrid model) leaves this cache
            // without tensors: its positions are bookkeeping only, so a rejected speculative draft is no inconsistency to warn about.
            const bool has_state =
                std::any_of(r_l.begin(), r_l.end(), [](const ggml_tensor * t) { return t != nullptr; }) ||
                std::any_of(s_l.begin(), s_l.end(), [](const ggml_tensor * t) { return t != nullptr; });
            if (has_state) {
                LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                    __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
            }
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}

bool llama_memory_recurrent::is_empty() const {
    const bool res = ctxs_bufs.empty();
    assert(!res || total_size() == 0);
    return res;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

size_t llama_memory_recurrent::size_p_bytes() const {
    size_t size_p_bytes = 0;

    for (const auto & p : p_l) {
        if (p != nullptr) {
            size_p_bytes += ggml_nbytes(p);
        }
    }

    return size_p_bytes;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    replay_commit_if_pending(); // k16-draft: the rows are current before they are read or copied

    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data);
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    replay_commit_if_pending(); // k16-draft: the rows are current before they are read or copied

    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    // save the head of the restored cells - could be needed to clear the state
    // the head is valid only when state_read_meta() succeeded
    const bool meta_read = state_read_meta(io, cell_count, seq_id);
    const uint32_t cell_head = head;

    res = res && meta_read;

    try {
        res = res && state_read_data(io, cell_count);
    } catch (...) {
        res = false;
    }

    if (!res) {
        state_clear(seq_id, cell_head, meta_read ? cell_count : 0);
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        set_rs_idx(seq_id, 0);
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cap_map != nullptr && seq_id >= 0 ? cap_pos : cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    // k15-pp: the captured row of a tensor (a single cell), see cap_map
    auto cap_tensor = [&](const ggml_tensor * t, size_t range_size) -> ggml_tensor * {
        if (cap_map == nullptr || range_size != 1) {
            return nullptr;
        }
        for (const auto & [src, snap] : *cap_map) {
            if (src == t) {
                return snap;
            }
        }
        return nullptr;
    };
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            if (ggml_tensor * c = cap_tensor(r_l[il], range_size)) {
                io.write_tensor(c, 0, buf_size);
                continue;
            }
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }

        // the PLE conv history is a second recurrent row, so it has to travel with the first
        if (p_l[il] != nullptr) {
            const uint64_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            io.write(&p_size_row, sizeof(p_size_row));

            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                if (ggml_tensor * c = cap_tensor(p_l[il], range_size)) {
                    io.write_tensor(c, 0, range_size * p_size_row);
                    continue;
                }
                io.write_tensor(p_l[il], range.first * p_size_row, range_size * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            // Write each logical cell row range. With pending recurrent rollback,
            // the logical current state may live in a rollback snapshot plane.
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                if (ggml_tensor * c = cap_tensor(s_l[il], range_size)) {
                    io.write_tensor(c, 0, buf_size);
                    continue;
                }
                io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == ubatch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= this->n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, this->n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            io.read_tensor(r_l[il], head * r_size_row, cell_count * r_size_row);
        }

        if (p_l[il] != nullptr) {
            uint64_t p_size_row_ref;
            io.read(&p_size_row_ref, sizeof(p_size_row_ref));
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            if (p_size_row != p_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched ple row size (%zu != %zu, layer %d)\n", __func__, p_size_row, (size_t) p_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                io.read_tensor(p_l[il], head * p_size_row, cell_count * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                io.read_tensor(s_l[il], head * s_size_row, cell_count * s_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    return true;
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
// the transposed s layout is not handled - state_read_data() rejects it before any write
void llama_memory_recurrent::state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count) {
    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    if (cell_count == 0) {
        return;
    }

    const uint32_t n_layer = hparams.n_layer();

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (r_l[il] != nullptr) {
            const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
            llama_clear_tensor_data(r_l[il], cell_head * r_size_row, cell_count * r_size_row);
        }

        if (s_l[il] != nullptr) {
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            llama_clear_tensor_data(s_l[il], cell_head * s_size_row, cell_count * s_size_row);
        }

        if (p_l[il] != nullptr) {
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            llama_clear_tensor_data(p_l[il], cell_head * p_size_row, cell_count * p_size_row);
        }
    }
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    // k16-draft: the previous window is committed before this ubatch reads (or copies) any state: by this ubatch's own graph when it
    // is a window of the same cells (replay_fuse), else (and for the conv snapshots and the other cells) by the commit graph, queued
    // here on the same streams as the graph of this ubatch
    // k19-npmtp: only the windows of the cells this ubatch reads, copies or moves (replay_commit_ubatch, find_slot); the other sequences'
    // windows stay pending, so a draft verified earlier in the same llama_decode can still be rolled back
    if (mem->replay && mem->replay_fused_ok) {
        mem->find_slot(ubatches[i_next]);
        mem->replay_fuse(ubatches[i_next]);
        mem->replay_commit_ubatch(ubatches[i_next], /*placed =*/ true);
    } else {
        mem->rp_fused = false;
        mem->replay_commit_ubatch(ubatches[i_next], /*placed =*/ false);
        mem->find_slot(ubatches[i_next]);
    }
    // k19-npmtp: whether a sequence of this ubatch reads its state from another row (LLAMA_RS_REPLAY_SCOPED=0: not written, as before)
    mem->rp_moved = false;
    if (mem->replay && mem->replay_scoped) {
        const llama_ubatch & ub = ubatches[i_next];
        for (uint32_t s = 0; s < ub.n_seqs && mem->head + s < mem->size; ++s) {
            mem->rp_moved |= mem->cells[mem->head + s].src0 != (int32_t) (mem->head + s);
        }
    }

    mem->replay_mark(ubatches[i_next]);

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_ubatch * llama_memory_recurrent_context::peek_ubatch(uint32_t k) const {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS || i_next + k >= ubatches.size()) {
        return nullptr;
    }

    return &ubatches[i_next + k];
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    return is_full ? mem->size : mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    return is_full ? 0 : mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    return is_full ? 0 : mem->rs_z;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

bool llama_memory_recurrent_context::get_replay() const { return mem->replay; }
uint32_t llama_memory_recurrent_context::get_n_win() const { return 1 + mem->n_rs_seq; }
ggml_tensor * llama_memory_recurrent_context::get_rk_l(int32_t il) const { return mem->replay ? mem->rk_l[il] : nullptr; }
ggml_tensor * llama_memory_recurrent_context::get_rv_l(int32_t il) const { return mem->replay ? mem->rv_l[il] : nullptr; }
ggml_tensor * llama_memory_recurrent_context::get_rg_l(int32_t il) const { return mem->replay ? mem->rg_l[il] : nullptr; }
ggml_tensor * llama_memory_recurrent_context::get_rb_l(int32_t il) const { return mem->replay ? mem->rb_l[il] : nullptr; }
void llama_memory_recurrent_context::set_rk_h(int32_t il, int32_t h) const { mem->rk_h[il] = h; }
bool llama_memory_recurrent_context::get_replay_fused() const { return mem->replay && mem->rp_fused; }
bool llama_memory_recurrent_context::get_replay_moved() const { return mem->replay && mem->rp_moved; }
void llama_memory_recurrent_context::set_replay_input(ggml_tensor * t) const {
    GGML_ASSERT(t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer));
    GGML_ASSERT(mem->rp_fused && (int64_t) mem->rp.size() == ggml_nelements(t));
    memcpy(t->data, mem->rp.data(), mem->rp.size()*sizeof(int32_t));
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_p_l(int32_t il) const {
    return mem->p_l[il];
}

int32_t llama_memory_recurrent_context::get_rs_alias(const llama_ubatch & ubatch) const {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_RS_ALIAS");
        return env == nullptr || atoi(env) != 0;
    }();

    if (!enabled || is_full || ubatch.n_seqs == 0 || ubatch.n_tokens != ubatch.n_seqs) {
        return -1;
    }

    const uint32_t n = ubatch.n_seqs;
    if (mem->head + n > mem->size) {
        return -1;
    }

    // every sequence must read the row it writes (src0 == its own cell) from the same rollback group,
    // so that no state is read after another sequence overwrote it
    int32_t base = -1;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t cell_idx = mem->head + i;
        const auto &   cell     = mem->cells[cell_idx];

        if (cell.src0 != (int32_t) cell_idx) {
            return -1;
        }

        uint32_t idx = 0;
        if (mem->n_rs_seq != 0 && !cell.seq_id.empty()) {
            const llama_seq_id seq = *cell.seq_id.begin();
            if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
                idx = mem->rs_idx[seq];
            }
        }

        const int32_t val = (int32_t)(idx * mem->size) + cell.src0;
        if (i == 0) {
            base = val;
        } else if (val != base + (int32_t) i) {
            return -1;
        }
    }

    return base;
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}
