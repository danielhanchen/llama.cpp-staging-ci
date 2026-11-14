// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <type_traits>

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-ext.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // k19-vgraph (LLAMA_KV_BUCKET=0: off, the 256 cell padding of every ubatch): a decode or verify ubatch of qwen4exp, whose every
    // attention layer is QSA, pads n_kv to an eighth of its octave instead of 256 cells, and the k-pool count with it, so its graph
    // and CUDA graph are not rebuilt and captured again each time the context grows past 256 cells (G4 Flash-Next: 8-10 verify
    // re-captures per 256 tokens, one per window size and boundary, 3-5 ms host each). Only where the bucket cannot change a result:
    // the attention gathers the selected cells (CUDA sparse flash attention: n_kv >= max(4096, 2*n_tokens*n_sel), a single
    // query tile up to 4 tokens), so its work list and split follow n_sel and not n_kv, the extra cells are masked and the extra
    // pools score -inf (top_k keeps the lowest of tied columns, which the padding at the end never is). =2 (opt-in, not exact:
    // the dense attention's KV split moves, a reassociation): any ubatch up to LLAMA_KV_BUCKET_MAX_TOKENS (8) tokens from 1024 cells.
    // LLAMA_KV_BUCKET_DIV (8): buckets per octave (G4 fn4 d2 16k, 256-token requests that never cross a boundary: 4 per octave,
    // n_kv 20480 for 16640, cost 0.5% decode in the extra mask / indexer / top_k cells)
    static const int mode = getenv("LLAMA_KV_BUCKET") ? atoi(getenv("LLAMA_KV_BUCKET")) : 1;
    const auto & hp = model.hparams;
    bool all_qsa = mode != 0 && mem_idx != nullptr && hp.indexer_kpool > 0 && hp.indexer_kpool_by_order && hp.indexer_top_k > 0;
    bool any_attn = false;
    for (uint32_t il = 0; all_qsa && il < hp.n_layer_all && il < LLAMA_MAX_LAYERS; ++il) {
        if (filter_attn == nullptr || filter_attn(il)) {
            any_attn = true;
            all_qsa  = hp.dsv4_compress_ratios[il] > 0;
        }
    }
    if (all_qsa && any_attn) {
        const uint32_t kpool = hp.indexer_kpool;
        const uint32_t n_sel = kpool*(hp.indexer_top_k/kpool) + kpool - 1; // build_qsa_sel at n_pool >= top_k/kpool
        llama_kv_cache::n_kv_bucket_params bp;
        if (mode == 2) {
            bp.n_tokens_max = getenv("LLAMA_KV_BUCKET_MAX_TOKENS") ? (uint32_t) std::max(1, atoi(getenv("LLAMA_KV_BUCKET_MAX_TOKENS"))) : 8;
            bp.n_kv_min     = 1024;
        } else {
            bp.n_tokens_max       = 4;
            bp.n_kv_min           = 4096;
            bp.n_kv_min_per_token = 2*n_sel;
        }
        bp.div = getenv("LLAMA_KV_BUCKET_DIV") ? (uint32_t) std::max(1, atoi(getenv("LLAMA_KV_BUCKET_DIV"))) : 8;
        // k19-graph (LLAMA_KV_BUCKET_DEV: unset = where the backend has it, 0 = off, 1 = forced, for CPU tests): the flash attention
        // reads the live length on the device (ggml_flash_attn_ext_set_kv_len, ggml-cuda: the bits of the unpadded node), so the
        // bucket moves no sum and every ubatch of up to LLAMA_KV_BUCKET_DEV_MAX_TOKENS (8, a DFlash2 window of 7 drafts) is bucketed,
        // LLAMA_KV_BUCKET_DEV_DIV (2) per octave, at least LLAMA_KV_BUCKET_DEV_STEP_MIN (16384, k19-fullcost; was 2048) cells: a verify width keeps its graph and
        // CUDA graph from one request to the next and while the context grows, re-captured only where a bucket starts (G4
        // Flash-Next DFlash2 / MTP: 8-10 re-captures of each width per request, ~4 ms each, plus an eager pass and instantiation for a
        // rarely used width). Only while flash attention is on (llama_context, set_n_kv_bucket_dev_allowed)
        const char * e_dev = getenv("LLAMA_KV_BUCKET_DEV");
        const int dev_mode = e_dev ? atoi(e_dev) : -1;
        const bool dev_backend = get_mem_attn()->fa_kv_len_supported();
        bp.dev_len = dev_mode == 1 || (dev_mode < 0 && dev_backend);
        if (const char * e = getenv("LLAMA_KV_BUCKET_DEV_MAX_TOKENS")) {
            bp.dev_n_tokens_max = (uint32_t) std::max(1, atoi(e));
        }
        if (const char * e = getenv("LLAMA_KV_BUCKET_DEV_DIV")) {
            bp.dev_div = (uint32_t) std::max(1, atoi(e));
        }
        if (const char * e = getenv("LLAMA_KV_BUCKET_DEV_STEP_MIN")) {
            bp.dev_step_min = (uint32_t) std::max(256, atoi(e));
            dev_step_min_env = true;
        }
        get_mem_attn()->set_n_kv_bucket(bp);
        mem_idx->set_n_kv_bucket(bp);
        LLAMA_LOG_INFO("%s: n_kv buckets for ubatches of up to %u tokens from %u cells (%u per token), %u per octave (LLAMA_KV_BUCKET=%d)\n",
            __func__, bp.n_tokens_max, bp.n_kv_min, bp.n_kv_min_per_token, bp.div, mode);
        if (bp.dev_len) {
            LLAMA_LOG_INFO("%s: with flash attention reading the KV length on the device: ubatches of up to %u tokens, %u per octave, "
                "at least %u cells (LLAMA_KV_BUCKET_DEV=%d)\n", __func__, bp.dev_n_tokens_max, bp.dev_div, bp.dev_step_min, dev_mode);
        }
    }
}

void llama_memory_hybrid_idx::set_n_kv_bucket_allowed(bool value) {
    get_mem_attn()->set_n_kv_bucket_allowed(value);
    if (mem_idx) {
        mem_idx->set_n_kv_bucket_allowed(value);
    }
}

void llama_memory_hybrid_idx::set_n_kv_bucket_dev_allowed(bool value) {
    get_mem_attn()->set_n_kv_bucket_dev_allowed(value);
    if (mem_idx) {
        mem_idx->set_n_kv_bucket_dev_allowed(value);
    }
}

void llama_memory_hybrid_idx::set_n_kv_bucket_dev_wide(bool wide) {
    if (dev_step_min_env) {
        return;
    }
    const uint32_t v = wide ? 16384 : 2048;
    get_mem_attn()->set_n_kv_bucket_dev_step_min(v);
    if (mem_idx) {
        mem_idx->set_n_kv_bucket_dev_step_min(v);
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
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

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single pool grouping
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported");

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
        // pooled keys, the destination rebuilds its layout and re-pools into the same rows
        mem_idx_stale_set(seq_id_dst, 0);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // every other sequence loses its cells, so their layouts must rebuild
        mem_idx_stale_set(-1, 0);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    if (mem_idx) {
        mem_idx->state_write(io, seq_id, flags);
    }
}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        // the indexer cache has the SWA type of the attention cache, so both read the same partial state
        get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if (mem_idx) {
            mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            // a restore that rewrites cells does it behind the pool layout's back
            if (!sinfos_attn.empty()) {
                mem_idx_stale_set(seq_id, 0);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;

        // k19-vwidth: the number of leading pools no update changed (members, cells, stream) since set_input_kpool last wrote this
        // sequence's pools into its kept arrays (it sets it back to pools.size())
        mutable size_t min_keep = 0;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;

    // k19-vwidth (LLAMA_KPOOL_INPUT_INC=0: off): the pool cells, members and end positions set_input_kpool last wrote, per global
    // pool index, and where each sequence's pools started there and how many were written (unified KV only)
    mutable std::vector<int32_t>   c_pcell;
    mutable std::vector<int32_t>   c_pidx;
    mutable std::vector<llama_pos> c_pend;
    mutable std::array<uint32_t, LLAMA_MAX_SEQ> c_start {};
    mutable std::array<uint32_t, LLAMA_MAX_SEQ> c_count {};
    mutable uint32_t c_kpool = 0;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    std::vector<uint32_t> rep_gen; // per global cell, the generation that last marked a pool with that rep
    uint32_t generation = 0;

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
    uint32_t n_pool_g    = 0; // k19-vgraph: graph pool count when n_kv is bucketed (0: kpool_pad(n_pool_real))
    bool     allfit      = false; // k19-p2qsa: see get_kpool_allfit
};

namespace {

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        // k19-vwidth: the leading pools this update keeps as they were
        size_t keep = sq.pools.size();
        if (sq.strm != (unified ? 0 : mem_idx->get_stream(s))) {
            keep = 0;
        }

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        }

        // k15-fn (LLAMA_KPOOL_INCREMENTAL=0: rebuild): an edit at p0 > pos_min (a rejected draft's seq_rm, every verify round) leaves
        // the cells before p0 and every pool that ends before p0 as they were (mem_idx_stale_set), so the layout keeps that prefix
        // and rescans from the end of its last kept pool: the greedy scan's decisions there depend only on those cells. The rebuild
        // walked the whole position set (8k context: 8k set nodes per ubatch). LLAMA_KPOOL_INCREMENTAL=2 checks it against a rebuild
        static const int incr = getenv("LLAMA_KPOOL_INCREMENTAL") ? atoi(getenv("LLAMA_KPOOL_INCREMENTAL")) : 1;
        bool incr_done = false;
        if (incr > 0 && mem_idx_stale[s] != POS_CLEAN && !sq.cells.empty() && !sp.empty() && sq.pos_min == sp.begin()->first &&
                mem_idx_stale[s] > sq.pos_min) {
            const llama_pos p0 = mem_idx_stale[s];
            const size_t nk = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(p0, 0u)) - sq.cells.begin();
            sq.cells.resize(nk);
            while (!sq.pools.empty() && sq.pools.back() + kpool > nk) {
                sq.pools.pop_back();
            }
            keep = std::min(keep, sq.pools.size());
            sq.j_next = sq.pools.empty() ? 0 : sq.pools.back() + kpool;
            for (auto it = sp.lower_bound(std::make_pair(p0, 0u)); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
            incr_done = sq.cells.size() == sp.size();
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (!incr_done && (sq.cells.size() != sp.size() || mem_idx_stale[s] != POS_CLEAN)) {
            sq.cells.assign(sp.begin(), sp.end());
            sq.pools.clear();
            sq.j_next  = 0;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
            keep = 0;
        }
        sq.min_keep = std::min(sq.min_keep, keep);

        // Pools start at the first valid token
        size_t j = sq.j_next;
        if (hparams_idx.indexer_kpool_by_order) {
            // consecutive cells in sequence order, whatever their positions
            for (; j + kpool <= sq.cells.size(); j += kpool) {
                sq.pools.push_back((uint32_t) j);
            }
        } else {
            while (j + kpool <= sq.cells.size()) {
                const llama_pos p0 = sq.cells[j].first;
                if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                    ++j;
                    continue;
                }
                bool ok = true;
                for (uint32_t k = 1; k < kpool; ++k) {
                    if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    sq.pools.push_back((uint32_t) j);
                    j += kpool;
                } else {
                    ++j;
                }
            }
        }
        sq.j_next = j;

        if (incr_done && incr >= 2) {
            // the check: a rebuild of this sequence gives the same cells, pools and scan state
            kpool_layout::seq ref = sq;
            ref.cells.assign(sp.begin(), sp.end());
            ref.pools.clear();
            ref.j_next  = 0;
            ref.pos_min = sp.begin()->first;
            size_t jr = 0;
            if (hparams_idx.indexer_kpool_by_order) {
                for (; jr + kpool <= ref.cells.size(); jr += kpool) {
                    ref.pools.push_back((uint32_t) jr);
                }
            } else {
                while (jr + kpool <= ref.cells.size()) {
                    const llama_pos q0 = ref.cells[jr].first;
                    if ((q0 - ref.pos_min) % (llama_pos) kpool != 0) {
                        ++jr;
                        continue;
                    }
                    bool ok = true;
                    for (uint32_t k = 1; k < kpool; ++k) {
                        if (ref.cells[jr + k].first != q0 + (llama_pos) k) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        ref.pools.push_back((uint32_t) jr);
                        jr += kpool;
                    } else {
                        ++jr;
                    }
                }
            }
            GGML_ASSERT(ref.cells == sq.cells && ref.pools == sq.pools && jr == sq.j_next && "LLAMA_KPOOL_INCREMENTAL=2: layout differs from a rebuild");
        }

        lay.n_pool_real += (uint32_t) sq.pools.size();
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
// k19-p2qsa: see llama_qsa_allfit_pause (llama-ext.h)
static std::atomic<int> g_qsa_allfit_pause { 0 };

void llama_qsa_allfit_pause(bool on) {
    g_qsa_allfit_pause.fetch_add(on ? 1 : -1);
}

void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    const auto *   idx     = mem->get_mem_idx();
    const uint32_t kv_size = idx->get_size();
    const uint32_t kpool   = mem->get_kpool();

    st.n_pool_real = lay.n_pool_real;
    st.n_new       = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(),  st.is_new.end(),  0);
        std::fill(st.rep_gen.begin(), st.rep_gen.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);
    st.rep_gen.resize((size_t) kv_size*idx->get_n_stream(), 0);

    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;

    // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
    // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
    auto mark = [&](llama_seq_id s, size_t k) {
        const auto & sq  = lay.seqs[s];
        const size_t rep = (size_t) sq.strm*kv_size + sq.cells[sq.pools[k] + kpool - 1].second;
        if (st.rep_gen[rep] != st.generation) {
            st.rep_gen[rep] = st.generation;
            st.is_new[pool_start[s] + k] = st.generation;
            ++st.n_new;
        }
    };

    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(s, it - sq.pools.begin());
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(s, (size_t) r / kpool);
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(s, it - sq.pools.begin());
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const uint32_t n_pool_max = kv_size / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});

    // k19-vgraph: a bucketed n_kv brings the pools with it (they change with the cells, every 256 of them), so the indexer, its
    // top_k and the pool inputs keep their shapes too; the extra pools are padding (sentinel members, -inf scores)
    // (only once the selection holds its full top_k/kpool pools without them: else n_sel would follow the padding)
    // (k19-graph: with the device KV length, always: the attention reads the selection over the live cells only, and the padded
    // pools are dead slots of the selection, so a longer selection marks the same live cells)
    // (LLAMA_KV_BUCKET_DEV_POOLS=0: only once the selection is full, as for the sparse-exact buckets)
    static const bool dev_pools = getenv("LLAMA_KV_BUCKET_DEV_POOLS") == nullptr || atoi(getenv("LLAMA_KV_BUCKET_DEV_POOLS")) != 0;
    st.n_pool_g = 0;
    if ((dev_pools && get_idx()->get_n_kv_live() > 0) || (get_idx()->get_n_kv_bucketed() && kpool_pad(st.n_pool_real) >= mem->get_kpool_top())) {
        const uint32_t n_kv = get_idx()->get_n_kv();
        st.n_pool_g = std::max(kpool_pad(st.n_pool_real), kpool_pad(n_kv/kpool));
    }

    // k19-p2qsa: a token sees at most the pools of its sequence (set_input_kpool fill_mask), and the graph's pool count is above the real
    // one (kpool_pad, or the bucket above), so with at most top pools in each sequence of the ubatch the top_k of the indexer keeps every
    // visible pool: the selection is known without scoring (LLAMA_QSA_ALLFIT=0: always score)
    static const bool allfit_on = getenv("LLAMA_QSA_ALLFIT") == nullptr || atoi(getenv("LLAMA_QSA_ALLFIT")) != 0;
    st.allfit = allfit_on && g_qsa_allfit_pause.load() <= 0 && mem->get_kpool_top() > 0;
    for (uint32_t i = 0; i < ubatch.n_tokens && st.allfit; ++i) {
        st.allfit = lay.seqs[ubatch.seq_id[i][0]].pools.size() <= mem->get_kpool_top();
    }
}

bool llama_memory_hybrid_idx_context::get_kpool_allfit() const {
    return kpool_cur().allfit;
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    const auto & st = kpool_cur();
    return st.n_pool_g > 0 ? st.n_pool_g : kpool_pad(st.n_pool_real);
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos, ggml_tensor * allfit_idx) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    // k19-pf: a graph that only writes the indexer cache (the qwen4exp MTP drafter's kv_only prompt pass) has no pool cells / idxs,
    // mask or tail inputs (pass all four as nullptr): only the new pools' members, reps and positions are filled
    // k19-p2qsa: an all-fit graph (allfit_idx) has none of the four either: the pools and the tail go to scratch, only allfit_idx is filled
    const bool no_sel      = pool_cells == nullptr;
    const bool writes_only = no_sel && allfit_idx == nullptr;
    GGML_ASSERT(no_sel == (pool_idxs == nullptr) && no_sel == (pool_mask == nullptr) && no_sel == (tail_idxs == nullptr));
    GGML_ASSERT(no_sel || ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(no_sel || ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(no_sel || ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(no_sel || ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = no_sel ? (st.n_pool_g > 0 ? st.n_pool_g : kpool_pad(st.n_pool_real)) : (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == (st.n_pool_g > 0 ? st.n_pool_g : kpool_pad(st.n_pool_real)));
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(no_sel || (pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens));
    GGML_ASSERT(no_sel || (tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens));
    GGML_ASSERT(no_sel || (pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool));
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // a cell of this ubatch, written before any read, so the padded pools read a finite K row
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // padding and absent cells point at the n_kv sentinel row, one past the live cells
    const int32_t sentinel = (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    // k19-p2qsa: the selection of an all-fit ubatch (see get_kpool_allfit), filled with the pool mask and the tail below
    int32_t * af = nullptr;
    uint32_t  af_sel = 0;
    uint32_t  af_top = 0;
    if (allfit_idx != nullptr) {
        GGML_ASSERT(!writes_only && st.allfit);
        GGML_ASSERT(ggml_backend_buffer_is_host(allfit_idx->buffer) && allfit_idx->type == GGML_TYPE_I32);
        GGML_ASSERT(allfit_idx->ne[1] == (int64_t) n_tokens && ggml_is_contiguous(allfit_idx));
        af_sel = (uint32_t) allfit_idx->ne[0];
        af_top = af_sel / kpool;
        // the top pools plus the tail, as build_inp_kpool sizes n_sel
        GGML_ASSERT(af_sel == kpool*std::min(n_pool, mem->get_kpool_top()) + kpool - 1);
        af = (int32_t *) allfit_idx->data;
    }
    if (sel_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask->buffer));
        GGML_ASSERT(sel_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask->ne[3] == (int64_t) n_tokens && sel_mask->ne[1] == 1 && sel_mask->ne[2] == 1);
        n_sel = (uint32_t) sel_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) sel_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    // writes_only: the pool cells / idxs to a scratch (the pad cells of the new pools read the reps); k19-p2qsa: all-fit too (allfit_idx
    // copies the members)
    std::vector<int32_t> pcell_w, pidx_w;
    if (no_sel) {
        pcell_w.resize(n_pool);
        pidx_w.resize((size_t) n_pool*kpool);
    }
    int32_t * pcell = no_sel ? pcell_w.data() : (int32_t *) pool_cells->data;
    int32_t * pidx  = no_sel ? pidx_w.data()  : (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    // k19-vwidth (LLAMA_KPOOL_INPUT_INC=0: off): with a unified KV the leading pools of a sequence that no layout update changed since
    // the last fill (lay.seqs[s].min_keep) and that start at the same global index are copied from the kept arrays instead of read
    // from the layout again; the rest are written as before and kept. LLAMA_KPOOL_INPUT_INC=2 checks the copies against the layout
    static const int inp_inc = getenv("LLAMA_KPOOL_INPUT_INC") ? atoi(getenv("LLAMA_KPOOL_INPUT_INC")) : 1;
    const bool use_inc = inp_inc > 0 && !writes_only && n_stream_kv == 1;
    if (use_inc && lay.c_kpool != kpool) {
        lay.c_kpool = kpool;
        lay.c_count.fill(0);
    }
    if (use_inc) {
        static std::atomic<bool> logged { false };
        if (!logged.exchange(true)) {
            LLAMA_LOG_INFO("%s: k19-vwidth: k-pool inputs keep the unchanged pools of the last fill (LLAMA_KPOOL_INPUT_INC=0: off)\n", __func__);
        }
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];

        size_t n_kept = 0;
        if (use_inc) {
            const uint32_t start = seq_pool_start[s];
            n_kept = lay.c_start[s] == start ? std::min<size_t>({ (size_t) lay.c_count[s], sq.min_keep, sq.pools.size() }) : 0;
            if (n_kept > 0) {
                GGML_ASSERT(start + n_kept < n_pool);
                memcpy(pcell + start, lay.c_pcell.data() + start, n_kept*sizeof(int32_t));
                memcpy(pidx + (size_t) start*kpool, lay.c_pidx.data() + (size_t) start*kpool, n_kept*kpool*sizeof(int32_t));
                pool_end.insert(pool_end.end(), lay.c_pend.begin() + start, lay.c_pend.begin() + start + n_kept);
                if (inp_inc >= 2) {
                    for (size_t pi = 0; pi < n_kept; ++pi) {
                        const uint32_t j  = sq.pools[pi];
                        const uint32_t ip = start + (uint32_t) pi;
                        GGML_ASSERT(pcell[ip] == (int32_t) gcell(sq, sq.cells[j + kpool - 1].second));
                        GGML_ASSERT(pool_end[ip] == sq.cells[j + kpool - 1].first);
                        for (uint32_t k = 0; k < kpool; ++k) {
                            GGML_ASSERT(pidx[(size_t) ip*kpool + k] == (int32_t) sq.cells[j + k].second && "LLAMA_KPOOL_INPUT_INC=2: kept pool differs");
                        }
                    }
                }
            }
        }

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = pi < n_kept ? seq_pool_start[s] + (uint32_t) pi : (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            if (pi >= n_kept) {
                pcell[ip] = (int32_t) gcell(sq, rep);

                for (uint32_t k = 0; k < kpool; ++k) {
                    pidx[(size_t) ip*kpool + k] = inert ? sentinel : (int32_t) sq.cells[j + k].second;
                }
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                nrep[i_new] = gcell(sq, rep);
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            if (pi >= n_kept) {
                pool_end.push_back(sq.cells[j + kpool - 1].first);
            }
        }

        if (use_inc) {
            // keep this sequence's pools as written
            const uint32_t start = seq_pool_start[s];
            const size_t   n     = sq.pools.size();
            if (lay.c_pcell.size() < start + n) {
                lay.c_pcell.resize(start + n);
                lay.c_pend.resize(start + n);
                lay.c_pidx.resize((size_t) (start + n)*kpool);
            }
            if (n > n_kept) {
                memcpy(lay.c_pcell.data() + start + n_kept, pcell + start + n_kept, (n - n_kept)*sizeof(int32_t));
                memcpy(lay.c_pidx.data() + (size_t) (start + n_kept)*kpool, pidx + (size_t) (start + n_kept)*kpool, (n - n_kept)*kpool*sizeof(int32_t));
                std::copy(pool_end.begin() + start + n_kept, pool_end.begin() + start + n, lay.c_pend.begin() + start + n_kept);
            }
            lay.c_start[s] = start;
            lay.c_count[s] = (uint32_t) n;
            sq.min_keep    = n;
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        // k15-fn (LLAMA_KPOOL_PAD_BITSET=0: sort): the reps as a bitset over the cells, the same smallest free cells as the sorted
        // search, without sorting every rep of the context per ubatch
        static const bool pad_bits = getenv("LLAMA_KPOOL_PAD_BITSET") == nullptr || atoi(getenv("LLAMA_KPOOL_PAD_BITSET")) != 0;
        const int64_t n_cells_all = (int64_t) kv_size*n_stream_kv;
        std::vector<int64_t>  reps;
        std::vector<uint64_t> rep_bits;
        if (pad_bits) {
            rep_bits.assign((size_t) (n_cells_all + 63)/64, 0);
            for (size_t ip = 0; ip < pool_end.size(); ++ip) {
                const int64_t c = pcell[ip];
                GGML_ASSERT(c >= 0 && c < n_cells_all);
                rep_bits[(size_t) c/64] |= uint64_t(1) << (c % 64);
            }
        } else {
            reps.assign(pcell, pcell + pool_end.size());
            std::sort(reps.begin(), reps.end());
        }
        const auto is_rep = [&](int64_t c) {
            if (pad_bits) {
                return c < n_cells_all && ((rep_bits[(size_t) c/64] >> (c % 64)) & 1) != 0;
            }
            return std::binary_search(reps.begin(), reps.end(), c);
        };

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (is_rep(pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    }

    if (writes_only) {
        GGML_ASSERT(gm == nullptr);
        return;
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data != nullptr ? data + (size_t) i*n_pool : nullptr;
            if (row != nullptr) {
                std::fill(row, row + n_pool, drop);
            }

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            if (row != nullptr) {
                std::fill(row + p0, row + p0 + nv, keep);
            }

            // k19-p2qsa: the members of the nv visible pools (the finite scores the top_k keeps, all of them here), then dead slots
            if (af != nullptr) {
                GGML_ASSERT(nv <= af_top);
                int32_t * arow = af + (size_t) i*af_sel;
                memcpy(arow, pidx + (size_t) p0*kpool, (size_t) nv*kpool*sizeof(int32_t));
                for (uint32_t k = nv*kpool; k < af_top*kpool; ++k) {
                    arow[k] = sentinel + (int32_t) k;
                }
            }

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask == nullptr) {
        fill_mask((float *) nullptr); // k19-p2qsa: all-fit, the visible pool counts only
    } else if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = tail_idxs != nullptr ? (int32_t *) tail_idxs->data : nullptr;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) c;
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            if (tidx != nullptr) {
                tidx[(size_t) i*(kpool - 1) + k] = cell;
            }
            if (af != nullptr) {
                // a tail cell is live unless it is the n_kv sentinel (build_qsa_sel's live_tail)
                const uint32_t slot = af_top*kpool + k;
                af[(size_t) i*af_sel + slot] = cell != sentinel ? cell : sentinel + (int32_t) slot;
            }

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}
