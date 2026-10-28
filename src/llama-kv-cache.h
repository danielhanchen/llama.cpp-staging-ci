// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "");

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_seq_max() const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // k19-fixB: whether token i of the ubatch sees a cell in [c0, c1) of its stream: a used cell (or a cell of the ubatch) of one of
    // its sequences at a position <= its own. A ubatch over several streams counts as seeing them
    bool exact_row_sees(const slot_info & sinfo, const llama_ubatch & ubatch, uint32_t i, uint32_t c0, uint32_t c1) const;

    // The stream holding seq_id's cells.
    uint32_t get_stream(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
    const slot_info_vec_t *   sinfos_in);

    // undo a state_read() of seq_id (-1 for the whole cache) that another memory module failed to complete
    void state_clear(llama_seq_id seq_id);

    //
    // graph_build API
    //

    // (k19-graph: n_kv_live, if given, gets the 256-padded n_kv before any bucket; 0 when the ubatch takes no device KV length)
    uint32_t get_n_kv(const slot_info & sinfo, bool * bucketed = nullptr, uint32_t * n_kv_live = nullptr) const;

    // k19-vgraph: coarser n_kv for small ubatches (decode, speculative verify), so their graphs keep their shapes, and their CUDA graphs
    // stay captured, while the context grows by up to an eighth instead of 256 cells (see llama_memory_hybrid_idx, the only user).
    // A ubatch of at most n_tokens_max tokens whose 256-padded n_kv is at least max(n_kv_min, n_tokens*n_kv_min_per_token) gets
    // n_kv padded to max(256, pow2floor(n_kv)/div) cells. Off when n_tokens_max == 0
    struct n_kv_bucket_params {
        uint32_t n_tokens_max       = 0;
        uint32_t n_kv_min           = 0;
        uint32_t n_kv_min_per_token = 0;
        uint32_t div                = 8;
        // k19-graph: the attention reads the live length on the device (ggml_flash_attn_ext_set_kv_len), so the bucket cannot move
        // its sums: every ubatch of up to dev_n_tokens_max tokens is bucketed from the first cell, dev_div buckets per octave of at
        // least dev_step_min cells (where the context allows it, set_n_kv_bucket_dev_allowed; else the fields above)
        bool     dev_len            = false;
        uint32_t dev_n_tokens_max   = 8;
        uint32_t dev_div            = 2;
        uint32_t dev_step_min       = 2048;
    };
    void set_n_kv_bucket(const n_kv_bucket_params & p) { n_kv_bucket = p; }
    // a context whose verify rows must equal single token decodes (llm_graph_exact_row_n_kv) keeps the 256 cell padding
    void set_n_kv_bucket_allowed(bool value) { n_kv_bucket_allowed = value; }
    // k19-graph: the device KV length buckets (n_kv_bucket_params::dev_len) only where flash attention runs (the context's choice)
    void set_n_kv_bucket_dev_allowed(bool value) { n_kv_bucket_dev_allowed = value; }
    // k19-fullcost: see llama_memory_hybrid_idx::set_n_kv_bucket_dev_wide
    void set_n_kv_bucket_dev_step_min(uint32_t value) { n_kv_bucket.dev_step_min = value; }
    // k19-graph: every attention layer's K / V sits on a device whose flash attention takes the KV length on the device
    bool fa_kv_len_supported() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // the cells of one stream that the SWA purge of apply_ubatch() changed, as they were before it
    struct purge_backup {
        uint32_t              strm;
        std::vector<uint32_t> idxs;
        llama_kv_cells        cells;
    };

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    // k19-rsbis: with `purged`, the cells the SWA purge changes outside the slot are saved there first (prepare() restores them)
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch, std::vector<purge_backup> * purged = nullptr);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

    // advisory get_prev_tokens() for a ubatch that is not applied yet (prefetching): the predecessors come from the
    // ubatch itself, then from `pending` (the ubatch right before it, not applied either, can be null), then from the cells
    // known[i] = 0 when a predecessor of token i cannot be resolved ahead (a position gap inside the batch)
    // returns false when positions do not give the token order (embd / mixed ubatch) or a token has several sequences
    bool get_prev_tokens_ahead(const llama_ubatch & ubatch, const llama_ubatch * pending, uint32_t n,
            std::vector<llama_token> & res, std::vector<uint8_t> & known) const;

    // token of seq_id at the largest cached position <= p, LLAMA_TOKEN_NULL if none or no cell ext
    llama_token seq_pos_tok_le(llama_seq_id seq_id, llama_pos p) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // k19-vgraph: see set_n_kv_bucket
    n_kv_bucket_params n_kv_bucket;
    bool n_kv_bucket_allowed = true;
    bool n_kv_bucket_dev_allowed = false;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    uint32_t n_rot_k = 0;
    uint32_t n_rot_v = 0;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // k17-small (LLAMA_KV_LAZY=1, opt-in): device KV buffers whose address range is reserved whole and backed with memory as the cells
    // in use grow (ggml_backend_buffer_commit of the CUDA backend), so a long context costs only the rows it has reached. One stream,
    // non-transposed V (flash attention) only: each layer's K and V rows of cells [0, n) are then one prefix of the tensor
    std::vector<ggml_backend_buffer_t> lazy_bufs;
    bool (*lazy_commit_fn)(ggml_backend_buffer_t, size_t, size_t) = nullptr;
    size_t (*lazy_committed_fn)(ggml_backend_buffer_t) = nullptr;
    uint32_t lazy_cells = 0;   // cells [0, lazy_cells) of every layer are backed
    // k17-small (lazy KV step 2): the MoE cache that gives device memory back as the cells grow (lazy_set_shrink), the free memory a
    // growth leaves on the device (LLAMA_KV_LAZY_MARGIN_MIB, at most what is free at the first growth; -1: not measured yet), and of a
    // measurement (no_alloc) the device KV bytes the lazy buffers would leave unbacked at load (see lazy_unbacked)
    std::function<size_t(ggml_backend_dev_t, size_t)> lazy_shrink_fn;
    std::map<ggml_backend_dev_t, int64_t> lazy_margin;
    std::map<ggml_backend_buffer_type_t, size_t> lazy_unbacked_est;
public:
    // back the rows of cells [0, n_cells) (rounded up to the step); false if the device has no memory for them. shrink_ok: between
    // batches (no graph pending), so the MoE cache may give memory back first (lazy_set_shrink)
    bool lazy_ensure(uint32_t n_cells, bool shrink_ok = false);
    bool lazy_on() const { return !lazy_bufs.empty(); }
    void lazy_set_shrink(std::function<size_t(ggml_backend_dev_t, size_t)> fn) { lazy_shrink_fn = std::move(fn); }
    // the device KV bytes that are not backed now, per buffer type (a measurement: the bytes the lazy buffers would leave unbacked at
    // load, reported only where the MoE cache shrinks, LLAMA_MOE_CACHE_SHRINK)
    std::map<ggml_backend_buffer_type_t, size_t> lazy_unbacked() const;
private:

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

public:
    // k19-vwidth: the first KQ mask row of a sequence in a decode, kept with the position of the cells' modification log it was read
    // at, so the next decode copies it and recomputes only the cells logged since then plus the cells of the sequence whose positions
    // lie between the two rows' positions (set_input_kq_mask)
    struct kq_row_entry {
        const void * cells = nullptr;
        uint32_t     strm  = 0;
        llama_seq_id seq   = -1;
        uint32_t     flags = 0;
        size_t       elem  = 0;
        llama_pos    p1 = 0, p1_x = 0, p1_y = 0, spm = 0;
        uint64_t     log_pos = 0;
        uint64_t     used    = 0;
        std::vector<uint8_t> row; // row.size()/elem cells are valid
    };
    struct kq_row_cache {
        std::vector<kq_row_entry> entries;
        uint64_t n_use  = 0;
        uint64_t n_hit  = 0;
        uint64_t n_miss = 0;
    };

private:
    mutable kq_row_cache kq_rows;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);

    void state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;
    const llama_ubatch * peek_ubatch(uint32_t k) const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    // k19-vgraph: whether this ubatch's n_kv was padded past 256 cells (llama_kv_cache::set_n_kv_bucket)
    bool get_n_kv_bucketed() const { return n_kv_bucketed; }

    // k19-graph: the cells the attention reads (256-padded, before the bucket) when it takes its length on the device, else 0
    uint32_t get_n_kv_live() const { return n_kv_live; }

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // see llama_kv_cache::exact_row_sees(), for the current ubatch
    bool exact_row_sees(const llama_ubatch & ubatch, uint32_t i, uint32_t c0, uint32_t c1) const;

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

    // see llama_kv_cache::get_prev_tokens_ahead()
    bool get_prev_tokens_ahead(const llama_ubatch & ubatch, const llama_ubatch * pending, uint32_t n,
            std::vector<llama_token> & res, std::vector<uint8_t> & known) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
    bool    n_kv_bucketed = false;
    uint32_t n_kv_live    = 0;
};
