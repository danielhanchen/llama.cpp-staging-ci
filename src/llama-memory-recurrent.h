// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <functional>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    // k19-rsbis: per seq, a rollback that rollback_copy already put on the live row (rs_idx stays 0); the snapshot groups still count
    // from the end of the last decode, so a second partial seq_rm before the next ubatch reads the seq is refused as with rs_idx != 0
    std::vector<uint8_t> rs_copied;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // k15-rsr: copies the per-token snapshot `group` of cell `cell` onto its live row (group 0) on the device; false when off or failed
    bool rollback_copy(uint32_t cell, uint32_t group);

    // k16-draft (LLAMA_RS_REPLAY, the commit-by-replay of Strata's verify window, include/strata/core/verify.hpp:13-18): the delta-net
    // states (s_l) keep one row per cell instead of 1 + n_rs_seq. A window of at most 1 + n_rs_seq tokens per sequence reads the row and
    // leaves it as it was, and stores its delta-net inputs (k, v, gate, beta; rk_l ...) per layer and cell; its conv histories (r_l, p_l)
    // keep their snapshot rows. The commit (llama_context::rs_replay_commit, run before the memory is read again: the next ubatch, a
    // state save, a sequence copy) advances each row by the window's accepted tokens with the same op and copies the conv snapshot of a
    // rollback onto the live row. Larger ubatches (prompts) write the final state as without drafting and cannot be rolled back.
    bool replay = false;
    std::vector<ggml_tensor *> rk_l, rv_l, rg_l, rb_l;
    mutable std::vector<int32_t> rk_h;  // per layer: the key heads the graph stored (H_k, or H_v where it repeats them)   // [S_k, H_v, n_win, size], [S_v, H_v, n_win, size], [1, H_v, n_win, size] x 2
    std::vector<uint32_t> pend_n;    // per cell: tokens of the window not committed yet (0: the row is current)
    std::vector<uint32_t> pend_rb;   // per cell: the tokens of that window rolled back (seq_rm)
    std::vector<uint8_t>  par;       // per cell: the half of the stores that holds its window
    std::vector<uint32_t> conv_rb;   // per cell: a conv snapshot still to copy onto the live row (its window committed by a fused ubatch)
    bool replay_fused_ok = false;    // LLAMA_RS_REPLAY_FUSED and a backend with ggml_gated_delta_net_replay
    bool rp_fused = false;           // the current ubatch commits the windows of its cells itself
    bool rp_moved = false;           // k19-npmtp: a cell of the current ubatch reads its state from another row (src0 != the cell)
    std::vector<int32_t> rp;         // its (kept tokens, half) per sequence
    bool replay_fuse(const llama_ubatch & ubatch);
    uint32_t get_n_win_store() const { return 2*(1 + n_rs_seq); } // token slots per cell in the stores (two halves)
    std::function<void(bool)> commit_cb; // set by llama_context: runs the commit graph (true: waits for it)
    uint64_t n_replay_commit = 0;
    bool has_pending() const;
    void replay_commit_if_pending(bool sync = true) const; // calls commit_cb (the state reads below call it first)
    // k19-npmtp: the windows of the cells a ubatch reads, copies or moves are committed before it, the others stay pending: a window is
    // rolled back (seq_rm) after the whole llama_decode, so a later ubatch of another sequence in the same decode (--parallel, a prompt
    // chunk beside a verify window) that committed every cell left nothing to roll back (seq_rm false, server abort).
    // LLAMA_RS_REPLAY_SCOPED=0: every pending window is committed before each ubatch (the old behaviour)
    bool replay_scoped = true;
    bool in_prepare    = false; // prepare() simulates find_slot on a copy of the cells: no commit from there
    void replay_commit_cells(const std::vector<uint8_t> & which, bool sync);
    void replay_commit_cell(int32_t c);
    void replay_commit_ubatch(const llama_ubatch & ubatch, bool placed);
    // k19-npmtp: find_slot moves the rollback rows of the cells it swaps with them: the snapshot groups 1..n_rs_seq (state and conv rows)
    // and, with replay, a pending window (instead of committing it). The ubatch graph copies only the live rows (s_copy), so a verify
    // window whose cell a later ubatch of the same decode moved rolled back from another cell's rows (snapshot rows: wrong state) or
    // could not roll back (replay). grp_moved: the cells whose rows moved in the last find_slot (s_copy reads their snapshot there)
    std::vector<uint8_t> grp_moved;
    std::function<void(const std::vector<std::pair<uint32_t, uint32_t>> &)> move_cb; // set by llama_context (rs_replay_move)
    // k19-fixB: set by llama_context: waits for the work queued on the state rows (decodes, replay commits, async state reads). Without
    // it rollback_copy is off (the indexed rollback): llama_context leaves it unset under -sm tensor, where each rollback would add views
    // to the meta backend's compute view arena of a reused graph
    std::function<void()> sync_cb;
    std::vector<std::pair<uint32_t, uint32_t>> rp_swaps;
    uint64_t n_replay_move = 0;
    uint64_t n_snap_move   = 0;
    // k19-b33mv: the move copies row to row (rs_move_rows) through one scratch unit per buffer type and shape (r, s, PLE conv rows;
    // with replay the k, v, gate/beta window slices of a cell), allocated with the cache (so the fit prices them) and only with several
    // sequences: no graph and no allocation at move time (the dup graph of 9c2e needed an unpriced compute buffer, ~900 MiB on G4 fn4)
    std::vector<ggml_tensor *> mv_tmp_r_l, mv_tmp_s_l, mv_tmp_p_l, mv_tmp_k_l, mv_tmp_v_l, mv_tmp_g_l;
    // k19-fixD: set by llama_context under -sm tensor: the move reads and writes the units of the parent tensors through host memory
    // instead of copying views (each view is kept in the meta backend's compute view arena, 16 per tensor of the buffer, until the
    // second graph rebuild after it: one find_slot of two swaps with 7 snapshot groups made 56 views on a buffer of 4 tensors)
    bool move_by_host = false;
    void rs_move_rows(const std::vector<std::pair<uint32_t, uint32_t>> & swaps);
    void replay_swap(int32_t a, int32_t b, const std::vector<uint8_t> & in_ubatch);
    void replay_mark(const llama_ubatch & ubatch);
    uint64_t n_rollback_copy = 0;

    // k19-rollback: a copy of one sequence's state rows (every r / s / PLE row of its cell) in host memory, pinned where the device has a
    // host buffer type. snap_save / snap_restore read and write the rows on the stream of the backend `be` gives for each tensor (ordered
    // with the decodes before and after, the host does not wait); `be` returns nullptr for rows the host reads directly. This is the rollback
    // of a drafted verify window that keeps no per-token rows on the device (n_rs_seq 0): the copy is taken before the window and put back
    // when a draft is not fully accepted, and the accepted tokens are decoded again (llama-server's speculative checkpoint, LLAMA_SPEC_RS_SNAP).
    // One sequence per memory only (a ubatch of several sequences may move the cell between the copy and its restore)
    bool snap_supported() const { return n_rs_seq == 0 && !replay && n_seq_max == 1 && !is_empty(); }
    bool snap_save   (llama_seq_id seq_id, const std::function<ggml_backend_t(const ggml_tensor *)> & be);
    bool snap_restore(llama_seq_id seq_id, const std::function<ggml_backend_t(const ggml_tensor *)> & be);
    size_t snap_bytes() const { return snap_size; }
    uint64_t n_snap_save = 0, n_snap_restore = 0;

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // k15-pp: a state write of one sequence's cell reads these tensors in place of the cell's rows and writes this position (the state
    // llama_set_state_capture kept); set around one state_write by llama_context, the io may read the tensors later
    const std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> * cap_map = nullptr;
    llama_pos cap_pos = -1;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // k16-draft: a no_alloc memory probe holds dummy buffers, sized from the tensors (LLAMA_RS_PROBE_DUMMY=0: real buffers)
    bool probe_sized = false;
    std::vector<ggml_backend_buffer_ptr> probe_base; // the probe's real allocation of the base rows (LLAMA_RS_PROBE_DUMMY=1)

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);

    // k19-rollback: see snap_save
    ggml_backend_buffer_ptr snap_buf;
    size_t       snap_size = 0;
    int32_t      snap_cell = -1;
    llama_seq_id snap_seq  = -1;
    llama_pos    snap_pos  = -1;
    bool snap_rows(int32_t cell, const std::function<void(ggml_tensor *, size_t off, size_t size, size_t off_host)> & fn) const;
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;
    const llama_ubatch * peek_ubatch(uint32_t k) const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    // k16-draft: the replay window's input stores (null when LLAMA_RS_REPLAY is off for this memory)
    bool          get_replay() const;
    uint32_t      get_n_win()  const;
    ggml_tensor * get_rk_l(int32_t il) const;
    ggml_tensor * get_rv_l(int32_t il) const;
    ggml_tensor * get_rg_l(int32_t il) const;
    ggml_tensor * get_rb_l(int32_t il) const;
    void          set_rk_h(int32_t il, int32_t h) const;
    bool          get_replay_fused() const;              // this ubatch's window commits the stored one (ggml_gated_delta_net_replay)
    bool          get_replay_moved() const;              // k19-npmtp: a sequence of this ubatch reads its state from another row
    void          set_replay_input(ggml_tensor * t) const; // I32 [2, n_seqs]: kept tokens of the stored window, its half

    int32_t s_copy(int i) const;

    // During decode (one token per sequence) every sequence usually reads its state from the row it
    // writes back to, so gathering the states is a copy of the cache onto itself. Returns the first
    // row of that in-place run when it covers all n_seqs sequences of the ubatch, or -1 when the
    // states have to be gathered. Only reads the cache bookkeeping: s_copy() clears the rollback
    // index as a side effect and must still run once per input.
    int32_t get_rs_alias(const llama_ubatch & ubatch) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
