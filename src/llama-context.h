// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-moe-cache.h"
#include "llama-prefetch.h"
#include "llama-weight-store.h"
#include "llama-sysmem.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;
class llama_memory_recurrent;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

// one tensor range of a state save: size bytes at offset of tensor go to dst
struct llama_state_write_info {
    ggml_tensor * tensor;
    uint8_t     * dst;
    size_t        size;
    size_t        offset;
};

// LLAMA_MTP_DEV_H: the MTP handoff on the device. The draft context owns `sink` [n_embd, n_rows + 1]; the target writes the h_nextn row of
// batch entry j to sink row 1 + j (device to device, on its own stream, after waiting ev_dft) instead of copying it to the host, and
// records ev_tgt; the draft graph reads its rows from the sink (row 0 = the pending row of the previous batch) after waiting ev_tgt,
// then copies its last row to row 0 and records ev_dft. No host copy of the hidden state and no host sync on the target in between.
struct llama_mtp_dev_h {
    llama_context * tgt = nullptr;
    llama_context * dft = nullptr;
    ggml_context * meta = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * sink = nullptr;
    ggml_backend_t backend_dft = nullptr; // the draft context's backend of the sink's device
    ggml_backend_dev_t dev = nullptr;
    ggml_backend_event_t ev_tgt = nullptr;
    ggml_backend_event_t ev_dft = nullptr;
    bool ev_tgt_rec = false;
    bool ev_dft_rec = false;
    bool usable = true;      // false after a target graph that cannot write the sink (other device, host buffer, layout)
    int32_t n_rows   = 0;    // batch rows the sink holds (row 0 is the pending row)
    int32_t min_rows = 64;   // smaller batches (decode, verify) keep the host path [LLAMA_MTP_DEV_H_MIN]
    int32_t last_rows = 0;   // rows the target's last decode wrote to the sink, 0 = the host path
    int64_t n_batches = 0;   // drafter passes that read the sink (engagement count, logged when the drafter is freed)
    int64_t n_rows_dev = 0;

    ~llama_mtp_dev_h();
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    // read back the PLE table pages the page cache gave up after the load's part read (see llama_ple_topup)
    size_t ple_topup();

    void synchronize();

    // synchronize() unless nothing was queued since the last one (see sync_pending): the output getters call this, so that
    // reading several outputs of one decode costs one stream synchronization instead of one per getter (LLAMA_SYNC_SKIP=0: always)
    void synchronize_outputs();
    void synchronize_pending();

    const llama_model   & get_model()   const;
    bool moe_cache_active() const { return moe_cache != nullptr; } //
    const llama_moe_cache * get_moe_cache() const { return moe_cache.get(); } //
    llama_moe_cache * get_moe_cache_mut() { return moe_cache.get(); } //
    bool moe_freeze_small(); // see llama_moe_freeze_small
    bool moe_pin_small();    // see llama_moe_pin_small
    bool moe_pin_small_set(bool on); // see llama_moe_pin_small_set
    void moe_request_begin();        // see llama_moe_request_begin
    // see llama_set_idle_abort
    bool (*idle_abort_fn)(void *) = nullptr;
    void * idle_abort_data        = nullptr;
    bool moe_cpu_miss_drop_slow(); // see llama_moe_cpu_miss_drop_slow
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);
    // the nextn rows [0, n) of the last decode on the host before the decode is done (see rows_ev)
    int64_t nextn_rows_wait(int64_t n);
    bool    mtp_weights_restore(); // kdp: see llama_mtp_draft_weights_restore
    bool    mtp_lend_prepare();    // kdp: the host copy at the first lend (host-RAM checked); false: not lendable
    bool    mtp_lend_warm();       // that copy at load where a prompt group is expected to borrow the drafter's weights
    bool    mtp_lend_active = false; // kdp: the last released layer-major group found too little free memory (llama_mtp_draft_lend_active)
    // an MTP context of a draft model that borrows token_embd / output from tgt's model (llama_model_params::model_shared): a
    // target lends only its own drafter's weights
    bool    mtp_drafts_for(const llama_context * tgt) const;
    float * nextn_rows_early();
    // llama_nextn_rows_ready: an event after the copies of the last decode's nextn rows to the host (nullptr when it made none)
    void *  nextn_rows_ready();

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_verify(bool value);
    void set_verify_exact(bool value);
    bool get_verify_exact() const; //
    bool restore_pp_plan(bool idle = false); // (see the definition); idle: from llama_prepare_prompt_plan
    uint32_t dsv41_far_rows(const llama_ubatch & ubatch, const std::vector<llama_pos> & ends, int32_t n_out) const; //
    void reserve_small_max(const llama_ubatch & ubatch); // (see the definition)
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);

    // LLAMA_MTP_DEV_H (see llama_mtp_dev_h); the draft context calls these
    bool    mtp_dev_h_attach(llama_context * tgt, int32_t n_rows);
    bool    mtp_dev_h_begin();
    void    mtp_dev_h_end(int32_t pending_row);
    bool    mtp_dev_h_get_row(int32_t row, float * dst);
    bool    mtp_dev_h_set_row(int32_t row, const float * src);
    int32_t mtp_dev_h_last_rows() const; // the target context

    std::unique_ptr<llama_mtp_dev_h> mtp_dev_h_own;  // draft: the sink it owns
    llama_mtp_dev_h * mtp_dev_h_sink   = nullptr;    // target: the sink of its drafter
    ggml_tensor     * mtp_dev_h_src    = nullptr;    // draft: the graph reads h from here while set (llm_graph_params::h_dev_src)
    bool              mtp_dev_h_this   = false;      // target: the current decode writes the sink
    bool              mtp_dev_h_waited = false;      // target: the current decode waited ev_dft
    int32_t           mtp_dev_h_done   = 0;          // target: rows of the current decode written to the sink
    std::vector<std::array<int64_t, 3>> mtp_dev_h_runs; // target: (host row offset, batch index, rows) written this decode

    // target: write the h_nextn rows of `ubatch` (host offset `offset`) to the sink; false = the host path for this ubatch
    bool mtp_dev_h_store(ggml_backend_t backend_h, ggml_tensor * t_h_nextn, const llama_ubatch & ubatch, int64_t offset);
    void mtp_dev_h_decode_begin(const llama_batch_ext & batch_inp);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);
    int64_t prompt_run_tokens(int64_t n_outputs_all, int64_t n_tokens_all); //

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    // LLAMA_STATE_SEQ_FLAGS_ASYNC: queue the tensor reads of a state save without synchronizing (see state_get_async)
    void state_get_async(const std::vector<llama_state_write_info> & winfos, bool no_grow = false);

    // wait until the last asynchronous state save has reached its destination
    void state_async_wait();
    // wait only until the device rows of the asynchronous save in flight are read (its D2H copies into the pinned
    // staging done), not for the worker's host copy into the destination (LLAMA_STATE_ASYNC_SPLIT_WAIT=0: the whole save)
    void state_async_wait_reads();

    // (LLAMA_RS_REPLAY): advances the delta-net state rows by the accepted tokens of the last window (see
    // llama_memory_recurrent::replay); called by the recurrent memory before it is read again
    void rs_replay_commit(bool sync);
    // decides rs_mode and reserves the fallback allocation of the largest commit graph at load (see rs_replay_commit)
    void rs_replay_setup();
    // moves the pending windows (stores and conv snapshot rows) of cells that find_slot swapped, in swap order
    void rs_replay_move(llama_memory_recurrent * mem, const std::vector<std::pair<uint32_t, uint32_t>> & swaps);
    // the host copy of a sequence's recurrent state rows for a speculative rollback (llama_memory_recurrent::snap_save)
    llama_memory_recurrent * rs_snap_mem() const;
    bool rs_snap_save(llama_seq_id seq_id);
    bool rs_snap_restore(llama_seq_id seq_id);
    // the commit graph of the pending cells (worst: every cell with a full window, the pending state untouched, for the reserve)
    void rs_commit_build(ggml_context * ctx, ggml_cgraph * gf, bool worst);
    ggml_gallocr_ptr         rs_ga_fb; // one backend: the reserved fallback allocation (null: not reserved)
    uint64_t                 rs_fb_used = 0;
    llama_memory_recurrent * rs_replay_mem = nullptr;
    llama_memory_recurrent * rs_move_mem   = nullptr; // any recurrent memory with rollback rows (rs_replay_move)
    ggml_backend_sched_ptr   rs_sched;
    int64_t                  rs_replay_us = 0;
    // the last commit graph, reused while the same cells commit the same tokens (LLAMA_RS_REPLAY_REUSE=0: rebuilt every time)
    struct rs_graph {
        ggml_context_ptr ctx;
        ggml_cgraph *    gf = nullptr;
        ggml_gallocr_ptr ga; // its own allocation on rs_backend (null: allocated by rs_sched or in rs_ga_fb)
        bool             shared = false; // allocated in the buffer reserved at load (rs_ga_fb), re-allocated on a key switch
    };
    std::map<std::vector<uint32_t>, rs_graph> rs_graphs; // per (cell, tokens, rolled back) set
    std::vector<uint32_t>    rs_key; // the graph the sched holds the allocation of
    uint64_t                 rs_reused = 0;
    uint64_t                 rs_conv_batched = 0; // conv-only commits done as one batched copy (LLAMA_RS_CONV_BATCH)
    uint64_t                 rs_sched_rebuilt = 0; // cached commit graphs built again (see rs_replay_commit)
    // every state and window tensor on one backend: the commit graphs run on it directly, each with its own small allocation, so a
    // switch between cached graphs neither re-allocates nor synchronizes (LLAMA_RS_REPLAY_SCHED=1: through rs_sched instead)
    int                        rs_mode = 0; // 0 not decided, 1 one backend, -1 sched
    ggml_backend_t             rs_backend = nullptr;
    ggml_backend_buffer_type_t rs_buft = nullptr;
    size_t                     rs_ga_bytes = 0, rs_ga_bytes_max = 0;

    // set when backend work is queued (graph_compute, async state copies), cleared by synchronize()
    bool sync_pending = true;

    // allocate the staging buffers of asynchronous saves with these flags up front (see llama_state_seq_reserve_async)
    size_t state_async_reserve(llama_state_seq_flags flags);

    // pinned staging for a whole-state save of `need` bytes (0: the bound of one sequence), see llama_state_seq_async_ext_reserve
    void state_async_ext_request(size_t need);

    void state_async_init_backends();

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    // (LLAMA_LM_CKPT_MERGE)
    void set_ubatch_breaks(const int32_t * breaks, int32_t n);
    bool set_batch_ahead(const llama_token * tokens, int32_t n, llama_pos pos0, llama_seq_id seq_id);
    bool set_state_capture(llama_seq_id seq_id, int32_t i_batch);
    bool state_capture_supported() const;
    bool state_captured(llama_seq_id seq_id, llama_pos * pos_max) const;
    bool state_captured_at(llama_seq_id seq_id, int32_t i_batch, llama_pos * pos_max);
    bool layer_major_on() const;
    bool layer_major_spans(int64_t n_before, int64_t n_after) const;

    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // an exact verify ubatch builds one attention per row (LLAMA_EXACT_FA_GROUP=0, or flash attention over a K / V cache that is not
    // F16); its rows are at most exact_rows_max(), the budget of graph_max_nodes (decode splits a longer verify batch)
    bool     exact_rows_per_query() const;
    uint32_t exact_rows_max() const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // sets the threadpool and the number of threads of the backends for a graph of a batch (batched) or of a single token
    void set_compute_threads(bool batched);

    // LLAMA_VRAM_WATCH=1 (diagnostic): the least free memory of each GPU seen after a graph computation (every 32 graphs and at the
    // end), logged when the context is freed
    void vram_watch_sample(bool force);
    int     vram_watch = -1; // -1 = not read yet
    int64_t vram_watch_n = 0;
    std::vector<size_t> vram_watch_min;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    // the host seed of a prompt does not fit: large prompt ubatches seed the MoE cache on the device instead (see the definition);
    // host_seed_fits = the host seed fits, large prompt ubatches may still seed on the device where its price is not wanted
    void moe_cache_devseed_fallback(bool host_seed_fits = false);

    // reserve the graph of a prefetched ubatch that seeds the MoE cache from its own ids if the devices have room for it (staging_pp:
    // staging memory of the pp graph per backend), returns false if it does not fit
    static std::atomic<bool> g_fit_seed_priced; // a memory probe priced the seed graph (sched_reserve)
    static bool fit_seed_price_on();             // LLAMA_FIT_SEED_PRICE (default on)
    static int64_t seed_own_min();               // smallest prefetched ubatch that seeds from its own ids: 1024 beside DFlash2 (LLAMA_MOE_CACHE_SEED_OWN_DFLASH), else 0
    bool seed_price_wanted() const;              // the cache holds the experts of few tokens (LLAMA_FIT_SEED_PRICE_MAX_TOKENS)
    bool sched_reserve_seed(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, const std::vector<size_t> & staging_pp);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);
    void set_sampler_raw_logits(llama_seq_id seq_id, bool raw);
    void set_sampler_keep_logits(llama_seq_id seq_id, bool keep);
    bool set_sampler_swap(llama_seq_id seq_id, llama_sampler * sampler);
    // the samplers the graph of a ubatch applies (graph_params): not those of the sequences copying raw logits, except in a reserve
    std::map<llama_seq_id, llama_sampler *> graph_samplers() const;
    bool graph_reserving = false;

    // sched_reserve: graph_reserve builds an embeddings ubatch (the target features of a DFlash / DSpark drafter), and the sizes of that
    // graph in a memory probe
    bool reserve_embd = false;
    std::vector<size_t> reserve_embd_sizes;

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // create a new backend scheduler for a graph of the specified size and register the callbacks below
    void sched_new(size_t max_nodes, bool parallel);

    // ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID and updates the MoE cache
    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);
    static bool sched_copy_experts_body(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);
    // the prompt map of an in-place prompt layer (llama_moe_cache::prompt_copy): records its routing, seeds as the staging copy does
    bool moe_prompt_copy(ggml_backend_t backend, ggml_tensor * dst, ggml_cgraph * graph, int32_t il);
    // ggml_backend_sched prefetch callback, selects how the weights in host memory are prefetched
    static ggml_backend_sched_prefetch_mode sched_prefetch_mode(const ggml_tensor * src, const ggml_tensor * dst, ggml_cgraph * graph, void * user_data);

    // weight store: the host weights of the prefetched copies of a graph, read ahead from the model files, and the bytes of the copies
    static void sched_stream_plan(const ggml_tensor * const * srcs, int n_srcs, void * user_data);
    static bool sched_stream_source(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, size_t offset, size_t size, void * user_data);

    // places the prefetched copies of large batches in slots lent by the MoE cache
    static void sched_lend(ggml_backend_t backend, ggml_backend_sched_lend_copy * copies, int n_copies, int depth, bool dry, void * user_data);

    // true if the copy of the experts used by a MUL_MAT_ID with these ids may skip reading the ids (all experts are copied)
    bool sched_copy_all_experts(const ggml_tensor * src, const ggml_tensor * ids) const;

    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;
    llama_moe_cache_ptr moe_cache;

    // advisory prefetch of the lazily read PLE rows, one ubatch ahead (see llama_ple_prefetch_mode)
    // the rows are predicted on the main thread before the ubatch before them is processed, the workers warm the
    // page cache meanwhile; set_input still computes the authoritative rows, so nothing here changes the results
    std::unique_ptr<llama_row_prefetcher> ple_prefetcher;
    // the next prompt batch announced before a decode (llama_set_batch_ahead): when that decode has queued its last ubatch, the
    // host gathers of the announced batch (token and PLE embedding rows) are converted on host threads while the GPU computes
    // (ggml_backend_cpu_get_rows_ahead); the next decode's gathers copy the rows that match. Same conversions, so exact
    struct batch_ahead_state {
        std::vector<llama_token> tok;
        llama_pos                pos0    = 0;
        llama_seq_id             seq_id  = 0;
        bool                     pending = false;
        std::vector<const ggml_tensor *> started; // tables with a result, dropped with the context
        int64_t                  n_started = 0;
        int64_t                  n_rows    = 0;
        int64_t                  t_us      = 0;
    } batch_ahead;
    void batch_ahead_start(const llama_memory_context_i * mctx);
    void batch_ahead_drop();
    // the PLE table read-in after the load (LLAMA_PLE_RESIDENT), stopped and joined when the context goes (also from a
    // constructor that throws later)
    struct ple_res_runner {
        std::thread       th;
        std::atomic<bool> stop{false};
        // the part of the table pinned by a GPU backend's host registration (LLAMA_PLE_RESIDENT_PARTIAL_PIN=1), released here
        void *            pin_addr  = nullptr;
        void (*pin_unmap)(void *)   = nullptr;
        size_t            pin_len   = 0; // published to llama_sysmem::pinned_file_bytes while pinned
        // the table's page span once a part read (not locked, not pinned) finished during the load, for ple_topup
        std::atomic<bool> part_done{false};
        uint8_t *         span_addr = nullptr;
        size_t            span_len  = 0;
        size_t            part_len  = 0; // the bytes of the span the load read
        int               fd        = -1;
        size_t            fd_offs   = 0; // file offset of span_addr + lead
        size_t            lead      = 0;
        ~ple_res_runner() {
            stop.store(true);
            if (th.joinable()) {
                th.join();
            }
            if (pin_addr != nullptr && pin_unmap != nullptr) {
                pin_unmap(pin_addr);
                llama_sysmem::pinned_file_add(-(int64_t) pin_len);
            }
        }
    } ple_res;

public:
    struct ple_state {
        bool enabled = false; // predict at all (prefetcher or verify)
        bool verify  = false; // LLAMA_PLE_PREFETCH_VERIFY=1: compare each prediction with the rows of set_input
        int  stats   = 0;     // LLAMA_PLE_PREFETCH_STATS=1: print the counters when the context is freed, 2: also per decode

        // the prediction for the current / the next ubatch (verify only), valid when non-empty
        std::vector<int32_t> pred_cur;
        std::vector<int32_t> pred_next;
        std::vector<int32_t> scratch;

        uint64_t n_predicted    = 0; // ubatches predicted
        uint64_t n_unpredicted  = 0; // ubatches the arch could not predict ahead
        uint64_t n_rows         = 0; // rows submitted
        uint64_t n_skipped_rows = 0; // rows not knowable ahead (-1)
        uint64_t n_checked      = 0; // ubatches verified
        uint64_t n_mismatch     = 0; // rows that differ from set_input
        int64_t  t_predict_us   = 0;

        // timing of set_inputs and page faults, with or without the prefetcher (stats only)
        int64_t  t_set_inputs_us = 0;
        uint64_t n_set_inputs    = 0;
        int64_t  t_decode_us     = 0;
        uint64_t n_decode        = 0;
        int64_t  majflt          = 0; // process major faults during decode(), the workers included
        int64_t  minflt          = 0;
    };

private:
    ple_state ple;

    void ple_prefetch_ahead(const llama_ubatch & ubatch, const llama_memory_context_i * mctx, bool first);
    void ple_prefetch_verify(const llama_ubatch & ubatch, const llama_memory_context_i * mctx);

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;
        // sequences whose raw logits are still copied although a sampler is set (llama_set_sampler_raw_logits): the sampler
        // stays in the graph (no new reserve) while a request that needs the host sampler runs
        std::set<llama_seq_id> raw;
        // sequences whose raw logits are copied too while their sampler stays in the graph (llama_set_sampler_keep_logits)
        std::set<llama_seq_id> keep;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // whether the reserve graph has a logits tensor, set in sched_reserve()
    // true until then, because the first output_reserve() runs before any graph is reserved
    bool graph_has_logits = true;

    // state of sched_copy_experts, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;
        std::vector<int32_t> counts; // tokens routed to each expert, empty if the ids were not read
        bool                 all = false; // used was set to every expert without reading the ids
        int64_t              sync_us = 0; // diagnostic: host time of the last routing read

        std::vector<std::pair<size_t, size_t>> uploads; // offset and size of the uploads of the current copy
        std::vector<ggml_backend_copy_region>   regions; // the copies of the current copy, when the backend batches them

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
            counts.clear();
            all = false;
        }
    };

    copy_experts_info copy_experts;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    // the graph allocator was re-planned for a small graph (a decode step or a verify window) since the worst-case pp
    // reservation: the next prompt reserves the pp graph again before its first ubatch (LLAMA_PP_RERESERVE=0: off)
    bool sched_plan_small = false;
    // the graph allocator holds the plan of a small host-KQ-mask graph of this many tokens (0: another plan; see reserve_small_max)
    uint32_t sched_plan_small_tokens = 0;
    int64_t n_pp_rereserve = 0;
    int32_t n_reallocs_diff = 0;

    // the MoE cache admits the most used experts at the end of a prompt (see moe_cache_seed_begin):
    // seed:      from the ids of this ubatch, read before its experts are copied
    // seed_prev: from the routing of an earlier ubatch of the prompt, the copies of this ubatch do not wait for its ids
    // record:    keep the routing outputs of this ubatch, for a later seed_prev
    bool moe_cache_seed         = false;
    bool moe_cache_seed_prev    = false;
    bool moe_cache_record       = false;
    bool moe_cache_seeded       = false; // the last ubatch seeded the MoE cache
    bool moe_cache_seed_enabled = true; // LLAMA_MOE_CACHE_SEED != 0
    bool moe_cache_seed_sync    = true;  // LLAMA_MOE_CACHE_SEED=2 (default): a prefetched ubatch seeds (earlier routing, else its own ids read first)
    bool moe_cache_seed_auto    = false; // LLAMA_MOE_CACHE_SEED unset, n_ubatch >= LLAMA_MOE_CACHE_SEED_AUTO: as LLAMA_MOE_CACHE_SEED=1
    std::unordered_set<const llm_graph_result *> devseed_zeroed; // graphs whose seed plans a disabled prepare zeroed since their build
    bool moe_cache_devseed_on   = false; // LLAMA_MOE_CACHE_DEVSEED: prefetched ubatches seed the cache on the device (see llama_moe_cache::devseed_plan)
    int64_t moe_cache_devseed_min = INT64_MAX; // beside the host seed (moe_cache_seed_sync), ubatches of this many tokens seed on the device
    bool moe_cache_devseed      = false; // the ubatch being computed seeds on the device
    bool moe_cache_devseed_graph = false; // the graph last built had its seed plans enabled
    // set once a lent prompt graph with outputs failed to allocate: such graphs keep their last layer at full width from then on (see
    // llm_graph_context::crop_full_last; LLAMA_MOE_LEND_FULL_LAST=0 never, 2 from the start)
    bool moe_full_last = false;
    bool moe_cache_seed_fits    = true;  // the graph of a prefetched ubatch that seeds from its own ids was reserved (see sched_reserve_seed)

    // device memory kept free for a model or context created after this one, per backend (llama_context_params::mem_reserve); only
    // while the context is created, 0 afterwards
    std::vector<size_t> mem_reserve;
    size_t sched_pf_headroom_def = SIZE_MAX; // the prefetch headroom of the scheduler without the reserve (SIZE_MAX: not raised)

    // value of moe_cache_seed when the current graph was allocated: the scheduler plans the copies of the experts with it
    bool moe_cache_seed_graph = false;
    bool moe_inplace_ok       = false; // this ubatch may read its experts in place (see moe_cache_seed_begin)
    int  moe_seed_early       = 0;     // after a per-request reset of the cache, seed before the end of the batch (see moe_request_begin)
    bool moe_inplace_graph    = false;
    int  moe_cache_hybrid_graph = 0; // llama_moe_cache::hybrid_mode of the previous graph

    // routing outputs of the last computed large ubatch (expert tensor, ids), valid until the next graph is built
    std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> moe_cache_ids;

    // the prompt the stored routing belongs to: its sequence and the position of its next token
    llama_seq_id moe_cache_seq     = -1;
    llama_pos    moe_cache_pos_next = -1;

    // layer-major prompts (LLAMA_MOE_LAYER_MAJOR=1): consecutive prompt ubatches of one decode() call run as one graph, layer by layer
    // (every ubatch through a layer before any goes on to the next), so each host expert is copied to the device once for all of them
    // while every ubatch keeps its own graph, kernels and order of operations (bit-identical to the ubatches run one after the other)
    bool lm_enabled = false;
    // (LLAMA_LM_CKPT_MERGE): ubatch breaks of the next decode() (batch indices), see llama_set_ubatch_breaks
    std::vector<int32_t> ubatch_breaks_next;
    bool lm_auto    = false;  // LLAMA_MOE_LAYER_MAJOR unset: on beside speculative decoding with a model
    int  lm_ub_max  = 16;     // ubatches per layer-major graph [LLAMA_MOE_LAYER_MAJOR_UB_MAX]
    int64_t lm_n_tail = 0;    // short last ubatches run as the last member of a group (LLAMA_LM_TAIL)
    bool lm_verbose = false;  // [LLAMA_MOE_LAYER_MAJOR_LOG]
    bool   lm_layer_inp_logged = false; // the first layer-major member that extracted layer inputs was logged
    size_t lm_layer_inp_rows   = 0;     // rows of layer inputs extracted from layer-major members
    bool lm_replay_logged = false; // the first replayed device seed is logged (see decode_layer_major)
    // beside the device seed of prompt ubatches (moe_cache_devseed_on): groups run [LLAMA_MOE_LAYER_MAJOR_DEVSEED, default 1] and the
    // group seeds from the routing of the ubatch before its last one, as with the host seed [LLAMA_MOE_LAYER_MAJOR_DEVSEED_SEED, default 1]
    bool lm_devseed      = true;
    bool lm_devseed_seed = true;
    std::vector<llm_graph_result_ptr>    lm_res;      // the graph of each ubatch of the group
    std::vector<ggml_backend_buffer_ptr> lm_inp_buf;  // unused (kept for the per-ubatch layout)
    ggml_backend_buffer_ptr              lm_arena;    // the inputs of the group's ubatches (host memory, set before the group is computed)
    std::vector<ggml_backend_buffer_ptr> lm_arena_old;
    size_t                               lm_arena_used = 0;
    std::vector<uint8_t>                 lm_meta;     // the merged graph
    int64_t lm_n_graphs = 0, lm_n_ubatches = 0, lm_n_fallback = 0, lm_n_seeded = 0, lm_n_split = 0, lm_n_released = 0, lm_n_shrunk = 0;
    // host copies of the routing kept for the next seed when the compute buffers are freed after a group (lm_ids_host_keep)
    // the smallest start position and size of a group the lend-first growth cap refused (later groups borrow at once)
    int64_t lm_lf_refused_pos = INT64_MAX, lm_lf_refused_n = INT64_MAX;
    ggml_context_ptr lm_ids_ctx;
    ggml_backend_buffer_ptr lm_ids_buf;
    void lm_ids_host_keep();
    // a graph of several ubatches is being allocated or computed: a weight copy on the regular path copies all experts (its split reads
    // the copy for every ubatch, the first product of the split routes only its own ubatch)
    bool lm_merged_active = false;
    // the fewest ubatches of a merged graph refused for its compute buffer growth (decode_layer_major)
    size_t lm_refused_min = SIZE_MAX;
    // per scheduler backend: the largest compute buffer a merged graph of decode_layer_major ran with (regrow rule)
    std::vector<size_t> lm_merged_peak;
    // the seed of the MoE cache at the end of a prompt, as the ubatch-by-ubatch order does it (moe_cache_seed_begin): the last large ubatch
    // seeds from the routing of the ubatch before it. In the merged graph that routing exists only once the graph reaches it, so the seed
    // of a layer is issued from the eval callback after that routing (or, for expert copies issued later, from the copy callback), on the
    // copy queue behind the expert copies it reads
    struct lm_seed_layer {
        ggml_tensor * ids;                      // the routing [n_expert_used, n_tokens] of the source ubatch
        const ggml_tensor * w_first;            // the expert tensor that keys the layer's seed counts
        std::vector<const ggml_tensor *> ws;    // the cached expert tensors of the layer
        bool done = false;
    };
    bool lm_seed_active = false;
    std::unordered_map<const ggml_tensor *, lm_seed_layer> lm_ask;                                // ask node -> layer
    std::unordered_map<const ggml_tensor *, std::pair<ggml_backend_t, const ggml_tensor *>> lm_staged; // expert tensor -> (copy backend, copy)
    std::unordered_set<const ggml_tensor *> lm_seeded;
    static bool lm_eval_cb(ggml_tensor * t, bool ask, void * user_data);
    // (LLAMA_MOE_LM_SEED_ASYNC, default on): the eval callback reads the routing without a device sync, into pinned memory behind an
    // event, and the seed of the layer is issued from the copy callback once the layer's expert copies are queued (lm_seed_resolve_due)
    struct lm_seed_pending {
        lm_seed_layer * L;
        int             ev;      // lm_async_ev
        size_t          off;     // bytes into lm_async_buf
        ggml_backend_t  be_ids;  // the backend that computes the routing
    };
    // llama_set_state_capture
    struct cap_region {
        ggml_tensor * t;     // a recurrent state tensor of the memory
        size_t        off;   // the row of the sequence's cell
        size_t        size;
        ggml_tensor * snap;  // its copy before the capture token
        ggml_tensor * view;  // the row as a tensor (view of t)
    };
    // up to LLAMA_STATE_CAPTURE_MAX captures per decode (e.g. the far and the near prompt checkpoint), each with its snapshots
    struct cap_slot {
        llama_seq_id seq     = -1;
        int32_t      idx     = -1;    // batch token index of the next / current decode
        bool         pending = false; // set for the next decode
        bool         done    = false;
        llama_pos    pos     = -1;    // position of the last token before idx
        uint32_t     cell    = 0;
        std::vector<cap_region> regions;
        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buf;
        ggml_backend_event_t    ev     = nullptr; // the compute stream right after the snapshot copies
        bool                    ev_set = false;
    };
    static constexpr int cap_max = 2;
    int cap_count() const;
    cap_slot cap[cap_max];
    int      cap_sel = -1;                     // the capture a LLAMA_STATE_SEQ_FLAGS_CAPTURED read takes (llama_state_seq_captured_at)
    bool cap_prepare(int s);                   // the regions of the sequence's cell and their snapshot tensors; false: unsupported
    size_t cap_reserve_bytes(ggml_backend_buffer_type_t * buft) const; // the snapshots fit counts (memory_breakdown)
    size_t cap_planned_bytes(ggml_backend_buffer_type_t * buft) const; // the snapshots captures will allocate (0: none planned)
    bool cap_starts(int s, const llama_ubatch & ub) const; // the ubatch starts at the batch row of capture s
    void cap_now(int s, llama_pos pos);        // copy the regions in stream order (state before the next ubatch)
    bool cap_any() const { for (const auto & c : cap) { if (c.idx >= 0 && !c.done) { return true; } } return false; }
    // (LLAMA_STATE_CAPTURE_SIDE=0 off): the read of a captured state's snapshots runs on a stream of its own, after an event
    // of the compute stream, so the output wait of the decode does not wait for it; the next capture copy waits for that read
    ggml_backend_t       cap_side    = nullptr;
    ggml_backend_event_t cap_side_ev = nullptr;   // the side stream's last read
    bool                 cap_side_pending = false;
    int                  cap_save = -1;           // the capture a CAPTURED save is reading (state_get_async)
    ggml_backend_t cap_backend() const;           // the compute backend of the snapshot device
    void cap_record(int s, ggml_backend_t be);
    bool lm_seed_async = false;
    ggml_backend_dev_t lm_async_dev = nullptr;
    std::vector<lm_seed_pending> lm_pending;
    std::vector<ggml_backend_event_ptr> lm_async_ev;
    ggml_backend_buffer_ptr lm_async_buf;
    size_t lm_async_used = 0;
    int    lm_async_nev  = 0;
    // (LLAMA_MTP_PROMPT_OVERLAP=0 off): when more ubatches follow a layer-major group in the same decode, an event after the
    // copies of the group's nextn rows to the host: an MTP drafter reads rows [0, rows_ev_n) while the rest of the decode computes
    ggml_backend_event_ptr rows_ev;
    ggml_backend_dev_t     rows_ev_dev = nullptr;
    int64_t                rows_ev_n   = 0;
    // the backend that queued the last decode's copies of nextn rows to the host (nullptr: none, or several backends)
    ggml_backend_t         nextn_host_be    = nullptr;
    bool                   nextn_host_multi = false;
    const ggml_cgraph * lm_copy_graph = nullptr; // the split graph of the last expert copy, lm_copy_seq counts its changes
    int64_t lm_copy_seq = 0;
    std::unordered_map<const ggml_tensor *, int64_t> lm_staged_seq; // expert tensor -> lm_copy_seq of its copy
    int64_t lm_n_async_wait_us = 0;
    int64_t lm_n_seed_unstaged = 0; // expert copies outside staging memory a layer-major seed skipped (LLAMA_MOE_LM_SEED_STAGED_ONLY)
    int64_t lm_n_async = 0;
    void lm_seed_resolve(lm_seed_pending & p);
    // pre = true: before the copies of a split (only the layers whose staged copies the split may overwrite); false: after them
    void lm_seed_resolve_due(bool pre);
    void lm_seed_resolve_all();
    // returns the number of ubatches it processed (0: none, the caller processes the current ubatch), < 0 on error (ret = decode() return)
    int decode_layer_major(llama_memory_context_i * mctx, int64_t n_outputs_all, int64_t n_tokens_all, int64_t & n_outputs_prev,
            int64_t & n_tokens_prev, bool & more, int & ret);

    void moe_cache_seed_begin(const llama_ubatch & ubatch, bool last_large, int64_t n_outputs_ub, bool early = false);
    void moe_cache_seed_end(const llama_ubatch & ubatch, int64_t n_outputs_ub);
    // the end of a prompt that did not seed: the lent slots that come back get the experts its last large ubatch used most
    // count: the same routing also goes to the RAM tier's refill and the self-learned profile, see
    // moe_cache_released_counts
    void moe_cache_refill_from_ids(bool count = false);
    void moe_cache_resident_record(const llm_graph_result * res); //
    void moe_cache_fill_from_ids(); //
    void moe_cache_autoprof_from_ids(); //
    bool moe_ap_counted = false;        // the recorded routing (moe_cache_ids) was counted
public:
    void moe_fill_idle(); // (llama_moe_host_fill_idle)
private:

    // ubatches of at least this many tokens prefetch the host weights on the scheduler copy queue, MoE experts whole [GGML_SCHED_PREFETCH_MIN_BATCH]
    // smaller ubatches (e.g. generation) copy only the routed experts or use the MoE cache, with no staging or event work
    int64_t prefetch_min_batch = 256;
    int64_t prefetch_dense_min_batch = 32; // dense host weights of smaller ubatches if n_ubatch >= prefetch_min_batch [GGML_SCHED_PREFETCH_DENSE_MIN: any n_ubatch]

    // tokens of the ubatch whose graph the scheduler is allocating, read by sched_prefetch_mode
    int64_t prefetch_n_tokens = 0;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // uploads of the mapped host weights from the model files through pinned memory [--weight-stream], nullptr if off
    // (declared after the backends and the scheduler: freed before them)
    std::unique_ptr<llama_weight_store> wstore;

    // host expert ranges registered with the disk prefetch engine (ggml-moe-prefetch.h), see disk_prefetch_init; each entry
    // unregisters its range when it goes (also when the constructor throws after the registration)
    struct disk_range_ref {
        const void * base = nullptr;
        explicit disk_range_ref(const void * p) : base(p) {}
        disk_range_ref(const disk_range_ref &) = delete;
        disk_range_ref & operator=(const disk_range_ref &) = delete;
        disk_range_ref(disk_range_ref && o) noexcept : base(o.base) { o.base = nullptr; }
        ~disk_range_ref();
    };
    std::vector<disk_range_ref> disk_ranges;
    void disk_prefetch_init();

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    // (LLAMA_MOE_HOST_THREADS=N | auto, opt-in): the threads of the ubatches whose missing experts the host computes (CPU misses,
    // hybrid MoE cache) instead of the decode threads; auto = every logical CPU the process may use (affinity, cgroup quota); 0 = off
    int32_t           n_threads_host       = 0;
    ggml_threadpool_t threadpool_host      = nullptr; // own pool when the caller attached threadpools (sized to their threads)
    void (*threadpool_host_free)(ggml_threadpool_t) = nullptr;
    int64_t           compute_n_tokens     = 0; // tokens of the ubatch whose graph is computed next
    int32_t host_threads_init() const;
    bool    host_threads_for(int64_t n_tokens) const;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes
    std::vector<size_t>                     backend_lend_freed;   // of a measurement, the compute lending frees (see sched_reserve)

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // the graphs of small output ubatches (1..8 tokens: decode and speculative verify windows) are kept per size, so a
    // switch between a decode and a verify window re-allocates the kept graph instead of building it again (LLAMA_GRAPH_SMALL_CACHE=0:
    // one graph for all sizes, as before). gf_res_epoch counts every invalidation of the kept graphs (memory updates, reserves)
    // four kept graphs per size: after a recurrent rollback the state cells and the restore copy differ for a few decodes
    static constexpr uint32_t GF_SMALL_MAX  = 8;
    static constexpr uint32_t GF_SMALL_WAYS = 4;
    static constexpr uint32_t GF_SMALL_N    = (GF_SMALL_MAX + 1)*GF_SMALL_WAYS;
    std::array<llm_graph_result_ptr, GF_SMALL_N> gf_res_small;
    std::array<uint64_t, GF_SMALL_N>             gf_res_small_epoch{};
    std::array<bool, GF_SMALL_N>                 gf_res_small_built{};
    std::array<uint64_t, GF_SMALL_N>             gf_res_small_used{};
    std::array<uint64_t, GF_SMALL_N>             gf_res_small_nuse{}; // uses since the kept graph was built
    std::array<std::map<llama_seq_id, llama_sampler *>, GF_SMALL_N> gf_res_small_smpl{}; // the sampler chains a kept graph was built with
    // the MoE cache modes a kept graph was built with (seed, hybrid, device seed, in place): the context-wide *_graph values follow
    // the graph built last, which can be another size's
    struct gf_small_modes {
        bool seed    = false;
        int  hybrid  = 0;
        bool devseed = false;
        bool inplace = false;
    };
    std::array<gf_small_modes, GF_SMALL_N> gf_res_small_modes{};
    uint64_t gf_res_epoch = 1;
    // set by restore_pp_plan around its graph_reserve: the kept graphs stay valid when the reserve replaced no buffer
    bool reserve_keep_small = false;
    // debug (LLAMA_GRAPH_SMALL_WHY=N): every invalidation of the kept graphs is logged with its source line
    void gf_res_epoch_bump(int line);
    uint64_t n_small_swaps = 0;
    uint64_t n_small_uses  = 0;
    // the scheduler rewrites a graph it allocates (split input copies in its own context, tensor addresses): each kept graph's
    // tensors as built, restored before the graph is allocated again, so the allocation sees the graph a build would give
    struct gf_tensor_state {
        ggml_tensor * t;
        ggml_tensor * src[GGML_MAX_SRC];
        void * data;
        ggml_backend_buffer_t buffer;
        void * extra;
        int32_t flags;
        ggml_backend_t backend; // the backend the graph builder assigned (nullptr: the scheduler's choice)
    };
    std::array<std::vector<gf_tensor_state>, GF_SMALL_N> gf_res_small_state;
    uint64_t n_small_backends = 0; // builder assignments given back to kept graphs allocated again
    uint64_t n_small_check_assigned = 0; // LLAMA_GRAPH_SMALL_CHECK: builder assignments seen at kept graph allocations
    uint64_t n_small_check_lost     = 0; //   of them not held by the scheduler at that allocation
    // the scheduler's split and allocation of each kept graph (LLAMA_SCHED_SIZE_CACHE, on by default): a swap-in restores it
    // instead of splitting and allocating the graph again, so the tensors keep their addresses and the CUDA graph is replayed
    std::array<ggml_backend_sched_plan_t, GF_SMALL_N> gf_res_small_plan{};
    uint64_t n_plan_restores = 0;
    uint64_t n_plan_saves    = 0;
    uint64_t n_host_fallbacks = 0; // graphs built again with the operations on host weights on the host
    uint64_t n_plan_invalid  = 0;
    void small_plan_free(uint32_t i);
    size_t release_device_compute(); // frees the device compute buffers (an MTP drafter beside its target's prompt group)

    // (common's tiny-cache n_ubatch rule, LLAMA_MOE_TINY_RELEASE set by common; LLAMA_MOE_PREFILL_UB_TINY_RELEASE=0 off): the
    // MoE cache banks are released while ubatches of more than tiny_rel_ub tokens run (their compute buffers grow in that memory) and
    // allocated and refilled before the next ubatch of at most tiny_rel_ub tokens, after this context's and its MTP drafters' device
    // compute buffers are freed
    int64_t tiny_rel_ub      = 0;     // > 0: on (the default n_ubatch the cache was sized for)
    int64_t tiny_rel_res_ub  = 0;     // > 0: sched_reserve reserves the graphs of at most this many tokens (also an MTP drafter of such a context)
    bool    tiny_rel_held    = false; // the banks are released by tiny_rel_enter
    bool    tiny_rel_routed  = false; // LLAMA_MOE_TINY_RELEASE_ROUTED=1: the slots come back with the experts the last large ubatch routed most
    bool place8_rel = false; // the release was set up by common's fit (LLAMA_MOE_PLACE8_UB), see tiny_rel_leave
    // (LLAMA_MOE_PLACE8_LEND, set by common's fit where it counted the MTP drafter's weights and compute buffers as lent):
    // a prompt ubatch that runs with the banks released borrows them (place8_lend_now); the drafter's restore frees this context's
    // compute buffers first (place8_free_compute)
    bool   place8_lend = false;
    void   place8_lend_now();
    size_t place8_free_compute();
    size_t place8_grow = 0;   // the compute buffer memory the fit gave the cache (LLAMA_MOE_PLACE8_MID, MiB), see decode_layer_major
    bool   place8_lean       = false; // small device (LLAMA_MOE_PLACE8_GROUPS): no short-prompt split, held-before refills of short prompts
    bool   place8_lean_group = false; //, LLAMA_MOE_PLACE8_GROUPS_LEAN=2: also held-before refills after a layer-major group
    // the prompt calls of one sequence that follow each other (a prompt and the server's separate pass of its last 4 tokens)
    // count as one prompt for the MoE cache's short-prompt rule (see decode, LLAMA_MOE_PROMPT_RUN)
    int64_t      prompt_run_n   = 0;
    llama_pos    prompt_run_end = -1;
    llama_seq_id prompt_run_seq = -1;
    int64_t tiny_rel_n_enter = 0;
    int64_t tiny_rel_n_leave = 0;
    double  tiny_rel_ms      = 0.0;   // host time of the restores (frees, trims, bank allocation, refill)
    size_t  pool_mark_trim_dev(bool trim); // mark / trim the CUDA pools of this context's backends (ggml_backend_pool_mark_trim)
    size_t  graph_evict_since_dev(int64_t t_us); // the backends' CUDA graphs used since t_us are evicted
    int64_t tiny_rel_t_mark = 0;            // when tiny_rel_enter marked the pools
    void    tiny_rel_enter(const char * why);
    void    tiny_rel_leave();

    // (decode extension, llama_moe_cache::ext_grow; LLAMA_MOE_DECODE_EXT=0 off): a ubatch of more than ext_ub() tokens runs
    // with the cache at its fitted slots (ext_park); the first verify window after it (or a single-token step without a drafter) frees
    // this context's and its MTP drafters' device compute buffers and grows the cache into the device memory then free, less
    // LLAMA_MOE_DECODE_EXT_KEEP_MIB (384) (ext_unpark)
    bool    ext_grown   = false;
    int64_t ext_n_grow  = 0;
    int64_t ext_n_park  = 0;
    double  ext_ms      = 0.0;
    size_t  ext_max     = 0;
    int64_t ext_ub() const;
    void    ext_park(const char * why);
    void    ext_unpark();

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;
    // unmasked next-n rows (MTP) in their own host buffer, sized by the batches that copy them to the host instead of n_batch
    // (a batch whose rows go to the device sink, LLAMA_MTP_DEV_H, needs none); LLAMA_NEXTN_LAZY=0: in the output buffer at n_batch
    ggml_backend_buffer_ptr buf_nextn;
    size_t  nextn_cap_rows  = 0;
    int64_t nextn_need_rows = 0;   // rows the current decode copies to the host (set before output_reserve)
    int64_t nextn_decode_rows = 0; // tokens of the current decode (a device-sink batch that falls back to the host needs them all)
    bool nextn_store_ensure(size_t rows);
    // a growth of the next-n store for a prompt batch is allocated on a thread while the prompt's graphs start
    // (LLAMA_NEXTN_ASYNC); nextn_join() installs it before the rows are first written or read
    std::thread             nextn_thr;
    ggml_backend_buffer_t   nextn_new      = nullptr;
    size_t                  nextn_new_rows = 0;
    bool                    nextn_pending  = false;
    bool                    nextn_async_ok = false; // set by decode() for the output_reserve of a prompt batch
    // the pending growth is started by nextn_launch() once a layer-major group is allocated, else made in nextn_join()
    bool                    nextn_started  = false;
    ggml_backend_buffer_type_t nextn_buft  = nullptr;
    size_t                  nextn_bytes    = 0;
    int64_t                 nextn_t_launch = 0;
    std::atomic<int64_t>    nextn_t_done   { 0 };
    void nextn_launch();
    void nextn_join();
    // (llama_set_nextn_hold): a reader of the next-n rows outside this context (an MTP drafter's prompt pass on a server worker)
    // is waited for, once, before this context next writes, reorders or frees those rows or frees the drafter's compute buffers
    void (*nextn_hold_fn)(void *) = nullptr;
    void * nextn_hold_ud = nullptr;
    void nextn_hold_wait() {
        if (nextn_hold_fn != nullptr) {
            auto * fn = nextn_hold_fn;
            nextn_hold_fn = nullptr;
            fn(nextn_hold_ud);
        }
    }
    // (llama_set_nextn_hold_free): a drafter pass outside this context that only the frees of the drafter's buffers wait for
    void (*nextn_hold_free_fn)(void *) = nullptr;
    void * nextn_hold_free_ud = nullptr;
    // before this context frees an MTP drafter's compute buffers or device weights: both holds
    void nextn_hold_wait_free() {
        nextn_hold_wait();
        if (nextn_hold_free_fn != nullptr) {
            auto * fn = nextn_hold_free_fn;
            nextn_hold_free_fn = nullptr;
            const int64_t t0 = ggml_time_us();
            fn(nextn_hold_free_ud);
            nextn_hold_free_log((ggml_time_us() - t0)/1e3);
        }
    }
    void nextn_hold_free_log(double ms);
public:
    void set_nextn_hold(void (*fn)(void *), void * ud) { nextn_hold_fn = fn; nextn_hold_ud = ud; }
    void set_nextn_hold_free(void (*fn)(void *), void * ud) { nextn_hold_free_fn = fn; nextn_hold_free_ud = ud; }
private:

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    // asynchronous state saves: per async-capable backend, a pinned staging buffer (reused, grown on demand) that the
    // device rows are read into in stream order, and an event recorded after the reads
    struct state_async_backend {
        ggml_backend_t          backend = nullptr;
        ggml_backend_dev_t      dev     = nullptr;
        ggml_backend_event_t    event   = nullptr;
        ggml_backend_buffer_ptr staging;
        // more pinned staging past `staging` (only added to, freed with the context), filled first fit after it
        std::vector<ggml_backend_buffer_ptr> ext;
    };
    std::vector<state_async_backend> state_async;
    bool state_async_init = false;

    // synchronous state saves read the device rows through two small pinned halves per backend (LLAMA_STATE_BOUNCE_MIB each,
    // 16): the next half's copies run while the previous half is copied to the destination (see state_get_bounced)
    struct state_bounce_backend {
        ggml_backend_buffer_ptr buf;
        ggml_backend_event_t    ev[2] = { nullptr, nullptr };
    };
    std::vector<state_bounce_backend> state_bounce;
    bool state_get_bounced(const std::vector<llama_state_write_info> & winfos);

    // waits for the events, then copies staging -> destination; joined by state_async_wait()
    std::thread state_async_worker;
    std::vector<ggml_backend_event_t> state_async_evs;        // the read events of the save in flight
    std::atomic<bool>                 state_async_reads_done { true };

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // true after the prism.hadamard coverage check passes on a reserved graph
    bool hadamard_verified = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
