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

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <cstdint>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;

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

    void synchronize();

    // synchronize() unless nothing was queued since the last one (see sync_pending): the output getters call this, so that
    // reading several outputs of one decode costs one stream synchronization instead of one per getter (LLAMA_SYNC_SKIP=0: always)
    void synchronize_outputs();

    const llama_model   & get_model()   const;
    bool moe_cache_active() const { return moe_cache != nullptr; } // k11-decode
    const llama_moe_cache * get_moe_cache() const { return moe_cache.get(); } // k13
    bool moe_freeze_small(); // k13: see llama_moe_freeze_small
    bool moe_pin_small();    // k13: see llama_moe_pin_small
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
    bool restore_pp_plan(); // k11-decode (see the definition)
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
    void state_get_async(const std::vector<llama_state_write_info> & winfos);

    // wait until the last asynchronous state save has reached its destination
    void state_async_wait();

    // set when backend work is queued (graph_compute, async state copies), cleared by synchronize()
    bool sync_pending = true;

    // allocate the staging buffers of asynchronous saves with these flags up front (see llama_state_seq_reserve_async)
    size_t state_async_reserve(llama_state_seq_flags flags);

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
    // k11 (LLAMA_LM_CKPT_MERGE)
    void set_ubatch_breaks(const int32_t * breaks, int32_t n);
    bool set_state_capture(llama_seq_id seq_id, int32_t i_batch);
    bool state_capture_supported() const;
    bool state_captured(llama_seq_id seq_id, llama_pos * pos_max) const;
    bool layer_major_on() const;

    uint32_t graph_max_nodes(uint32_t n_tokens) const;

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

    // the host seed of a prompt does not fit: large prompt ubatches seed the MoE cache on the device instead (see the definition)
    void moe_cache_devseed_fallback();

    // reserve the graph of a prefetched ubatch that seeds the MoE cache from its own ids if the devices have room for it (staging_pp:
    // staging memory of the pp graph per backend), returns false if it does not fit
    bool sched_reserve_seed(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, const std::vector<size_t> & staging_pp);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

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
    // k11-decode: the graph allocator was re-planned for a small graph (a decode step or a verify window) since the worst-case pp
    // reservation: the next prompt reserves the pp graph again before its first ubatch (LLAMA_PP_RERESERVE=0: off)
    bool sched_plan_small = false;
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
    bool moe_cache_devseed_on   = false; // LLAMA_MOE_CACHE_DEVSEED: prefetched ubatches seed the cache on the device (see llama_moe_cache::devseed_plan)
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
    // k11 (LLAMA_LM_CKPT_MERGE): ubatch breaks of the next decode() (batch indices), see llama_set_ubatch_breaks
    std::vector<int32_t> ubatch_breaks_next;
    bool lm_auto    = false;  // LLAMA_MOE_LAYER_MAJOR unset: on beside speculative decoding with a model
    int  lm_ub_max  = 16;     // ubatches per layer-major graph [LLAMA_MOE_LAYER_MAJOR_UB_MAX]
    bool lm_verbose = false;  // [LLAMA_MOE_LAYER_MAJOR_LOG]
    bool lm_replay_logged = false; // k11: the first replayed device seed is logged (see decode_layer_major)
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
    // k13-small: host copies of the routing kept for the next seed when the compute buffers are freed after a group (lm_ids_host_keep)
    ggml_context_ptr lm_ids_ctx;
    ggml_backend_buffer_ptr lm_ids_buf;
    void lm_ids_host_keep();
    // a graph of several ubatches is being allocated or computed: a weight copy on the regular path copies all experts (its split reads
    // the copy for every ubatch, the first product of the split routes only its own ubatch)
    bool lm_merged_active = false;
    // the fewest ubatches of a merged graph refused for its compute buffer growth (decode_layer_major)
    size_t lm_refused_min = SIZE_MAX;
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
    // k15-pp (LLAMA_MOE_LM_SEED_ASYNC, default on): the eval callback reads the routing without a device sync, into pinned memory behind an
    // event, and the seed of the layer is issued from the copy callback once the layer's expert copies are queued (lm_seed_resolve_due)
    struct lm_seed_pending {
        lm_seed_layer * L;
        int             ev;      // lm_async_ev
        size_t          off;     // bytes into lm_async_buf
        ggml_backend_t  be_ids;  // the backend that computes the routing
    };
    // k15-pp: llama_set_state_capture
    struct cap_region {
        ggml_tensor * t;     // a recurrent state tensor of the memory
        size_t        off;   // the row of the sequence's cell
        size_t        size;
        ggml_tensor * snap;  // its copy before the capture token
        ggml_tensor * view;  // the row as a tensor (view of t)
    };
    llama_seq_id cap_seq     = -1;
    int32_t      cap_idx     = -1;    // batch token index of the next / current decode
    bool         cap_pending = false; // set for the next decode
    bool         cap_done    = false;
    llama_pos    cap_pos     = -1;    // position of the last token before cap_idx
    uint32_t     cap_cell    = 0;
    std::vector<cap_region> cap_regions;
    ggml_context_ptr        cap_ctx;
    ggml_backend_buffer_ptr cap_buf;
    bool cap_prepare();             // the regions of the sequence's cell and their snapshot tensors; false: unsupported
    void cap_now(llama_pos pos);    // synchronize, then copy the regions (state before the next ubatch)
    bool lm_seed_async = false;
    ggml_backend_dev_t lm_async_dev = nullptr;
    std::vector<lm_seed_pending> lm_pending;
    std::vector<ggml_backend_event_ptr> lm_async_ev;
    ggml_backend_buffer_ptr lm_async_buf;
    size_t lm_async_used = 0;
    int    lm_async_nev  = 0;
    const ggml_cgraph * lm_copy_graph = nullptr; // the split graph of the last expert copy, lm_copy_seq counts its changes
    int64_t lm_copy_seq = 0;
    std::unordered_map<const ggml_tensor *, int64_t> lm_staged_seq; // expert tensor -> lm_copy_seq of its copy
    int64_t lm_n_async_wait_us = 0;
    int64_t lm_n_async = 0;
    void lm_seed_resolve(lm_seed_pending & p);
    // pre = true: before the copies of a split (only the layers whose staged copies the split may overwrite); false: after them
    void lm_seed_resolve_due(bool pre);
    void lm_seed_resolve_all();
    // returns the number of ubatches it processed (0: none, the caller processes the current ubatch), < 0 on error (ret = decode() return)
    int decode_layer_major(llama_memory_context_i * mctx, int64_t n_outputs_all, int64_t n_tokens_all, int64_t & n_outputs_prev,
            int64_t & n_tokens_prev, bool & more, int & ret);

    void moe_cache_seed_begin(const llama_ubatch & ubatch, bool last_large, int64_t n_outputs_ub);
    void moe_cache_seed_end(const llama_ubatch & ubatch, int64_t n_outputs_ub);
    // the end of a prompt that did not seed: the lent slots that come back get the experts its last large ubatch used most
    void moe_cache_refill_from_ids();

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

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    // k11 (LLAMA_MOE_HOST_THREADS=N | auto, opt-in): the threads of the ubatches whose missing experts the host computes (CPU misses,
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

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // k11-decode: the graphs of small output ubatches (1..8 tokens: decode and speculative verify windows) are kept per size, so a
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
    uint64_t gf_res_epoch = 1;
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
    };
    std::array<std::vector<gf_tensor_state>, GF_SMALL_N> gf_res_small_state;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    // asynchronous state saves: per async-capable backend, a pinned staging buffer (reused, grown on demand) that the
    // device rows are read into in stream order, and an event recorded after the reads
    struct state_async_backend {
        ggml_backend_t          backend = nullptr;
        ggml_backend_dev_t      dev     = nullptr;
        ggml_backend_event_t    event   = nullptr;
        ggml_backend_buffer_ptr staging;
    };
    std::vector<state_async_backend> state_async;
    bool state_async_init = false;

    // waits for the events, then copies staging -> destination; joined by state_async_wait()
    std::thread state_async_worker;

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
