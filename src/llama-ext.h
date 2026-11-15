// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers
    size_t lend_freed = 0; // k15-fn: of a measurement (no_alloc), the staging arena the MoE cache lends for, not in total() (see sched_reserve)
    size_t kv_lazy    = 0; // k17-small: the bytes of context a lazy KV cache leaves unbacked (a measurement: at load), in total()

    size_t total() const {
        return model + context + compute;
    }
};

// k19-vramlend: lazy KV mode (LLAMA_KV_LAZY): 0 = off (=0, the kill switch), 1 = on (=1: every eligible device KV cache backs its cells
// lazily, also without a MoE cache to lend to), 2 = auto (unset: a fit measures the KV lazily and gives the MoE cache the bytes the first
// cells leave unbacked; the KV cache is then lazy only where common set LLAMA_KV_LAZY_GRANTED=1 for that fit, see common_init_result)
// (k19-vramlend-optin: unset = 0, the automatic mode is LLAMA_KV_LAZY=auto)
int  llama_kv_lazy_mode();
// whether a device KV cache allocated now backs its cells lazily (LLAMA_KV_LAZY_SUSPEND=1 always backs it whole)
bool llama_kv_lazy_alloc_on();
// whether a measurement (no_alloc) reports the KV bytes a lazy KV cache leaves unbacked at load (mb.kv_lazy)
bool llama_kv_lazy_measure_on();

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// k15-fn: with a sampler set for seq_id (llama_set_sampler), raw = true still copies the raw logits of its rows to the host (the host
// sampler of a request that the device sampler cannot serve), raw = false copies only what the sampler outputs. No new graph reserve
LLAMA_API void llama_set_sampler_raw_logits(struct llama_context * ctx, llama_seq_id seq_id, bool raw);
// k19-gramdev: keep = true copies the raw logits of seq_id's rows to the host as well, while its sampler stays in the graph (a device
// top-k whose host chain may need the full row of one position: grammar, reasoning budget); read them with llama_get_raw_logits_ith
LLAMA_API void llama_set_sampler_keep_logits(struct llama_context * ctx, llama_seq_id seq_id, bool keep);
// k19-gramdev: the raw logits of output i (llama_get_logits_ith returns the sampler's candidates when it output some); only valid for
// rows whose raw logits were copied (no sampler, llama_set_sampler_raw_logits or llama_set_sampler_keep_logits)
LLAMA_API float * llama_get_raw_logits_ith(struct llama_context * ctx, int32_t i);
// k15-fn: reserve the graphs now if a change (e.g. llama_set_sampler) asked for it, instead of at the next decode
LLAMA_API void llama_sched_reserve_now(struct llama_context * ctx);
// k19-fnttft: once the whole load is over (drafter, warm-ups), read back the pages of a PLE table read in part at the load that the page
// cache gave up since, up to the reclaimable RAM less the part read's reserve; page cache only, exact. Returns the bytes read (0: nothing
// to do or not applicable). LLAMA_PLE_TOPUP=0: off
LLAMA_API size_t llama_ple_topup(struct llama_context * ctx);
// k19-l4fill: the server is idle (no request runs or waits): the exclusive RAM tier of the MoE cache reads the experts the last prompts
// routed most that neither tier holds into the slots it freed, in the background until the next batch. Placement only. LLAMA_MOE_HOST_FILL=0: off
LLAMA_API void llama_moe_host_fill_idle(struct llama_context * ctx);
// k15-fn: replace the backend sampler of seq_id, which has one, without reserving the graphs again: the caller reserved them with a
// sampler whose graph is at least as large (e.g. the masked greedy of an ignore_eos request, swapped for the plain one). False: nothing
// was changed (no sampler set for seq_id, or this one cannot run on the backend)
LLAMA_API bool llama_set_sampler_swap(struct llama_context * ctx, llama_seq_id seq_id, struct llama_sampler * sampler);
// k15-fn: greedy with the given tokens excluded (their logit is -inf, as a -inf logit bias), the lowest id on a tie; on the backend
// it masks the row in the samplers' padded copy of the logits
LLAMA_API struct llama_sampler * llama_sampler_init_greedy_masked(int32_t n_tokens, const llama_token * tokens);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Marks the next decodes as speculative verify batches (the caller clears it again). With LLAMA_MTP_EXACT=1, every
// verify ubatch of more than one token is computed batch invariant: each row gets the bytes a decode of that token alone
// would give (the same kernels, splits and reduction order per row), so greedy speculative decoding gives exactly the
// output of plain decoding. Prompt batches and single token decodes are not affected.
LLAMA_API void llama_set_verify(struct llama_context * ctx, bool value);

// k11 (LLAMA_LM_CKPT_MERGE): the next llama_decode() of a single sequence batch ends ubatches at these batch indices (each segment is split as
// a batch of its own would be: the same ubatches as separate decode() calls, in one layer-major group); cleared by that call
LLAMA_API void llama_set_ubatch_breaks(struct llama_context * ctx, const int32_t * breaks, int32_t n);
// k15-pp: the next llama_decode() keeps the partial state of seq_id (the recurrent rows) as it is before batch token i_batch, for a
// checkpoint there that the decode does not stop at (the merged prompt checkpoint of llama-server, LLAMA_LM_CKPT_MERGE). After the
// decode, llama_state_seq_captured() says whether it was taken, and llama_state_seq_get_size_ext / get_data_ext with
// LLAMA_STATE_SEQ_FLAGS_CAPTURED serialise it as the state of the sequence at that position (PARTIAL_ONLY only, no SWA KV cache).
// Unsupported memories (SWA attention, several cells per sequence) take nothing. Cleared by the decode after the next one
#define LLAMA_STATE_SEQ_FLAGS_CAPTURED 8
LLAMA_API bool llama_set_state_capture(struct llama_context * ctx, llama_seq_id seq_id, int32_t i_batch);
LLAMA_API bool llama_state_seq_captured(const struct llama_context * ctx, llama_seq_id seq_id, llama_pos * pos_max);
// k15-pp2: up to two captures per decode (llama_set_state_capture once per index; LLAMA_STATE_CAPTURE_MAX=1: one). The capture at
// batch index i_batch taken by the last decode; selects it for the next LLAMA_STATE_SEQ_FLAGS_CAPTURED read (default: the lowest index)
LLAMA_API bool llama_state_seq_captured_at(struct llama_context * ctx, llama_seq_id seq_id, int32_t i_batch, llama_pos * pos_max);
// k15-ttft: whether llama_set_state_capture can take a capture in this context (a recurrent memory, beside a non-SWA KV cache if hybrid)
LLAMA_API bool llama_state_capture_supported(const struct llama_context * ctx);
// layer-major prompt groups can run in this context (a MoE cache with layer-major prompts on)
LLAMA_API bool llama_layer_major_on(const struct llama_context * ctx);
// k15-pp: a batch of n_before + n_after prompt tokens of one sequence with a ubatch break (llama_set_ubatch_breaks) after n_before would
// run the first ubatch after the break in the same layer-major group as the one before it
LLAMA_API bool llama_layer_major_spans(const struct llama_context * ctx, int32_t n_before, int32_t n_after);
// k15-idlewait: with LLAMA_STATE_SEQ_FLAGS_ASYNC, take the asynchronous path only where the pinned staging buffers already hold the
// save (no new pinned memory); otherwise the save is synchronous
#define LLAMA_STATE_SEQ_FLAGS_ASYNC_NO_GROW 16
// k11: wait for the asynchronous state save in flight (LLAMA_STATE_SEQ_FLAGS_ASYNC), before its destination is read or freed
LLAMA_API void llama_state_seq_async_wait(struct llama_context * ctx);
// k19-host30k: pin now enough staging that a later asynchronous save of `bytes` (0: the largest save of one sequence) reads every row
// asynchronously, also with LLAMA_STATE_SEQ_FLAGS_ASYNC_NO_GROW. Added to the staging, kept with the context (LLAMA_STATE_ASYNC_EXT=0: no-op).
// Call it while the context is idle (e.g. after loading)
LLAMA_API void llama_state_seq_async_ext_reserve(struct llama_context * ctx, size_t bytes);

// LLAMA_MTP_EXACT=1 is set (read once)
LLAMA_API bool llama_exact_verify_enabled(void);

// k11-decode: this context computes its verify batches batch invariant as LLAMA_MTP_EXACT=1 does, without the env (the server's
// automatic n-gram drafting sets it, so drafting on by default leaves greedy output byte equal to plain decoding)
LLAMA_API void llama_set_verify_exact(struct llama_context * ctx, bool value);

// k13: a MoE cache with CPU misses (LLAMA_MOE_CPU_MISS) freezes its small batches (up to 8 tokens): nothing is fetched there, the host
// computes every miss, no timing trials; the split of a token is then the same in a verify window as in its single token decode.
// Returns true if frozen (false without such a cache or before its calibration)
LLAMA_API bool llama_moe_freeze_small(struct llama_context * ctx);

// k13: like llama_moe_freeze_small for the decisions only: small batches always take the host path for their misses (no timing trials,
// no miss-rate windows), but still fetch their share and admit, so the cache follows the decode. Not exact under drafts. True if pinned
LLAMA_API bool llama_moe_pin_small(struct llama_context * ctx);
// k15-lfm2: lifts (on = false) or restores (on = true) the pin of llama_moe_pin_small, e.g. for a request that cannot draft; a cache
// the pin did not freeze is left as it is. Returns the pinned state
LLAMA_API bool llama_moe_pin_small_set(struct llama_context * ctx, bool on);
// k15-lfm2: a new request starts (the CPU-miss rule restarts from the launch's decisions with LLAMA_MOE_CPU_MISS_PER_REQUEST=1)
LLAMA_API void llama_moe_request_begin(struct llama_context * ctx);
// k15-idlewait: the idle work of llama_prepare_prompt_plan (the MoE cache's pending re-calibration of the CPU-miss host timings) polls
// fn(data) and stops, staying pending, when it returns true (a request arrived); NULL: it runs to the end
LLAMA_API void llama_set_idle_abort(struct llama_context * ctx, bool (*fn)(void *), void * data);
// k15-lfm2: turns the CPU-miss rule off on a host that cannot compute experts faster than it uploads them (few threads, host time per
// expert not well below the upload time) where the cache holds most host experts: the misses are fetched, every expert runs on the device
// from a slot (verify rows are single token bytes again). Returns true if it turned the rule off. LLAMA_MOE_CPU_MISS_DROP=0: never
LLAMA_API bool llama_moe_cpu_miss_drop_slow(struct llama_context * ctx);

// k13: 0 = no MoE cache or its host never computes an expert, 1 = hybrid path with a fixed share (small caches), 2 = CPU-miss rule
LLAMA_API int32_t llama_moe_host_mode(const struct llama_context * ctx);
// k19-tokcost: pause the MoE cache's CPU-miss rule statistics (miss-rate windows) while the caller decodes dummy batches (no-op without a cache)
LLAMA_API void llama_moe_cache_stats_pause(struct llama_context * ctx, bool on);
// k19-prewexact: snapshot what a decode moves in the MoE cache before dummy batches (slots, replacement state, device counters, rule
// state; the rules stay quiet until the end) and take it back after them, so the next request sees the cache it would have seen without
// them. begin: false without a cache or if refused; end: false if nothing was taken back (refused: logged, the state stays as moved)
LLAMA_API bool llama_moe_cache_prewarm_begin(struct llama_context * ctx);
LLAMA_API bool llama_moe_cache_prewarm_end(struct llama_context * ctx);
// k19-prewexact diagnostic (LLAMA_MOE_CACHE_FP_LOG=1): logs hashes of the MoE cache state (no-op without a cache or without the env)
LLAMA_API void llama_moe_cache_fingerprint(struct llama_context * ctx, const char * when);

// k19-p2qsa: while paused (calls nest), QSA graphs score their selection even where every pool fits the top_k (LLAMA_QSA_ALLFIT): a server
// prewarms its decode / verify graphs in the scored form, the one requests past 512 pools need, as before the all-fit graphs (process wide)
LLAMA_API void llama_qsa_allfit_pause(bool on);
// k19-int3fix: the context's memory selects QSA pools (Flash-Next indexer with a pool top_k): the all-fit graph form exists for it
LLAMA_API bool llama_qsa_allfit_capable(const struct llama_context * ctx);

// k15-ttft: a MoE cache runs (experts stream from host memory into device slots)
LLAMA_API bool llama_moe_cache_active(const struct llama_context * ctx);

// k11-decode: restore the worst-case prompt plan of the graph allocator if a small graph (a verify window) re-planned it; a server
// calls it when its slots go idle, so the next prompt does not pay for it (no-op otherwise; LLAMA_PP_RERESERVE=0: off)
LLAMA_API bool llama_prepare_prompt_plan(struct llama_context * ctx);

// k19-rollback: the recurrent state of a sequence copied to host memory and back, on the streams that compute it (no host wait, no
// per-token rows on the device): llama_rs_snap_save before a drafted verify window, llama_rs_snap_restore when it is not fully accepted
// (the state and the sequence's recurrent position go back to the copy; remove the positions after it with llama_memory_seq_rm, then
// decode the accepted tokens again). Supported for a context with a recurrent memory without rollback rows (n_rs_seq 0) and one
// sequence, not under -sm tensor. False when unsupported or when the restore does not match the copy
LLAMA_API bool llama_rs_snap_supported(const struct llama_context * ctx);
// k19-rollback: layer weights of the model are in host memory while others are on a device (host experts, a MoE cache, layers on the CPU)
LLAMA_API bool llama_model_weights_spill(const struct llama_model * model);
LLAMA_API bool llama_rs_snap_save     (struct llama_context * ctx, llama_seq_id seq_id);
LLAMA_API bool llama_rs_snap_restore  (struct llama_context * ctx, llama_seq_id seq_id);

// k11-decode: what a speculative rollback of the architecture (general.architecture) keeps: 0 nothing but KV cells, 1 a small
// recurrent state (LFM2 short conv), 2 a large recurrent / compressor state (Mamba, GDN, DeepSeek-V4: a snapshot per drafted
// token is tens of MiB), -1 unknown
LLAMA_API int32_t llama_arch_rollback_state(const char * arch_name);

// k11-decode: every layer weight of the context's model is in device memory and no MoE cache runs: the placement where batch
// invariant verify gives the bytes of single token decodes
LLAMA_API bool llama_verify_exact_placement(const struct llama_context * ctx);

// k17-strict: the most tokens a batch invariant verify ubatch may hold so that every MoE cache layer still reads its experts from the
// device slots (n_tokens * n_expert_used <= slots, as a single token decode does); -1 when no cap applies (no MoE cache, verify
// batches not computed batch invariant, or LLAMA_SPEC_EXACT_DEVICE_CAP=0). A larger window falls back to the layer's host expert
// tensors and runs on the CPU, whose kernels are not those of the single token decode on the device (a numeric format change)
LLAMA_API int32_t llama_verify_device_max_tokens(const struct llama_context * ctx);
// k19-df2pp: the tokens whose experts the MoE cache's device slots hold at once (llama_moe_cache::n_tokens_cached, at most 32), -1 without a cache
LLAMA_API int32_t llama_moe_cache_tokens(const struct llama_context * ctx);
// k19-df2cost: the cached share of the MoE cache's host expert bytes (0..1), -1 without a cache or before its slots are sized
LLAMA_API double llama_moe_cache_frac(const struct llama_context * ctx);
// k19-bigverify: a MoE cache that holds at least LLAMA_MOE_CACHE_WIDE_GRAPHS_FRAC (0.6; 0 = off, as before) of its host expert bytes
// keeps the verify graphs of a no-cache target: the server prewarms every verify width at load (server_spec_prewarm, as
// LLAMA_SPEC_PREWARM_CACHED=1) and device KV buckets are at least 16384 cells (k19-fullcost), so a request replays the captured widths
// instead of building and capturing each one. k19-widefix: only where its host experts are at least LLAMA_MOE_CACHE_WIDE_GRAPHS_MIN_MIB
// (32768; 0 = any size). False without a cache or before its slots are sized
LLAMA_API bool llama_moe_cache_wide_graphs(const struct llama_context * ctx);

// k15-ttft: a prompt run as one ubatch instead of [n-4] [4] differs from the split run only by IEEE reassociation (plans/k11.md
// "Equivalence rule"): every layer weight is computed on a CUDA device with no host-computed expert (llama_verify_exact_placement and
// llama_moe_host_mode == 0), and no F16 matmul weight (cuBLAS accumulates F16 products in F16 above 16 rows, MMF in F32 below).
// On such a placement both shapes use the same formats: q8_1 activations into int8 MMVQ / MMQ with F32 accumulation, TF32 for F32
// weights (MMF and cuBLAS SGEMM under CUBLAS_TF32_TENSOR_OP_MATH), BF16 in / F32 out, FlashAttention MMA F16 (at least 4 rows)
LLAMA_API bool llama_single_pass_equivalent(const struct llama_context * ctx);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// LLAMA_MTP_DEV_H: the MTP prompt handoff on the device (llama_mtp_dev_h in src/llama-context.h). attach: ctx_dft allocates a device
// sink of n_rows + 1 rows and ctx_tgt writes the h_nextn rows of single-sequence batches of LLAMA_MTP_DEV_H_MIN (64) .. n_rows tokens
// there instead of to the host (false: the host path stays). last_rows: rows ctx_tgt's last decode wrote (0 = the host buffer holds
// them). begin / end bracket a ctx_dft decode that reads its h rows from the sink (row i = batch entry i, row 0 the pending row), end
// copies row pending_row to row 0. get_row / set_row: blocking copies of one sink row.
LLAMA_API bool    llama_mtp_dev_h_attach   (struct llama_context * ctx_dft, struct llama_context * ctx_tgt, int32_t n_rows);
LLAMA_API int32_t llama_mtp_dev_h_last_rows(struct llama_context * ctx_tgt);
LLAMA_API bool    llama_mtp_dev_h_begin    (struct llama_context * ctx_dft);
LLAMA_API void    llama_mtp_dev_h_end      (struct llama_context * ctx_dft, int32_t pending_row);
LLAMA_API bool    llama_mtp_dev_h_get_row  (struct llama_context * ctx_dft, int32_t row, float * dst);
LLAMA_API bool    llama_mtp_dev_h_set_row  (struct llama_context * ctx_dft, int32_t row, const float * src);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
// k11: room for n_tokens entries, n_state_tokens of them with a state row, before they are added
LLAMA_API void llama_batch_ext_reserve(struct llama_batch_ext * batch, int32_t n_tokens, int32_t n_state_tokens);
// k17: frees the state rows a batch prefaulted at its creation (LLAMA_BATCH_STATE_PREFAULT); later state rows grow it again
LLAMA_API void llama_batch_ext_release_state(struct llama_batch_ext * batch);

LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// k15-pp3 (LLAMA_MTP_PROMPT_OVERLAP=0 off): the nextn rows [0, n_rows) of the last decode, on the host before that decode is done.
// When a layer-major group was followed by more ubatches in the same decode, waits for the group's rows only and returns how many rows
// are on the host (>= n_rows); returns 0 when it cannot (the caller then synchronizes as before). Unmasked nextn, one sequence only.
LLAMA_API int64_t llama_nextn_rows_wait(struct llama_context * ctx, int64_t n_rows);
// the nextn rows without a synchronization (unmasked: row i at i*n_embd); nullptr when outputs are to be reordered.
// Valid for rows [0, llama_nextn_rows_wait(...)) until the next decode
LLAMA_API float * llama_get_embeddings_nextn_early(struct llama_context * ctx);
// k19-host: fn(user_data) is called once, on the calling thread of a later decode / encode / output read, before the context writes,
// reorders or frees its next-n rows, or frees an MTP drafter's compute buffers (a reader of the rows outside the context, e.g. a drafter's
// prompt pass on another thread, is waited for there). nullptr clears it. Not thread-safe: set it from the thread that decodes
LLAMA_API void llama_set_nextn_hold(struct llama_context * ctx, void (*fn)(void *), void * user_data);
// k19-q4x262: as llama_set_nextn_hold, but fn(user_data) is called only before the context frees an MTP drafter's compute buffers or
// device weights (MoE cache banks coming back after a released prompt, the decode extension, a layer-major group's drafter release /
// lend), not before it writes its own next-n rows: a drafter pass on another thread beside a short decode of this context (whose rows
// are stashed) ends before the buffers it computes in go. nullptr clears it. Not thread-safe: set it from the thread that decodes
LLAMA_API void llama_set_nextn_hold_free(struct llama_context * ctx, void (*fn)(void *), void * user_data);

// kdp (LLAMA_MTP_DRAFT_LEND): maps the device weights of a drafter context's model back if a layer-major group of its target borrowed
// their memory (same addresses, copied back from pinned host memory); a no-op otherwise. false: no device memory for them
LLAMA_API bool llama_mtp_draft_weights_restore(struct llama_context * ctx_dft);
// kdp: true where the target's last released layer-major group was below LLAMA_MTP_DRAFT_LEND_FREE_MIB free (it lends, or would but
// for a drafter pass in flight): the server skips the drafter's pass over the early prompt batches only then
LLAMA_API bool llama_mtp_draft_lend_active(const struct llama_context * ctx_tgt);
// k19-np12c: makes the MTP drafter's pinned host copy now where the target's prompt groups are expected to borrow its weights (LLAMA_MTP_DRAFT_LEND_EAGER=0 off)
LLAMA_API bool llama_mtp_draft_lend_warm(struct llama_context * ctx_tgt);
// kdp: the drafter's weights are lent out now (a remap is due before its next pass)
LLAMA_API bool llama_mtp_draft_weights_evicted(const struct llama_context * ctx_dft);
// kdp: the outcome of a request whose drafter weights were remapped: ok resets the failure count, a failure (remap, or the drafter's pass
// after it) counts, and LLAMA_MTP_DRAFT_LEND_MAX_FAILS (3) consecutive ones turn the lend off for the process (one WARN line)
LLAMA_API void llama_mtp_draft_lend_note(struct llama_context * ctx_dft, bool ok);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// the backend of the device can map host memory for the device lookup of the MoE cache (moe_cache_size)
LLAMA_API bool llama_moe_cache_device_lookup_supported(ggml_backend_dev_t dev);
// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);
