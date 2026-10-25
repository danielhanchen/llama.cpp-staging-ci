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

    size_t total() const {
        return model + context + compute;
    }
};

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
// k15-ttft: whether llama_set_state_capture can take a capture in this context (a recurrent memory, beside a non-SWA KV cache if hybrid)
LLAMA_API bool llama_state_capture_supported(const struct llama_context * ctx);
// layer-major prompt groups can run in this context (a MoE cache with layer-major prompts on)
LLAMA_API bool llama_layer_major_on(const struct llama_context * ctx);
// k15-pp: a batch of n_before + n_after prompt tokens of one sequence with a ubatch break (llama_set_ubatch_breaks) after n_before would
// run the first ubatch after the break in the same layer-major group as the one before it
LLAMA_API bool llama_layer_major_spans(const struct llama_context * ctx, int32_t n_before, int32_t n_after);
// k11: wait for the asynchronous state save in flight (LLAMA_STATE_SEQ_FLAGS_ASYNC), before its destination is read or freed
LLAMA_API void llama_state_seq_async_wait(struct llama_context * ctx);

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
// k15-lfm2: turns the CPU-miss rule off on a host that cannot compute experts faster than it uploads them (few threads, host time per
// expert not well below the upload time) where the cache holds most host experts: the misses are fetched, every expert runs on the device
// from a slot (verify rows are single token bytes again). Returns true if it turned the rule off. LLAMA_MOE_CPU_MISS_DROP=0: never
LLAMA_API bool llama_moe_cpu_miss_drop_slow(struct llama_context * ctx);

// k13: 0 = no MoE cache or its host never computes an expert, 1 = hybrid path with a fixed share (small caches), 2 = CPU-miss rule
LLAMA_API int32_t llama_moe_host_mode(const struct llama_context * ctx);

// k11-decode: restore the worst-case prompt plan of the graph allocator if a small graph (a verify window) re-planned it; a server
// calls it when its slots go idle, so the next prompt does not pay for it (no-op otherwise; LLAMA_PP_RERESERVE=0: off)
LLAMA_API bool llama_prepare_prompt_plan(struct llama_context * ctx);

// k11-decode: what a speculative rollback of the architecture (general.architecture) keeps: 0 nothing but KV cells, 1 a small
// recurrent state (LFM2 short conv), 2 a large recurrent / compressor state (Mamba, GDN, DeepSeek-V4: a snapshot per drafted
// token is tens of MiB), -1 unknown
LLAMA_API int32_t llama_arch_rollback_state(const char * arch_name);

// k11-decode: every layer weight of the context's model is in device memory and no MoE cache runs: the placement where batch
// invariant verify gives the bytes of single token decodes
LLAMA_API bool llama_verify_exact_placement(const struct llama_context * ctx);

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

LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

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
