// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// --fit-kv-resident: placement for a dense model that does not fit entirely in device memory
//   - whole layers: the first layers go to system memory with their KV cache, attention over that KV cache runs on the CPU
//   - KV-resident: every layer keeps its attention and its KV cache (at the requested context) in device memory, the FFN tensors
//     of the last layers go to system memory first
enum common_fit_kv_resident {
    COMMON_FIT_KV_RESIDENT_AUTO = 0, // KV-resident if the whole-layer placement would spill little KV cache per byte of weights (default)
    COMMON_FIT_KV_RESIDENT_ON   = 1, // KV-resident whenever the model does not fit
    COMMON_FIT_KV_RESIDENT_OFF  = 2, // whole layers only
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level,             // minimum log level to print during fitting, lower levels go to debug log
             common_fit_kv_resident   kv_resident = COMMON_FIT_KV_RESIDENT_AUTO); // placement for dense models that do not fit

// k11 (LLAMA_MOE_FIT_MARGIN_MIB=N; k17-models: 256 by default on devices with 10+ GiB, see common_fit_moe_margin): the margin the fit keeps free beside a MoE cache. The fit sizes the cache with the full
// margin (--fit-target, default 1024 MiB), so every decision that follows the cache size (n_ubatch, lending, the resident rule) is the same
// as before; then the cache gets the part of the default margin beyond N MiB. Off by default: the 8 GB tiers use that margin at run time
// (G4: 70 MiB free at the low point of Qwen3.6 UD-Q4_K_XL + MTP; N = 384 ran out of memory there). An explicit --fit-target is kept as
// given; LLAMA_MOE_FIT_MARGIN=0 keeps the full margin.
//   - auto = false: the user gave --fit-target (call before each fit)
//   - common_fit_moe_margin_gain: bytes of the margin the last fit leaves to its MoE cache (0 if the fit did not shrink a cache)
void    common_fit_set_moe_margin_auto(bool auto_margin);
int64_t common_fit_moe_margin_gain();

// k19-int3fix2: the server prewarms N verify CUDA graphs (sampler chains x verify widths, server_spec_prewarm) beside a MoE cache that
// holds at least LLAMA_MOE_CACHE_WIDE_GRAPHS_FRAC of the host experts; each keeps device memory no fit measure sees. With N > 0 (set
// before a fit with a draft model or MTP heads, 0 = none) the fit keeps N x LLAMA_FIT_PREWARM_GRAPH_MIB (default 40, 0 = off) more free
// on every device beside such a cache, taken from the cache
void    common_fit_set_prewarm_graphs(int n_graphs);

// k13-small: the fit kept every layer on a small device with the reduced margin (LLAMA_FIT_SMALL_MARGIN_MIB); set once per process
bool    common_fit_small_device();
// k19-place8: the last fit sized a small device's MoE cache for decode, prompts release its banks (LLAMA_MOE_PLACE8_UB set)
bool    common_fit_place8();
// k19-l4poly: the last fit gave a host-bound small device (every layer fit within the default margin beside a cache of at most
// LLAMA_FIT_HOSTBOUND_MAX_FRAC of the host experts) the small-device placement; implies common_fit_small_device()
bool    common_fit_hostbound();
double  common_fit_hostbound_max_frac();
// k19-placemid: the k19-place8 placement on a device of the regular fit (12 / 16 GB class): drafting and layer-major prompts as usual
bool    common_fit_place8_mid();

// k19-tierfix-fit: the run-time reserve of a small-device placement, in MiB added to its reduced margin and taken from its cache.
// The placement is measured at the reserve sizes of its buffers; what runs later allocates more in the margin: the CUDA pool of
// full-ubatch expert products, the CUDA graphs and streams of the decode and prompt paths, the layer-major groups (which merge only
// with 128 MiB of headroom left). G4 capped at 8 GB, Qwen3.8-Flash-Next UD-IQ4_XS, -ub 2048: margin 256 MiB, 277 MiB projected free,
// 110.6 MiB free at the first 8k group with the route tail skip and none with it off (CUDA graphs failed to instantiate in the chat
// decode, then a stream creation aborted). The cache keeps at least floor_mib (a cache group needs n_expert_used slots):
//   - common_fit_small_reserve_plan: min(reserve_mib, cache_mib - floor_mib), 0 when the cache is not above the floor
//   - common_fit_small_reserve_mib: the plan with LLAMA_FIT_SMALL_RESERVE_MIB (128, 0 = off: the margin of before) and
//     LLAMA_FIT_SMALL_RESERVE_MIN_CACHE_MIB (64)
int64_t common_fit_small_reserve_plan(int64_t cache_mib, int64_t reserve_mib, int64_t floor_mib);
int64_t common_fit_small_reserve_mib(int64_t cache_mib);

// k17-small (lazy KV): the bytes a fit added to a MoE cache of this final size for the KV cells not backed at load (0: none)
size_t  common_fit_kv_lazy_grant(size_t moe_cache_size);

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

// k19-fitplain: the default-margin step-down of a plain launch (no draft model, no MTP heads) of a dense model whose fit spills
// whole layers to the CPU at the 1024 MiB margin (see common_init_result). Pure helpers, unit-tested in test-fit-plain-spill:
//   - common_fit_margin_steps_mib: a comma list of MiB ("512,448,256,192") as bytes; empty tokens skipped, negatives as 0
//   - common_fit_margin_trials: per step the margins lowered to it (a margin already at or below a step stays); a step that
//     lowers none of them is left out, so the trials run from the largest margin down and never above the given margins
//   - common_fit_plain_spill_gate: whether the step-down may act given each used device's display signal (1 headless, 0 a
//     display / watchdog, -1 unknown): only if every device is headless; force "1" acts anyway, "0" never (LLAMA_FIT_PLAIN_HEADLESS)
//   - common_fit_devices_headless: that signal for each device (the CUDA backend's ggml_backend_dev_headless, -1 elsewhere)
std::vector<size_t>              common_fit_margin_steps_mib(const char * spec);
std::vector<std::vector<size_t>> common_fit_margin_trials(const std::vector<size_t> & margins, const std::vector<size_t> & steps);
bool                             common_fit_plain_spill_gate(const std::vector<int> & headless, const char * force);
std::vector<int>                 common_fit_devices_headless(const std::vector<ggml_backend_dev_t> & devs);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
