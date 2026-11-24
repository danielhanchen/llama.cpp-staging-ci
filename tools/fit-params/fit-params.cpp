// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "llama.h"
#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"

#include <cinttypes>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// satisfies -Wmissing-declarations
int llama_fit_params(int argc, char ** argv);

int llama_fit_params(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // --moe-cache-mib auto: the fit below sizes the cache, the fitted size is printed
    const bool moe_cache_auto = common_moe_cache_auto_resolve(params, !params.fit_params_print);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (!params.fit_params_print) {
        // as common_init_from_params: an explicit -ub is kept by the fit without flash attention
        common_fit_set_ubatch_user(params.n_ubatch_set);
        common_fit_faoff_ubatch_reset();
        // as common_init_from_params: an explicit --fit-target is kept beside a MoE cache (the printed arguments are replayed with it)
        common_fit_set_moe_margin_auto(!params.fit_params_target_set);
        // K/V in host chunks is only allocated by a launch that went through common_init_from_params (LLAMA_KV_SEL_GRANTED, set
        // there): the printed arguments cannot carry it, so the fit measures the placement without them
        common_fit_kv_sel_allow(false);
        // the small-device placement of the fit (LLAMA_FIT_PLACE8) sizes the MoE cache for prompt buffers that only this process
        // shrinks (LLAMA_MOE_PLACE8_UB, set in its environment): the printed arguments cannot carry that, so a launch with them would
        // reserve the full prompt buffers beside the larger cache. Off here unless set explicitly
        if (getenv("LLAMA_FIT_PLACE8") == nullptr) {
            common_set_env("LLAMA_FIT_PLACE8", "0");
        }
        const uint32_t n_batch_in  = cparams.n_batch;
        const uint32_t n_ubatch_in = cparams.n_ubatch;
        const common_params_fit_status status = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                nullptr,
                params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR,
                params.fit_kv_resident);
        if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            LOG_ERR("%s: failed to fit CLI arguments to free memory, exiting...\n", __func__);
            exit(1);
        }

        LOG_INF("%s: printing fitted CLI arguments to stdout...\n", __func__);
        common_log_flush(common_log_main());
        printf("-c %" PRIu32 " -ngl %" PRIi32, cparams.n_ctx, mparams.n_gpu_layers);

        size_t nd = llama_max_devices();
        while (nd > 1 && mparams.tensor_split[nd - 1] == 0.0f) {
            nd--;
        }
        if (nd > 1) {
            for (size_t id = 0; id < nd; id++) {
                if (id == 0) {
                    printf(" -ts ");
                }
                printf("%s%" PRIu32, id > 0 ? "," : "", uint32_t(mparams.tensor_split[id]));
            }
        }

        const size_t ntbo = llama_max_tensor_buft_overrides();
        bool any_tbo = false;
        for (size_t itbo = 0; itbo < ntbo && mparams.tensor_buft_overrides[itbo].pattern != nullptr; itbo++) {
            if (itbo == 0) {
                printf(" -ot \"");
            }
            printf("%s%s=%s", itbo > 0 ? "," : "", mparams.tensor_buft_overrides[itbo].pattern, ggml_backend_buft_name(mparams.tensor_buft_overrides[itbo].buft));
            any_tbo = true;
        }
        printf("%s", any_tbo ? "\"" : "");
        if (moe_cache_auto && cparams.moe_cache_size >= params.moe_cache_size) {
            // auto that the fit did not size: no cache
            printf(" --moe-cache-mib 0");
        } else if (params.moe_cache_size > 0) {
            // the fit may reduce the MoE cache, print the fitted size
            printf(" --moe-cache-mib %zu", cparams.moe_cache_size/(1024*1024));
        }
        if (cparams.n_batch != n_batch_in) {
            printf(" -b %" PRIu32, cparams.n_batch);
        }
        if (cparams.n_ubatch != n_ubatch_in) {
            // the fit chose a smaller n_ubatch (flash attention off): the placement above was measured at it
            printf(" -ub %" PRIu32, cparams.n_ubatch);
        }
        printf("\n");
    } else {
        LOG_INF("%s: printing estimated memory in MiB to stdout (device, model, context, compute) ...\n", __func__);
        common_log_flush(common_log_main());

        common_fit_print(params.model.path.c_str(), &mparams, &cparams);
    }

    return 0;
}
