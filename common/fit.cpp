#include "fit.h"

#include "json.h"
#include "log.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <vector>

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

// --fit-kv-resident auto: use the KV-resident placement for a dense model only if the whole-layer placement would put less than this much
// KV cache (incl. recurrent state) in system memory per byte of weights it puts there. Making room in device memory for that KV cache moves
// as many bytes of FFN weights to system memory, which every token reads, while the whole-layer placement reads only the used cells.
// Calibrated on k6 (Colab G4 RTX PRO 6000, EPYC 9B45 48 vCPU at -t 24 and pinned to 8 / 6 physical cores; Qwen3.8-27B, gemma-4-12B,
// gemma-4-31B, Llama-3.1-8B; VRAM 4-16 GB, -c 8192 / 32768; llama-server, 2048-token prompt, medians of 3): below 0.13 (ratios
// 0.045-0.120, 8 cells x 3 thread counts) FFN first decoded +0.0..+8.8% faster and processed prompts +0.8..+21% faster; from 0.141 up
// (0.141-1.012, 8 cells) it decoded up to 34% slower: slower in every cell at 6 threads (-0.6..-33%), faster only in the two
// Qwen3.8-27B 32K cells at 24 threads (+2.2 / +3.1%) and 8 threads (+0.1%). After a 16384-token prompt at -c 32768 FFN first decoded
// +5..+45% faster in every cell auto decides: see LLAMA_FIT_KV_RESIDENT_CTX.
static constexpr double FIT_KV_RESIDENT_AUTO_MAX_KV_RATIO = 0.13;

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

static bool    g_moe_margin_auto = true;
static int64_t g_moe_margin_gain = 0;

void common_fit_set_moe_margin_auto(bool auto_margin) {
    g_moe_margin_auto = auto_margin;
}

int64_t common_fit_moe_margin_gain() {
    return g_moe_margin_gain;
}

static bool g_small_device = false;

bool common_fit_small_device() {
    return g_small_device;
}

// the margin of a device beside a MoE cache for the default margin m (see common_fit_set_moe_margin_auto)
static int64_t common_fit_moe_margin(int64_t m) {
    constexpr int64_t MiB_ = 1024*1024;
    constexpr int64_t m_default = 1024*MiB_;
    if (!g_moe_margin_auto || m < m_default) {
        return m;
    }
    // opt-in (k11 G4 timing): the 8 GB tiers use the default margin at run time (layer-major groups, prefetch depth, the MTP context's
    // staging, the CUDA pools: 70 MiB free at the low point of Qwen3.6 UD-Q4_K_XL + MTP); 384 MiB ran out of device memory there and on
    // Flash-Next UD-IQ1_S, and halved the 8k prompt of Qwen3.6 Q3_K_M
    const char * env = getenv("LLAMA_MOE_FIT_MARGIN");
    const char * e   = getenv("LLAMA_MOE_FIT_MARGIN_MIB");
    if ((env != nullptr && atoi(env) == 0) || e == nullptr) {
        return m;
    }
    const int64_t keep = std::max<int64_t>(0, atoll(e))*MiB_;
    // what was added to the default (e.g. a multimodal projector) stays
    return m - m_default + std::min(keep, m_default);
}

static std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model);
    if (mparams->load_mtp) {
        hp_ngl    += llama_model_n_layer_nextn(model);
    }
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    common_memory_breakdown_print(ctx);

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, const common_fit_extra_model * extra, enum ggml_log_level log_level,
        enum common_fit_kv_resident kv_resident) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // with non-unified kv, we need to take into account n_streams
    // for example, if memory can hold more than model's trained context size, we must extend the n_ctx to hold enough n_streams
    const uint32_t n_streams  = cparams->kv_unified ? 1 : std::max<uint32_t>(1, cparams->n_seq_max);
    const bool     n_ctx_auto = cparams->n_ctx == 0;

    dmds_t   dmds_extra;       // memory of the extra model, laid out on the devices of the main model
    uint32_t n_ctx_extra = 0;  // context that memory was measured at

    // the extra model competes for the same memory as the main model, add it to every measurement
    // its memory is measured again whenever the context it follows changes
    auto add_extra_memory = [&](dmds_t & dmds) {
        if (extra == nullptr) {
            return;
        }

        if (dmds_extra.empty() || n_ctx_extra != cparams->n_ctx) {
            std::vector<ggml_backend_dev_t> devs_extra;
            uint32_t ngl_extra = 0;
            uint32_t nct_extra = 0;
            uint32_t nex_extra = 0;

            extra->cparams->n_ctx = cparams->n_ctx;

            LOG_TRC("%s: getting device memory data for the extra model at a context size of %" PRIu32 ":\n",
                __func__, cparams->n_ctx);

            dmds_t measured;
            try {
                measured = common_get_device_memory_data_impl(
                    extra->path_model, extra->mparams, extra->cparams, devs_extra, ngl_extra, nct_extra, nex_extra, log_level);
            } catch (const std::runtime_error & e) {
                // the extra model is optional, fit the main model alone rather than giving up
                LOG_WRN("%s: failed to measure the memory of the extra model, fitting without it: %s\n", __func__, e.what());
                dmds_extra = dmds_t(devs.size() + 1);
                n_ctx_extra = cparams->n_ctx;
                return;
            }

            dmds_extra = dmds_t(devs.size() + 1);
            dmds_extra.back().mb = measured.back().mb;
            for (size_t je = 0; je < devs_extra.size(); je++) {
                for (size_t id = 0; id < devs.size(); id++) {
                    if (devs_extra[je] == devs[id]) {
                        dmds_extra[id].mb.model   += measured[je].mb.model;
                        dmds_extra[id].mb.context += measured[je].mb.context;
                        dmds_extra[id].mb.compute += measured[je].mb.compute;
                        break;
                    }
                }
            }
            if (extra->shares_model) {
                for (llama_device_memory_data & dmd : dmds_extra) {
                    dmd.mb.model = 0;
                }
            }

            n_ctx_extra = cparams->n_ctx;
        }

        for (size_t id = 0; id < dmds.size(); id++) {
            dmds[id].mb.model   += dmds_extra[id].mb.model;
            dmds[id].mb.context += dmds_extra[id].mb.context;
            dmds[id].mb.compute += dmds_extra[id].mb.compute;
        }
    };

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    dmds_t dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

    // saturate instead of overflowing, this also preserves the UINT32_MAX sentinel of n_ctx_min:
    const uint32_t n_ctx_max       = (uint32_t) std::min<uint64_t>(uint64_t(hp_nct)    * n_streams, UINT32_MAX);
    const uint32_t n_ctx_min_total = (uint32_t) std::min<uint64_t>(uint64_t(n_ctx_min) * n_streams, UINT32_MAX);

    // llama_context would use only hp_nct in total for n_ctx == 0, resolve the context before measuring anything else:
    if (n_ctx_auto) {
        cparams->n_ctx = n_ctx_max;
        if (n_streams > 1) {
            LOG_TRC("%s: context size unset and KV cache not unified -> using %" PRIu32 " for %" PRIu32 " sequences:\n",
                __func__, n_ctx_max, n_streams);
            dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        }
    }
    add_extra_memory(dmds_full);

    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    // the MoE cache (--moe-cache-mib) has the lowest priority: it gets the device memory that is left once the layers are placed,
    // the requested size is only a cap; LLAMA_FIT_MOE_CACHE=0 restores the old behaviour, which kept the requested size and placed the layers around it
    // user tensor overrides (-ot, --cpu-moe, --n-cpu-moe) are then kept as constraints instead of aborting the fit
    const char * fit_moe_cache_env = getenv("LLAMA_FIT_MOE_CACHE");
    const bool   fit_moe_cache     = nd > 0 && hp_nex > 0 && cparams->moe_cache_size > 0 &&
        (fit_moe_cache_env == nullptr || atoi(fit_moe_cache_env) != 0);

    // dense models that do not fit (--fit-kv-resident, kv_resident): a layer that is moved to system memory takes its attention and its
    // KV cache with it, while an FFN tensor in system memory only adds a weight upload (prompt batches) or a CPU matrix-vector product
    // (token generation). The KV-resident placement (FFN first) takes the MoE path of the fit with the FFN tensors in the place of the
    // experts: every layer keeps its attention, norms and KV cache in device memory and the FFN tensors of the last layers are moved to
    // system memory; if the layers do not fit even without their FFN tensors, the layers in device memory keep only their attention,
    // norms and KV cache and the first layers go to system memory whole.
    //   - off:  whole layers, as before
    //   - on:   FFN first
    //   - auto: FFN first if the layers fit without their FFN tensors and the whole-layer placement would put less than
    //           FIT_KV_RESIDENT_AUTO_MAX_KV_RATIO bytes of KV cache in system memory per byte of weights, otherwise whole layers.
    //           The KV cache is counted at the requested context, which is never cut for it. Decode reads only the used cells, so a
    //           context that is mostly used favours FFN first at any ratio: LLAMA_FIT_KV_RESIDENT_CTX=N declares an expected used context
    //           of N tokens per sequence, and N at least half the context per sequence picks FFN first (Qwen3.8-27B at -c 131072 / 16 GB,
    //           ratio 0.62, decoded 11.9 t/s after a 2K prompt with the KV cache resident vs 18.4 with whole layers, break-even near 30K
    //           used cells; at -c 32768 every cell auto decides decoded faster with FFN first after a 16K prompt).
    //           User tensor overrides make auto behave as off.
    // LLAMA_FIT_DENSE_FFN overrides kv_resident (kill switch): 0 = off, 1 = on, 2 = on but whole layers if the layers do not fit without
    // their FFN tensors. With on, user tensor overrides are kept as constraints.
    // G4 (RTX PRO 6000), gemma-4 E2B UD-IQ2_M at a 2 GB cap, -c 16384, medians of 3: FFN first (34/36 layers) decoded 121.0 tok/s vs 131.6
    // with whole layers (11/36) for 2k prompts at 13641 vs 11048 tok/s, so on is not the default; auto keeps whole layers there.
    enum { DENSE_FFN_OFF = 0, DENSE_FFN_ON = 1, DENSE_FFN_STRICT = 2, DENSE_FFN_AUTO = 3 };
    int fit_dense_ffn_mode = kv_resident == COMMON_FIT_KV_RESIDENT_ON   ? DENSE_FFN_ON   :
                             kv_resident == COMMON_FIT_KV_RESIDENT_AUTO ? DENSE_FFN_AUTO : DENSE_FFN_OFF;
    if (const char * fit_dense_ffn_env = getenv("LLAMA_FIT_DENSE_FFN")) {
        const int v = atoi(fit_dense_ffn_env);
        fit_dense_ffn_mode = v == 0 ? DENSE_FFN_OFF : v == 2 ? DENSE_FFN_STRICT : DENSE_FFN_ON;
    }
    if (fit_dense_ffn_mode == DENSE_FFN_AUTO && mparams->tensor_buft_overrides &&
            (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        fit_dense_ffn_mode = DENSE_FFN_OFF;
    }
    const bool fit_dense_ffn = nd > 0 && hp_nex == 0 && fit_dense_ffn_mode != DENSE_FFN_OFF;

    // the user overrides come first in every placement written below: the first matching pattern wins
    std::vector<llama_model_tensor_buft_override> tbo_user;
    if ((fit_moe_cache || fit_dense_ffn) && mparams->tensor_buft_overrides) {
        for (const llama_model_tensor_buft_override * tbo = mparams->tensor_buft_overrides; tbo->pattern != nullptr; tbo++) {
            tbo_user.push_back(*tbo);
        }
    }

    // shrinks the MoE cache until the measured placement dmds (model params mp) meets the targets, returns the remaining total surplus
    // the cache memory is measured (probe without the cache), the slot sizing itself stays in llama_moe_cache
    // with several devices every device must meet its own target: the cache is split over the devices of its layers, so a device short of
    // memory needs the total shrunk by its deficit scaled by its share of the cache (a total surplus can hide a device that misses its target)
    auto fit_moe_cache_size = [&](dmds_t & dmds, const llama_model_params * mp, const bool use_margins) -> int64_t {
        g_moe_margin_gain = 0; // the last sizing of the cache decides
        auto margin_of = [&](size_t id) -> int64_t {
            return use_margins ? int64_t(margins[id]) : 0;
        };
        auto surplus_of = [&](const dmds_t & d) {
            int64_t s = 0;
            for (size_t id = 0; id < nd; id++) {
                s += d[id].free;
                s -= int64_t(d[id].mb.total()) + margin_of(id);
            }
            return s;
        };
        auto devices_ok = [&](const dmds_t & d) {
            for (size_t id = 0; id < nd; id++) {
                if (int64_t(d[id].free) - int64_t(d[id].mb.total()) - margin_of(id) < 0) {
                    return false;
                }
            }
            return true;
        };
        auto measure = [&](const llama_context_params * cp) {
            dmds_t res = common_get_device_memory_data_impl(path_model, mp, cp, devs, hp_ngl, hp_nct, hp_nex, log_level);
            add_extra_memory(res);
            return res;
        };
        const int64_t surplus = surplus_of(dmds);
        if (surplus >= 0 && devices_ok(dmds)) {
            return surplus;
        }
        const int64_t size_requested = cparams->moe_cache_size;

        llama_context_params cparams_nc = *cparams;
        cparams_nc.moe_cache_size = 0;
        LOG_TRC("%s: getting device memory data without the MoE cache:\n", __func__);
        dmds_t dmds_nc = measure(&cparams_nc);
        const int64_t surplus_nc  = surplus_of(dmds_nc);
        const int64_t cache_bytes = surplus_nc - surplus; // device memory the cache takes in this placement
        if (cache_bytes <= 0) {
            return surplus; // no layer of this placement uses the cache
        }
        // k11: the part of the margin beside the cache it may take once every other decision is made (common_fit_moe_margin_gain)
        int64_t margin_gain = 0;
        for (size_t id = 0; id < nd; id++) {
            if (use_margins && int64_t(dmds[id].mb.total()) > int64_t(dmds_nc[id].mb.total())) {
                margin_gain += int64_t(margins[id]) - common_fit_moe_margin(int64_t(margins[id]));
            }
        }

        // bytes of cache to give back so that the total and every device meet their targets, INT64_MAX if a device misses its target
        // even without its share of the cache
        auto shortfall = [&](const dmds_t & d) -> int64_t {
            int64_t need = -surplus_of(d);
            int64_t cache_total = 0;
            for (size_t id = 0; id < nd; id++) {
                cache_total += std::max<int64_t>(0, int64_t(d[id].mb.total()) - int64_t(dmds_nc[id].mb.total()));
            }
            for (size_t id = 0; id < nd; id++) {
                const int64_t s_id = int64_t(d[id].free) - int64_t(d[id].mb.total()) - margin_of(id);
                if (s_id >= 0) {
                    continue;
                }
                const int64_t cache_id = std::max<int64_t>(0, int64_t(d[id].mb.total()) - int64_t(dmds_nc[id].mb.total()));
                if (cache_id <= 0 || cache_total <= 0) {
                    return INT64_MAX;
                }
                need = std::max(need, int64_t(double(-s_id) * double(cache_total) / double(cache_id)));
            }
            return need;
        };

        const int64_t need0 = shortfall(dmds);
        int64_t size = need0 == INT64_MAX ? 0 : std::min(size_requested, cache_bytes) - need0;
        for (int it = 0; it < 6 && size > 0 && surplus_nc > 0; it++) {
            cparams->moe_cache_size = size;
            dmds_t dmds_test;
            try {
                dmds_test = measure(cparams);
            } catch (const std::runtime_error &) {
                break; // too small to hold the experts of one token
            }
            const int64_t need = shortfall(dmds_test);
            if (need <= 0) {
                // k15-fn: the first size that fits can leave memory above the targets (the cache does not take exactly the bytes the
                // linear estimate gave it: G4 Qwen3.8-Flash-Next UD-IQ4_XS + MTP at 32 GB left 856 MiB above the 1024 MiB margin);
                // grow it by the measured surplus while it still fits (LLAMA_MOE_FIT_FILL=0: keep the first size)
                static const bool fill = getenv("LLAMA_MOE_FIT_FILL") == nullptr || atoi(getenv("LLAMA_MOE_FIT_FILL")) != 0;
                for (int g = 0; fill && g < 3 && size < size_requested; g++) {
                    const int64_t extra = std::min<int64_t>(surplus_of(dmds_test) - 16*MiB, size_requested - size);
                    if (extra < 64*MiB || !devices_ok(dmds_test)) {
                        break;
                    }
                    cparams->moe_cache_size = size + extra;
                    dmds_t dmds_grow;
                    try {
                        dmds_grow = measure(cparams);
                    } catch (const std::runtime_error &) {
                        break;
                    }
                    if (shortfall(dmds_grow) > 0) {
                        break;
                    }
                    LOG_TRC("%s: MoE cache grown by %" PRId64 " MiB into the measured surplus\n", __func__, extra/MiB);
                    size      = size + extra;
                    dmds_test = dmds_grow;
                }
                cparams->moe_cache_size = size;
                LOG_TRC("%s: MoE cache reduced from %" PRId64 " to %" PRId64 " MiB to keep the layers in device memory\n",
                    __func__, size_requested/MiB, size/MiB);
                if (margin_gain > 0) {
                    g_moe_margin_gain = std::min<int64_t>(margin_gain, size_requested - size);
                }
                dmds = dmds_test;
                return surplus_of(dmds_test);
            }
            if (need == INT64_MAX) {
                break;
            }
            size -= need + MiB;
        }
        LOG_TRC("%s: MoE cache disabled (requested %" PRId64 " MiB) to keep the layers in device memory\n", __func__, size_requested/MiB);
        cparams->moe_cache_size = 0;
        dmds = dmds_nc;
        return surplus_nc;
    };

    // a placement given by the user (host experts through overrides) already holds the cache: shrink it before the context or the layers
    // if the user also fixed the layers the fit aborts below, so only the free memory itself is the target there
    if (fit_moe_cache && !tbo_user.empty()) {
        fit_moe_cache_size(dmds_full, mparams, mparams->n_gpu_layers == default_mparams.n_gpu_layers);
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    // MoE models can move their experts to system memory in step 3, the KV cache only pays for context.
    // Before the context is cut below the kept size (LLAMA_FIT_CTX_KEEP, default 32768, 0 = training context),
    // check whether that context fits with all experts in system memory; if so keep it and let step 3 spill instead.
    // LLAMA_FIT_CTX_FIRST=1 restores the old order (context first).
    // Returns true if cparams->n_ctx was set to a context > n_ctx_reduced, in which case step 3 has to run.
    auto keep_ctx_by_spilling = [&](const uint32_t n_ctx_reduced, const int64_t sum_used_target) -> bool {
        if (hp_nex == 0 || nd == 0) {
            return false;
        }
        const char * env_first = getenv("LLAMA_FIT_CTX_FIRST");
        if (env_first != nullptr && atoi(env_first) != 0) {
            return false;
        }
        // step 3 can only spill if it is allowed to change the weight placement:
        if (mparams->n_gpu_layers != default_mparams.n_gpu_layers || !tensor_buft_overrides ||
                (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft) && tbo_user.empty()) ||
                (nd > 1 && (!tensor_split || mparams->split_mode == LLAMA_SPLIT_MODE_ROW))) {
            return false;
        }
        uint64_t keep = 32768;
        if (const char * env_keep = getenv("LLAMA_FIT_CTX_KEEP")) {
            keep = strtoull(env_keep, nullptr, 10);
            if (keep == 0) {
                keep = hp_nct;
            }
        }
        const uint32_t n_ctx_keep = (uint32_t) std::min<uint64_t>(n_ctx_max, std::max<uint64_t>(n_ctx_min_total, keep * n_streams));
        if (n_ctx_keep <= n_ctx_reduced) {
            return false;
        }

        static const std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        // the user overrides stay first, the MoE cache is left out: it only gets what remains after the layers and the context
        std::vector<llama_model_tensor_buft_override> tbo_spill = tbo_user;
        tbo_spill.push_back({pattern_moe_all.c_str(), ggml_backend_cpu_buffer_type()});
        tbo_spill.push_back({nullptr, nullptr});
        llama_model_params mparams_spill = *mparams;
        mparams_spill.tensor_buft_overrides = tbo_spill.data();
        llama_context_params cparams_spill = *cparams;
        if (fit_moe_cache) {
            cparams_spill.moe_cache_size = 0;
        }
        auto used_with_experts_on_host = [&](const uint32_t n_ctx) -> int64_t {
            cparams->n_ctx      = n_ctx; // the extra model follows it
            cparams_spill.n_ctx = n_ctx;
            dmds_t dmds_spill = common_get_device_memory_data_impl(
                path_model, &mparams_spill, &cparams_spill, devs, hp_ngl, hp_nct, hp_nex, log_level);
            add_extra_memory(dmds_spill);
            int64_t sum = 0;
            for (size_t id = 0; id < nd; id++) {
                sum += dmds_spill[id].mb.total();
            }
            return sum;
        };

        LOG_TRC("%s: getting device memory data with all MoE tensors in system memory and context %" PRIu32 ":\n", __func__, n_ctx_keep);
        const int64_t used_keep = used_with_experts_on_host(n_ctx_keep);
        if (used_keep <= sum_used_target) {
            cparams->n_ctx = n_ctx_keep;
            LOG_TRC("%s: context size %" PRIu32 " fits with MoE tensors in system memory (%" PRId64 " <= %" PRId64 " MiB) -> keeping it, spilling weights instead of reducing it to %" PRIu32 "\n",
                __func__, n_ctx_keep, used_keep/MiB, sum_used_target/MiB, n_ctx_reduced);
            return true;
        }
        const int64_t used_min = used_with_experts_on_host(n_ctx_min_total);
        uint32_t n_ctx = n_ctx_min_total;
        if (sum_used_target > used_min && used_keep > used_min) {
            n_ctx += uint32_t(int64_t(n_ctx_keep - n_ctx_min_total) * (sum_used_target - used_min) / (used_keep - used_min));
            const uint32_t align = 256 * n_streams;
            n_ctx = std::max(n_ctx - n_ctx % align, n_ctx_min_total);
        }
        if (n_ctx <= n_ctx_reduced) {
            cparams->n_ctx = n_ctx_reduced;
            LOG_TRC("%s: even with MoE tensors in system memory only a context of %" PRIu32 " fits -> no gain from spilling\n", __func__, n_ctx);
            return false;
        }
        cparams->n_ctx = n_ctx;
        LOG_TRC("%s: with MoE tensors in system memory a context of %" PRIu32 " fits (instead of %" PRIu32 ") -> spilling weights\n",
            __func__, n_ctx, n_ctx_reduced);
        return true;
    };

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (n_ctx_auto) {
                if (n_ctx_max > n_ctx_min_total) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min_total;
                    dmds_t dmds_min_ctx = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    add_extra_memory(dmds_min_ctx);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (n_ctx_max - n_ctx_min_total) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        // round down context for CUDA backend, keep it divisible by the number of streams:
                        const uint32_t align = 256 * n_streams;
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % align, n_ctx_min_total);

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (n_ctx_max - n_ctx_min_total);
                        const int64_t memory_reduction = (n_ctx_max - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                        if (!keep_ctx_by_spilling(cparams->n_ctx, sum_used_target)) {
                            if (nd <= 1) {
                                LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                                return;
                            }
                            LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                        }
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                        keep_ctx_by_spilling(cparams->n_ctx, sum_used_target);
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, n_ctx_max);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, n_ctx_max, n_ctx_min_total);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft) && tbo_user.empty()) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory
    //   - for a dense model with fit_dense_ffn, same as a MoE model with the FFN tensors in the place of the MoE tensors

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        if (hp_nex == 0) {
            // dense FFN tensors, anchored at the end of the tensor type so that e.g. ffn_norm is never matched:
            //   ATTN and MOE: up, gate and down in system memory, UP: gate and down, GATE: down
            static std::array<std::string, n_strings> patterns_all;
            static std::array<std::string, n_strings> patterns_up;
            static std::array<std::string, n_strings> patterns_gate;
            std::string * p     = nullptr;
            const char  * types = nullptr;
            switch (lf) {
                case LAYER_FRACTION_ATTN:
                case LAYER_FRACTION_MOE:  p = &patterns_all[il];  types = "(up|gate|down)"; break;
                case LAYER_FRACTION_UP:   p = &patterns_up[il];   types = "(gate|down)";    break;
                case LAYER_FRACTION_GATE: p = &patterns_gate[il]; types = "down";           break;
                default: GGML_ABORT("fatal error");
            }
            if (p->empty()) {
                *p = "blk\\." + std::to_string(il) + "\\.ffn_" + types + "\\.";
            }
            return p->c_str();
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (const llama_model_tensor_buft_override & tbo : tbo_user) {
            tensor_buft_overrides[itbo++] = tbo;
        }
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory data per device (+ system memory) for given numbers of layers per device
    auto get_dmds_for_layers = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> dmds_t {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        dmds_t dmd_nl = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmd_nl);
        return dmd_nl;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        const dmds_t dmd_nl = get_dmds_for_layers(ngl_per_device, overflow_bufts);

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0; // total surplus with all MoE tensors (dense: FFN tensors) in system memory
    bool    partial_layers         = hp_nex > 0; // whether layers may keep their MoE (dense: FFN) tensors in system memory
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        const size_t n_user = tbo_user.size();
        if (n_user + 2 > llama_max_tensor_buft_overrides()) {
            throw common_params_fit_exception("too many tensor_buft_overrides set by user, abort");
        }
        for (size_t i = 0; i < n_user; i++) {
            tensor_buft_overrides[i] = tbo_user[i];
        }
        tensor_buft_overrides[n_user + 0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[n_user + 1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        dmds_t dmds_cpu_moe = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmds_cpu_moe);

        if (fit_moe_cache) {
            // order: the dense weights of all layers, then the cache (up to the requested size), then experts in device memory
            const size_t size_before = cparams->moe_cache_size;
            const dmds_t dmds_cpu_moe_req = dmds_cpu_moe; // k13-small: the measurement at the requested cache size, for the retry below
            fit_moe_cache_size(dmds_cpu_moe, mparams, true);
            bool fits_per_device = cparams->moe_cache_size > 0;
            for (size_t id = 0; id < nd; id++) {
                fits_per_device = fits_per_device && int64_t(dmds_cpu_moe[id].free) - int64_t(dmds_cpu_moe[id].mb.total()) >= margins[id];
            }
            // k13-small: a small device (G4 ballasted to 4 GB, Qwen3.6-35B-A3B Q3_K_M + MTP: the dense weights of every layer, the context and
            // a cache fit only inside the 1024 MiB margin). Without this the fit moves 13 of 41 whole layers (attention, KV, recurrent state)
            // to the CPU and drops the cache, so every token pays a graph split per CPU layer and per MoE layer (k12: 91 tok/s against
            // PolyStrata's 177, which keeps every dense layer on the device). Here the margin of the one device drops to
            // LLAMA_FIT_SMALL_MARGIN_MIB (256) instead: every layer stays on the device beside a small cache, whose host misses the
            // single-graph path computes (no graph split). The placement is the one of -fitt 256; one GPU only, default margin only (an
            // explicit smaller -fitt is the user's), LLAMA_FIT_SMALL_MARGIN_MIB=0 keeps the k12 placement.
            if (!fits_per_device && nd == 1 && size_before > 0) {
                const char * e_sm = getenv("LLAMA_FIT_SMALL_MARGIN_MIB");
                const int64_t small_mib = e_sm != nullptr ? std::max<int64_t>(0, atoll(e_sm)) : 256;
                constexpr int64_t m_default = 1024*MiB;
                // opt-in: the cache of a small device that holds only a few experts anyway is capped (LLAMA_FIT_SMALL_CACHE_MIB=48)
                // where the uncapped fit gives it at most LLAMA_FIT_SMALL_CACHE_CAP_BELOW_MIB (256): what it does not take stays free for
                // the layer-major prompt groups, which read each host expert once per group instead of once per ubatch. G4 4 GB q3m + MTP
                // (111 MiB uncapped, 76 MiB free, every group of 2+ ubatches split): prompt 2k 1885 -> 4712, 8k 1937 -> 4103 tok/s,
                // decode -9 / -2 / -5% (chat / 2k / 8k, the smaller cache); gpt-oss-20b 4 GB (476 MiB) lost 4-10% decode capped: not capped.
                // Off by default: the server KLD run (k13kld3, 8k prompts) ran out of device memory in a merged prompt group at the cap
                const char * e_sc = getenv("LLAMA_FIT_SMALL_CACHE_MIB");
                const int64_t cap_mib = e_sc != nullptr ? std::max<int64_t>(0, atoll(e_sc)) : 0; // opt-in (LLAMA_FIT_SMALL_CACHE_MIB=48): see below
                const char * e_cb = getenv("LLAMA_FIT_SMALL_CACHE_CAP_BELOW_MIB");
                const int64_t cap_below_mib = e_cb != nullptr ? std::max<int64_t>(0, atoll(e_cb)) : 256;
                // opt-in second, smaller margin when the first one leaves no cache (LLAMA_FIT_SMALL_MARGIN_MIN_MIB=N): G4 4 GB Qwen3.6
                // UD-Q4_K_XL + MTP placed every layer at 128 MiB but ran out of device memory in its first request (CUDA error, run k13r2)
                const char * e_sm2 = getenv("LLAMA_FIT_SMALL_MARGIN_MIN_MIB");
                const int64_t small_min_mib = std::min<int64_t>(small_mib, e_sm2 != nullptr ? std::max<int64_t>(0, atoll(e_sm2)) : small_mib);
                if (g_moe_margin_auto && small_mib > 0 && small_mib*MiB < m_default && margins[0] >= m_default) {
                    const int64_t margin_saved = margins[0];
                    const dmds_t  dmds_nc_saved = dmds_cpu_moe;
                    // one sizing of the cache from the measurement d at the requested size req, margin m_mib: ok if every layer fits
                    auto try_fit = [&](size_t req, const dmds_t & d, int64_t m_mib) {
                        margins[0] = margin_saved - m_default + m_mib*MiB; // what was added to the default (a projector) stays
                        dmds_cpu_moe = d;
                        cparams->moe_cache_size = req;
                        fit_moe_cache_size(dmds_cpu_moe, mparams, true);
                        return cparams->moe_cache_size > 0 && cparams->moe_cache_size != size_before &&
                            int64_t(dmds_cpu_moe[0].free) - int64_t(dmds_cpu_moe[0].mb.total()) >= margins[0];
                    };
                    bool    ok    = false;
                    int64_t m_ok  = 0;
                    for (const int64_t m_mib : { small_mib, small_min_mib }) {
                        if (!ok && m_mib > 0) {
                            ok   = try_fit(size_before, dmds_cpu_moe_req, m_mib);
                            m_ok = m_mib;
                        }
                    }
                    if (ok && cap_mib > 0 && cparams->moe_cache_size > size_t(cap_mib)*MiB && cparams->moe_cache_size <= size_t(cap_below_mib)*MiB) {
                        const size_t  size_unc = cparams->moe_cache_size;
                        const dmds_t  dmds_unc = dmds_cpu_moe;
                        const int64_t gain_unc = g_moe_margin_gain;
                        cparams->moe_cache_size = size_t(cap_mib)*MiB;
                        dmds_t dmds_cap = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                        add_extra_memory(dmds_cap);
                        if (!try_fit(size_t(cap_mib)*MiB, dmds_cap, m_ok)) {
                            margins[0] = margin_saved - m_default + m_ok*MiB;
                            cparams->moe_cache_size = size_unc;
                            dmds_cpu_moe = dmds_unc;
                            g_moe_margin_gain = gain_unc;
                        }
                    }
                    if (ok) {
                        LOG_INF("%s: small device: every layer stays in device memory beside a MoE cache of %zu MiB, the fit margin drops from "
                            "%" PRId64 " to %" PRId64 " MiB (otherwise layers move to the CPU; LLAMA_FIT_SMALL_MARGIN_MIB=0 keeps that)\n",
                            __func__, cparams->moe_cache_size/MiB, margin_saved/MiB, margins[0]/MiB);
                        fits_per_device = true;
                        g_small_device  = true;
                    } else {
                        margins[0] = margin_saved;
                        dmds_cpu_moe = dmds_nc_saved;
                        cparams->moe_cache_size = 0;
                        g_moe_margin_gain = 0;
                    }
                }
            }
            bool keep = fits_per_device && cparams->moe_cache_size != size_before;
            if (fits_per_device && !keep) {
                // the cache has its requested size: steps 3 and 4 may add experts in device memory if they fit next to it,
                // their first placement keeps the experts of one layer in device memory, without room for it they would drop layers instead
                std::vector<ngl_t> ngl_all(nd);
                ngl_all[nd - 1].n_layer = hp_ngl + 1;
                ngl_all[nd - 1].n_part  = hp_ngl;
                std::vector<float> tensor_split_saved; // set_ngl_tensor_split_tbo writes the shared tensor_split buffer
                if (nd > 1) {
                    tensor_split_saved.assign(tensor_split, tensor_split + nd);
                }
                llama_model_params mparams_all = *mparams;
                set_ngl_tensor_split_tbo(ngl_all, std::vector<ggml_backend_buffer_type_t>(nd, cpu_buft), mparams_all);
                std::copy(tensor_split_saved.begin(), tensor_split_saved.end(), tensor_split);
                LOG_TRC("%s: getting device memory data with all layers in device memory and the experts of one layer:\n", __func__);
                dmds_t dmds_all = common_get_device_memory_data_impl(
                    path_model, &mparams_all, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                add_extra_memory(dmds_all);
                int64_t surplus_all = 0;
                for (size_t id = 0; id < nd; id++) {
                    surplus_all += int64_t(dmds_all[id].free) - int64_t(dmds_all[id].mb.total()) - margins[id];
                }
                keep = surplus_all < 0;
                // restore the overrides of the probe with all MoE tensors in system memory
                for (size_t i = 0; i < n_user; i++) {
                    tensor_buft_overrides[i] = tbo_user[i];
                }
                tensor_buft_overrides[n_user + 0] = {pattern_moe_all.c_str(), cpu_buft};
                tensor_buft_overrides[n_user + 1] = {nullptr, nullptr};
                mparams->tensor_buft_overrides = tensor_buft_overrides;
            }
            if (keep) {
                // k13-small: the same placement reached under an explicit --fit-target (Studio sends one, at least 512 MiB) is a small
                // device too when it leaves less than the default 1024 MiB free beside a cache of at most 256 MiB: the run-time rules keyed
                // on common_fit_small_device (no n_ubatch growth, the checkpoint merge) then apply there as well. The margin itself is the
                // caller's and is not touched.
                if (!g_small_device && nd == 1 && cparams->moe_cache_size <= size_t(256)*MiB &&
                        int64_t(dmds_cpu_moe[0].free) - int64_t(dmds_cpu_moe[0].mb.total()) < 1024*MiB) {
                    g_small_device = true;
                    LOG_INF("%s: small device: every layer stays in device memory beside a MoE cache of %zu MiB within the margin of %" PRId64 " MiB\n",
                        __func__, cparams->moe_cache_size/MiB, margins[0]/MiB);
                }
                // every layer in device memory with all experts in system memory and the cache: the placement of --cpu-moe
                LOG_TRC("%s: all layers in device memory, MoE tensors in system memory, MoE cache of %zu MiB\n",
                    __func__, cparams->moe_cache_size/MiB);
                for (size_t id = 0; id < nd; id++) {
                    LOG_TRC("%s:   - %s: %6" PRId64 " MiB used, %6" PRId64 " MiB free\n", __func__, dev_names[id].c_str(),
                        int64_t(dmds_cpu_moe[id].mb.total())/MiB, (int64_t(dmds_cpu_moe[id].free) - int64_t(dmds_cpu_moe[id].mb.total()))/MiB);
                }
                return;
            }
        }
        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[n_user] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    } else if (fit_dense_ffn) {
        const static std::string pattern_ffn_all = "blk\\.\\d+\\.ffn_(up|gate|down)\\."; // matches all dense FFN tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        const size_t n_user = tbo_user.size();
        if (n_user + 2 > llama_max_tensor_buft_overrides()) {
            throw common_params_fit_exception("too many tensor_buft_overrides set by user, abort");
        }
        for (size_t i = 0; i < n_user; i++) {
            tensor_buft_overrides[i] = tbo_user[i];
        }
        tensor_buft_overrides[n_user + 0] = {pattern_ffn_all.c_str(), cpu_buft};
        tensor_buft_overrides[n_user + 1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all FFN tensors moved to system memory:\n", __func__);
        dmds_t dmds_cpu_ffn = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmds_cpu_ffn);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_ffn[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_ffn[id].mb.total()) + margins[id];
        }
        partial_layers = global_surplus_cpu_moe > 0 || fit_dense_ffn_mode == DENSE_FFN_ON;
        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with the FFN tensors in system memory there is a total surplus of %" PRId64 " MiB, spilling FFN tensors before layers\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else if (partial_layers) {
            LOG_TRC("%s: with the FFN tensors in system memory there is still a total deficit of %" PRId64 " MiB, "
                "spilling the FFN tensors of the layers in device memory and the other layers whole\n", __func__, -global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with the FFN tensors in system memory there is still a total deficit of %" PRId64 " MiB, spilling whole layers\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset to the user overrides
        tensor_buft_overrides[n_user] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // fills ngl_per_device and mem, part: layers may keep their MoE (dense: FFN) tensors in system memory
    const char * fit_func = __func__; // __func__ inside a lambda would be operator()
    auto fill_back_to_front = [&](const bool part) {
        // optimize the number of layers per device using the method of false position:
        //   - ngl_per_device has 0 layers for each device, lower bound
        //   - try a "high" configuration where a device is given all unassigned layers
        //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
        //   - check memory use of our guess, replace either the low or high bound
        //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
        //   - the last device has the output layer, which cannot be a partial layer
        if (hp_nex > 0) {
            LOG_TRC("%s: filling dense-only layers back-to-front:\n", fit_func);
        } else if (part) {
            LOG_TRC("%s: filling layers without FFN tensors back-to-front:\n", fit_func);
        } else {
            LOG_TRC("%s: filling dense layers back-to-front:\n", fit_func);
        }
        for (int id = nd - 1; id >= 0; id--) {
            uint32_t n_unassigned = hp_ngl + 1;
            for (size_t jd = id + 1; jd < nd; ++jd) {
                assert(n_unassigned >= ngl_per_device[jd].n_layer);
                n_unassigned -= ngl_per_device[jd].n_layer;
            }

            std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
            ngl_per_device_high[id].n_layer = n_unassigned;
            if (part) {
                ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
            }
            if (ngl_per_device_high[id].n_layer > 0) {
                std::vector<int64_t> mem_high = get_memory_for_layers(fit_func, ngl_per_device_high, overflow_bufts);
                if (mem_high[id] > targets[id]) {
                    assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                    uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                    LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", fit_func, id, delta);
                    while (delta > 1) {
                        uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                        step_size = std::max(step_size, uint32_t(1));
                        step_size = std::min(step_size, delta - 1);

                        std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                        ngl_per_device_test[id].n_layer += step_size;
                        if (part) {
                            ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                                step_size - 1 : step_size; // the first layer is the output layer which must always be full
                        }
                        const std::vector<int64_t> mem_test = get_memory_for_layers(fit_func, ngl_per_device_test, overflow_bufts);

                        if (mem_test[id] <= targets[id]) {
                            ngl_per_device = ngl_per_device_test;
                            mem            = mem_test;
                            LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", fit_func, id, ngl_per_device[id].n_layer);
                        } else {
                            ngl_per_device_high = ngl_per_device_test;
                            mem_high            = mem_test;
                            LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", fit_func, id, ngl_per_device_high[id].n_layer);
                        }
                        delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                    }
                } else {
                    assert(ngl_per_device_high[id].n_layer == n_unassigned);
                    ngl_per_device = ngl_per_device_high;
                    mem            = mem_high;
                    LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", fit_func, id, ngl_per_device[id].n_layer);
                }
            }

            const int64_t projected_margin = dmds_full[id].free - mem[id];
            LOG_TRC(
                "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
                fit_func, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
        }
    };
    // dense auto: fill whole layers first, then decide whether to refill with the FFN tensors first
    const bool kvr_auto = fit_dense_ffn && partial_layers && fit_dense_ffn_mode == DENSE_FFN_AUTO;
    fill_back_to_front(partial_layers && !kvr_auto);

    if (kvr_auto) {
        // auto: compare the device memory of the whole-layer placement just filled with every layer in device memory, the difference is
        // what it puts in system memory (counted on the device side: weights in a CPU buffer that is not a host buffer, e.g. repacked
        // weights, are not part of the system memory data)
        std::vector<ngl_t> ngl_ref(nd);
        ngl_ref[nd - 1].n_layer = hp_ngl + 1;
        const dmds_t dmds_ref  = get_dmds_for_layers(ngl_ref,        overflow_bufts);
        const dmds_t dmds_base = get_dmds_for_layers(ngl_per_device, overflow_bufts);
        int64_t kv_spill = 0;
        int64_t w_spill  = 0;
        for (size_t id = 0; id < nd; id++) {
            kv_spill += int64_t(dmds_ref[id].mb.context) - int64_t(dmds_base[id].mb.context);
            w_spill  += int64_t(dmds_ref[id].mb.model)   - int64_t(dmds_base[id].mb.model);
        }
        // an expected used context of at least half the context per sequence: FFN first whatever the ratio
        bool long_ctx = false;
        if (const char * env_ctx = getenv("LLAMA_FIT_KV_RESIDENT_CTX")) {
            const uint64_t n_ctx_used = strtoull(env_ctx, nullptr, 10) * n_streams;
            long_ctx = n_ctx_used > 0 && 2*n_ctx_used >= cparams->n_ctx;
        }
        const double ratio   = w_spill > 0 ? double(kv_spill) / double(w_spill) : 0.0;
        const bool   use_kvr = long_ctx || ratio < FIT_KV_RESIDENT_AUTO_MAX_KV_RATIO;
        uint32_t n_layer_dev = 0;
        for (size_t id = 0; id < nd; id++) {
            n_layer_dev += ngl_per_device[id].n_layer;
        }
        LOG_INF("%s: --fit-kv-resident auto: at a context of %" PRIu32 " the whole-layer placement (%" PRIu32 "/%" PRIu32 " layers) spills "
            "%" PRId64 " MiB of weights and %" PRId64 " MiB of KV cache (ratio %.3f, FFN first below %.3f%s) -> %s\n", __func__, cparams->n_ctx,
            n_layer_dev, hp_ngl + 1, w_spill/MiB, kv_spill/MiB, ratio, FIT_KV_RESIDENT_AUTO_MAX_KV_RATIO,
            long_ctx ? ", LLAMA_FIT_KV_RESIDENT_CTX: long context" : "",
            use_kvr ? "FFN tensors first (KV-resident)" : "whole layers");
        if (!use_kvr) {
            set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
            return;
        }
        ngl_per_device = std::vector<ngl_t>(nd);
        mem            = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);
        fill_back_to_front(true);
    }
    if (!partial_layers || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit (dense model: all tensors but the FFN tensors),
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else if (hp_nex > 0) { // for a dense model LAYER_FRACTION_ATTN is the placement of the other partial layers
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        const common_fit_extra_model * extra,
        ggml_log_level log_level,
        common_fit_kv_resident kv_resident) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    g_moe_margin_gain = 0;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, extra, log_level, kv_resident);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());

    // same data as the table below, for --log-jsonl consumers
    common_json rows = common_json::array();
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});

        rows.push_back({
            {"kind",        "device"},
            {"name",        name},
            {"description", desc},
            {"total",       total / MiB},
            {"free",        free / MiB},
            {"self",        self / MiB},
            {"model",       mb.model / MiB},
            {"context",     mb.context / MiB},
            {"compute",     mb.compute / MiB},
            {"unaccounted", unaccounted / static_cast<int64_t>(MiB)},
        });
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted

        rows.push_back({
            {"kind",    "host"},
            {"name",    "Host"},
            {"self",    self / MiB},
            {"model",   mb_host.model / MiB},
            {"context", mb_host.context / MiB},
            {"compute", mb_host.compute / MiB},
        });
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted

        rows.push_back({
            {"kind",    "buffer_type"},
            {"name",    name},
            {"self",    self / MiB},
            {"model",   mb.model / MiB},
            {"context", mb.context / MiB},
            {"compute", mb.compute / MiB},
        });

        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }

    LOG_JSON("fit_memory_breakdown", common_json({
        {"unit", "MiB"},
        {"rows", rows},
    }));
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
