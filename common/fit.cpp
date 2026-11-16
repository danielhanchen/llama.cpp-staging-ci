// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "fit.h"

#include "common.h"
#include "json.h"
#include "log.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <cstring>
#include <array>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/vfs.h>
#endif

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
// k19-kvsel: measurements count the K/V bytes of host chunks (mb.kv_sel) on their device, as without LLAMA_KV_SEL_HOST (see common_fit_impl)
static bool    g_fit_sel_as_dev  = false;

// k19-fitfaoff: the last measured context ran attention without flash attention (-fa off, or auto resolved off)
static bool    g_fit_fa_off      = false;
// k19-fitfaoff: bytes of one F16 copy of the largest KQ matrix of the last measured context without flash attention (0 with it)
static int64_t g_fit_faoff_kq    = 0;

bool common_fit_last_flash_attn() {
    return !g_fit_fa_off;
}

int64_t common_fit_moe_margin_fa_plan(int64_t m, int64_t m_reduced, bool flash_attn, bool keep_faoff) {
    return !flash_attn && keep_faoff ? m : m_reduced;
}

int64_t common_fit_faoff_kq_bytes(int64_t n_head, int64_t n_ubatch, int64_t n_ctx) {
    return std::max<int64_t>(0, n_head)*std::max<int64_t>(0, n_ubatch)*std::max<int64_t>(0, n_ctx)*2;
}

// k19-fitfaoff: -ub / LLAMA_ARG_UBATCH given (common_fit_set_ubatch_user): the fit keeps that n_ubatch without flash attention
static bool     g_fit_ub_user    = false;
// k19-fitfaoff: the smallest n_ubatch a fit without flash attention chose since the last reset (0: none chosen)
static uint32_t g_fit_faoff_ub   = 0;

void common_fit_set_ubatch_user(bool ub_user) {
    g_fit_ub_user = ub_user;
}

uint32_t common_fit_faoff_ubatch() {
    return g_fit_faoff_ub;
}

void common_fit_faoff_ubatch_reset() {
    g_fit_faoff_ub = 0;
}

uint32_t common_fit_faoff_ubatch_plan(uint32_t ub, uint32_t ub_floor, const std::function<bool(uint32_t)> & fits) {
    while (ub > ub_floor && !fits(ub)) {
        ub = std::max(ub_floor, ub/2);
    }
    return ub;
}

// LLAMA_FIT_FAOFF_UB: 1 (default) without flash attention and without -ub the fit lowers n_ubatch before moving layers, 0 = off
static bool common_fit_faoff_ub_on() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_FIT_FAOFF_UB");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

bool common_fit_faoff_pin_ubatch(bool ub_user, bool fa_off, bool keep, bool ub_on, bool has_gpu) {
    return !ub_user && fa_off && keep && ub_on && has_gpu;
}

// LLAMA_FIT_FAOFF_MARGIN: 1 (default) the full margin beside a MoE cache without flash attention, 0 = the reduced one as before
static bool common_fit_faoff_keep() {
    static const bool keep = [] {
        const char * e = getenv("LLAMA_FIT_FAOFF_MARGIN");
        return e == nullptr || atoi(e) != 0;
    }();
    return keep;
}

void common_fit_set_moe_margin_auto(bool auto_margin) {
    g_moe_margin_auto = auto_margin;
}

int64_t common_fit_moe_margin_gain() {
    return g_moe_margin_gain;
}

static int g_prewarm_graphs = 0;

void common_fit_set_prewarm_graphs(int n_graphs) {
    g_prewarm_graphs = std::max(0, n_graphs);
}

// k19-l4gap2: the rows of a speculative verify window (draft n_max + 1; 1 without a drafter), set before the fit
static int g_verify_rows = 1;

void common_fit_set_verify_rows(int n_rows) {
    g_verify_rows = std::max(1, n_rows);
}

static bool g_small_device = false;

bool common_fit_small_device() {
    return g_small_device;
}

// k19-place8: see the small-device branch of common_params_fit_impl. Skipped where the user set LLAMA_MOE_TINY_RELEASE_FORCE (the test
// knob of the same release) before the first fit; LLAMA_FIT_PLACE8=0 off; LLAMA_FIT_PLACE8_UB (31): the largest ubatch the decode-sized
// context runs beside the whole cache
static bool g_place8 = false;

bool common_fit_place8() {
    return g_place8;
}

// whether the model file lives on a memory file system (tmpfs / ramfs, Linux): its pages are in RAM whatever the available memory reads
bool common_fit_file_in_ram_fs(const std::string & path) {
#if defined(__linux__)
    struct statfs sfs;
    if (statfs(path.c_str(), &sfs) == 0) {
        const unsigned long t = (unsigned long) sfs.f_type;
        return t == 0x01021994UL || t == 0x858458f6UL; // TMPFS_MAGIC, RAMFS_MAGIC
    }
#else
    GGML_UNUSED(path);
#endif
    return false;
}

static bool common_fit_place8_on() {
    static const bool user_env = getenv("LLAMA_MOE_TINY_RELEASE_FORCE") != nullptr;
    static const bool on = getenv("LLAMA_FIT_PLACE8") == nullptr || atoi(getenv("LLAMA_FIT_PLACE8")) != 0;
    return on && !user_env;
}

// the default is one token below the batch from which the CUDA backend copies host weights to compute an op on the device
// (GGML_OP_OFFLOAD_MIN_BATCH, 32): a reserve at 32 tokens holds the expert copies of a split (G4 8 GB fn4: 1709 MiB of compute, the
// cache grew 65 -> 669 MiB only), one at 31 computes the experts its slots do not hold on the host
static uint32_t common_fit_place8_ub() {
    static const uint32_t ub = [] {
        if (const char * e = getenv("LLAMA_FIT_PLACE8_UB")) {
            return (uint32_t) std::max(1, atoi(e));
        }
        const char * e = getenv("GGML_OP_OFFLOAD_MIN_BATCH");
        return (uint32_t) std::max(1, (e != nullptr ? atoi(e) : 32) - 1);
    }();
    return ub;
}

// read by the context (llama_context, see tiny_rel_ub) when it is created: set for the loaded context, never for a measurement
// mid: k19-placemid (a device of the regular fit, not the small-device branch): the context keeps its layer-major prompt groups, the
// cache keeps multi-token hybrid batches (LLAMA_MOE_PLACE8_MID) and the draft model is fitted as usual
static bool g_place8_mid = false;

bool common_fit_place8_mid() {
    return g_place8 && g_place8_mid;
}

// grow_mib (mid only): the compute buffer memory the cache took, which a released layer-major group may grow back into
// groups: the k19-pgroups small device (LLAMA_MOE_PLACE8_GROUPS, read by the context), which keeps the layer-major groups of a mid-size
// device but not its short-prompt split or its routed refills (see llama_context, k19-pgroups2)
static void common_fit_place8_env(uint32_t ub, bool mid = false, int64_t grow_mib = 0, bool groups = false) {
    common_set_env("LLAMA_MOE_PLACE8_UB", ub > 0 ? std::to_string(ub) : "");
    common_set_env("LLAMA_MOE_PLACE8_MID", ub > 0 && mid ? std::to_string(std::max<int64_t>(1, grow_mib)) : "");
    common_set_env("LLAMA_MOE_PLACE8_GROUPS", ub > 0 && mid && groups ? "1" : "");
    g_place8     = ub > 0;
    g_place8_mid = ub > 0 && mid;
}

// k19-l4poly: a host-bound small device. A device with less than LLAMA_FIT_PLACE8_MIN_FREE_MIB (6144) free whose every layer fits
// within the default margin, but beside a MoE cache of at most LLAMA_FIT_HOSTBOUND_MAX_FRAC (0.10, the share of the hybrid gate,
// LLAMA_MOE_CACHE_HYBRID_MAX_FRAC) of the host experts: the host computes most misses of every decode step and a larger cache is what
// shortens it. Colab L4 capped at 4 GB, LFM2.5-24B-A2B UD-Q4_K_XL ctx 16384 (3696 MiB free, 13371 MiB of host experts): a 1157 MiB
// cache at the 1024 MiB margin, 663 MiB of it after the tiny-cache n_ubatch rule (117 slots, 4.8%), against PolyStrata's 322 experts
// (1.73 GiB, 352 MiB left free). Such a device takes the small-device placement (the 256 MiB margin + the run-time reserve) and its
// k19-place8 decode-sized cache. LLAMA_FIT_HOSTBOUND=0 off
static bool g_hostbound = false;

bool common_fit_hostbound() {
    return g_hostbound;
}

double common_fit_hostbound_max_frac() {
    static const double f = getenv("LLAMA_FIT_HOSTBOUND_MAX_FRAC") != nullptr ? std::max(0.0, atof(getenv("LLAMA_FIT_HOSTBOUND_MAX_FRAC"))) : 0.10;
    return f;
}

// the k19-place8 multi-token guard (LLAMA_MOE_PLACE8_SINGLE) is off on a host-bound device: k19-nanfix (the cause) is merged, and its
// verify windows otherwise run every expert on the host. Only set while the user has not set it, and taken back by the next fit
static void common_fit_hostbound_env(bool on) {
    static bool set_by_fit = false;
    if (on && getenv("LLAMA_MOE_PLACE8_SINGLE") == nullptr) {
        common_set_env("LLAMA_MOE_PLACE8_SINGLE", "0");
        set_by_fit = true;
    } else if (!on && set_by_fit) {
        common_set_env("LLAMA_MOE_PLACE8_SINGLE", "");
        set_by_fit = false;
    }
    g_hostbound = on;
}

static bool common_fit_hostbound_place(int64_t free_dev, int64_t cache, int64_t host_experts) {
    static const bool on = getenv("LLAMA_FIT_HOSTBOUND") == nullptr || atoi(getenv("LLAMA_FIT_HOSTBOUND")) != 0;
    static const int64_t max_free = int64_t(getenv("LLAMA_FIT_PLACE8_MIN_FREE_MIB") != nullptr ?
        std::max(0, atoi(getenv("LLAMA_FIT_PLACE8_MIN_FREE_MIB"))) : 6144)*1024*1024;
    return on && g_moe_margin_auto && host_experts > 0 && cache > 0 && free_dev < max_free &&
        double(cache) <= common_fit_hostbound_max_frac()*double(host_experts);
}

int64_t common_fit_small_reserve_mib(int64_t cache_mib) {
    static const int64_t reserve_mib = [] {
        const char * e = getenv("LLAMA_FIT_SMALL_RESERVE_MIB");
        return e != nullptr ? std::max<int64_t>(0, atoll(e)) : 128;
    }();
    static const int64_t floor_mib = [] {
        const char * e = getenv("LLAMA_FIT_SMALL_RESERVE_MIN_CACHE_MIB");
        return e != nullptr ? std::max<int64_t>(0, atoll(e)) : 64;
    }();
    return common_fit_small_reserve_plan(cache_mib, reserve_mib, floor_mib);
}

int64_t common_fit_small_reserve_plan(int64_t cache_mib, int64_t reserve_mib, int64_t floor_mib) {
    if (reserve_mib <= 0 || cache_mib <= floor_mib) {
        return 0;
    }
    return std::min(reserve_mib, cache_mib - floor_mib);
}

// k17-small (lazy KV): the MoE cache sizes a fit raised by the unbacked KV, and by how much (every fit of the process)
static std::vector<std::pair<size_t, size_t>> g_kv_lazy_grants;

size_t common_fit_kv_lazy_grant(size_t moe_cache_size) {
    for (const auto & [after, grant] : g_kv_lazy_grants) {
        if (after == moe_cache_size) {
            return grant;
        }
    }
    return 0;
}

// the margin of a device beside a MoE cache for the default margin m (see common_fit_set_moe_margin_auto)
static int64_t common_fit_moe_margin_impl(int64_t m, int64_t free_dev, int64_t cache_want, bool has_drafter);

// k19-fitfaoff (LLAMA_FIT_FAOFF_MARGIN=0 off): without flash attention the reduced margin beside a MoE cache is not taken. The
// attention then runs as KQ / KQV mat-muls whose F16 copies of the KQ matrix (n_kv x n_tokens x n_head) come from the CUDA pool at
// run time, outside the measured compute buffer; the reduced margin (256 MiB, gated with flash attention on) does not hold them.
// Kaggle 2x T4, DeepSeek-V4.1 fake dsf8, -fa off, --moe-cache-mib auto, ctx 16384: fit projected 1297 MiB free on CUDA1 at the 1024 MiB
// margin, the cache then took +768 MiB per device (+1536 total), and the first 2100-token prompt aborted with CUDA out of memory in
// the pool (cuMemCreate). The measured compute buffer itself matched the run (CUDA1 10768 MiB both)
// k19-q4cap (LLAMA_FIT_MOE_HEADLESS=0 off, =1 also on a card with a display): on a headless card (the CUDA backend's NVML display signal,
// as k19-fitplain) with a device budget of at least LLAMA_FIT_MOE_HEADLESS_MIN_GIB (10), the default 1024 MiB margin beside a MoE cache
// steps down to LLAMA_FIT_MOE_HEADLESS_MIB (512) where the rule above keeps it (beside a drafter or MTP heads, or a cache share over
// its ratio). Nothing else allocates on such a card, and from 10 GiB the host weight prefetch staging of the main context lives in lent
// cache memory (L4 12 GB: 354 MiB lent) instead of the margin; below that the staging (T4 8 GB: 386 MiB, depth 2) and the prompt
// graphs of the released banks need the margin: Colab T4 8 GB, Qwen3.6-35B-A3B UD-Q4_K_XL + MTP, margin 512: the main context's
// staging was not allocated (chat -5.7..-8%) and a 30k prompt aborted (3445 MiB compute buffer, cudaMalloc out of memory). Gains with
// it (Qwen3.6-35B-A3B UD-Q4_K_XL + MTP, 3 launches, chat / 8k decode): G4 @16 +4.7 / +2.3%, L4 12 GB +1.3 / +6.0% (30k +10.7%), L4 16 GB 0 / +0.8%, T4 12 GB
// +1.2 / +3.3%; KLD short192 / long48 inside the -fa off band (L4 12 GB, T4 8 GB). Cards with a display keep 1024 (other processes)
static int64_t common_fit_moe_margin_headless(int64_t m, int64_t r, int64_t free_dev, int headless) {
    constexpr int64_t MiB_ = 1024*1024;
    constexpr int64_t m_default = 1024*MiB_;
    const char * e = getenv("LLAMA_FIT_MOE_HEADLESS");
    const int mode = e != nullptr ? atoi(e) : -1;
    if (mode == 0 || r != m || !g_moe_margin_auto || m < m_default || getenv("LLAMA_MOE_FIT_MARGIN_MIB") != nullptr ||
            (getenv("LLAMA_MOE_FIT_MARGIN") != nullptr && atoi(getenv("LLAMA_MOE_FIT_MARGIN")) == 0)) {
        return r;
    }
    if (!(headless == 1 || mode == 1)) {
        return r;
    }
    const char * eg = getenv("LLAMA_FIT_MOE_HEADLESS_MIN_GIB");
    const int64_t min_free = (eg != nullptr ? std::max<int64_t>(0, atoll(eg)) : 10)*1024*MiB_;
    if (free_dev < min_free) {
        return r;
    }
    const char * em = getenv("LLAMA_FIT_MOE_HEADLESS_MIB");
    const int64_t keep = std::min<int64_t>(m_default, (em != nullptr ? std::max<int64_t>(0, atoll(em)) : 512)*MiB_);
    return m - m_default + keep;
}

static int64_t common_fit_moe_margin(int64_t m, int64_t free_dev, int64_t cache_want, bool has_drafter, int headless = -1) {
    const int64_t r0 = common_fit_moe_margin_impl(m, free_dev, cache_want, has_drafter);
    const int64_t r  = common_fit_moe_margin_headless(m, r0, free_dev, headless);
    if (r != r0) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_INF("%s: k19-q4cap: headless device: the margin beside the MoE cache steps down to %" PRId64 " MiB (LLAMA_FIT_MOE_HEADLESS=0: %" PRId64 " MiB)\n",
                __func__, r/(1024*1024), r0/(1024*1024));
        }
    }
    const int64_t k = common_fit_moe_margin_fa_plan(m, r, !g_fit_fa_off, common_fit_faoff_keep());
    if (k != r) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_INF("%s: flash attention off: the full margin beside the MoE cache stays (%" PRId64 " MiB, not %" PRId64 "; LLAMA_FIT_FAOFF_MARGIN=0 for the reduced one)\n",
                __func__, k/(1024*1024), r/(1024*1024));
        }
    }
    return k;
}

static int64_t common_fit_moe_margin_impl(int64_t m, int64_t free_dev, int64_t cache_want, bool has_drafter) {
    constexpr int64_t MiB_ = 1024*1024;
    constexpr int64_t m_default = 1024*MiB_;
    if (!g_moe_margin_auto || m < m_default) {
        return m;
    }
    // k17-models: on by default (256 MiB stay free beside the cache) where the device budget is at least LLAMA_MOE_FIT_MARGIN_MIN_GIB (10)
    // and the host experts the cache would hold are at most LLAMA_MOE_FIT_MARGIN_MAX_RATIO (1.3; k18: 1.8 cost Flint 16 GB (1.37) +71 / +181 ms
    // G4 / L4 chat TTFT) times that budget: a cache that holds a
    // small share of a large expert set stages many misses per prompt ubatch at run time and needs the margin.
    // Gate (k17m_l4mm / g4mm / g4mg / g4mh, chat to 30k prompts, log grep for allocation failures and fallbacks), experts / budget:
    //   pass with 256: LFM2.5 Q4_K_XL 12 GB (1.12), q4x + MTP 12 / 16 GB (1.58 / 1.17), Flint Q4_K_XL 16 GB (1.37),
    //     Flash-Next IQ1_S + MTP 16 / 24 GB (2.38 / 1.58), IQ4_XS + MTP 24 GB (2.37), q4x 8 GB (2.40)
    //   fail with 256: Flint 12 GB (1.85: no allocation failure, but its pinned text follows the cache size and short192 KLD is x1.03 over
    //     a one-point G4 band), Flint 8 GB (2.82, prefetch staging allocation failed), Flash-Next IQ4_XS 12 GB (4.78, CUDA out of memory abort)
    //     and 16 GB (3.55, 1275 MiB staging allocation failed)
    // Gain: LFM2.5 12 GB cache 9236 -> 10000 MiB, long48 +6..+9% (L4 76.4 -> 83.6, G4 257.4 -> 273.1), chat +6..+10%; q4x 12 GB 0..+12%.
    // LLAMA_MOE_FIT_MARGIN=0: the full margin everywhere; LLAMA_MOE_FIT_MARGIN_MIB keeps its meaning (below)
    if (getenv("LLAMA_MOE_FIT_MARGIN_MIB") == nullptr) {
        const char * en  = getenv("LLAMA_MOE_FIT_MARGIN");
        const char * eg  = getenv("LLAMA_MOE_FIT_MARGIN_MIN_GIB");
        const char * er  = getenv("LLAMA_MOE_FIT_MARGIN_MAX_RATIO");
        const int64_t mn = (eg != nullptr ? std::max<int64_t>(0, atoll(eg)) : 10)*1024*MiB_;
        const double  mr = er != nullptr ? std::max(0.0, atof(er)) : 1.3;
        // k18-margin: never beside a draft model or MTP heads (LLAMA_MOE_FIT_MARGIN_DRAFT=1 allows it). The drafter's runtime pools
        // (mul_mat_q, CUDA graphs) live in that slack once the MTP draft is priced exactly: k15 head b741bad76b7 + the rule aborted with
        // CUDA out of memory in every G4 q4x + MTP 12 / 16 GB launch (about 130 MiB free after the cache took +768 MiB), fn1 + MTP 24 GB
        // at 8k / 16k and 1 of 2 L4 q4x + MTP 12 GB launches; the kill switch and base had 0 aborts
        const char * ed = getenv("LLAMA_MOE_FIT_MARGIN_DRAFT");
        const bool draft_ok = ed != nullptr && atoi(ed) != 0;
        if ((en != nullptr && atoi(en) == 0) || free_dev < mn || (double) cache_want > mr*(double) free_dev || (has_drafter && !draft_ok)) {
            return m;
        }
        return m - m_default + 256*MiB_;
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

// k17-small (LLAMA_FIT_MTP_PRICE, default 1): the fit prices an MTP drafter as it runs. Once the draft attaches, the target's graphs
// also return the hidden state before the output norm (llama_set_embeddings_nextn), and the drafter's graphs return it for their
// output rows. Both re-reserve larger compute buffers that the measures did not have: Kaggle T4 4 GB, Qwen3.6-35B-A3B UD-Q4_K_XL + MTP
// at ctx 16384, the main context grew 474 -> 506 MiB and q3m 428 -> 466 MiB. g_fit_nextn = 1 while a fit for an MTP target measures
// the main model; an MTP context's own measure always has it (see below). The drafter without a MoE cache of its own (its share too
// small) also keeps one prefetch split in its compute, see sched_reserve. =0: the measures before k17-small, =1: on every device.
// Default: on where the server's post-draft floor is (an NVIDIA GPU below compute capability 8.0, see LLAMA_DRAFT_MIN_FREE_MIB), so
// that the fit rather than that floor decides the drafter there; off elsewhere. G4 (cc 12.0) runs q4x 4 GB 16k + MTP with 18.6 MiB
// free after the draft (decode 170 tok/s); the exact price would drop the drafter there for the 256 MiB margin (116 tok/s without it).
// The default is decided once and exported as LLAMA_FIT_MTP_PRICE, which the measure in sched_reserve reads
// an NVIDIA GPU below compute capability 8.0 among the devices
static bool common_fit_pre_ampere() {
    static const bool pre = [] {
        bool any = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            // an integrated GPU counts as the server's floor counts it (a pre-Ampere Jetson pairs the price with the floor as a T4 does)
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto cc_fn = reg ? (int (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
            const int cc = cc_fn != nullptr ? cc_fn(dev) : -1;
            any = any || (cc >= 0 && cc < 800);
        }
        return any;
    }();
    return pre;
}

static bool common_fit_mtp_price_on() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_FIT_MTP_PRICE");
        if (e != nullptr && *e != '\0') {
            return atoi(e) != 0;
        }
        const bool pre_ampere = common_fit_pre_ampere();
        // k17-small round 9: on every GPU by default. G4 4 GB, Qwen3.6-35B-A3B UD-Q3_K_M + MTP at ctx 32768 aborted in its first prompt
        // with the old price (2 of 2 launches, as base) and runs with the exact one (the fit drops the drafter: 143 tok/s); where the old
        // price kept the drafter and it ran (q4x 16k / 20k), the main context had lost its host weight prefetch to a failed allocation
        // after the draft attached (8k prompt 1662-1888 tok/s against 9200-9502 without the drafter, decode 11-19% higher with it).
        // LLAMA_FIT_MTP_PRICE_PRE_AMPERE_ONLY=1: only below compute capability 8.0 (054ee0034e5)
        const char * e_pa = getenv("LLAMA_FIT_MTP_PRICE_PRE_AMPERE_ONLY");
        const bool on = (e_pa != nullptr && atoi(e_pa) != 0) ? pre_ampere : true;
        common_set_env("LLAMA_FIT_MTP_PRICE", on ? "1" : "0");
        if (on) {
            LOG_INF("%s: an MTP drafter is priced as it runs (LLAMA_FIT_MTP_PRICE=0 off)\n", __func__);
        }
        return on;
    }();
    return on;
}
static thread_local int g_fit_nextn = 0;

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

    const bool mtp_price = common_fit_mtp_price_on(); // decided (and exported for sched_reserve) before the context is created

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
    g_fit_fa_off = !llama_get_flash_attn(ctx); // k19-fitfaoff: as resolved by the context (auto included)
    g_fit_faoff_kq = g_fit_fa_off ? common_fit_faoff_kq_bytes(llama_model_n_head(model), llama_n_ubatch(ctx), llama_n_ctx(ctx)) : 0;

    // k17-small: measure the graphs the context runs once an MTP draft is attached (unmasked for the target, masked for the drafter,
    // as common_speculative sets them)
    if (mtp_price) {
        const bool is_mtp = cparams->ctx_type == LLAMA_CONTEXT_TYPE_MTP;
        if (is_mtp || g_fit_nextn == 1) {
            llama_set_embeddings_nextn(ctx, true, /*masked =*/ is_mtp);
            llama_sched_reserve_now(ctx);
        }
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
                ret[i].mb.lend_freed += mb.lend_freed;
                ret[i].mb.kv_lazy    += mb.kv_lazy;
                ret[i].mb.kv_sel     += mb.kv_sel;
                break;
            }
        }
    }
    if (g_fit_sel_as_dev) {
        for (size_t i = 0; i < nd; i++) {
            ret[i].mb.context += ret[i].mb.kv_sel;
        }
    }

    // k15-fn: the staging arena a lending MoE cache takes over (mb.lend_freed) is memory the measure counted nowhere and the loaded
    // context does not take: count it off the device (compute first, then context), as the fit counts the arena of a context that
    // does not lend (not at all). fit_moe_cache_size gives it to the cache. Only with LLAMA_MOE_CACHE_LEND_SAVE=1 (nothing reported otherwise)
    for (size_t i = 0; i < nd; i++) {
        const size_t c = std::min(ret[i].mb.compute, ret[i].mb.lend_freed);
        ret[i].mb.compute -= c;
        ret[i].mb.context -= std::min(ret[i].mb.context, ret[i].mb.lend_freed - c);
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
    common_fit_hostbound_env(false); // k19-l4poly: decided again by every fit
    // k19-place8: every fit measures without the release a previous fit set up (the small-device branch decides it again)
    if (g_place8) {
        common_fit_place8_env(0);
    }
    // k17-small (lazy KV): a fit beside a drafter measures the KV cache as backed whole, as without LLAMA_KV_LAZY (the KV cache is then
    // allocated so, see common_init_result): measured lazily, G4 q3m 4 GB 16k + MTP got a 151 MiB cache against 152 without lazy KV;
    // a fit without one (none asked, or the trial that drops it) measures it lazily. LLAMA_KV_LAZY_FIT_DRAFT=1: lazily beside a drafter too
    {
        const char * e_kv = getenv("LLAMA_KV_LAZY");
        const char * e_fd = getenv("LLAMA_KV_LAZY_FIT_DRAFT");
        if (e_kv != nullptr && (strcmp(e_kv, "auto") == 0 || atoi(e_kv) != 0)) {   // k19-vramlend: LLAMA_KV_LAZY=auto or 1
            const char * v = extra != nullptr && !(e_fd != nullptr && atoi(e_fd) != 0) ? "1" : "0";
#if defined(_WIN32)
            _putenv_s("LLAMA_KV_LAZY_SUSPEND", v);
#else
            setenv("LLAMA_KV_LAZY_SUSPEND", v, 1);
#endif
        }
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

    // k17-small: the main model of a fit with an MTP context on its weights is measured as an MTP target (see g_fit_nextn)
    struct nextn_scope {
        int prev;
        explicit nextn_scope(bool on) : prev(g_fit_nextn) { if (on) { g_fit_nextn = 1; } }
        ~nextn_scope() { g_fit_nextn = prev; }
    } nextn_guard(extra != nullptr && extra->shares_model && common_fit_mtp_price_on());
    const bool has_drafter = extra != nullptr || mparams->load_mtp; // a draft model or MTP heads beside the target (k18-margin)

    dmds_t      dmds_extra;    // memory of the extra model, laid out on the devices of the main model
    std::string key_extra;     // context (and placement, see below) that memory was measured at

    // k16-fit: an extra model that shares the weights of the main model (an MTP context) runs them in the placement the fit gives the
    // main model, and takes its share of the main model's MoE cache budget (llama_moe_cache::budget, layer_set::nextn). Measured with
    // its own parameters (every weight on the device, no cache) it priced the MTP context of Qwen3.6-35B-A3B UD-Q4_K_XL at 323 MiB of
    // compute; with the experts in system memory it takes 714 MiB of compute (the staging of its host experts) and a 250 MiB cache share,
    // so the loaded drafter left 474 MiB free against the 1024 MiB target (40 MiB at the end of a run). It is measured in the placement
    // and at the cache size of the probe it is added to (one extra measure per distinct placement). LLAMA_FIT_EXTRA_FOLLOW=0: its own
    // parameters, as before
    static const bool extra_follow = getenv("LLAMA_FIT_EXTRA_FOLLOW") == nullptr || atoi(getenv("LLAMA_FIT_EXTRA_FOLLOW")) != 0;
    auto placement_key = [&](const llama_model_params * mp, const llama_context_params * cp) {
        std::string k = " ngl " + std::to_string(mp->n_gpu_layers) + " sm " + std::to_string(int(mp->split_mode)) +
            " mg " + std::to_string(mp->main_gpu) + " cache " + std::to_string(cp->moe_cache_size) + " ts";
        if (mp->tensor_split != nullptr) {
            for (size_t i = 0; i < llama_max_devices(); i++) {
                k += " " + std::to_string(mp->tensor_split[i]);
            }
        }
        k += " tbo";
        for (const llama_model_tensor_buft_override * o = mp->tensor_buft_overrides; o != nullptr && o->pattern != nullptr; o++) {
            k += std::string(" ") + o->pattern + "=" + (o->buft != nullptr ? ggml_backend_buft_name(o->buft) : "-");
        }
        return k;
    };

    // the extra model competes for the same memory as the main model, add it to every measurement
    // its memory is measured again whenever the context it follows changes (and, sharing the model, its placement or cache size)
    auto add_extra_memory = [&](dmds_t & dmds, const llama_model_params * mp_cur = nullptr, const llama_context_params * cp_cur = nullptr,
            bool decode_sized = false) {
        if (extra == nullptr) {
            return;
        }
        if (mp_cur == nullptr) {
            mp_cur = mparams;
        }
        if (cp_cur == nullptr) {
            cp_cur = cparams;
        }
        const bool follow = extra->shares_model && extra_follow;
        const std::string key = std::to_string(cparams->n_ctx) + (follow ? placement_key(mp_cur, cp_cur) : std::string());

        if (dmds_extra.empty() || key_extra != key) {
            std::vector<ggml_backend_dev_t> devs_extra;
            uint32_t ngl_extra = 0;
            uint32_t nct_extra = 0;
            uint32_t nex_extra = 0;

            extra->cparams->n_ctx = cparams->n_ctx;

            llama_model_params   mp_x = *extra->mparams;
            llama_context_params cp_x = *extra->cparams;
            if (follow) {
                mp_x.n_gpu_layers          = mp_cur->n_gpu_layers;
                mp_x.split_mode            = mp_cur->split_mode;
                mp_x.main_gpu              = mp_cur->main_gpu;
                mp_x.tensor_split          = mp_cur->tensor_split;
                mp_x.tensor_buft_overrides = mp_cur->tensor_buft_overrides;
                cp_x.moe_cache_size        = cp_cur->moe_cache_size;
            }

            LOG_TRC("%s: getting device memory data for the extra model at a context size of %" PRIu32 "%s:\n",
                __func__, cparams->n_ctx, follow ? ", in the placement of the main model" : "");

            dmds_t measured;
            bool estimated = false; // a DFlash draft priced from its GGUF (below)
            const int nextn_main = g_fit_nextn;
            g_fit_nextn = 0; // the extra model is not the target
            try {
                measured = common_get_device_memory_data_impl(
                    extra->path_model, &mp_x, &cp_x, devs_extra, ngl_extra, nct_extra, nex_extra, log_level);
                g_fit_nextn = nextn_main;
            } catch (const std::runtime_error & e) {
                g_fit_nextn = nextn_main;
                dmds_extra = dmds_t(devs.size() + 1);
                key_extra = key;
                // a DFlash / DSpark draft needs its target to be created: its estimate (weights, K/V, compute and the target's growth for
                // the layer inputs it extracts) goes to the first GPU of the main model. Fitted without it, the MoE cache of
                // Qwen3.8-Flash-Next at 32 GB kept its size and the main context's pp graph failed to allocate once the draft was loaded
                size_t w = 0, kv = 0, comp = 0;
                if (!extra->shares_model && mp_x.n_gpu_layers != 0 &&
                        common_fit_dflash_draft_bytes(extra->path_model, cp_x, path_model, cparams->n_ubatch, w, kv, comp)) {
                    for (size_t id = 0; id < devs.size(); id++) {
                        if (ggml_backend_dev_type(devs[id]) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                            dmds_extra[id].mb.model   += w;
                            dmds_extra[id].mb.context += kv;
                            dmds_extra[id].mb.compute += comp;
                            LOG_INF("%s: the extra model (a DFlash draft) is priced from its GGUF on %s: weights %.2f, K/V %.2f, compute %.2f MiB\n",
                                __func__, ggml_backend_dev_name(devs[id]), w/1048576.0, kv/1048576.0, comp/1048576.0);
                            estimated = true;
                            break;
                        }
                    }
                }
                if (!estimated) {
                    // the extra model is optional, fit the main model alone rather than giving up
                    LOG_WRN("%s: failed to measure the memory of the extra model, fitting without it: %s\n", __func__, e.what());
                    return;
                }
            }

            if (!estimated) {
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
            }

            key_extra = key;
        }

        // a lent extra model (common_fit_extra_model::lent) holds only its context while the main model's prompt ubatches run
        const bool lent = extra->lent && !decode_sized;
        for (size_t id = 0; id < dmds.size(); id++) {
            dmds[id].mb.model   += lent ? 0 : dmds_extra[id].mb.model;
            dmds[id].mb.context += dmds_extra[id].mb.context;
            dmds[id].mb.compute += lent ? 0 : dmds_extra[id].mb.compute;
        }
    };

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    dmds_t dmds_full;
    try {
        dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
    } catch (const std::runtime_error &) {
        // k18-fit-small: an explicit MoE cache too small for the experts of one token fails the probe context; the real context runs
        // without the cache (llama_context, LLAMA_MOE_CACHE_TOO_SMALL_FAIL=1 fails it), so fit what it runs: one warning instead of an
        // error line (T4 Granite 4.0-H tiny Q4_K_M --moe-cache-mib 4). Only when the same measure without the cache succeeds
        const char * e_fail = getenv("LLAMA_MOE_CACHE_TOO_SMALL_FAIL");
        if (cparams->moe_cache_size == 0 || (e_fail != nullptr && atoi(e_fail) != 0)) {
            throw;
        }
        const size_t size_asked = cparams->moe_cache_size;
        cparams->moe_cache_size = 0;
        try {
            dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        } catch (const std::runtime_error &) {
            cparams->moe_cache_size = size_asked;
            throw;
        }
        static bool warned = false; // the trial fits of common.cpp measure the same parameters
        if (!warned) {
            warned = true;
            LOG_WRN("%s: a MoE cache of %.2f MiB cannot hold the experts of one token: the host experts run without a MoE cache "
                "(LLAMA_MOE_CACHE_TOO_SMALL_FAIL=1 to fail instead)\n", __func__, size_asked/(1024.0*1024.0));
        }
    }

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

    // k19-fitfaoff (LLAMA_FIT_FAOFF_UB=0 off, LLAMA_FIT_FAOFF_MARGIN=0 both off): without flash attention and without -ub, the KQ copy
    // reserve below scales with n_ubatch, so the fit first halves n_ubatch (down to 512) while the whole model at that ubatch misses a
    // GPU's margin plus the reserve, and only then moves layers / experts to the host. Kaggle 2x T4 dsf8 -fa off ctx 16384: at -ub 2048
    // the 4096 MiB reserve put 8 of 9 layers on the host (4.5 / 2.5 tok/s chat / 8k), at -ub 512 (1024 MiB) 52.7 / 17.1 without an abort
    bool has_gpu = false;
    for (size_t id = 0; id < nd; id++) {
        has_gpu = has_gpu || ggml_backend_dev_type(devs[id]) == GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    if (common_fit_faoff_pin_ubatch(g_fit_ub_user, g_fit_fa_off, common_fit_faoff_keep(), common_fit_faoff_ub_on(), has_gpu)) {
        constexpr uint32_t ub_floor = 512;
        const uint32_t ub0 = cparams->n_ubatch;
        if (ub0 > ub_floor) {
            auto fits = [&](uint32_t ub) -> bool {
                llama_context_params cp = *cparams;
                cp.n_ubatch       = ub;
                cp.moe_cache_size = 0; // the whole model on its devices, as placed; the cache only takes what is left
                dmds_t d;
                try {
                    d = common_get_device_memory_data_impl(path_model, mparams, &cp, devs, hp_ngl, hp_nct, hp_nex, log_level);
                } catch (const std::runtime_error &) {
                    return false;
                }
                add_extra_memory(d, mparams, &cp);
                const int64_t kq = g_fit_faoff_kq;
                for (size_t id = 0; id < nd; id++) {
                    if (ggml_backend_dev_type(devs[id]) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                            int64_t(d[id].free) - int64_t(d[id].mb.total()) - margins[id] - kq < 0) {
                        return false;
                    }
                }
                return true;
            };
            const uint32_t ub = common_fit_faoff_ubatch_plan(ub0, ub_floor, fits);
            if (ub != ub0) {
                cparams->n_ubatch = ub;
                if (cparams->n_batch < ub) {
                    cparams->n_batch = ub;
                }
                dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                add_extra_memory(dmds_full);
                LOG_INF("%s: flash attention off: n_ubatch %u -> %u before moving layers to the host (the KQ copies scale with it; "
                    "LLAMA_FIT_FAOFF_UB=0 keeps it)\n", __func__, ub0, ub);
            }
        }
        g_fit_faoff_ub = g_fit_faoff_ub == 0 ? cparams->n_ubatch : std::min(g_fit_faoff_ub, cparams->n_ubatch);
    }

    // k19-fitfaoff (LLAMA_FIT_FAOFF_MARGIN=0 off): without flash attention the KQ / KQV mat-muls take F16 copies of the KQ matrix
    // (n_head x n_ubatch x n_kv) from the CUDA pool at run time; the measured compute buffer holds the F32 KQ but not those copies.
    // Each GPU's margin gets one copy at the full context. Kaggle 2x T4, DeepSeek-V4.1 fake dsf8, -fa off, ctx 16384, -ub 2048: with
    // the 1024 MiB margin (1297 MiB projected free on CUDA1) the 7986-token prompt still aborted in a pool allocation (cuMemCreate)
    if (g_fit_fa_off && g_fit_faoff_kq > 0 && common_fit_faoff_keep() && nd > 0) {
        for (size_t id = 0; id < nd; id++) {
            if (ggml_backend_dev_type(devs[id]) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                margins[id] += g_fit_faoff_kq;
            }
        }
        LOG_INF("%s: flash attention off: each GPU keeps %" PRId64 " MiB more free for the F16 copies of the KQ matrix (n_head x n_ubatch x "
            "n_ctx; LLAMA_FIT_FAOFF_MARGIN=0 off)\n", __func__, g_fit_faoff_kq/(1024*1024));
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
        int64_t pw_reserve = 0; // k19-int3fix2: prewarmed verify graphs beside a wide-graphs cache (common_fit_set_prewarm_graphs)
        auto margin_of = [&](size_t id) -> int64_t {
            return use_margins ? int64_t(margins[id]) + pw_reserve : 0;
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
            add_extra_memory(res, mp, cp);
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
        const std::vector<int> dev_headless = common_fit_devices_headless(devs); // k19-q4cap
        for (size_t id = 0; id < nd; id++) {
            if (use_margins && int64_t(dmds[id].mb.total()) > int64_t(dmds_nc[id].mb.total())) {
                margin_gain += int64_t(margins[id]) - common_fit_moe_margin(int64_t(margins[id]), int64_t(dmds[id].free),
                    int64_t(dmds[id].mb.total()) - int64_t(dmds_nc[id].mb.total()), has_drafter, id < dev_headless.size() ? dev_headless[id] : -1);
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

        // k19-int3fix2: the server's verify graph prewarm (server_spec_prewarm) runs beside a cache that holds at least the wide-graphs
        // share of the host experts (llama_moe_cache_wide_graphs) and keeps its graphs' device memory, which no measure here sees: A100
        // 80 GB with 48 GB free, Qwen3.8-Flash-Next + DFlash2 (n_max 6), 3 chains x 7 widths took 828 -> 64 MiB of the 1024 MiB margin
        // (22-66 MiB per graph), and the few-row MMQ graphs the rest (2 MiB, CUDA out of memory at the first request). That memory is
        // kept free here from the cache, estimated per graph (LLAMA_FIT_PREWARM_GRAPH_MIB, opt-in, e.g. 40); the server's free-memory
        // floor (LLAMA_SPEC_PREWARM_MIN_FREE_MIB, default 512) is the default guard
        if (use_margins && has_drafter && g_prewarm_graphs > 0) {
            // opt-in (default 0): measured at 48 GB the reserve lets the server prewarm all 21 graphs, and those windows replay slower
            // than the ones the 512 MiB floor leaves eager: G4 Flash-Next DFlash2 chat 254.5 tok/s with 40 vs 274.6 without (base
            // 262.2), A100 80 GB 110.7 vs 120.4 (base 115.6); both run (the reserve alone also kept the A100 alive with the floor off)
            static const double graph_mib = getenv("LLAMA_FIT_PREWARM_GRAPH_MIB") != nullptr ?
                std::max(0.0, atof(getenv("LLAMA_FIT_PREWARM_GRAPH_MIB"))) : 0.0;
            static const double wide_frac = getenv("LLAMA_MOE_CACHE_WIDE_GRAPHS_FRAC") != nullptr ?
                atof(getenv("LLAMA_MOE_CACHE_WIDE_GRAPHS_FRAC")) : 0.6;
            static const double wide_min = (getenv("LLAMA_MOE_CACHE_WIDE_GRAPHS_MIN_MIB") != nullptr ?
                std::max(0.0, atof(getenv("LLAMA_MOE_CACHE_WIDE_GRAPHS_MIN_MIB"))) : 32768.0)*1024.0*1024.0;
            const double  experts = (double) common_moe_expert_bytes(path_model);
            const int64_t need_pl = shortfall(dmds);
            const double  cache_e = need_pl == INT64_MAX ? 0.0 : (double) (std::min(size_requested, cache_bytes) - need_pl);
            if (graph_mib > 0.0 && wide_frac > 0.0 && experts > 0.0 && experts >= wide_min && cache_e >= wide_frac*experts) {
                pw_reserve = int64_t(graph_mib*(double) g_prewarm_graphs)*MiB;
                LOG_INF("%s: k19-int3fix2: %" PRId64 " MiB kept free from the MoE cache for %d prewarmed verify graphs (cache %.0f MiB = %.1f%% of "
                    "%.0f MiB of host experts; LLAMA_FIT_PREWARM_GRAPH_MIB=0: off)\n", __func__, pw_reserve/MiB, g_prewarm_graphs,
                    cache_e/MiB, 100.0*cache_e/experts, experts/MiB);
            }
        }
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
                // grow it by the measured surplus while it still fits. Opt-in (LLAMA_MOE_FIT_FILL=1): G4 k15f_g4, Qwen3.6 UD-Q4_K_XL + MTP,
                // +338 / +532 MiB of cache at 12 / 16 GB gave +7..+10% decode at 2k-16k (12 GB) and +-2% (16 GB), but a 42-token prompt's
                // TTFT went 165 -> 178 ms and 125 -> 139 ms; Flash-Next UD-IQ4_XS at 16 / 32 GB and LFM2.5 16 GB had no surplus to fill
                static const bool fill = getenv("LLAMA_MOE_FIT_FILL") != nullptr && atoi(getenv("LLAMA_MOE_FIT_FILL")) != 0;
                // k15-fn: where the cache lends because that frees memory (llama_moe_cache lend_saves), the measure counts the device
                // that much lower (mb.lend_freed). That is a fixed amount, not one that grows with the cache, so the first size, which
                // takes the whole difference to the measure without the cache as the cache's, misses it: grow by it (at most), as the
                // fill does by any surplus. Only with LLAMA_MOE_CACHE_LEND_SAVE=1 (opt-in, see llama_moe_cache lend_save)
                int64_t lent_left = 0;
                for (size_t id = 0; id < nd; id++) {
                    lent_left += int64_t(dmds_test[id].mb.lend_freed);
                }
                for (int g = 0; (fill || lent_left >= 64*MiB) && g < 3 && size < size_requested; g++) {
                    int64_t extra = std::min<int64_t>(surplus_of(dmds_test) - 16*MiB, size_requested - size);
                    if (!fill) {
                        extra = std::min(extra, lent_left);
                    }
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
                    LOG_TRC("%s: MoE cache grown by %" PRId64 " MiB into the measured surplus%s\n", __func__, extra/MiB,
                        fill ? "" : " the lent staging copies leave");
                    size      = size + extra;
                    dmds_test = dmds_grow;
                    lent_left -= extra;
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
        // k16-fit: --moe-cache-mib auto (an unbounded request) beside user host experts (--cpu-moe, -ot) where the whole cache fits:
        // every host expert gets a slot, so the request becomes the bytes the cache measures plus the surplus (the slots stop at all
        // experts anyway). Without this the request stayed unbounded, the fit returned "no changes needed" and
        // common_init_result dropped the cache (T4 gpt-oss-20b --cpu-moe auto on a free card: CPU experts at 14 tok/s against 41 with a
        // 4000 MiB cache). Only the user placement: the fit's own all-experts-on-the-host probe keeps the unbounded request to tell
        // "the cache fits whole" (then experts move to the device). LLAMA_FIT_MOE_CACHE_ALLFIT=0 keeps the old behaviour
        const char * e_allfit = getenv("LLAMA_FIT_MOE_CACHE_ALLFIT");
        const bool   allfit   = e_allfit == nullptr || atoi(e_allfit) != 0;
        if (allfit && cparams->moe_cache_size >= (size_t(1) << 40)) { // COMMON_MOE_CACHE_AUTO_BYTES: not shrunk, so the whole cache fits
            llama_context_params cparams_nc = *cparams;
            cparams_nc.moe_cache_size = 0;
            LOG_TRC("%s: getting device memory data without the MoE cache (the whole cache fits):\n", __func__);
            dmds_t dmds_nc = common_get_device_memory_data_impl(path_model, mparams, &cparams_nc, devs, hp_ngl, hp_nct, hp_nex, log_level);
            add_extra_memory(dmds_nc, mparams, &cparams_nc);
            int64_t cache_bytes = 0;
            int64_t surplus     = 0;
            for (size_t id = 0; id < nd; id++) {
                cache_bytes += int64_t(dmds_full[id].mb.total()) - int64_t(dmds_nc[id].mb.total());
                surplus     += int64_t(dmds_full[id].free) - int64_t(dmds_full[id].mb.total()) - int64_t(margins[id]);
            }
            // the request also carries the surplus beyond the targets: common_moe_cache_auto_ubatch funds a larger n_ubatch from the
            // request, and with the bare cache bytes it took them from the slots (T4 gpt-oss-20b k16fit_t4a: 9783 -> 9421 MiB, 9417 of
            // 9720 MiB cached). The cache stops at every expert, so the device use stays the projection's plus what n_ubatch takes
            if (cache_bytes > 0) {
                cparams->moe_cache_size = size_t(cache_bytes + std::max<int64_t>(surplus, 16*MiB));
                LOG_INF("%s: --moe-cache-mib auto: every host expert fits in device memory, MoE cache of %" PRId64 " MiB, "
                    "request %zu MiB with the surplus beyond the target\n", __func__, cache_bytes/MiB, cparams->moe_cache_size/MiB);
            }
        }
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
            add_extra_memory(dmds_spill, &mparams_spill, &cparams_spill);
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
        add_extra_memory(dmd_nl, &mparams_copy, cparams);
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
            // k19-l4poly: a host-bound small device (see common_fit_hostbound_place): every layer fits within the default margin, but
            // beside a cache of at most a tenth of the host experts, so the host computes most misses of every token. It takes the
            // small-device placement below (smaller margin, the k19-place8 decode-sized cache) although it fits
            const size_t  size_fit_hb = cparams->moe_cache_size;
            const int64_t gain_fit_hb = g_moe_margin_gain;
            // k19-kvsel: K/V cells in host chunks (LLAMA_KV_SEL_HOST=1) leave device memory free that the cache should get, but a device
            // that fits only because of them would take the regular path at the default margin (G4 10 GB, Qwen3.8-Flash-Next ctx 131K
            // q8_0: of 1200 MiB freed, k19-placemid gave the cache 560). The path is chosen as without host chunks: the cache is sized
            // once more with their bytes counted on the device, and where that leaves no cache the small-device placement below runs on
            // the real measurement, so the cache gets what the host chunks free. LLAMA_KV_SEL_FIT=0: the path of the real measurement
            bool sel_small = false;
            if (fits_per_device && nd == 1 && size_before > 0 && dmds_cpu_moe_req[0].mb.kv_sel > 0) {
                static const bool sel_fit_on = getenv("LLAMA_KV_SEL_FIT") == nullptr || atoi(getenv("LLAMA_KV_SEL_FIT")) != 0;
                if (sel_fit_on) {
                    dmds_t d_dev = dmds_cpu_moe_req;
                    for (size_t id = 0; id < nd; id++) {
                        d_dev[id].mb.context += d_dev[id].mb.kv_sel;
                    }
                    cparams->moe_cache_size = size_before;
                    g_fit_sel_as_dev = true;
                    fit_moe_cache_size(d_dev, mparams, true);
                    g_fit_sel_as_dev = false;
                    bool fits_dev = cparams->moe_cache_size > 0;
                    for (size_t id = 0; id < nd; id++) {
                        fits_dev = fits_dev && int64_t(d_dev[id].free) - int64_t(d_dev[id].mb.total()) >= margins[id];
                    }
                    sel_small = !fits_dev;
                    LOG_INF("%s: kv sel: %zu MiB of K/V in host chunks; counted on the device the cache would be %zu MiB (%s), so the %s "
                        "placement runs (LLAMA_KV_SEL_FIT=0: the regular one)\n", __func__, dmds_cpu_moe_req[0].mb.kv_sel/MiB,
                        cparams->moe_cache_size/MiB, fits_dev ? "fits" : "does not fit", sel_small ? "small-device" : "regular");
                    cparams->moe_cache_size = size_fit_hb;
                    g_moe_margin_gain       = gain_fit_hb;
                }
            }
            // not beside a draft model or MTP heads (as the k18 margin rule): their run-time pools were never measured at this margin
            const bool    hostbound   = fits_per_device && nd == 1 && size_before > 0 && !has_drafter &&
                common_fit_hostbound_place(int64_t(dmds_cpu_moe[0].free), int64_t(size_fit_hb),
                    int64_t(dmds_cpu_moe.back().mb.model) - int64_t(dmds_full.back().mb.model));
            // k13-small: a small device (G4 ballasted to 4 GB, Qwen3.6-35B-A3B Q3_K_M + MTP: the dense weights of every layer, the context and
            // a cache fit only inside the 1024 MiB margin). Without this the fit moves 13 of 41 whole layers (attention, KV, recurrent state)
            // to the CPU and drops the cache, so every token pays a graph split per CPU layer and per MoE layer (k12: 91 tok/s against
            // PolyStrata's 177, which keeps every dense layer on the device). Here the margin of the one device drops to
            // LLAMA_FIT_SMALL_MARGIN_MIB (256) instead: every layer stays on the device beside a small cache, whose host misses the
            // single-graph path computes (no graph split). The placement is the one of -fitt 256; one GPU only, default margin only (an
            // explicit smaller -fitt is the user's), LLAMA_FIT_SMALL_MARGIN_MIB=0 keeps the k12 placement.
            if ((!fits_per_device || hostbound || sel_small) && nd == 1 && size_before > 0) {
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
                // k17-small: with the MTP drafter priced as it runs (common_fit_mtp_price_on) the second margin is 160 MiB by default
                // (LLAMA_FIT_MTP_SMALL_MARGIN_MIB). The 256 MiB margin was tuned with a drafter priced 129 + 32 MiB short, so at the exact price
                // it leaves no cache beside the drafter and the fit drops it where it runs: Colab T4 4 GB, Qwen3.6-35B-A3B UD-Q4_K_XL + MTP at
                // ctx 16384 drafted with 126.9 MiB free after load at the old price (95 MiB once the target grew for the drafter), while a
                // Kaggle T4 at 74.8 MiB (about 43 at the first prompt; its driver keeps ~53 MiB more) ran out of memory. 160 MiB priced
                // exactly leaves ~160 on the Colab host and ~107 on the Kaggle one at the first prompt, the cache taking the rest
                const bool mtp_priced = extra != nullptr && extra->shares_model && common_fit_mtp_price_on();
                const char * e_sm3 = getenv("LLAMA_FIT_MTP_SMALL_MARGIN_MIB");
                // only below compute capability 8.0 (the T4 hosts it was measured on): G4 (cc 12.0) 4 GB q4x 16k at 160 MiB kept the
                // drafter with 130.6 MiB free after it, the target grew 474 -> 506 MiB, and the first prompt ran out of memory creating a
                // cuBLAS handle (rs_replay_commit, 2 of 2 launches); at 256 the fit drops the drafter there and every cell runs (k17s_g4i)
                const int64_t mtp_min_mib = e_sm3 != nullptr ? std::max<int64_t>(0, atoll(e_sm3)) : (common_fit_pre_ampere() ? 160 : 0);
                const char * e_sm2 = getenv("LLAMA_FIT_SMALL_MARGIN_MIN_MIB");
                const int64_t small_min_mib = std::min<int64_t>(small_mib, e_sm2 != nullptr ? std::max<int64_t>(0, atoll(e_sm2)) :
                    (mtp_priced && mtp_min_mib > 0 ? mtp_min_mib : small_mib));
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
                    // k19-kvwin6: beside a KV window ring (whose prompt ubatches need more compute than its decode), no cache may fit at
                    // the full n_ubatch while every layer does with none: k19-place8 below is then tried from a zero cache. Its safety is
                    // that of the placement it starts from: every layer and the full-n_ubatch compute buffers fit within the small margin,
                    // so a released prompt ubatch has the memory it was measured with; the cache only takes the difference to the graphs
                    // reserved at its small ubatches. Without the k19-place8 cache the placement of before stays.
                    // LLAMA_FIT_PLACE8_WIN=0 off
                    bool p8_zero = false;
                    const char * e_kw = getenv("LLAMA_KV_WIN");
                    const char * e_ks = getenv("LLAMA_KV_WIN_SINKS");
                    const int    kw_n = e_kw != nullptr ? atoi(e_kw) : (e_ks != nullptr ? std::max(0, cparams->kv_win) : cparams->kv_win);
                    const int    ks_n = e_ks != nullptr ? std::max(0, atoi(e_ks)) : cparams->kv_win_sinks;
                    const bool kv_win_on = kw_n > 0 || (kw_n == 0 && ks_n > 0); // as llama_kv_win_params_from
                    {
                        static const bool p8w_on = getenv("LLAMA_FIT_PLACE8_WIN") == nullptr || atoi(getenv("LLAMA_FIT_PLACE8_WIN")) != 0;
                        if (!ok && p8w_on && kv_win_on && small_mib > 0 && common_fit_place8_on() && cparams->n_ubatch > common_fit_place8_ub()) {
                            margins[0] = margin_saved - m_default + small_mib*MiB;
                            cparams->moe_cache_size = 0;
                            dmds_t d0;
                            bool   d0_ok = true;
                            try {
                                d0 = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                            } catch (const std::runtime_error &) {
                                d0_ok = false;
                            }
                            if (d0_ok) {
                                add_extra_memory(d0);
                            }
                            if (d0_ok && int64_t(d0[0].free) - int64_t(d0[0].mb.total()) >= margins[0]) {
                                dmds_cpu_moe      = d0;
                                g_moe_margin_gain = 0;
                                m_ok              = small_mib;
                                ok                = true;
                                p8_zero           = true;
                            } else {
                                margins[0]              = margin_saved;
                                dmds_cpu_moe            = dmds_nc_saved;
                                cparams->moe_cache_size = size_before;
                            }
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
                    // k19-tierfix-fit: the run-time reserve (common_fit_small_reserve_mib) is added to the reduced margin, taken from the
                    // cache down to its floor; the placement of before is kept where that one does not fit. Only at the first reduced
                    // margin: a placement that needed the second one (LLAMA_FIT_SMALL_MARGIN_MIN_MIB, the MTP margin) has no room for it
                    if (ok && m_ok == small_mib && !p8_zero) {
                        const int64_t extra_mib = common_fit_small_reserve_mib(int64_t(cparams->moe_cache_size/MiB));
                        if (extra_mib > 0) {
                            const size_t  size_ok   = cparams->moe_cache_size;
                            const dmds_t  dmds_ok   = dmds_cpu_moe;
                            const int64_t gain_ok   = g_moe_margin_gain;
                            const int64_t margin_ok = margins[0];
                            if (try_fit(size_before, dmds_cpu_moe_req, m_ok + extra_mib)) {
                                LOG_INF("%s: small device: %" PRId64 " MiB of the MoE cache kept free for the run-time buffers (CUDA pool, "
                                    "graphs, streams): cache %zu -> %zu MiB (LLAMA_FIT_SMALL_RESERVE_MIB=0 keeps the cache)\n", __func__,
                                    extra_mib, size_ok/MiB, cparams->moe_cache_size/MiB);
                                m_ok += extra_mib;
                            } else {
                                cparams->moe_cache_size = size_ok;
                                dmds_cpu_moe            = dmds_ok;
                                g_moe_margin_gain       = gain_ok;
                                margins[0]              = margin_ok;
                            }
                        }
                    }
                    // k19-place8: the compute buffer is sized for the largest prompt ubatch (G4 at 8 GB, Qwen3.8-Flash-Next UD-IQ4_XS with -ub
                    // 2048: 2312 MiB, 1275 of them the staging arena of the prefetched expert copies) beside a cache of a few dozen slots
                    // (65 MiB, ~1% decode hits), and it sits idle during decode. Here the cache takes that memory instead: the context
                    // reserves its graphs at LLAMA_FIT_PLACE8_UB (31) tokens, small batches whose missing experts run on the host, and a larger ubatch runs with the
                    // cache banks released while its compute buffers grow into their memory; the next small ubatch frees the compute buffers
                    // and refills the banks with the experts they held (the k19-tiny-ub release, llama_context::tiny_rel_enter / leave,
                    // byte-exact at a fixed placement). The cache grows by the measured difference between the two placements, so a released
                    // prompt uses what the fit measured for it. Kept only if the larger cache meets the margin; otherwise the placement above
                    // stays. One device only (the small-device rule). LLAMA_FIT_PLACE8=0 off
                    // measured on G4 8 GB only: devices with less free memory than LLAMA_FIT_PLACE8_MIN_FREE_MIB (6144) keep the old placement
                    static const int64_t place8_min_free = int64_t(getenv("LLAMA_FIT_PLACE8_MIN_FREE_MIB") != nullptr ?
                        std::max(0, atoi(getenv("LLAMA_FIT_PLACE8_MIN_FREE_MIB"))) : 6144)*MiB;
                    bool p8_got = false;
                    if (ok && common_fit_place8_on() && cparams->n_ubatch > common_fit_place8_ub() &&
                            (int64_t(dmds_cpu_moe[0].free) >= place8_min_free || hostbound)) {
                        const uint32_t ub_full = cparams->n_ubatch;
                        const uint32_t ub_rel  = common_fit_place8_ub();
                        const size_t   size_ok = cparams->moe_cache_size;
                        const dmds_t   dmds_ok = dmds_cpu_moe;
                        bool           got     = false;
                        int64_t        grow    = 0;
                        cparams->n_ubatch = ub_rel;
                        common_fit_place8_env(ub_rel); // the cache of a measurement reads the gate of its small batches (place8_single)
                        auto measure8 = [&](size_t cache, dmds_t & d) {
                            cparams->moe_cache_size = cache;
                            try {
                                d = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                            } catch (const std::runtime_error &) {
                                return false;
                            }
                            add_extra_memory(d, nullptr, nullptr, /*decode_sized =*/ true);
                            return true;
                        };
                        dmds_t d_s;
                        // k19-kvwin6: context that grows with n_ubatch (the KV window ring holds n_ubatch slots beside its window) keeps the
                        // size of the full n_ubatch at run time (only the graphs are reserved at ub_rel): not part of the growth, added back below
                        int64_t ctx_ub = 0;
                        if (measure8(size_ok, d_s)) {
                            ctx_ub = kv_win_on ? std::max<int64_t>(0, int64_t(dmds_ok[0].mb.context) - int64_t(d_s[0].mb.context)) : 0;
                            grow   = int64_t(dmds_ok[0].mb.total()) - int64_t(d_s[0].mb.total()) - ctx_ub;
                        }
                        // k19-l4poly: a host-bound device takes even a smaller difference (LLAMA_FIT_HOSTBOUND_PLACE8_MIN_MIB, 256 as elsewhere)
                        static const int64_t hb_min = int64_t(getenv("LLAMA_FIT_HOSTBOUND_PLACE8_MIN_MIB") != nullptr ?
                            std::max(1, atoi(getenv("LLAMA_FIT_HOSTBOUND_PLACE8_MIN_MIB"))) : 256)*MiB;
                        if (grow >= (hostbound ? hb_min : 256*MiB)) {
                            int64_t req = int64_t(size_ok) + grow;
                            for (int it = 0; it < 3 && req >= int64_t(size_ok) + 128*MiB; it++) {
                                dmds_t d8;
                                if (!measure8(size_t(req), d8)) {
                                    break;
                                }
                                const int64_t s8 = int64_t(d8[0].free) - int64_t(d8[0].mb.total()) - ctx_ub - margins[0];
                                LOG_TRC("%s: k19-place8: ubatches of up to %u tokens beside a cache of %" PRId64 " MiB: compute %zu MiB, %" PRId64
                                    " MiB over the margin\n", __func__, ub_rel, req/MiB, d8[0].mb.compute/MiB, s8/MiB);
                                if (s8 >= 0) {
                                    d8[0].mb.context += size_t(ctx_ub);
                                    dmds_cpu_moe = d8;
                                    got = true;
                                    break;
                                }
                                req += s8 - 16*MiB;
                            }
                        }
                        cparams->n_ubatch = ub_full;
                        p8_got = got;
                        if (got) {
                            LOG_INF("%s: small device: the MoE cache takes the memory of the prompt compute buffers (%" PRId64 " MiB): cache %zu -> "
                                "%zu MiB; ubatches of more than %u tokens run with the cache banks released (k19-place8; LLAMA_FIT_PLACE8=0 off)\n",
                                __func__, grow/MiB, size_ok/MiB, cparams->moe_cache_size/MiB, ub_rel);
                            // k19-pgroups: prompts keep their layer-major groups beside the decode-sized cache, as k19-placemid does on a
                            // mid-size device: a group releases the banks for its merged graph (one pass over each layer's experts for up to
                            // LLAMA_MOE_LAYER_MAJOR_UB_MAX ubatches) instead of every prompt ubatch running alone with the banks released (G4 10 /
                            // 12 GB, Flash-Next UD-IQ4_XS ctx 131K / 262K: 493 tok/s prompts vs 3669 with LLAMA_FIT_PLACE8=0, which loses 20-25%
                            // decode with its 160 MiB cache). The cache, its size and the reserved graphs stay those of k19-place8; the draft rule of
                            // the small device stays (no device drafter). Devices with at least LLAMA_FIT_PLACE8_GROUPS_MIN_FREE_MIB (9216) free
                            // (8 GB cards keep k19-place8 as it was), not host-bound. LLAMA_FIT_PLACE8_GROUPS=0 off
                            static const bool groups_on = getenv("LLAMA_FIT_PLACE8_GROUPS") == nullptr || atoi(getenv("LLAMA_FIT_PLACE8_GROUPS")) != 0;
                            static const int64_t groups_min_free = int64_t(getenv("LLAMA_FIT_PLACE8_GROUPS_MIN_FREE_MIB") != nullptr ?
                                std::max(0, atoi(getenv("LLAMA_FIT_PLACE8_GROUPS_MIN_FREE_MIB"))) : 9216)*MiB;
                            if (groups_on && !hostbound && int64_t(dmds_cpu_moe[0].free) >= groups_min_free) {
                                common_fit_place8_env(ub_rel, true, grow/MiB, true);
                                g_place8_mid = false; // the small device's draft rule (common_fit_draft_on_cpu) stays
                                LOG_INF("%s: small device: prompts keep their layer-major groups beside the decode-sized cache, each group releases "
                                    "the banks for its merged graph (k19-pgroups; LLAMA_FIT_PLACE8_GROUPS=0 off)\n", __func__);
                            }
                        } else {
                            common_fit_place8_env(0);
                            cparams->moe_cache_size = size_ok;
                            dmds_cpu_moe            = dmds_ok;
                            LOG_INF("%s: small device: the MoE cache keeps its size (k19-place8: %" PRId64 " MiB of prompt compute buffers, no "
                                "larger cache within the margin)\n", __func__, grow/MiB);
                        }
                    }
                    if (p8_zero && !p8_got) {
                        ok = false; // the zero-cache start only stands with the k19-place8 cache: the placement of before (below)
                    } else if (p8_zero) {
                        LOG_INF("%s: small device beside the KV window ring: every layer fits at the full n_ubatch without a cache, the k19-place8 "
                            "cache takes the prompt compute memory (LLAMA_FIT_PLACE8_WIN=0 off)\n", __func__);
                    }
                    if (ok && hostbound) {
                        LOG_INF("%s: small device, host-bound: every layer fits within the margin of %" PRId64 " MiB beside a MoE cache of %zu MiB "
                            "(at most %.0f%% of the host experts); the margin drops to %" PRId64 " MiB and the cache grows to %zu MiB (k19-l4poly; "
                            "LLAMA_FIT_HOSTBOUND=0 off)\n", __func__, margin_saved/MiB, size_fit_hb/MiB, 100.0*common_fit_hostbound_max_frac(),
                            margins[0]/MiB, cparams->moe_cache_size/MiB);
                        g_small_device = true;
                        common_fit_hostbound_env(true);
                    } else if (ok) {
                        LOG_INF("%s: small device: every layer stays in device memory beside a MoE cache of %zu MiB, the fit margin drops from "
                            "%" PRId64 " to %" PRId64 " MiB (otherwise layers move to the CPU; LLAMA_FIT_SMALL_MARGIN_MIB=0 keeps that)\n",
                            __func__, cparams->moe_cache_size/MiB, margin_saved/MiB, margins[0]/MiB);
                        fits_per_device = true;
                        g_small_device  = true;
                    } else {
                        // k19-l4poly: a host-bound device that fitted keeps that placement (k19-kvsel: as does one with host chunks)
                        common_fit_place8_env(0);
                        margins[0] = margin_saved;
                        dmds_cpu_moe = dmds_nc_saved;
                        cparams->moe_cache_size = hostbound || sel_small ? size_fit_hb : 0;
                        g_moe_margin_gain = hostbound || sel_small ? gain_fit_hb : 0;
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
                add_extra_memory(dmds_all, &mparams_all, cparams);
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
                // k19-placemid: the k19-place8 placement on a mid-size device (G4 12 / 16 GB, Qwen3.8-Flash-Next UD-IQ4_XS ctx 32768 -ub 2048:
                // a 2264 MiB compute buffer, 1275 MiB of it the staging arena of the prefetched expert copies, beside a cache of 3068 MiB,
                // idle during decode). The graphs are reserved at LLAMA_FIT_PLACE8_UB (31) tokens and the cache takes the difference; prompts
                // keep their layer-major groups, which release the banks for the merged graph (as they already do where it does not fit),
                // and a prompt ubatch outside a group releases them as at 8 GB. Only where the freed memory is at least
                // LLAMA_FIT_PLACEMID_MIN_FRAC (0.10) of the cache: at 24 / 48 GB the staging copies are already lent from the cache (989 MiB
                // compute) and every prompt would refill tens of GB of banks. One device; LLAMA_FIT_PLACEMID=0 (or LLAMA_FIT_PLACE8=0) off
                static const bool placemid_on = getenv("LLAMA_FIT_PLACEMID") == nullptr || atoi(getenv("LLAMA_FIT_PLACEMID")) != 0;
                static const double placemid_min_frac = getenv("LLAMA_FIT_PLACEMID_MIN_FRAC") != nullptr ?
                    std::max(0.0, atof(getenv("LLAMA_FIT_PLACEMID_MIN_FRAC"))) : 0.10;
                // k19-l4gap (opt-in, LLAMA_FIT_PLACEMID_DISK_SKIP=1): no k19-placemid on a disk tier whose cache holds the experts of fewer
                // than LLAMA_FIT_PLACEMID_DISK_MIN_TOKENS (4, the RAM tier's device minimum, LLAMA_MOE_HOST_MIN_TOKENS) tokens before the
                // growth: the host experts do not fit in RAM (more than 0.8 of the available memory, the rule of
                // llama_context::disk_prefetch_init, the file not on a memory file system), and the short prompts that k19-placemid runs as
                // ubatches of up to 31 tokens exceed the few tokens the device lookups serve (4 there), so their expert products run on the
                // CPU from the mapping, whose pages the disk serves at its read rate. With the compute buffers kept a prompt ubatch copies its
                // routed experts from the RAM tier instead. Colab L4 12 GB, Qwen3.8-Flash-Next UD-IQ4_XS from the VM disk (57.6 GB of
                // experts, 51 GB of RAM) + MTP n2, 3 interleaved launches (plans/k19_l4gap.md): 42-token chat TTFT 28.1 -> 17.1 s, 8k TTFT
                // 90.7 -> 85.4 s, but chat decode 5.94 -> 5.59 tok/s (the cache keeps 2256 instead of 4173 MiB): answers longer than ~1000
                // tokens get slower, so it stays opt-in. Unset: k19-placemid as before (no extra work, no log line)
                // k19-l4gap2 (default; LLAMA_FIT_PLACEMID_DISK_UB=0 off): on the same disk tier k19-placemid keeps its cache but reserves the
                // graphs at, and splits short prompts into, ubatches of LLAMA_FIT_PLACEMID_DISK_UB (4) tokens: the tokens the device lookups
                // serve there, so a short prompt's experts come from the device slots and the RAM tier instead of the CPU products of the
                // mapping, and the decode keeps the larger cache. Colab L4 12 GB, same model, held-out texts, 3 interleaved launches
                // (plans/k19_l4gap.md section 15d): chat TTFT -25%, chat decode +15%, a second chat TTFT -60%, a 1536-token thinking answer
                // decode +2%; the skip above lost 9% on that answer. Only where a speculative verify window has at most that many rows (MTP
                // n2: 3), set by common_fit_set_verify_rows: a wider one would release and refill the banks in every pass
                static const bool placemid_disk_skip = getenv("LLAMA_FIT_PLACEMID_DISK_SKIP") != nullptr && atoi(getenv("LLAMA_FIT_PLACEMID_DISK_SKIP")) != 0;
                static const int  placemid_disk_ub   = getenv("LLAMA_FIT_PLACEMID_DISK_UB") != nullptr ? std::max(0, atoi(getenv("LLAMA_FIT_PLACEMID_DISK_UB"))) : 4;
                bool     placemid_ram = true;
                uint32_t ub_disk      = 0;
                if ((placemid_disk_skip || placemid_disk_ub > 0) && placemid_on && cparams->moe_cache_size > 0) {
                    static const double placemid_disk_tokens = getenv("LLAMA_FIT_PLACEMID_DISK_MIN_TOKENS") != nullptr ?
                        std::max(0.0, atof(getenv("LLAMA_FIT_PLACEMID_DISK_MIN_TOKENS"))) : 4.0;
                    const uint64_t exp_b  = common_moe_expert_bytes(path_model);
                    const uint64_t avail  = llama_available_memory();
                    const double   tokens = common_moe_cache_tokens(path_model, cparams->moe_cache_size);
                    if (exp_b > 0 && avail > 0 && (double) exp_b > 0.8*(double) avail && tokens >= 0.0 && tokens < placemid_disk_tokens &&
                            !common_fit_file_in_ram_fs(path_model)) {
                        if (placemid_disk_skip) {
                            placemid_ram = false;
                            LOG_INF("%s: k19-l4gap: no k19-placemid (LLAMA_FIT_PLACEMID_DISK_SKIP=1): %.0f MiB of host experts do not fit in %.0f MiB of "
                                "available RAM and the cache of %zu MiB holds the experts of %.1f tokens (below %.1f): short prompts keep one ubatch that "
                                "copies its experts from the RAM tier\n", __func__, exp_b/1048576.0, avail/1048576.0, cparams->moe_cache_size/MiB, tokens,
                                placemid_disk_tokens);
                        } else if (g_verify_rows > placemid_disk_ub) {
                            // a verify window wider than the ubatch would run with the banks released every pass (DFlash2 n6: 7 rows)
                            LOG_INF("%s: k19-l4gap2: k19-placemid stays at %u-token ubatches: verify windows of %d rows exceed %d tokens\n", __func__,
                                common_fit_place8_ub(), g_verify_rows, placemid_disk_ub);
                        } else {
                            ub_disk = (uint32_t) placemid_disk_ub;
                            LOG_INF("%s: k19-l4gap2: k19-placemid at ubatches of %u tokens: %.0f MiB of host experts do not fit in %.0f MiB of available "
                                "RAM and the cache of %zu MiB holds the experts of %.1f tokens (below %.1f), so short prompts run in ubatches the device "
                                "lookups serve (LLAMA_FIT_PLACEMID_DISK_UB=0: %u)\n", __func__, ub_disk, exp_b/1048576.0, avail/1048576.0,
                                cparams->moe_cache_size/MiB, tokens, placemid_disk_tokens, common_fit_place8_ub());
                        }
                    }
                }
                const uint32_t ub_mid = ub_disk > 0 ? std::min(ub_disk, common_fit_place8_ub()) : common_fit_place8_ub();
                if (!g_small_device && nd == 1 && placemid_on && placemid_ram && common_fit_place8_on() && cparams->moe_cache_size > 0 &&
                        cparams->n_ubatch > ub_mid) {
                    const uint32_t ub_full = cparams->n_ubatch;
                    const uint32_t ub_rel  = ub_mid;
                    const size_t   size_ok = cparams->moe_cache_size;
                    const dmds_t   dmds_ok = dmds_cpu_moe;
                    bool           got     = false;
                    int64_t        grow    = 0;
                    cparams->n_ubatch = ub_rel;
                    common_fit_place8_env(ub_rel, true);
                    auto measure_mid = [&](size_t cache, dmds_t & d) {
                        cparams->moe_cache_size = cache;
                        try {
                            d = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                        } catch (const std::runtime_error &) {
                            return false;
                        }
                        add_extra_memory(d);
                        return true;
                    };
                    dmds_t d_s;
                    if (measure_mid(size_ok, d_s)) {
                        grow = int64_t(dmds_ok[0].mb.total()) - int64_t(d_s[0].mb.total());
                    }
                    if (grow >= 256*MiB && double(grow) >= placemid_min_frac*double(size_ok)) {
                        int64_t req = int64_t(size_ok) + grow;
                        for (int it = 0; it < 3 && req >= int64_t(size_ok) + 128*MiB; it++) {
                            dmds_t dm;
                            if (!measure_mid(size_t(req), dm)) {
                                break;
                            }
                            const int64_t sm = int64_t(dm[0].free) - int64_t(dm[0].mb.total()) - margins[0];
                            LOG_TRC("%s: k19-placemid: ubatches of up to %u tokens beside a cache of %" PRId64 " MiB: compute %zu MiB, %" PRId64
                                " MiB over the margin\n", __func__, ub_rel, req/MiB, dm[0].mb.compute/MiB, sm/MiB);
                            if (sm >= 0) {
                                dmds_cpu_moe = dm;
                                got = true;
                                break;
                            }
                            req += sm - 16*MiB;
                        }
                    }
                    cparams->n_ubatch = ub_full;
                    if (got) {
                        common_fit_place8_env(ub_rel, true, grow/MiB);
                        LOG_INF("%s: the MoE cache takes the memory of the prompt compute buffers (%" PRId64 " MiB): cache %zu -> %zu MiB; "
                            "ubatches of more than %u tokens run with the cache banks released, layer-major groups as before (k19-placemid; "
                            "LLAMA_FIT_PLACEMID=0 off)\n", __func__, grow/MiB, size_ok/MiB, cparams->moe_cache_size/MiB, ub_rel);
                    } else {
                        common_fit_place8_env(0);
                        cparams->moe_cache_size = size_ok;
                        dmds_cpu_moe            = dmds_ok;
                        LOG_INF("%s: the MoE cache keeps its size (k19-placemid: %" PRId64 " MiB of prompt compute buffers beside a cache of %zu "
                            "MiB, at least %.0f%% of it and 256 MiB wanted, LLAMA_FIT_PLACEMID_MIN_FRAC)\n", __func__, grow/MiB, size_ok/MiB,
                            100.0*placemid_min_frac);
                    }
                }
                // k17-small (lazy KV step 2, PolyStrata's rule): LLAMA_KV_LAZY=1 backs only the first LLAMA_KV_LAZY_INIT (4096) cells of
                // the KV cache at load (mb.kv_lazy: the rest, reported only where the MoE cache can shrink). The cache was fitted above
                // beside the full-context KV, so that size is its floor; it gets the unbacked KV bytes on top (up to the size requested)
                // and gives them back as the cells grow (llama_moe_cache::shrink), so every cell of the context still fits.
                // LLAMA_MOE_CACHE_SHRINK=0: the cache keeps the floor (step 1)
                // (LLAMA_KV_LAZY_FIT=0: the cache keeps the floor while still shrinking for the KV cache, a debug knob)
                static const bool kv_lazy_fit = getenv("LLAMA_KV_LAZY_FIT") == nullptr || atoi(getenv("LLAMA_KV_LAZY_FIT")) != 0;
                // beside a fitted drafter the cache keeps the floor (LLAMA_KV_LAZY_FIT_DRAFT=1 gives it the unbacked KV there too): the bigger
                // cache lent the drafter's staging, its compute took 506 MiB and left 14 MiB (G4 q4x 4 GB 16k + MTP: out of memory at the
                // first request 3/3; q3m 16k + MTP 5-17% slower end to end)
                static const bool kv_lazy_fit_draft = getenv("LLAMA_KV_LAZY_FIT_DRAFT") != nullptr && atoi(getenv("LLAMA_KV_LAZY_FIT_DRAFT")) != 0;
                const bool kv_lazy_draft_ok = extra == nullptr || kv_lazy_fit_draft;
                if (kv_lazy_fit && !kv_lazy_draft_ok && nd == 1 && dmds_cpu_moe[0].mb.kv_lazy > 0) {
                    LOG_INF("%s: lazy KV: a drafter is fitted, the MoE cache keeps %zu MiB, not the %zu MiB of KV cells not backed at load on top "
                        "(LLAMA_KV_LAZY_FIT_DRAFT=1 to give them)\n", __func__, cparams->moe_cache_size/MiB, dmds_cpu_moe[0].mb.kv_lazy/MiB);
                }
                // k19-vramlend: in the automatic mode (LLAMA_KV_LAZY unset) only where the unbacked KV is at least LLAMA_KV_LAZY_AUTO_MIN_FRAC
                // (0.10) of the fitted cache: Colab T4 q4x ctx 32768 no draft, 4 GB (grant 4x the cache) decode +9..+17%, 8 GB (+14%) +2..+5%;
                // L4 12 GB (+7%) -2..-6% (the larger cache tipped the CPU-miss rule off for 1-token batches; exact env +2..+3%)
                static const double kv_lazy_auto_min = getenv("LLAMA_KV_LAZY_AUTO_MIN_FRAC") != nullptr ?
                    std::max(0.0, atof(getenv("LLAMA_KV_LAZY_AUTO_MIN_FRAC"))) : 0.10;
                const char * e_kvl = getenv("LLAMA_KV_LAZY");
                const bool   kv_lazy_auto = e_kvl != nullptr && strcmp(e_kvl, "auto") == 0;
                const size_t kv_lazy_add  = cparams->moe_cache_size > 0 && cparams->moe_cache_size < size_before ?
                    std::min(dmds_cpu_moe[0].mb.kv_lazy, size_before - cparams->moe_cache_size) : 0;
                const bool   kv_lazy_gate = !kv_lazy_auto || double(kv_lazy_add) >= kv_lazy_auto_min*double(cparams->moe_cache_size);
                if (kv_lazy_fit && kv_lazy_draft_ok && nd == 1 && kv_lazy_add > 0 && !kv_lazy_gate) {
                    LOG_INF("%s: lazy KV (auto): the %zu MiB of KV cells not backed at load are under %.0f%% of the MoE cache (%zu MiB): the KV cache "
                        "is backed whole (LLAMA_KV_LAZY_AUTO_MIN_FRAC, LLAMA_KV_LAZY=1 lends them anyway)\n", __func__, kv_lazy_add/MiB,
                        100.0*kv_lazy_auto_min, cparams->moe_cache_size/MiB);
                }
                if (kv_lazy_fit && kv_lazy_draft_ok && kv_lazy_gate && nd == 1 && dmds_cpu_moe[0].mb.kv_lazy > 0 && cparams->moe_cache_size > 0 && cparams->moe_cache_size < size_before) {
                    const size_t floor = cparams->moe_cache_size;
                    cparams->moe_cache_size += std::min(dmds_cpu_moe[0].mb.kv_lazy, size_before - floor);
                    g_kv_lazy_grants.emplace_back(cparams->moe_cache_size, cparams->moe_cache_size - floor);
                    LOG_INF("%s: lazy KV: the MoE cache takes the %zu MiB of KV cells not backed at load: %zu -> %zu MiB, it gives them back as the "
                        "context grows (LLAMA_MOE_CACHE_SHRINK=0 to disable)\n", __func__, dmds_cpu_moe[0].mb.kv_lazy/MiB, floor/MiB,
                        cparams->moe_cache_size/MiB);
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
        // LLAMA_MTP_EXACT=1: no such layer (LLAMA_FIT_PARTIAL_LAYER=1 keeps it, =0 drops it everywhere). Its host part is no MoE cache
        // expert set (UP / GATE split the experts, ATTN also moves the router and the shared expert), so the scheduler computes it on
        // the CPU for small batches: the greedy text then followed which layer the fit split, i.e. the cache size and a draft model's share
        static const bool partial_layer = [] {
            const char * e = getenv("LLAMA_FIT_PARTIAL_LAYER");
            if (e != nullptr) {
                return atoi(e) != 0;
            }
            const char * x = getenv("LLAMA_MTP_EXACT");
            return !(x != nullptr && atoi(x) != 0);
        }();
        if (partial_layer && ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
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

// k19-fitplain: see fit.h
std::vector<size_t> common_fit_margin_steps_mib(const char * spec) {
    std::vector<size_t> steps;
    const std::string str = spec != nullptr ? spec : "";
    size_t pos = 0;
    while (pos < str.size()) {
        const size_t c = str.find(',', pos);
        const std::string tok = str.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
        if (!tok.empty()) {
            steps.push_back((size_t) std::max(0, atoi(tok.c_str())) * 1024 * 1024);
        }
        if (c == std::string::npos) {
            break;
        }
        pos = c + 1;
    }
    return steps;
}

std::vector<std::vector<size_t>> common_fit_margin_trials(const std::vector<size_t> & margins, const std::vector<size_t> & steps) {
    std::vector<std::vector<size_t>> trials;
    for (const size_t step : steps) {
        std::vector<size_t> mg = margins;
        bool lower = false;
        for (auto & m : mg) {
            if (m > step) {
                m = step;
                lower = true;
            }
        }
        if (lower && (trials.empty() || trials.back() != mg)) {
            trials.push_back(mg);
        }
    }
    return trials;
}

bool common_fit_plain_spill_gate(const std::vector<int> & headless, const char * force) {
    if (force != nullptr && force[0] != '\0') {
        return atoi(force) != 0;
    }
    return !headless.empty() && std::all_of(headless.begin(), headless.end(), [](int h) { return h == 1; });
}

std::vector<int> common_fit_devices_headless(const std::vector<ggml_backend_dev_t> & devs) {
    std::vector<int> res;
    for (ggml_backend_dev_t dev : devs) {
        if (dev == nullptr) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto fn = reg ? (int (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_headless") : nullptr;
        res.push_back(fn != nullptr ? fn(dev) : -1);
    }
    return res;
}
