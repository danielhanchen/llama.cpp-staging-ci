// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-decision.h"
#include "server-http.h"
#include "server-task.h"
#include "server-prompt-ssd.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "gguf.h"
#include "../../src/llama-ext.h" // staging API: llama_set_verify
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "spec-governor.h"
#include "server-spec-auto.h"
#include "server-tpp.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cinttypes>
#include <cstring>
#include <exception>
#include <memory>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>
#include <list>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <functional>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__linux__)
#include <sys/vfs.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// k19-tpp: bytes of a GGUF file and its splits (NAME-0000k-of-0000n.gguf), 0 if unknown
static uint64_t tpp_gguf_bytes(const std::string & path) {
    std::error_code ec;
    if (path.empty()) {
        return 0;
    }
    int k = 0, n = 0;
    const size_t at = path.rfind("-of-");
    if (at != std::string::npos && at >= 6 && path.size() >= at + 4 + 5 + 5 &&
            sscanf(path.c_str() + at - 5, "%5d-of-%5d.gguf", &k, &n) == 2 && n > 1) {
        const std::string prefix = path.substr(0, at - 6);
        uint64_t sum = 0;
        for (int i = 0; i < n; ++i) {   // llama_split_path numbers from 0 (split_no 0 = NAME-00001-of-0000n.gguf)
            char sp[4096];
            llama_split_path(sp, sizeof(sp), prefix.c_str(), i, n);
            const auto b = std::filesystem::file_size(sp, ec);
            if (ec) {
                return 0;
            }
            sum += b;
        }
        return sum;
    }
    const auto b = std::filesystem::file_size(path, ec);
    return ec ? 0 : (uint64_t) b;
}

// k19-fnmtp2: true if a file lives on a memory file system (tmpfs / ramfs on Linux): its pages are host memory that MemAvailable does
// not count as available, so a model staged there is resident though the available memory reads lower than its size
static bool tpp_file_in_ram_fs(const std::string & path) {
#if defined(__linux__)
    struct statfs st;
    if (path.empty() || statfs(path.c_str(), &st) != 0) {
        return false;
    }
    return (unsigned long) st.f_type == 0x01021994ul || (unsigned long) st.f_type == 0x858458f6ul; // TMPFS_MAGIC, RAMFS_MAGIC
#else
    (void) path;
    return false;
#endif
}

// k19-bigverify: bytes of the GGUF's per-layer token embedding table (per_layer_token_embd.weight), an input-layer GET_ROWS table that
// the loader keeps in host memory whatever the device split (Qwen3.8-Flash-Next UD-IQ4_XS: 26.8 GiB of its 87.2 GiB). 0 if absent or
// unreadable (the caller then counts the whole file, as before)
static uint64_t tpp_host_input_bytes(const std::string & path) {
    if (path.empty()) {
        return 0;
    }
    std::vector<std::string> files = { path };
    int k = 0, n = 0;
    const size_t at = path.rfind("-of-");
    if (at != std::string::npos && at >= 6 && path.size() >= at + 4 + 5 + 5 &&
            sscanf(path.c_str() + at - 5, "%5d-of-%5d.gguf", &k, &n) == 2 && n > 1) {
        const std::string prefix = path.substr(0, at - 6);
        files.clear();
        for (int i = 0; i < n; ++i) {
            char sp[4096];
            llama_split_path(sp, sizeof(sp), prefix.c_str(), i, n);
            files.emplace_back(sp);
        }
    }
    uint64_t sum = 0;
    for (const std::string & f : files) {
        struct gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
        gguf_context * g = gguf_init_from_file(f.c_str(), gp);
        if (g == nullptr) {
            return 0;
        }
        const int64_t id = gguf_find_tensor(g, "per_layer_token_embd.weight");
        if (id >= 0) {
            sum += gguf_get_tensor_size(g, id);
        }
        gguf_free(g);
    }
    return sum;
}

// k19-vwidth (LLAMA_SPEC_PREWARM=N: widths 1..N, opt-in): a target with a drafter and no MoE cache decodes, at load, a
// dummy prompt of LLAMA_SPEC_PREWARM_POS (640) tokens, then windows of every width 1..N, LLAMA_SPEC_PREWARM_REPS (10) times each,
// appended after it (a 1-row decode as a plain decode, wider ones as verify windows, as the server runs them), then forgets them: each
// width's kept graph (llama_context gf_res_small) is built and its CUDA graph warmed up, captured and launched at least 8 times in the
// KV bucket of a short request. The prompt matters: graphs built on an empty memory take other branches (no pooled keys: G4 fn4
// 7570 vs 7613 nodes) and a request could not reuse them. 8 launches keep a CUDA graph through idle time (ggml-cuda evicts the
// others after 10 s, GGML_CUDA_GRAPH_KEEP) and let a new KV bucket re-capture it directly (GGML_CUDA_GRAPH_DIRECT_RECAPTURE, no eager
// pass). A width a request uses for the first time (a confidence rule's, or a draft the n_predict limit cut at the end of a request)
// then replays a captured graph instead of an eager pass plus a capture (~15-20 ms each on G4 Flash-Next). The memory is cleared
// after; no value of a later request depends on these decodes
// k19-fullcost: on by default on a GPU target (widths 1..8; LLAMA_SPEC_PREWARM=N up to 8, =0 off; ~2.6 s of load time on G4 Flash-Next
// with DFlash2: 7 widths x 3 sampler chains x 9 decodes). Before k19-fullcost the idle re-reserve of the pp graph (restore_pp_plan)
// invalidated every kept graph at each request's start, so no prewarmed graph was ever used (vw_t2a / vw_t4: on = off within noise);
// with the kept graphs surviving it (and one KV bucket for chat and 8k, LLAMA_KV_BUCKET_DEV_STEP_MIN) a request replays them: G4 fn4 full
// card (fc_r3, 2 launches): DFlash2 n6 chat 329.1 -> 343.1, the width rule over every width (LLAMA_DFLASH_EV) chat 325.2 -> 372.5, 8k
// 266.4 -> 292.7 tok/s, same text
// (widths up to the outputs the context takes per sequence, the server's widest window)
// (the graphs carry the slot's device sampler: each chain of `chains` is swapped in for its own windows, raw logits off as for a request
// it serves; `restore` / `restore_raw` are put back after. No chain: the windows run once with the context's sampler state)
static void server_spec_prewarm(llama_context * ctx, bool gpu, int32_t n_out_max, const std::vector<llama_sampler *> & chains,
        llama_sampler * restore, bool restore_raw) {
    const char * e = getenv("LLAMA_SPEC_PREWARM");
    int w_max = e != nullptr ? std::max(0, std::min(8, atoi(e))) : (gpu ? 8 : 0);
    if (n_out_max > 0) {
        w_max = std::min(w_max, (int) n_out_max);
    }
    static const int reps  = getenv("LLAMA_SPEC_PREWARM_REPS") ? std::max(2, atoi(getenv("LLAMA_SPEC_PREWARM_REPS"))) : 9;
    static const int n_pre = getenv("LLAMA_SPEC_PREWARM_POS")  ? std::max(0, atoi(getenv("LLAMA_SPEC_PREWARM_POS")))  : 640;
    // k19-fullcost (LLAMA_SPEC_PREWARM_CACHED=1, opt-in): beside a MoE cache too; LLAMA_SPEC_PREWARM_WIDTHS="1,3,7" prewarms only those
    // widths (default 1..w_max). LLAMA_SPEC_PREWARM_MEM=1 logs the device's free memory after each width (the graph memory per width)
    // (k19-bigverify: unset, on beside a cache that holds most of the experts, llama_moe_cache_wide_graphs; =0 off there too)
    // k19-g4cap (opt-in, LLAMA_SPEC_PREWARM_CACHED_FRAC=f, e.g. 0.3; unset / 0 = only the wide-graph rule, as before): also beside a cache
    // that holds at least f of its host expert bytes, where the widest window is at least LLAMA_SPEC_PREWARM_CACHED_MIN_W (5) rows (DFlash2:
    // MTP's 1..3-row graphs are captured by the first request's own short windows). Only the prewarm: the device KV buckets keep their
    // 2048-cell step there. Colab G4 Qwen3.8-Flash-Next UD-IQ4_XS at a 32 GB cap (41.4% cached): the warm-up request leaves the 4..7-row
    // verify widths to be built and captured inside the next one (36-129 ms per width); prewarmed, with the CPU-miss windows paused (below),
    // DFlash2 n6 chat 203.1 -> 224.1 tok/s (same text), 8k 172.7 -> 190.3 (other text: the prewarm moves the cached experts, so the
    // host-computed ones; srv_kld long48 inside the -fa off band), 3 launches on one VM with the CPU-miss calibration pinned for both
    // builds. At 16 GB (15.5% cached) keep it off: level with the pause, -24% without it. Opt-in because the 8k text changes (exact in chat)
    // k19-prewexact: with the cache taken back after the windows (exact, below) the text is that of no prewarm in chat and 8k: same VM,
    // CPU-miss calibration pinned, 3 launches (px_c32): chat 213.1 (int3) / 209.8 (FRAC unset) -> 231.5 tok/s, 8k 180.5 / 180.5 -> 179.9,
    // md5 equal in every launch; the 8k gain of the prewarm without the restore (195.8) came with its other text. 24 GB (27.1% cached)
    // is under 0.3, so no prewarm there
    // (k19-prewexact: on by default at 0.3 now that the prewarm is exact, LLAMA_SPEC_PREWARM_CACHED_FRAC=0 off)
    static const double cached_frac_min = getenv("LLAMA_SPEC_PREWARM_CACHED_FRAC") != nullptr ?
        atof(getenv("LLAMA_SPEC_PREWARM_CACHED_FRAC")) : 0.3;
    static const int cached_min_w = getenv("LLAMA_SPEC_PREWARM_CACHED_MIN_W") != nullptr ?
        atoi(getenv("LLAMA_SPEC_PREWARM_CACHED_MIN_W")) : 5;
    llama_moe_cache_fingerprint(ctx, "prewarm point"); // k19-prewexact diagnostic (LLAMA_MOE_CACHE_FP_LOG=1)
    const double cache_frac = llama_moe_cache_frac(ctx);
    const bool wide_graphs  = llama_moe_cache_wide_graphs(ctx);
    const bool band_ok      = !wide_graphs && cached_frac_min > 0.0 && cache_frac >= cached_frac_min && w_max >= cached_min_w;
    const bool cached_ok = getenv("LLAMA_SPEC_PREWARM_CACHED") != nullptr ? atoi(getenv("LLAMA_SPEC_PREWARM_CACHED")) != 0 :
        (wide_graphs || band_ok);
    static const bool mem_log   = getenv("LLAMA_SPEC_PREWARM_MEM") != nullptr && atoi(getenv("LLAMA_SPEC_PREWARM_MEM")) != 0;
    std::vector<int> widths;
    if (const char * ew = getenv("LLAMA_SPEC_PREWARM_WIDTHS")) {
        for (const char * q = ew; *q; ) {
            const int w = atoi(q);
            if (w >= 1 && w <= w_max && std::find(widths.begin(), widths.end(), w) == widths.end()) {
                widths.push_back(w);
            }
            while (*q && *q != ',') { ++q; }
            if (*q == ',') { ++q; }
        }
        std::sort(widths.begin(), widths.end());
    } else {
        for (int w = 1; w <= w_max; ++w) {
            widths.push_back(w);
        }
    }
    int sum_w = 0;
    for (int w : widths) {
        sum_w += w;
    }
    const int n_chain = std::max<int>(1, (int) chains.size());
    // k19-p2qsa / k19-int3fix: the QSA form of the prewarmed decode / verify graphs. LLAMA_SPEC_PREWARM_ALLFIT unset = the all-fit form
    // beside a MoE cache (the host tiers, where the first request building its all-fit widths costs: G4 Flash-Next 48 GB DFlash2 chat 281.0
    // tok/s with the all-fit prewarm vs 253.9 with the scored one, base without all-fit 265.8) and the scored form without one (full card:
    // chat 391.6 vs 382.8, 8k 236.1 vs 232.0: a request past 512 pools then replays them); 0 = scored only (k19-p2qsa 7f298b321bd), 1 =
    // all-fit only, 2 = both when the model selects QSA pools (twice the graphs: out of device memory at G4 48 GB without the floor below,
    // 245.2 tok/s with it)
    static const int pw_env  = getenv("LLAMA_SPEC_PREWARM_ALLFIT") != nullptr ? atoi(getenv("LLAMA_SPEC_PREWARM_ALLFIT")) : -1;
    const int  pw_mode  = pw_env >= 0 ? pw_env : (llama_moe_cache_active(ctx) ? 1 : 0);
    const bool pw_both  = pw_mode >= 2 && llama_qsa_allfit_capable(ctx);
    const int  n_forms  = pw_both ? 2 : 1;
    const int  n_end    = n_pre + n_forms*n_chain*reps*sum_w;
    if (widths.empty() || ctx == nullptr || (llama_moe_cache_active(ctx) && !cached_ok) ||
            llama_n_batch(ctx) < (uint32_t) std::max(w_max, std::min(n_pre, 1)) || (int) llama_n_ctx_seq(ctx) < n_end + 64) {
        return;
    }
    if (cached_ok && llama_moe_cache_active(ctx) && !wide_graphs) {
        SRV_INF("k19-g4cap: verify graphs prewarmed beside a MoE cache that holds %.1f%% of the host experts (LLAMA_SPEC_PREWARM_CACHED_FRAC "
                "%.2f)\n", 100.0*cache_frac, cached_frac_min);
    }
    auto dev_free_mib = []() -> double {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                size_t f = 0, t = 0;
                ggml_backend_dev_memory(dev, &f, &t);
                return f/1048576.0;
            }
        }
        return 0.0;
    };
    double free_prev = mem_log ? dev_free_mib() : 0.0;
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const llama_token   tok   = llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_bos(vocab) : 0;
    llama_memory_t mem = llama_get_memory(ctx);
    const int64_t t0 = ggml_time_us();
    const int n_b = std::max(w_max, (int) std::min<uint32_t>(llama_n_batch(ctx), (uint32_t) std::max(n_pre, 1)));
    llama_batch b = llama_batch_init(n_b, 0, 1);
    llama_pos pos = 0;
    bool ok = true;
    auto add = [&](int n, bool all_out) {
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            b.token[i]     = tok;
            b.pos[i]       = pos + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = all_out || i == n - 1;
        }
        pos += n;
    };
    // k19-tokcost: the dummy windows (one repeated token) must not feed the MoE cache's CPU-miss rule: one such miss-rate window
    // (0.7%) turned the host path off for the next request (fn4 24 GB, fc_r13: P_CUM 0.5 chat 153 vs 184 tok/s; 16 GB prewarm -20%).
    // k19-g4cap: LLAMA_SPEC_PREWARM_CM_PAUSE unset = paused beside a cache under the wide-graph share (llama_moe_cache_wide_graphs), where
    // the host path carries the decode (G4 Flash-Next 32 GB: 8k 172.7 -> 193.8 tok/s and chat TTFT 0.26 -> 0.22 s with the pause; 16 GB
    // forced prewarm chat 125.8 -> 162.8). Over the share the prewarm keeps counting, as before: there the windows it feeds turn the host
    // path off, which a timing trial confirms (G4 48 GB, 14.7 vs 16.6 ms per small batch), and the pause would move the warm phase and
    // the trial into the first request (gc_ab1 48 GB DFlash2 chat 269.8 -> 246.1). =1 always, =0 never
    // k19-prewexact: beside a MoE cache the dummy windows also move what the cache holds (the 640-token prompt seeds it, every window
    // admits and evicts by the clock), so the next requests compute other experts on the host: G4 Flash-Next 32 GB with the band prewarm
    // gave chat the same text but 8k another one (832d4d83 vs 6974c3ad). Exact: the cache's state is snapshotted before the windows and
    // taken back after them (the experts of the slots that changed are copied back into the banks, slots / clock / device counters /
    // seed and refill bookkeeping restored) and its rules are quiet meanwhile (no CPU-miss window, timing trial, overlap rule or trial
    // moves), so the first request sees the cache it would have seen without the prewarm and only the captured graphs remain
    // (G4 Flash-Next 32 GB: 337 slots get their expert back, 795 MiB, per launch; the state fingerprints at every request begin equal
    // those of a launch without the prewarm, LLAMA_MOE_CACHE_FP_LOG=1). Not the default for the wide-graph prewarm (>= 60% cached, G4
    // 48 GB): there its windows end the CPU-miss warm phase and timing trial before the first request, and exact moves them into it
    // (px_ab48b DFlash2 chat 261.7 -> 245.0 tok/s); the 48 GB 8k text varies per launch with or without a prewarm (the timing trial).
    // LLAMA_SPEC_PREWARM_EXACT unset = exact for the band prewarm (a cache under the wide-graph share), 1 = also over it, 0 = never (the
    // k19-tokcost pause below then)
    const char * e_ex  = getenv("LLAMA_SPEC_PREWARM_EXACT");
    const int    ex_env = e_ex != nullptr ? atoi(e_ex) : -1;
    const bool   exact  = llama_moe_cache_active(ctx) && (ex_env > 0 || (ex_env < 0 && !wide_graphs)) &&
                          llama_moe_cache_prewarm_begin(ctx);
    const char * e_cmp = getenv("LLAMA_SPEC_PREWARM_CM_PAUSE");
    const bool cm_pause = !exact && (e_cmp != nullptr ? atoi(e_cmp) != 0 : !wide_graphs);
    if (cm_pause) {
        llama_moe_cache_stats_pause(ctx, true);
    }
    // k19-p2qsa: the prewarmed graphs in the scored QSA form (LLAMA_QSA_ALLFIT), the kept graphs a request past 512 pools replays: an all-fit
    // prewarm left only all-fit graphs, so the first verify / decode of such a request rebuilt every width inside its decode (G4 Flash-Next
    // full card, MTP 8k after chat: 4 builds, -2.9%). The first short request builds its all-fit graphs beside them instead (its own ways),
    // a one-time cost per width. LLAMA_SPEC_PREWARM_ALLFIT=1: prewarm the all-fit form instead
    // the first (or only) form is the scored one unless LLAMA_SPEC_PREWARM_ALLFIT=1; with both forms each chain prewarms its scored widths,
    // then its all-fit widths (the greedy chain first, so a memory floor below cuts the rarely used chains first)
    bool pw_paused = false;
    auto pw_pause = [&](bool on) {
        if (on != pw_paused) {
            llama_qsa_allfit_pause(on);
            pw_paused = on;
        }
    };
    pw_pause(pw_mode != 1);
    // k19-int3fix: every captured verify graph keeps device memory the fit does not count (A100 80 GB with 48 GB free, Flash-Next + DFlash2
    // MoE cache: 22-66 MiB per graph, 3 chains x 7 widths took 828 -> 64 MiB free on ec9072e9789 and -> 2 MiB with the few-row MMQ, whose
    // first request then died in a pool alloc). LLAMA_SPEC_PREWARM_MIN_FREE_MIB (default 512, 0 = no floor): no further width is
    // prewarmed below that much free device memory; the widths left out are captured when a request first needs them
    static const double min_free = getenv("LLAMA_SPEC_PREWARM_MIN_FREE_MIB") != nullptr ? atof(getenv("LLAMA_SPEC_PREWARM_MIN_FREE_MIB")) : 512.0;
    bool floor_hit = false;
    int  n_skipped = 0;
    llama_set_verify(ctx, false);
    for (int done = 0; done < n_pre && ok; ) {
        const int n = std::min(n_b, n_pre - done);
        add(n, false);
        ok = llama_decode(ctx, b) == 0;
        done += n;
    }
    int n_ok = 0;
    for (int c = 0; c < n_chain && ok; ++c) {
        if (!chains.empty()) {
            ok = llama_set_sampler_swap(ctx, 0, chains[c]);
            llama_set_sampler_raw_logits(ctx, 0, false);
        }
        for (int form = 0; form < n_forms && ok; ++form) {
            pw_pause(pw_mode != 1 && form == 0);
            n_ok = 0;
            for (int w : widths) {
                if (!ok) {
                    break;
                }
                if (min_free > 0.0 && gpu && (floor_hit || dev_free_mib() < min_free)) {
                    floor_hit = true;
                    n_skipped++;
                    continue;
                }
                for (int rep = 0; rep < reps && ok; ++rep) {
                    add(w, true);
                    llama_set_verify(ctx, w > 1);
                    ok = llama_decode(ctx, b) == 0;
                    llama_synchronize(ctx);
                }
                n_ok += ok;
                if (mem_log) {
                    const double fr = dev_free_mib();
                    SRV_INF("k19-fullcost: prewarm chain %d form %d width %d: device free %.1f MiB (%+.1f)\n", c, form, w, fr, fr - free_prev);
                    free_prev = fr;
                }
            }
        }
    }
    if (floor_hit) {
        SRV_INF("k19-int3fix: verify graph prewarm left %d of %d chain x form x width graphs to the first request that needs them: device free "
                "memory under %.0f MiB (LLAMA_SPEC_PREWARM_MIN_FREE_MIB=0: no floor)\n", n_skipped, n_chain*n_forms*(int) widths.size(), min_free);
    }
    if (pw_both && ok) {
        SRV_INF("k19-int3fix: verify graphs prewarmed in both QSA forms, scored and all-fit, %d widths (LLAMA_SPEC_PREWARM_ALLFIT=0: scored only)\n", (int) widths.size());
    }
    if (!ok) {
        SRV_WRN("k19-vwidth: verify graph prewarm stopped at %d rows (a decode or sampler swap failed)\n", n_ok + 1);
    }
    if (!chains.empty()) {
        llama_set_sampler_swap(ctx, 0, restore);
        llama_set_sampler_raw_logits(ctx, 0, restore_raw);
    }
    llama_set_verify(ctx, false);
    llama_batch_free(b);
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);
    pw_pause(false);
    if (cm_pause) {
        llama_moe_cache_stats_pause(ctx, false);
    }
    if (exact) {
        llama_moe_cache_prewarm_end(ctx);
    }
    llama_moe_cache_fingerprint(ctx, "after the prewarm");
    SRV_INF("k19-vwidth: the decode / verify graphs of 1..%d rows warmed up after a %d-token prompt, %d decodes each, %d sampler chain(s) (%.1f ms; LLAMA_SPEC_PREWARM=0: off)\n",
            n_ok, n_pre, reps, (int) chains.size(), (ggml_time_us() - t0)/1000.0);
}

// k19-l4reg: see server_slot::get_n_draft_max (default on, LLAMA_SPEC_END_FULL_WINDOW=0: the draft is cut to the tokens left as before)
static bool spec_end_full_window() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_SPEC_END_FULL_WINDOW");
        const bool on = e == nullptr || atoi(e) != 0;
        LOG_INF("k19-l4reg: the last pass before the n_predict limit keeps the full draft window: %s (LLAMA_SPEC_END_FULL_WINDOW=0: off)\n",
            on ? "on" : "off");
        return on;
    }();
    return v;
}

// k19-q4xaccept: a request with penalties (Studio sends presence 1.5 for every Qwen3.5 / 3.6 / 3.8 request; the server's window is
// the last 64 tokens) drafts the argmax of the draft's candidates AFTER those penalties (window = the target's last tokens plus the
// ones drafted before them), the token the target most likely emits; before, the draft's raw argmax was often a token the penalty had
// just pushed down (a word of the last 64 tokens), which the target then rejected. Drafts only: the verify is unchanged (greedy:
// argmax match, sampled: sample-and-match with the target's own RNG), so the output is unchanged.
// LLAMA_SPEC_DRAFT_PEN=0: the raw argmax (as before)
static bool spec_draft_pen_on() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_SPEC_DRAFT_PEN");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

static bool spec_replay_reverify() {
    static const bool v = getenv("LLAMA_SPEC_REPLAY_REVERIFY") != nullptr && atoi(getenv("LLAMA_SPEC_REPLAY_REVERIFY")) != 0;
    return v;
}

// k19-sharednoise: drafts take the target's shared noise when the target draws with it (LLAMA_SHARED_NOISE=1);
// LLAMA_SHARED_NOISE_DRAFT=0: they do not (the target's draws stay counter-based)
// k19-small1115 (LLAMA_SPEC_TPP_ACTIVE, default on; 0 = off): the MTP window rules of a --parallel 1 server on a --parallel N one,
// decided per round from the active slots (load_model, update_slots)
static bool spec_tpp_active_on() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_SPEC_TPP_ACTIVE");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

static bool spec_sn_draft_on() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_SHARED_NOISE_DRAFT");
        return common_shared_noise_enabled() && (e == nullptr || atoi(e) != 0);
    }();
    return v;
}

// k19-host30k (LLAMA_STATE_ASYNC_EXT=0: off): the prompt-cache save of a long prompt (larger than LLAMA_PROMPT_SAVE_ASYNC_MAX_MIB, or than
// the staging at a request) was synchronous on the server thread: G4 Qwen3.8-Flash-Next UD-IQ4_XS, the 642 MiB save of a 16k prompt
// took 175-360 ms of the next (30k) request's TTFT at 24 / 32 GB (a MoE cache: saved at the request) and ~140 ms at full card (saved at
// idle, which the request sent right after the response waits for). The contexts pin at load the staging a whole-sequence save needs at
// most (llama_state_seq_async_ext_reserve), and the save then reads its rows asynchronously: the same bytes into the entry. Pinned on a
// thread while the slot generated instead (k19hx_g4), the pinning cost the decode 5-17% (cudaMallocHost beside the decode)
static bool server_state_async_ext() {
    static const bool v = getenv("LLAMA_STATE_ASYNC_EXT") == nullptr || atoi(getenv("LLAMA_STATE_ASYNC_EXT")) != 0;
    return v;
}

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<llama_token> server_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < draft.size(); ++i) {
        // the token is discarded - the call is what advances the sampler over this position
        common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, draft[i], true);
        result.push_back(draft[i]);
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

// k19-handoff (diagnostic, LLAMA_PASS_TIMING=1): host wall time between the steps of a speculative pass of one generating slot,
// averaged per request (printed with the timings): draft call, the host work up to the target's decode call, the decode call, its
// wait, the drafter's pass over the verify rows, the accept, the accepted tokens' handling and the loop back to the next draft
struct server_pass_timer {
    enum { DRAFT0, DRAFT1, DEC0, DEC1, DEC2, PROC0, PROC1, ACC0, ACC1, TOK1, N_STAMP };
    bool    on = false;
    int64_t t[N_STAMP] = {};
    int32_t width = 0;
    double  sum[N_STAMP] = {};
    int64_t n = 0;
    double  w_sum[32] = {};
    int64_t w_n[32] = {};
    server_pass_timer() {
        const char * e = getenv("LLAMA_PASS_TIMING");
        on = e != nullptr && atoi(e) != 0;
    }
    void stamp(int k) {
        if (!on) {
            return;
        }
        const int64_t now = ggml_time_us();
        if (k == DRAFT0) {
            bool full = t[DRAFT0] > 0;
            for (int i = 1; i < N_STAMP && full; ++i) {
                full = t[i] >= t[i - 1];
            }
            if (full) {
                for (int i = 0; i + 1 < N_STAMP; ++i) {
                    sum[i] += (double) (t[i + 1] - t[i]);
                }
                sum[N_STAMP - 1] += (double) (now - t[TOK1]);
                n += 1;
            }
            for (int i = 0; i < N_STAMP; ++i) {
                t[i] = 0;
            }
        }
        t[k] = now;
        if (k == DEC2 && t[DEC0] > 0 && width > 0 && width < 32) {
            w_sum[width] += (double) (now - t[DEC0]);
            w_n[width]   += 1;
        }
    }
    std::string report() {
        static const char * names[N_STAMP] = { "draft", "to_dec", "dec_call", "dec_wait", "to_proc", "proc", "to_acc", "acc", "tokens", "to_draft" };
        std::string r;
        char b[96];
        if (n == 0) {
            return r;
        }
        double tot = 0.0;
        for (int i = 0; i < N_STAMP; ++i) {
            tot += sum[i];
        }
        snprintf(b, sizeof(b), "pass timing: %" PRId64 " passes, %.3f ms/pass:", n, tot / n / 1e3); r += b;
        for (int i = 0; i < N_STAMP; ++i) {
            snprintf(b, sizeof(b), " %s %.3f", names[i], sum[i] / n / 1e3); r += b;
        }
        r += " | verify ms by width:";
        for (int w = 1; w < 32; ++w) {
            if (w_n[w] > 0) {
                snprintf(b, sizeof(b), " w%d %.3f (%" PRId64 ")", w, w_sum[w] / w_n[w] / 1e3, w_n[w]); r += b;
            }
        }
        const auto & dg = common_dflash_diag();
        if (dg.n_draft > 0) {
            snprintf(b, sizeof(b), " | dflash call: launch %.3f wait %.3f walk %.3f ms (%" PRId64 ")", dg.t_launch / 1e3 / dg.n_draft,
                    dg.t_wait / 1e3 / dg.n_draft, dg.t_walk / 1e3 / dg.n_draft, dg.n_draft); r += b;
        }
        return r;
    }
    void reset() {
        for (int i = 0; i < N_STAMP; ++i) {
            sum[i] = 0; t[i] = 0;
        }
        for (int w = 0; w < 32; ++w) {
            w_sum[w] = 0; w_n[w] = 0;
        }
        n = 0;
        auto & dg = common_dflash_diag();
        dg.t_launch = dg.t_wait = dg.t_walk = dg.n_draft = 0;
    }
};
// k19-handoff: the server yields its task queue to a worker while a decode runs; ending a yield waits for that worker to wake and
// stop (two thread hand-offs), which a short speculative pass pays three times (draft, verify, the drafter's pass over the rows).
// LLAMA_SERVER_YIELD_SMALL=0: work of at most 32 tokens runs without the yield (queued tasks then wait for the end of the pass);
// default 1 = always yield, as before
static bool srv_yield_small() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_SERVER_YIELD_SMALL");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}
template <typename Q, typename F>
static void srv_yield_or_run(Q & q, bool small, F && f) {
    if (small && !srv_yield_small()) {
        f();
        return;
    }
    q.yield_to_queue(std::forward<F>(f));
}

static server_pass_timer & srv_pass_timer() {
    static server_pass_timer t;
    return t;
}

struct server_batch {
    common_batch view; // the rendered sub-batch [off, off + n_tokens), see render()

    struct token {
        int32_t id_slot;
        llama_token token;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        bool output;
        bool is_prompt; // for stats tracking
        int32_t decision_order = 0;
        int32_t i_embd = -1; // row in embd, -1 if this is a token
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // a slot added speculative draft tokens to verify (see llama_set_verify)
    bool has_draft = false;

    // embedding entries, they can be mixed with tokens if the context supports it
    std::vector<float> embd; // n_embd per row

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    void init(llama_context * ctx, int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        view = common_batch(ctx);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, { pos, 0, 0, 0 }, output, is_prompt });
        return true;
    }

    // embd_in has n_embd values, pos has GGML_MROPE_SECTIONS values
    bool add_embd(int32_t id_slot, const float * embd_in, const llama_pos * pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, { pos[0], pos[1], pos[2], pos[3] }, output, is_prompt });
        tokens.back().i_embd = (int32_t) (embd.size() / n_embd);
        embd.insert(embd.end(), embd_in, embd_in + n_embd);
        return true;
    }

    bool has_embd() const {
        return !embd.empty();
    }

    void clear() {
        tokens.clear();
        embd.clear();
        view.clear();
        slot_batched      = nullptr;
        has_draft         = false;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // remove the entries after the first n
    void truncate(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= size());
        for (int32_t i = n; i < size(); i++) {
            if (tokens[i].i_embd >= 0) {
                embd.resize((size_t) tokens[i].i_embd * n_embd);
                break;
            }
        }
        tokens.resize(n);
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void set_decision_order(int32_t idx, int32_t order) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].decision_order = order;
    }

    // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
    void render(int32_t off, int32_t n_tokens) {
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        view.clear();
        for (int32_t i = off; i < off + n_tokens; i++) {
            const auto & t = tokens[i];
            if (t.i_embd >= 0) {
                view.add_embd({ embd.data() + (size_t) t.i_embd * n_embd, 1, (size_t) n_embd }, t.pos.data(), t.id_slot, t.output);
            } else {
                view.add(t.token, t.pos[0], t.id_slot, t.output);
            }
            view.tokens.back().decision_order = t.decision_order;
        }
    }
};

// Returning the buffer of a context checkpoint to the OS takes ~10 ms per 200 MiB (munmap), and checkpoints are dropped
// right before a prompt is processed (invalidated by the new prompt, or evicted to make room for a new one).
// The dropped checkpoints are freed on a background thread instead; env LLAMA_CKPT_FREE_ASYNC=0 frees them in place.
static void checkpoints_free_async(std::list<common_prompt_checkpoint> && dropped) {
    if (dropped.empty()) {
        return;
    }

    // their buffers go back to the checkpoint pool first (common_state_buf_recycle, LLAMA_CKPT_POOL)
    for (auto & ckpt : dropped) {
        common_state_buf_recycle(std::move(ckpt.data_tgt));
        common_state_buf_recycle(std::move(ckpt.data_dft));
    }

    static const bool disabled = [] {
        const char * e = getenv("LLAMA_CKPT_FREE_ASYNC");
        return e != nullptr && atoi(e) == 0;
    }();

    if (disabled) {
        dropped.clear();
        return;
    }

    std::thread([dropped = std::move(dropped)]() mutable {
        dropped.clear();
    }).detach();
}

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // k19-host (LLAMA_MTP_PROMPT_BG): joins the drafter's prompt pass running on a worker before the slot touches the draft context
    std::function<void()> spec_bg_join;
    void spec_bg_wait() const {
        if (spec_bg_join) {
            spec_bg_join();
        }
    }

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;
    // k19-sharednoise: the key and the mirrored chain of the target's shared-noise draws, read by the drafter
    common_shared_noise_draft spec_sn;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    std::vector<completion_token_output> spec_probs; // k13: n_probs of the accepted tokens, read from their verify rows before any rollback
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    // k19-rollback: the speculative checkpoint of the target is the host copy of its recurrent rows (llama_rs_snap_save), not spec_ckpt's data
    bool spec_rs_snap = false;

    // k9-draft: speculation cost governor (common/spec-governor.h); gov_step: what the current step is for it
    // (0 = not governed, 1 = plain because the governor held drafting back, 2 = a draft is verified, 3 = drafting gave no draft)
    common_spec_governor gov;
    int64_t gov_t_last = 0;
    int     gov_step   = 0;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    common_chat_input generated;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    // ckpt_keep_max >= -1 (k15-ttft): the caller knows the next prompt of this slot drops every checkpoint with pos_max above it, so
    // those are moved into the cache entry instead of copied (opt-in, LLAMA_PCS_MOVE_CKPT=1), or copied on a thread (at_request, default)
    // k15-ttft: the slot's prompt is in the prompt cache already, saved while the server was idle (LLAMA_PCS_IDLE_SAVE)
    bool prompt_saved_idle = false;
    bool prompt_saved_idle_new = false; // ... into a new entry (not one the cache held already)
    // k15-idlewait: that save moved the slot's checkpoints into the entry instead of copying them (they come back at the next request)
    bool ckpt_lent = false;

    bool prompt_save(server_prompt_cache & prompt_cache, llama_pos ckpt_keep_max = -2, bool force_async = false, bool lend_ckpt = false,
                     bool at_request = false) {
        spec_bg_wait();
        if (prompt.tokens.size() == 0) {
            return false;
        }
        // k15-idlewait: an idle save of the only slot moves the checkpoints into the entry (no copy on the server loop: G4 q4x, 63 MiB for
        // the 640-token warm prompt); ckpt_return gives them back at the next request (server_context_impl::get_available_slot)
        std::list<common_prompt_checkpoint> lent;
        if (lend_ckpt) {
            lent.splice(lent.end(), prompt.checkpoints);
        }
        struct lent_back {
            std::list<common_prompt_checkpoint> & from; std::list<common_prompt_checkpoint> & to;
            ~lent_back() { if (!from.empty()) { to.splice(to.begin(), from); } }
        } lent_guard { lent, prompt.checkpoints };

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        // opt-in (LLAMA_PCS_MOVE_CKPT=1): local q4x spilled multi-turn (mt_local, k15r36) ended +900 MiB RssAnon with the move on
        static const bool move_ckpt = getenv("LLAMA_PCS_MOVE_CKPT") != nullptr && atoi(getenv("LLAMA_PCS_MOVE_CKPT")) != 0;
        const bool mv = move_ckpt && ckpt_keep_max >= -1;
        // k15-idlewait (LLAMA_PCS_CKPT_COPY_ASYNC=0: copied here, as before): at a request save the checkpoints the next prompt erases
        // are copied on a thread of the prompt cache into pooled buffers (the same buffers and copies as before, the originals go back
        // to the pool once copied, as the prompt's erase would have put them); the entry waits for it like a state save in flight.
        // G4 q4x 12 GB chat: the copies of 2 checkpoints (125.6 MiB) were ~20 ms of the request.
        // k17: default on again (LLAMA_PCS_CKPT_COPY_ASYNC=0: off). Its +577 MiB end-of-run RssAnon (T4 LFM2.5 multi-turn) was the state
        // pool: oversized pooled buffers held by small checkpoints and refills racing the originals (common_state_buf_pending_add,
        // LLAMA_STATE_POOL_SMALL_FRESH, LLAMA_STATE_POOL_REL); with those the run ends below the copies in place
        static const bool copy_async = getenv("LLAMA_PCS_CKPT_COPY_ASYNC") == nullptr || atoi(getenv("LLAMA_PCS_CKPT_COPY_ASYNC")) != 0;
        const bool ca = !mv && copy_async && at_request && ckpt_keep_max >= -1;
        prompt_cache.clone_async_next = ca;
        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft, (mv || ca) ? &prompt.checkpoints : nullptr, ckpt_keep_max);
        if (cur == nullptr) {
            return false;
        }
        if (!lent.empty()) {
            for (const auto & c : lent) {
                common_state_buf_prefault(c.data_tgt.size(), 2); // as the copy would have asked (prompt_checkpoint_clone)
            }
            SLT_INF(*this, "idle save: %zu checkpoint(s) lent to the cache entry\n", lent.size());
            cur->prompt.checkpoints.splice(cur->prompt.checkpoints.end(), lent);
            ckpt_lent = true;
        }

        // k11 (LLAMA_PROMPT_SAVE_ASYNC=1, opt-in: on G4 it took 5 ms off the chat TTFT but the copies then delay the prompt's first
        // ubatch, Qwen3.6 MTP 32 GB pp -35% chat .. -3% 2k with md5 unchanged): the device copies are queued in stream order before the next decode, the host
        // copy into the cache entry runs on the context's state worker; the cache waits for it before it reads, erases or evicts
        // the entry (server_prompt_cache::wait_inflight). Larger states than LLAMA_PROMPT_SAVE_ASYNC_MAX_MIB (512) are saved as before.
        static const bool async_on = getenv("LLAMA_PROMPT_SAVE_ASYNC") != nullptr && atoi(getenv("LLAMA_PROMPT_SAVE_ASYNC")) != 0;
        static const size_t async_max = (size_t) (getenv("LLAMA_PROMPT_SAVE_ASYNC_MAX_MIB") ? std::max(0, atoi(getenv("LLAMA_PROMPT_SAVE_ASYNC_MAX_MIB"))) : 512) << 20;
        llama_state_seq_flags fl = (async_on || force_async) && cur_size <= async_max ? LLAMA_STATE_SEQ_FLAGS_ASYNC : LLAMA_STATE_SEQ_FLAGS_NONE;
        // k15-idlewait (LLAMA_PCS_REQ_SAVE_ASYNC=0: synchronous, as before): a save made at the request (where no idle save ran: a spilled
        // tier, a MoE cache) reads the state rows into the pinned staging buffers the context already holds (the checkpoint saves reserve
        // them) and the state worker copies them into the entry while the prompt runs; the next memory op waits for the reads only. G4
        // q4x 12 GB chat: the synchronous reads into the fresh entry (77 MiB, 82 calls with a sync each) were most of the 40 ms the request
        // waited before its prompt. Only where the staging holds the save already (LLAMA_STATE_SEQ_FLAGS_ASYNC_NO_GROW): no new pinned memory
        static const bool req_async = getenv("LLAMA_PCS_REQ_SAVE_ASYNC") == nullptr || atoi(getenv("LLAMA_PCS_REQ_SAVE_ASYNC")) != 0;
        if (fl == LLAMA_STATE_SEQ_FLAGS_NONE && req_async && at_request && (cur_size <= async_max || server_state_async_ext())) {
            fl = LLAMA_STATE_SEQ_FLAGS_ASYNC | LLAMA_STATE_SEQ_FLAGS_ASYNC_NO_GROW;
        }
        // k19-host30k: a save larger than LLAMA_PROMPT_SAVE_ASYNC_MAX_MIB, at a request or at idle, is asynchronous where the staging
        // the context pinned beforehand (server_state_async_ext) holds it, its rows past the staging synchronous as before
        if (fl == LLAMA_STATE_SEQ_FLAGS_NONE && force_async && server_state_async_ext()) {
            fl = LLAMA_STATE_SEQ_FLAGS_ASYNC | LLAMA_STATE_SEQ_FLAGS_ASYNC_NO_GROW;
        }

        const int64_t t_get0 = ggml_time_us();
        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, fl);
        SRV_TRC(" - prompt state read in %.2f ms\n", (ggml_time_us() - t_get0) / 1000.0);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, fl);
        }
        if (fl & LLAMA_STATE_SEQ_FLAGS_ASYNC) {
            prompt_cache.inflight = cur;
        }

        return true;
    }

    // k15-ttft: the RAM prompt-cache save of this slot (state + a clone of each checkpoint) writes into fresh host memory when the
    // buffer pool holds nothing large enough, and faulting those pages in made most of the save (local B200, Qwen3.6-35B-A3B MTP,
    // 640-token prompt: 77 MiB state + 2 x 62 MiB checkpoints, 80-290 ms against 8-16 ms into resident buffers). Once the prompt is
    // processed (and again when the slot goes idle) the prefault worker is asked for resident pool buffers of the sizes the next save
    // needs, so it runs while the slot generates or waits. Same copies into other buffers: the saved bytes are unchanged.
    // Opt-in (LLAMA_PCS_PREFAULT_NEXT=1): the prepared buffers stay resident while the slot idles (G4 q4x 32 GB: +113 MiB RSS), and
    // the 2 MiB pages of common_state_thp_alloc take most of the fault cost without holding memory
    void prefault_next_save() const {
        static const bool on = getenv("LLAMA_PCS_PREFAULT_NEXT") != nullptr && atoi(getenv("LLAMA_PCS_PREFAULT_NEXT")) != 0;
        if (!on || prompt.tokens.size() == 0) {
            return;
        }
        const size_t n_tgt = llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t n_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        common_state_buf_prefault(n_tgt, 1);
        // checkpoint clones (prompt_checkpoints_clone): one more buffer per checkpoint size than the state request leaves
        size_t ckpt_max = 0;
        int    n_ckpt   = 0;
        for (const auto & c : prompt.checkpoints) {
            ckpt_max = std::max(ckpt_max, c.data_tgt.size());
            n_ckpt  += c.data_tgt.size() > 0;
        }
        if (n_ckpt > 0) {
            common_state_buf_prefault(ckpt_max, n_ckpt + (n_tgt >= ckpt_max ? 1 : 0));
        }
        if (n_dft > 0) {
            common_state_buf_prefault(n_dft, 1);
        }
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        spec_bg_wait();
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        spec_bg_wait();
        prompt_saved_idle = false; prompt_saved_idle_new = false; ckpt_lent = false;
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;
    // k15-fn: the device greedy chain of this slot (set in the context while greedy requests run, see launch_slot_with_task)
    llama_sampler_ptr smpl_dev;
    llama_sampler_ptr smpl_dev_m; // the same with the EOG tokens excluded (ignore_eos requests), swapped in for them
    llama_sampler_ptr smpl_dev_k; // a device top-k (dev_topk_k candidates) for sampled requests the host chain can finish from them
    bool smpl_dev_set = false;

    // the device chain with the largest graph, set when the graphs are reserved (the others are swapped in without a reserve)
    llama_sampler * dev_reserve_chain() const {
        return smpl_dev_k ? smpl_dev_k.get() : smpl_dev_m ? smpl_dev_m.get() : smpl_dev.get();
    }

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        // kdp: a request with drafting off (the drafter's remap failed) ends here; the next one remaps again
        spec_off_req   = false;
        spec_lent_req  = false;
        spec_is_replay = false;
        spec_rs_snap   = false;

        last_nl_pos    = 0;
        generated      = {};
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();
        n_spec_pen = 0;

        n_predict_max = -1;

        // the device greedy chain lives with the slot and stays set for the next greedy request (setting a sampler reserves the graphs)
        if (!smpl_dev_set) {
            llama_set_sampler(ctx_tgt, id, nullptr);
        } else {
            llama_set_sampler_raw_logits(ctx_tgt, id, true);
        }
        llama_set_sampler_keep_logits(ctx_tgt, id, false);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool can_split() const {
        GGML_ASSERT(task);
        // a decision over a non-causal context reads the whole prompt in one batch
        if (task->type == SERVER_TASK_TYPE_DECISION && !llama_get_causal_attn(ctx_tgt)) {
            return false;
        }
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!task->need_embd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (!llama_get_memory(ctx_tgt)) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const auto pooling = llama_pooling_type(ctx_tgt);
        if (pooling == LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == LLAMA_POOLING_TYPE_RANK && llama_get_causal_attn(ctx_tgt)) {
            return true;
        }
        // a decision task reads its outputs from the last batch
        if (task->type == SERVER_TASK_TYPE_DECISION) {
            return true;
        }
        return false;
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        // a joint decision head reads the whole batch
        if (!task->decision.order.empty() || !other_slot.task->decision.order.empty()) {
            return false;
        }

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    // kdp (LLAMA_MTP_DRAFT_LEND): spec_off_req = the drafter's weights or compute buffers could not be mapped back after a lend, so
    // this request decodes without drafts (the output is the same); spec_lent_req = this request remapped them
    bool spec_off_req  = false;
    bool spec_lent_req = false;

    bool can_speculate() const {
        return !!spec && !spec_off_req;
    }

    int32_t n_spec_pen = 0; // k19-q4xaccept: passes drafted with the request's penalties (this request)

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        // k19-l4reg (LLAMA_SPEC_END_FULL_WINDOW=0: off): near the n_predict limit the draft is no longer cut to the tokens left, so the
        // last pass of a request runs the width every other pass ran instead of a new one (process_token stops at the limit and drops
        // the rest). A width a request never ran before builds its graph at that pass: Colab L4 12 GB Qwen3.8-Flash-Next UD-IQ4_XS MTP,
        // the 1-row decode a text left for its last token took 0.86-1.30 s against 30 ms for a 3-row verify window
        if (n_remaining() > 0 && n_remaining() - 1 < n_draft_max) {
            if (!spec_end_full_window()) {
                n_draft_max = n_remaining() - 1;
            } else if (n_remaining() - 1 < common_speculative_n_max(spec)) {
                static std::atomic<int64_t> n_kept{0};
                const int64_t k = ++n_kept;
                if (k <= 2 || (k & (k - 1)) == 0) {
                    SLT_INF(*this, "k19-l4reg: %d token(s) left before the n_predict limit: the draft window is kept (engaged %" PRId64 " times)\n",
                        n_remaining(), k);
                }
            }
        }

        // k17-strict: batch invariant verify under a MoE cache: the window (the sampled token + the drafts) stays within the tokens
        // whose experts the cache serves on the device, as the single token decodes it must equal (llama_verify_device_max_tokens);
        // a larger window would run its experts on the CPU from host weights. LLAMA_SPEC_EXACT_DEVICE_CAP=0: off
        if (const int32_t cap = llama_verify_device_max_tokens(ctx_tgt); cap > 0 && n_draft_max > cap - 1) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                SLT_INF(*this, "exact verify: draft windows capped at %d tokens, the most whose experts the MoE cache serves on the device "
                    "(a larger window runs them on the CPU from host weights; LLAMA_SPEC_EXACT_DEVICE_CAP=0: off)\n", cap);
            }
            n_draft_max = cap - 1;
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                // text embeddings broadcast the same position across the M-RoPE sections
                const llama_pos p = prompt.tokens.pos_next();
                const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
                GGML_ASSERT((int32_t) inp_embd.size() == batch.n_embd);
                add_ok &= batch.add_embd(id, inp_embd.data(), pos, true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            batch.has_draft = true;

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        spec_bg_wait();
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (task->is_child() || task->is_stateless()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            if (task && !task->is_child() && !task->is_stateless()) {
                prefault_next_save();
            }
            if (getenv("LLAMA_PCS_MEM_LOG") != nullptr) { // debug: what the slot and the buffer pool hold as the slot goes idle
                size_t csz = 0, ccap = 0;
                for (const auto & c : prompt.checkpoints) { csz += c.data_tgt.size() + c.data_dft.size(); ccap += c.data_tgt.capacity() + c.data_dft.capacity(); }
                SLT_INF(*this, "pcs mem: at release %zu ckpts, %.1f / %.1f MiB (size / capacity), pool %.1f MiB [%s]\n", prompt.checkpoints.size(),
                    csz/1048576.0, ccap/1048576.0, common_state_buf_pool_bytes()/1048576.0, common_state_buf_pool_desc().c_str());
            }

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        {
            // k19-hostgap: engagement of the task-queue yield's fast end (LLAMA_SERVER_YIELD_FAST=0: off), counts since start
            int64_t n_fast = 0, n_total = 0;
            server_queue_yield_counts(n_fast, n_total);
            SLT_INF(*this, "k19-hostgap: %" PRId64 " of %" PRId64 " task-queue yields ended without waiting for the worker (LLAMA_SERVER_YIELD_FAST=0 off)\n",
                    n_fast, n_total);
        }
        if (srv_pass_timer().on) {
            const std::string r = srv_pass_timer().report();
            if (!r.empty()) {
                SLT_INF(*this, "%s\n", r.c_str());
            }
            srv_pass_timer().reset();
        }
        {
            // k19-gramdev: device top-k rows of a grammar / reasoning budget request
            const auto ds = common_sampler_get_dev_stats(smpl.get());
            if (ds.n_rows > 0) {
                SLT_INF(*this, "device top-k grammar: %" PRId64 " rows, %" PRId64 " rejected, %" PRId64 " fallback, %" PRId64 " forced, %" PRId64 " full rows\n",
                        ds.n_rows, ds.n_rejected, ds.n_fallback, ds.n_forced, ds.n_full);
            }
        }
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        SLT_INF(*this,
                "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
                llama_perf_context(ctx_tgt).n_reallocs_same, llama_perf_context(ctx_tgt).n_reallocs_diff);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            if (n_spec_pen > 0) {
                SLT_INF(*this, "penalty-aware drafts: %d passes drafted after the request's penalties (LLAMA_SPEC_DRAFT_PEN)\n", n_spec_pen);
            }
            if (gov.enabled) {
                SLT_INF(*this, "spec governor: %" PRId64 " wins, %" PRId64 " losses, %" PRId64 " parked steps, %" PRId64 " drafted / %" PRId64
                        " plain steps (session)\n", gov.n_wins, gov.n_losses, gov.n_parked, gov.n_spec_steps, gov.n_plain_steps);
            }
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated.empty() ? debug_generated_text : generated.text;
            }
        }

        return res;
    }

    // the other slot continues from the tokens processed so far
    void copy_prompt_to(server_slot & other) const {
        spec_bg_wait();
        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.prompt = prompt.clone();
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        spec_bg_wait();
        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// encode the mtmd chunk at idx, batched with as many of the next chunks as possible
// returns 0 on success
static int encode_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx) {
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);

    mbatch.reset(mtmd_batch_init(mctx));
    int32_t res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return 0;
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                struct cb_data_t {
                    common_speculative * spec;
                    llama_context * ctx;
                } cb_data = { slot.spec, slot.ctx_tgt };

                static auto cb = [](const mtmd_helper_embd_batch * b, void * user_data) {
                    const auto * data = static_cast<cb_data_t *>(user_data);

                    common_batch batch(data->ctx);
                    for (int32_t i = 0; i < b->n_tokens; ++i) {
                        llama_pos pos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < b->n_pos; ++j) {
                            pos[j] = b->pos[j * b->n_tokens + i];
                        }
                        batch.add_embd({ b->embd + (size_t) i * b->n_embd, 1, (size_t) b->n_embd }, pos, b->seq_id, false);
                    }

                    return common_speculative_process(data->spec, batch) ? 0 : 1;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    &cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    if (encode_mtmd_chunk(slot, mbatch, idx) != 0) {
        return -1;
    }

    return try_decode();
}

// add an encoded mtmd chunk to the batch of the text tokens, instead of decoding it on its own like process_mtmd_chunk()
// embd is the output of the encoder for this chunk
// returns false if the batch is full
static bool add_mtmd_chunk(const server_slot & slot, const mtmd_input_chunk * chunk, const float * embd, server_batch & batch) {
    // positions are the ones of mtmd_helper_decode_image_chunk()
    const auto * image    = mtmd_input_chunk_get_tokens_image(chunk);
    const bool   is_mrope = mtmd_decode_use_mrope(slot.mctx);
    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_embd   = batch.n_embd;
    const llama_pos pos_0 = slot.prompt.tokens.pos_next();

    for (size_t i = 0; i < n_tokens; i++) {
        const llama_pos p = pos_0 + (llama_pos) i;
        llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, p };
        if (is_mrope && image) {
            const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
            pos[0] = rel.t;
            pos[1] = rel.y;
            pos[2] = rel.x;
            pos[3] = rel.z;
        }
        if (!batch.add_embd(slot.id, embd + i * n_embd, pos, slot.need_embd(), /* is_prompt */ true)) {
            return false;
        }
    }
    return true;
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;
    bool device_greedy_ok = false; // k15-fn: greedy requests sample on the device (see launch_slot_with_task)
    std::vector<llama_token> dev_greedy_mask; // the EOG tokens (ignore_eos), ascending: the mask of the slots' masked device chain
    int32_t dev_topk_k = 0; // k15-fn: the candidates of the device top-k chain (0: off, LLAMA_SERVER_DEVICE_TOPK=0)

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_decision_context decision;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            prompt_ssd_save_all();
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
        prompt_ssd.reset();
    }

    // write the slots and the RAM prompt cache to the SSD tier and wait for it (shutdown, sleep)
    void prompt_ssd_save_all() {
        if (!prompt_ssd || !prompt_cache || ctx_tgt == nullptr) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        for (auto & slot : slots) {
            if (slot.prompt.tokens.size() > 0) {
                slot.prompt_save(*prompt_cache);
            }
        }
        while (!prompt_cache->states.empty()) {
            prompt_cache->evict_front();
        }
        prompt_ssd->flush();
        SRV_INF("prompt cache: SSD tier holds %zu prompts (%.3f MiB), saving took %.1f ms\n",
                prompt_ssd->count(), prompt_ssd->size() / (1024.0*1024.0), (ggml_time_us() - t0) / 1000.0);
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    // k15-lfm2: spill mode 3 pinned the small batches (llama_moe_pin_small); a request that cannot draft (a prompt of long_prompt+ tokens
    // whose shortest repeat is longer than any match) lifts the pin for its decode and the next request restores it before its prompt
    bool spill_pinned     = false;
    bool spill_pin_lifted = false;

    common_speculative_ptr spec;

    // k19-host (LLAMA_MTP_PROMPT_BG=0 off): the MTP drafter's pass over a prompt batch runs on a worker beside the target's next decode when
    // that decode is the slot's last few prompt tokens (the [n-4] [4] split of a long prompt: the [4] waits for its experts over PCIe while
    // the drafter computes). The worker runs the same process_rows calls on the same rows in the same order, then the draft-context steps
    // the main thread queued for it (the checkpoint's draft part, the draft's seq_rm), so the draft's KV, drafts and text are those of
    // the in-line pass. The main thread joins it before any other use of the draft context or the speculative state
    struct spec_bg_state_t {
        std::thread                        thr;
        std::mutex                         m;
        std::condition_variable            cv;
        std::deque<std::function<void()>>  q;       // draft-context steps queued by the main thread, run after the segments
        bool                               closing = false;
        bool                               live    = false; // main thread: a worker is running
        bool                               hold    = false; // main thread: the next update_slots continues the slot's prompt beside it
        bool                               group   = false; // main thread: beside the next prompt batch of any size (the target holds)
        bool                               free_hold = false; // main thread: the target's free-only hold is set (k19-q4x262)
        bool                               fail    = false; // main thread: a pass the target's hold joined failed (reported at the next join)
        int                                slot_id = -1;    // main thread: the slot whose prompt batch the pass computes (k19-studio-np4)
        std::atomic<bool>                  ok      { true };
        int64_t                            t_launch = 0;
    } spec_bg;

    static constexpr int32_t spec_bg_tail_max = 64; // the next target decode it may run beside: at most this many tokens

    // k19-q4x262 (LLAMA_MTP_PROMPT_BG_FREE_HOLD=0 off): a pass launched beside the tail decode (not group mode) sets the target's
    // free-only hold (llama_set_nextn_hold_free). Without it, a tail decode that took the MoE cache banks back after a released prompt
    // freed the drafter's compute buffers and graph results while the pass computed in them (G4 q4x 12 GB ctx 262144 MTP, 8192-token
    // prompt: the server died silently at the first decode, base and head alike)
    static bool spec_bg_free_hold_on() {
        static const bool v = getenv("LLAMA_MTP_PROMPT_BG_FREE_HOLD") == nullptr || atoi(getenv("LLAMA_MTP_PROMPT_BG_FREE_HOLD")) != 0;
        return v;
    }

    // joins the worker (a no-op without one); false: its pass failed (the caller reports it, as the in-line pass's failure)
    bool spec_bg_join() {
        bool ok = spec_bg_join_impl();
        if (spec_bg.fail) {
            spec_bg.fail = false;
            ok = false;
        }
        // kdp: a drafter pass that failed after a lend (its compute buffers did not fit again) turns drafting off for the request
        if (!ok) {
            return spec_fail_soft("the drafter's pass on a worker failed");
        }
        for (const auto & sl : slots) {
            if (sl.is_processing() && sl.spec_lent_req && !sl.spec_off_req) {
                llama_mtp_draft_lend_note(ctx_dft, true); // the remapped drafter computed its pass
                break;
            }
        }
        return true;
    }

    // kdp: a failed drafter step in a request whose drafter weights were lent and remapped (or the failed remap itself): the processing
    // slots decode without drafts for the rest of the request instead of failing it (true); not after a lend: false (the caller fails)
    bool spec_fail_soft(const char * what, bool remap = false) {
        bool lent = remap;
        for (const auto & sl : slots) {
            lent |= sl.is_processing() && sl.spec_lent_req;
        }
        if (!lent) {
            return false;
        }
        for (auto & sl : slots) {
            if (sl.is_processing()) {
                sl.spec_off_req = true;
                sl.spec_draft.clear();
                sl.spec_i_batch.clear();
            }
        }
        llama_mtp_draft_lend_note(ctx_dft, false);
        SRV_WRN("%s after the MTP drafter weight lend: drafting is off for this request, the remap is retried on the next one "
                "(LLAMA_MTP_DRAFT_LEND=0 off)\n", what);
        return true;
    }

    // k19-host (LLAMA_MTP_PROMPT_BG_GROUP): the target context's next-n hold (llama_set_nextn_hold), on the server thread inside the target's
    // decode, before it writes / frees the rows the pass reads or frees the drafter's compute buffers
    static void spec_bg_hold_cb(void * ud) {
        auto * self = (server_context_impl *) ud;
        if (!self->spec_bg_join_impl()) {
            self->spec_bg.fail = true;
        }
    }

    bool spec_bg_join_impl() {
        if (!spec_bg.live) {
            return true;
        }
        if (spec_bg.group) {
            llama_set_nextn_hold(ctx_tgt, nullptr, nullptr);
        }
        if (spec_bg.free_hold) {
            llama_set_nextn_hold_free(ctx_tgt, nullptr, nullptr);
            spec_bg.free_hold = false;
        }
        const int64_t t0 = ggml_time_us();
        {
            std::lock_guard<std::mutex> lock(spec_bg.m);
            spec_bg.closing = true;
        }
        spec_bg.cv.notify_all();
        spec_bg.thr.join();
        spec_bg.live  = false;
        spec_bg.hold  = false;
        spec_bg.group = false;
        common_speculative_rows_ext_clear(spec.get());
        static const bool srv_phase = getenv("LLAMA_DECODE_PHASE_LOG") != nullptr && atoi(getenv("LLAMA_DECODE_PHASE_LOG")) != 0;
        if (srv_phase) {
            SRV_INF("srv phase: drafter joined, waited %.2f ms, %.2f ms after its launch\n", (ggml_time_us() - t0)/1e3, (ggml_time_us() - spec_bg.t_launch)/1e3);
        }
        return spec_bg.ok.load();
    }

    // k19-studio-np4 (LLAMA_MTP_PROMPT_BG_MULTI=0 off): the slot whose prompt the drafter's background pass may accompany: the only
    // slot (-np 1), or with several slots the only one that is processing (the others idle: Studio's -np 4 with one user). Before, the
    // pass ran on a worker only at -np 1; at -np 4 each 8192-token group of a long prompt waited for its drafter pass in line (G4 Flash-Next
    // + Q8_0 MTP 16 GB, 30k: 93-117 ms per group, TTFT 6.356 vs 6.162 s at -np 1). Every other slot that starts work joins the worker
    // first (pre_decode, and the slot's prompt_load / prompt_save / prompt_clear / release), as the draft context is shared
    server_slot * spec_bg_slot() {
        if (slots.size() == 1) {
            return &slots[0];
        }
        static const bool multi_env = getenv("LLAMA_MTP_PROMPT_BG_MULTI") == nullptr || atoi(getenv("LLAMA_MTP_PROMPT_BG_MULTI")) != 0;
        if (!multi_env) {
            return nullptr;
        }
        server_slot * one = nullptr;
        for (auto & sl : slots) {
            if (sl.is_processing()) {
                if (one != nullptr) {
                    return nullptr;
                }
                one = &sl;
            }
        }
        return one;
    }

    // for callers that cannot throw: a failed pass ends every slot, as decode()'s throw does
    void spec_bg_join_or_abort() {
        if (!spec_bg.live) {
            return;
        }
        if (!spec_bg_join()) {
            SRV_ERR("%s", "failed to process speculative batch\n");
            abort_all_slots("failed to process speculative batch");
        }
    }

    // a draft-context step of the main thread while the worker runs: queued behind its segments, else run here
    void spec_bg_push(std::function<void()> && fn) {
        if (!spec_bg.live) {
            fn();
            return;
        }
        {
            std::lock_guard<std::mutex> lock(spec_bg.m);
            spec_bg.q.push_back(std::move(fn));
        }
        spec_bg.cv.notify_all();
    }

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_ssd>   prompt_ssd; // SSD tier of the prompt cache (--cache-ssd-dir)
    std::unique_ptr<server_prompt_cache> prompt_cache;

    // k11 (LLAMA_LM_CKPT_MERGE): absolute batch indices where the server would have ended a prompt batch for the checkpoint
    // 4 + n_ubatch tokens before the end; the batch goes on, its ubatches still end there (llama_set_ubatch_breaks), so a
    // layer-major prompt reads the host experts once for both (the checkpoint itself is not made)
    std::vector<int32_t> lm_breaks;
    // k15-pp: the checkpoint at the first merged break, taken inside the decode (llama_set_state_capture): slot, its token count there
    // k15-pp2: up to two (the far and the near checkpoint of one prompt batch)
    struct lm_cap_req_t {
        int     slot;
        int64_t n_tokens;
        int32_t batch;    // absolute batch index of the break
    };
    std::vector<lm_cap_req_t> lm_caps;
    // LLAMA_STATE_CAPTURE_MAX=1: one capture per decode (the near checkpoint is then a separate call beside a far capture)
    int lm_cap_max() const {
        static const int n = getenv("LLAMA_STATE_CAPTURE_MAX") ? std::max(1, std::min(2, atoi(getenv("LLAMA_STATE_CAPTURE_MAX")))) : 2;
        return n;
    }

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;
    int32_t spec_active_cap = 0; // k19-small1115: drafts per window while another slot is active (0: no cut; load_model)
    bool    spec_g4t_raised = false; // k19-g4tail: the rule raised the MTP windows to n_def drafts (load_model)
    bool    spec_chat_deep  = false; // the MTP windows of the chat bucket draft deeper (load_model)
    int32_t spec_g4t_cap    = 0;     // k19-g4tail2: drafts per window when the placed MoE cache is outside the band (0: no cut)

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        if (spec_bg.live) {
            spec_bg_join();
        }
        if (prompt_cache) {
            prompt_cache->wait_inflight(); // an asynchronous prompt save is complete before the contexts go
        }
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            prompt_ssd_save_all();
            prompt_ssd.reset();
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        // k11-decode: n-gram drafting on by default where nothing else drafts (server-spec-auto.h, LLAMA_SPEC_AUTO=0: off)
        server_spec_auto_apply(params);

        // k19-tpp (LLAMA_SPEC_TPP=0: off): an MTP drafter at 2 drafts (Studio's --spec-draft-n-max 2, greedy, no p-min) keeps those 2 and
        // adds a 3rd where the measured tokens per ms of a 4-row verify beat the 3-row one (common_spec_mtp_cost::tpp_more / tpp_keep).
        // Verify rows of 3 and 4 tokens run the same kernels (LLAMA_SPEC_TPP_VERIFY), so the greedy text stays that of 2 drafts
        {
            const char * e = getenv("LLAMA_SPEC_TPP");
            auto & sp = params.speculative;
            // (types starts as { NONE } and --spec-type appends: the only drafting type must be draft-mtp)
            const bool mtp_only = std::count_if(sp.types.begin(), sp.types.end(), [](common_speculative_type t) {
                return t != COMMON_SPECULATIVE_TYPE_NONE;
            }) == 1 && std::find(sp.types.begin(), sp.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != sp.types.end();
            // by default only where every GPU is NVIDIA Ampere or newer (compute capability >= 8.0): there a 4-row verify row runs the
            // kernels of a 3-row one (mmvq keeps its 4 warps up to 4 columns, F32 weights stay on mmvf under the verify tag). Turing's
            // mmvq table halves the q6_K warps at 4 columns, so a T4 window's head rows sum in another order (Kaggle T4 q4x + MTP heads:
            // +3% decode, but chat text varied launch to launch); AMD and other GPUs report no capability. LLAMA_SPEC_TPP=1 forces it on
            // k19-small1115 (LLAMA_SPEC_TPP_ACTIVE, default on; 0 = only with --parallel 1, as before): with --parallel N the windows
            // are decided per round from the active slots: a lone request drafts as on a --parallel 1 server (the wider windows below),
            // and while another slot is active every window is cut to the floor of 2 drafts (update_slots), the windows of a
            // --parallel N server without them. Colab A100 40 GB Qwen3.6-35B-A3B UD-Q3_K_XL in-file MTP n2 (ks_a1): one request on
            // --parallel 4 decoded 324.3 tok/s vs 342.6 on --parallel 1, the same text
            const bool mtp_shape = mtp_only && sp.draft.n_max == 2 && sp.draft.n_min == 0 && sp.draft.p_min <= 0.0f && !sp.draft.probabilistic &&
                (params.n_parallel == 1 || spec_tpp_active_on());
            // the GPUs the fit will use (server-tpp.h: --device's list, else every GPU; with --split-mode none the main GPU only)
            std::vector<ggml_backend_dev_t> all_devs;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                all_devs.push_back(ggml_backend_dev_get(i));
            }
            std::vector<ggml_backend_dev_t> req_devs;
            for (ggml_backend_dev_t dev : params.devices) {
                if (dev != nullptr) {
                    req_devs.push_back(dev);
                }
            }
            const std::vector<ggml_backend_dev_t> gpus = tpp_fit_gpus(req_devs, all_devs,
                [](ggml_backend_dev_t d) {
                    const enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
                    return t == GGML_BACKEND_DEVICE_TYPE_GPU ? TPP_GPU : t == GGML_BACKEND_DEVICE_TYPE_IGPU ? TPP_IGPU : TPP_NOT_GPU;
                },
                [](ggml_backend_dev_t a, ggml_backend_dev_t b) { return ggml_backend_dev_backend_reg(a) == ggml_backend_dev_backend_reg(b); },
                params.split_mode == LLAMA_SPLIT_MODE_NONE, params.main_gpu);
            int n_gpu = 0, cc_min = INT_MAX, cc_max = -1;
            for (ggml_backend_dev_t dev : gpus) {
                if (!mtp_shape) {
                    break;
                }
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                auto cc_fn = reg ? (int (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
                const int cc = cc_fn != nullptr ? cc_fn(dev) : -1;
                n_gpu++;
                cc_min = std::min(cc_min, cc);
                cc_max = std::max(cc_max, cc);
            }
            const bool dev_ok = n_gpu > 0 && cc_min >= 800;
            // The restricted DFlash2 draft head beside a MoE cache that holds most of the experts (LLAMA_DFLASH_DRAFT_VOCAB_WIDE, see
            // llama_context and the fit) is on where every GPU the fit uses has compute capability 12.0, the only cards it is measured on:
            // Colab G4, Qwen3.8-Flash-Next UD-IQ4_XS + DFlash2 n6, 48 GB, 3 launches: chat 281.2 -> 285.8 tok/s (+1.6%), the same text.
            // An explicit LLAMA_DFLASH_DRAFT_VOCAB_WIDE keeps its value
            if (getenv("LLAMA_DFLASH_DRAFT_VOCAB_WIDE") == nullptr && sp.has_dft() && !gpus.empty()) {
                bool all_120 = true;
                for (ggml_backend_dev_t dev : gpus) {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    auto cc_fn = reg ? (int (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
                    all_120 = all_120 && cc_fn != nullptr && cc_fn(dev) == 1200;
                }
                if (all_120) {
                    common_set_env("LLAMA_DFLASH_DRAFT_VOCAB_WIDE", "1");
                }
            }
            // no wider window where the experts will not all fit on the devices: there the fit runs an MoE cache, whose residency a
            // wider window moves, and sizing the outputs for 4 rows instead of 3 already takes from the cache (G4 fn4 16 GB, ktp_g4e: MoE
            // cache 5528 vs 5538 MiB, text != base, 0.94-0.96 of base at 16k / 30k). Decided before the fit, from the GGUF bytes (target
            // and drafter + 1.5 GiB) against the devices' free memory; the speculative side also turns the window off if a cache runs
            // anyway. LLAMA_SPEC_TPP_CACHED=1: on there too
            // (an integrated GPU's free memory is the host's available memory, ggml-cuda's UMA path: several of them count it once)
            size_t dev_free = 0;
            size_t igpu_free = 0;
            for (ggml_backend_dev_t dev : gpus) {
                if (!mtp_shape) {
                    break;
                }
                size_t f = 0, t = 0;
                ggml_backend_dev_memory(dev, &f, &t);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    igpu_free = std::max(igpu_free, f);
                } else {
                    dev_free += f;
                }
            }
            dev_free += igpu_free;
            // k19-bigverify (opt-in, LLAMA_SPEC_TPP_NEED_PLE=0): without tensor overrides the per-layer embedding table never takes device
            // memory, so it can be left out of the need. Colab A100 80 GB, Qwen3.8-Flash-Next UD-IQ4_XS + MTP drafter: need 94815 MiB with
            // its 26.8 GiB table against 80725 MiB free keeps the windows at 2 drafts although the fit places every expert on the card and
            // runs no MoE cache. Left out (bv_abh), k19-tpp + k19-depth's 4-draft windows there: chat 156.1 -> 159.7 tok/s (+2.3%) but 8k
            // 148.2 -> 120.6 (-18.6%: each verify row costs ~2.9 ms on that card, 4.06 / 3.05 tokens per pass at 25.5 / 25.3 ms). So the
            // default (unset / 1) still counts the whole file until the deep windows are priced per device
            const uint64_t tgt_bytes = mtp_shape ? tpp_gguf_bytes(params.model.path) : 0;
            const uint64_t ple_bytes = mtp_shape && tgt_bytes > 0 && !tpp_host_experts(params.tensor_buft_overrides, params.moe_cache_size, params.moe_cache_auto) &&
                getenv("LLAMA_SPEC_TPP_NEED_PLE") != nullptr && atoi(getenv("LLAMA_SPEC_TPP_NEED_PLE")) == 0 ? tpp_host_input_bytes(params.model.path) : 0;
            const uint64_t need = mtp_shape ? tgt_bytes - std::min(ple_bytes, tgt_bytes) + tpp_gguf_bytes(sp.draft.mparams.path) + (1536ull << 20) : 0;
            if (ple_bytes > 0) {
                SRV_INF("k19-bigverify: the per-layer embedding table (%.0f MiB) stays in host memory: not counted in the device need of the "
                    "draft window rule (LLAMA_SPEC_TPP_NEED_PLE=0; unset: counted)\n", ple_bytes/1048576.0);
            }
            // (no GPU: no MoE cache. LLAMA_SPEC_TPP_FORCE_CACHED=1, a test hook: as if the experts did not fit)
            const char * ce = getenv("LLAMA_SPEC_TPP_CACHED");
            const char * fe = getenv("LLAMA_SPEC_TPP_FORCE_CACHED");
            // (experts placed on the host by hand, --cpu-moe / --n-cpu-moe / -ot, or a fixed --moe-cache-mib: a cache tier too)
            const bool host_experts = tpp_host_experts(params.tensor_buft_overrides, params.moe_cache_size, params.moe_cache_auto);
            const bool cache_expected = (fe != nullptr && atoi(fe) != 0) || (n_gpu > 0 && (host_experts || !(need > (1536ull << 20) && need <= dev_free)));
            const bool exact_mtp = [] {
                const char * x = getenv("LLAMA_MTP_EXACT");
                return x != nullptr && atoi(x) != 0;
            }();
            // Only where the routed experts fit in host RAM: where they do not (Colab L4, 52 GB of RAM for the 57.6 GB of experts of fn4n)
            // the cache's misses are read from the disk and each verify row costs ~27% of a pass (fm_l1: 4 drafts -5% chat), so the windows
            // stay at 2 there. LLAMA_SPEC_DEPTH_CACHED_DISK=1: on there too; LLAMA_SPEC_DEPTH_CACHED_RAM_MIB=N: N MiB of RAM instead of
            // MemAvailable (test hook)
            const uint64_t ram_avail = [] {
                const char * r = getenv("LLAMA_SPEC_DEPTH_CACHED_RAM_MIB");
                return r != nullptr ? (uint64_t) (std::max(0.0, atof(r))*1048576.0) : llama_available_memory();
            }();
            bool disk_ok = [] {
                const char * d = getenv("LLAMA_SPEC_DEPTH_CACHED_DISK");
                return d != nullptr && atoi(d) != 0;
            }();
            const char * dc_env = getenv("LLAMA_SPEC_DEPTH_CACHED");
            int depth_cached = dc_env != nullptr ? atoi(dc_env) : 0;
            // k19-fnmtp2: the default where a verify row beside the cache is cheap (LLAMA_SPEC_DEPTH_CACHED unset; =0: 2 drafts, =N: N).
            // An MTP drafter of its own (-md) at Studio's shape drafts LLAMA_SPEC_DEPTH_CACHED_N (4) per window below LLAMA_SPEC_DEPTH_POS
            // when (a) the cache is expected to hold at least LLAMA_SPEC_DEPTH_CACHED_SHARE (0.30) of the routed experts, estimated before
            // the fit as 1 - (device need without the host-only per-layer embedding table - free) / expert bytes, and (b) the whole target
            // GGUF is held in host memory (on a memory file system, or its bytes within the available memory), so the rows' expert misses
            // and per-layer embedding rows come from RAM, not the disk. G4 RTX PRO 6000 96 GB, Qwen3.8-Flash-Next UD-IQ4_XS from RAM + its
            // MTP drafter, Studio argv, 3 launches, this rule against its kill switch: 48 GB (cache 67%) chat
            // +10.1% / 8k +18.0%, 32 GB (41%) +10.0% / +3.0%; at 16 GB (14%: rule off) 4 drafts measured chat +7.8% (12.8% launch spread) /
            // 8k -6.7%. A verify row there costs 2.2-2.5 ms of a 12.6-13.0 ms pass at 48 / 32 GB, ~3.5 ms of 15.2 at 16 GB. Colab A100 80 GB
            // at 48 GB chat +17.7% / 8k +1.6%. The A100 40 GB VM of round 1 (93.7 GB model, 83 GB of RAM: rows read from the disk) stays at 2
            // (decided before the cache share bands below: where these deeper windows are the default, those bands leave the
            // windows to this rule, see share_deep)
            const bool deep_try = dc_env == nullptr && mtp_shape && cache_expected && sp.has_dft() && !exact_mtp && !host_experts &&
                (e != nullptr ? atoi(e) != 0 : dev_ok) && !(ce != nullptr && atoi(ce) != 0);
            const uint64_t exp_b    = deep_try ? common_moe_expert_bytes(params.model.path) : 0;
            const uint64_t ple_b    = deep_try ? tpp_host_input_bytes(params.model.path) : 0;
            const uint64_t need_dev = need - std::min(need, ple_b);
            const uint64_t short_b  = need_dev > dev_free ? need_dev - dev_free : 0;
            const double   deep_share    = exp_b > 0 ? 1.0 - (double) short_b/(double) exp_b : 0.0;
            const double   deep_share_min = [] {
                const char * x = getenv("LLAMA_SPEC_DEPTH_CACHED_SHARE");
                return x != nullptr ? atof(x) : 0.30;
            }();
            const int n_def = [] {
                const char * x = getenv("LLAMA_SPEC_DEPTH_CACHED_N");
                return x != nullptr ? std::clamp(atoi(x), 3, 8) : 4;
            }();
            const bool in_ram_fs = deep_try && tpp_file_in_ram_fs(params.model.path);
            const bool resident  = in_ram_fs || (tgt_bytes > 0 && ram_avail > 0 && tgt_bytes <= ram_avail);
            // (short_b == 0: every routed expert is expected on the devices and the need only reads high from the drafter and margin,
            // k19-bigverify's A100 80 GB full card, where 4 drafts measured 8k -18.6%: not this rule's tier)
            const bool def_on    = deep_try && exp_b > 0 && short_b > 0 && deep_share >= deep_share_min && resident;
            // k19-g4tail (LLAMA_MOE_G4TAIL=0: off, the int3 rule above alone): a cache expected to hold LLAMA_MOE_G4TAIL_LO .. _HI (0.07 ..
            // 0.17, the estimate; G4 16 GB reads 10.8% here, 14.4% once placed) of the routed experts, on compute capability 12.0 cards
            // only, with the target in RAM: the windows draft n_def (4) too, and the MoE cache's cross-layer prefetch is on in its band
            // (LLAMA_MOE_PREFETCH_AUTO=2 for llama_moe_cache: top 10 predicted experts, at most 1 copied per layer, generation and verify
            // windows only, where the placed cache holds 10-20%). With the route tail skip off (int3 c5d7c3bc40d) every rank 7-9 miss is a
            // host expert or a copy; the two together, Colab G4 Flash-Next UD-IQ4_XS MTP n2 16 GB, 3 launches on two VMs: chat +8.3 / +9.0%
            // (held-out text +9.5%), 8k +8.0 / +10.8% (held-out -1.3%); long48 KLD inside the -fa off band on all 8 statistics.
            // An explicit LLAMA_SPEC_DEPTH_CACHED / LLAMA_MOE_PREFETCH / LLAMA_MOE_PREFETCH_AUTO keeps its value
            const bool g4tail_on = [] {
                const char * x = getenv("LLAMA_MOE_G4TAIL");
                return x == nullptr || atoi(x) != 0;
            }();
            const double g4t_lo = getenv("LLAMA_MOE_G4TAIL_LO") != nullptr ? atof(getenv("LLAMA_MOE_G4TAIL_LO")) : 0.07;
            const double g4t_hi = getenv("LLAMA_MOE_G4TAIL_HI") != nullptr ? atof(getenv("LLAMA_MOE_G4TAIL_HI")) : 0.17;
            // k19-g4tail2: not above LLAMA_FIT_LONGCTX_MIN_CTX (32768, where --fit's long-context ladder takes over; n_ctx 0 = the
            // trained context counts as above): there the deeper windows changed the fit itself. G4 16 GB ctx 262144: priced at 4
            // drafts the requantized drafter was left out, and the placement moved (49 layers, n_ubatch 2048, cache 6233 MiB against
            // 50 layers, n_ubatch 512, 3915 MiB with the rule off): decode chat / 8k 87.4 / 82.3 tok/s against 133.0 / 125.2. Above the
            // bound the rule sets nothing, as LLAMA_MOE_G4TAIL=0
            const int64_t g4t_ctx_max = [] {
                const char * x = getenv("LLAMA_FIT_LONGCTX_MIN_CTX");
                return x != nullptr && x[0] != 0 ? (int64_t) atoll(x) : (int64_t) 32768;
            }();
            const bool g4t_ctx_ok = params.n_ctx > 0 && (int64_t) params.n_ctx <= g4t_ctx_max;
            const bool g4t_band = deep_try && !def_on && g4tail_on && exp_b > 0 && short_b > 0 && resident && deep_share >= g4t_lo && deep_share < g4t_hi &&
                n_gpu > 0 && cc_min == 1200 && cc_max == 1200;
            const bool g4t = g4t_band && g4t_ctx_ok;
            // Between the two rules (from LLAMA_MOE_G4TAIL_HI, 0.17, up to LLAMA_SPEC_DEPTH_CACHED_SHARE, 0.30) the windows stayed at 2
            // drafts on compute capability 12.0 cards. There the deep windows' rule applies too, under the same conditions as the band
            // below it (target in RAM, n_ctx not above LLAMA_FIT_LONGCTX_MIN_CTX), drafting LLAMA_SPEC_DEPTH_MID_N (default 3) per
            // window; no prefetch and no host-path change (both stay as at the rule's floor). Colab G4, Qwen3.8-Flash-Next UD-IQ4_XS +
            // its MTP drafter, Studio argv, 24 GB (25.0% expected), 3 launches: 3 drafts chat 185.7 -> 194.8 tok/s, 8k 163.0 -> 174.8;
            // 4 drafts chat 191.5, 8k 155.5 (each extra verify row misses more experts at this share). LLAMA_SPEC_DEPTH_MID=0,
            // LLAMA_MOE_G4TAIL=0 or an explicit LLAMA_SPEC_DEPTH_CACHED_SHARE: off
            const bool mid_on = [] {
                const char * x = getenv("LLAMA_SPEC_DEPTH_MID");
                return x == nullptr || atoi(x) != 0;
            }();
            const int n_mid = [] {
                const char * x = getenv("LLAMA_SPEC_DEPTH_MID_N");
                return x != nullptr ? std::clamp(atoi(x), 3, 8) : 3;
            }();
            const bool def_mid = deep_try && !def_on && !g4t_band && mid_on && g4tail_on && getenv("LLAMA_SPEC_DEPTH_CACHED_SHARE") == nullptr &&
                exp_b > 0 && short_b > 0 && resident && deep_share >= g4t_hi && deep_share < deep_share_min && g4t_ctx_ok &&
                n_gpu > 0 && cc_min == 1200 && cc_max == 1200;
            // The same 3-draft windows just above that floor, up to LLAMA_SPEC_DEPTH_MID_HI (0.33; 0: the deep windows' own depth from
            // the floor): there a 4th verify row still misses enough experts that the narrower window pays. Colab G4, Qwen3.8-Flash-Next
            // UD-IQ4_XS + its MTP drafter, Studio argv, 3 launches on three texts each, 3 against 4 drafts: 27 GB (30.4% expected) chat
            // +2..+8%, 8k -3..+14%; 28 GB (32.1%) chat +1..+8%, 8k +2..+38%; from 29 GB (33.9%) on 4 drafts win chat on every text
            // (29 / 30 / 32 / 36 GB)
            const double mid_hi = [] {
                const char * x = getenv("LLAMA_SPEC_DEPTH_MID_HI");
                return x != nullptr ? atof(x) : 0.33;
            }();
            const bool def_mid_hi = def_on && mid_on && g4tail_on && getenv("LLAMA_SPEC_DEPTH_CACHED_SHARE") == nullptr &&
                getenv("LLAMA_SPEC_DEPTH_CACHED_N") == nullptr && deep_share < mid_hi && g4t_ctx_ok && n_gpu > 0 && cc_min == 1200 && cc_max == 1200;
            // k19-sharegate (LLAMA_SPEC_TPP_SHAREGATE=0: off, the blanket rule above): a cache expected to hold nearly every routed
            // expert (estimated share = 1 - (need - free) / expert bytes, at least LLAMA_SPEC_TPP_CACHED_SHARE, default 0.85) keeps the
            // priced 3-draft windows, as LLAMA_SPEC_TPP_CACHED=1. The rule above came from fn4 at a 9-14% share; at ~92% (q4x on a 24 GB
            // L4) a 4th verify row misses few experts. Not for experts placed on the host by hand, not with LLAMA_MTP_EXACT=1, only
            // where the routed experts fit in host RAM (misses not read from the disk)
            const char * sg = getenv("LLAMA_SPEC_TPP_SHAREGATE");
            const bool share_try = mtp_shape && cache_expected && ce == nullptr && !(fe != nullptr && atoi(fe) != 0) &&
                !host_experts && n_gpu > 0 && (sg == nullptr || atoi(sg) != 0) &&
                !(getenv("LLAMA_MTP_EXACT") != nullptr && atoi(getenv("LLAMA_MTP_EXACT")) != 0) &&
                (e != nullptr ? atoi(e) != 0 : dev_ok);
            const uint64_t share_exp = share_try ? common_moe_expert_bytes(params.model.path) : 0;
            const double share_min = [] {
                const char * s = getenv("LLAMA_SPEC_TPP_CACHED_SHARE");
                return s != nullptr ? atof(s) : 0.85;
            }();
            const uint64_t share_short = need > dev_free ? need - dev_free : 0;
            const double share_est = share_exp > 0 ? 1.0 - (double) share_short/(double) share_exp : 0.0;
            const uint64_t share_ram = share_exp > 0 ? llama_available_memory() : 0;
            const bool share_hi = share_exp > 0 && share_est >= share_min && (share_ram == 0 || share_exp <= share_ram);
            // k19-q4cap (LLAMA_SPEC_TPP_CACHED_LOW=0 off): from LLAMA_SPEC_TPP_CACHED_SHARE_LOW (0.30) of the routed experts up to the
            // share above, the 3rd draft only in the windows of the chat bucket (KV positions below 1536, LLAMA_SPEC_TPP_CTX_EXTRA
            // "1,0,0,0,0" unless given): longer contexts keep 2 drafts. Qwen3.6-35B-A3B UD-Q4_K_XL + MTP (3 launches, priced
            // windows at every position): chat G4 @16 (61%) +8.8%, L4 16 GB (61%) +6.3%, L4 12 GB (39%) +9.6%, L4 8 GB (17%) 0%; 8k
            // 0..+2.2% (G4 @16 with the larger cache -2.7%), 30k -8.5% (L4 12 GB): so the priced 3rd draft stays off past the chat bucket.
            // Qwen3.8 Flash-Next UD-IQ4_XS at a 9-14% share (where the blanket rule above was measured) stays below it. Same output class (MTP verify), KLD inside the -fa off band (L4 12 GB)
            const char * lo_e = getenv("LLAMA_SPEC_TPP_CACHED_LOW");
            const double share_lo = [] {
                const char * s = getenv("LLAMA_SPEC_TPP_CACHED_SHARE_LOW");
                return s != nullptr ? atof(s) : 0.30;
            }();
            // Neither band below caps an MTP drafter of its own (-md) where the deeper windows above are its default (def_on / g4t): their
            // estimate counts the host-only per-layer embedding table as device need and reads far below the cache's real share there.
            // Colab G4 48 GB, Qwen3.8-Flash-Next UD-IQ4_XS + its MTP drafter: 20% read against 67.7% held, so the chat band set 3-draft
            // windows in place of 6 / 4 (chat 262.2 -> 235.2 tok/s, 8k 220.8 -> 211.3). Built-in MTP heads (no -md) never take the deep
            // windows and keep both bands
            const bool share_deep = def_on || g4t || def_mid;
            const bool share_low = !share_hi && !share_deep && share_exp > 0 && (lo_e == nullptr || atoi(lo_e) != 0) && share_est >= share_lo &&
                share_lo < share_min && (share_ram == 0 || share_exp <= share_ram);
            // LLAMA_SPEC_TPP_CACHED_SHARE_CHAT (default 0.15, 0 = off): from that share up to the band above, the 3rd draft in the chat
            // bucket only (the longer buckets keep 2 drafts). L4 capped at 8 GB (17%), Qwen3.6-35B-A3B UD-Q4_K_XL + MTP, three VMs, 10
            // interleaved launches per arm: chat +2.9% (2.69 -> 3.28 tokens per pass), 8k level; the 8k bucket's 3rd draft did not pay there
            const double share_chat_lo = [] {
                const char * s = getenv("LLAMA_SPEC_TPP_CACHED_SHARE_CHAT");
                return s != nullptr ? atof(s) : 0.15;
            }();
            const bool share_chat = !share_hi && !share_low && !share_deep && share_exp > 0 && (lo_e == nullptr || atoi(lo_e) != 0) &&
                share_chat_lo > 0.0 && share_est >= share_chat_lo && share_chat_lo < share_lo && (share_ram == 0 || share_exp <= share_ram);
            if (share_deep && !share_hi && share_exp > 0 && (lo_e == nullptr || atoi(lo_e) != 0) &&
                    share_est >= std::min(share_lo, share_chat_lo > 0.0 ? share_chat_lo : share_lo)) {
                SRV_INF("MoE cache expected to hold %.1f%% of the routed experts by the device need, %.1f%% without the host-only tables: "
                    "the deeper MTP windows are the default here, the 3rd-draft share bands stay off\n", 100.0*share_est, 100.0*deep_share);
            }
            if (share_chat && getenv("LLAMA_SPEC_TPP_CTX_EXTRA") == nullptr) {
                common_set_env("LLAMA_SPEC_TPP_CTX_EXTRA", "1,0,0,0,0");
                SRV_INF("MoE cache expected to hold %.1f%% of the routed experts (from %.2f): the 3rd draft in the chat bucket only "
                    "(LLAMA_SPEC_TPP_CACHED_SHARE_CHAT=0: off)\n", 100.0*share_est, share_chat_lo);
            }
            const bool share_ok = share_hi || share_low || share_chat;
            // LLAMA_SPEC_TPP_CACHED_LOW_MID (default 1, =0 off): in that band the windows of the 6144..12288 bucket take the 3rd draft
            // too. Their 4-row windows carry the verify tag (3-row kernels); beside a cache holding most experts the extra row adds few
            // misses while a pass carries ~0.6 more tokens. L4 capped at 16 GB (61%), Qwen3.6-35B-A3B UD-Q4_K_XL + MTP, 8k prompt, 3
            // interleaved launches: decode 96.9 -> 110.0 tok/s (2.59 -> 3.24 tokens per pass), same text there; on other cards the cache
            // residency can differ and change the text within the -fa off KLD band; chat, 16k and 30k use other buckets
            static const bool low_mid = getenv("LLAMA_SPEC_TPP_CACHED_LOW_MID") == nullptr || atoi(getenv("LLAMA_SPEC_TPP_CACHED_LOW_MID")) != 0;
            if (share_low && getenv("LLAMA_SPEC_TPP_CTX_EXTRA") == nullptr) {
                common_set_env("LLAMA_SPEC_TPP_CTX_EXTRA", low_mid ? "1,0,1,0,0" : "1,0,0,0,0");
                if (low_mid) {
                    SRV_INF("%s", "3rd draft also in the 6144..12288 bucket beside the MoE cache (LLAMA_SPEC_TPP_CACHED_LOW_MID=0: off)\n");
                }
            }
            if (share_low) {
                SRV_INF("k19-q4cap: MoE cache expected to hold %.1f%% of the routed experts (from %.2f): the 3rd draft in the chat bucket only "
                    "(below 1536 tokens; LLAMA_SPEC_TPP_CACHED_LOW=0: off)\n", 100.0*share_est, share_lo);
            }
            if (share_try) {
                SRV_INF("k19-sharegate: MoE cache expected to hold %.1f%% of the routed experts (%.0f MiB short of %.0f MiB, %.0f MiB of RAM): "
                    "priced 3-draft windows %s (LLAMA_SPEC_TPP_CACHED_SHARE=%.2f; LLAMA_SPEC_TPP_SHAREGATE=0: off)\n", 100.0*share_est,
                    share_short/1048576.0, share_exp/1048576.0, share_ram/1048576.0, share_ok ? "on" : "off", share_min);
            }
            if (share_ok) {
                // the speculative ctor's cache rule and the verify tag (llama-context.cpp) both read it
                common_set_env("LLAMA_SPEC_TPP_CACHED", "1");
                ce = getenv("LLAMA_SPEC_TPP_CACHED");
            }
            const bool fits_all = (ce != nullptr && atoi(ce) != 0) || !cache_expected;
            // LLAMA_SPEC_TPP=1 forces past the device rule, not the MoE cache rule
            const bool on = (e != nullptr ? atoi(e) != 0 : dev_ok) && fits_all;
            // k19-fnmtp (LLAMA_SPEC_DEPTH_CACHED=N drafts, N = 3..8; 0: off; unset: the k19-fnmtp2 default below): beside a MoE cache an MTP drafter of its own
            // (-md) drafts N in every window below LLAMA_SPEC_DEPTH_POS (12288), the windows from there are k19-tpp's priced ones (2, a 3rd
            // where it pays at the cache's cost ratio). The 4-row windows then get the verify tag (LLAMA_SPEC_VERIFY4_CACHED=1 is set unless
            // given), so they run the 3-row kernels instead of cuBLAS. Not with LLAMA_MTP_EXACT=1 (beside a cache the exact text follows the
            // window where the CPU-miss host path runs): there the windows stay at 2.
            // Not on by itself because the deeper window does not pay on every measured cached tier (Colab A100 40 GB
            // Qwen3.8-Flash-Next UD-IQ4_XS from the VM disk, 3 launches, with the PLE table read in part): 3 drafts full card chat +3.4% /
            // 8k -3.8%, 16 GB chat -6% / 8k +7.6%; 4 drafts full card +1.8% / +6.8%, 16 GB -16% / -14%. A verify row costs ~10 ms there
            // (PLE rows read from the disk and expert misses copied at ~10 GB/s), against 3.4 -> 4.0 tokens per pass
            if (deep_try && !on) {
                if (g4t_band && !g4t_ctx_ok) {
                    SRV_INF("k19-g4tail2: n_ctx %d is above LLAMA_FIT_LONGCTX_MIN_CTX (%lld): the g4tail rule stays off (MTP windows and "
                        "the prefetch as with LLAMA_MOE_G4TAIL=0)\n", params.n_ctx, (long long) g4t_ctx_max);
                }
                spec_g4t_raised = g4t;
                if (def_on || g4t || def_mid) {
                    depth_cached = def_mid || def_mid_hi ? n_mid : n_def;
                    // (a model on a memory file system: its experts are in RAM whatever MemAvailable reads)
                    disk_ok = disk_ok || in_ram_fs;
                }
                // Where the cache is expected to hold most of the routed experts (at least LLAMA_SPEC_HOSTPATH_OFF_SHARE, default 0.60;
                // 0: off) on compute capability 12.0 cards, the CPU-miss host path stays off (as LLAMA_MOE_CPU_MISS=0): the few misses of
                // the deep verify windows are fetched instead of computed on the host. There the host path ran inside the decode call
                // (6-7 ms of host time in an 18 ms pass) and its trial in the first chat request. Colab G4, Qwen3.8-Flash-Next UD-IQ4_XS +
                // its MTP drafter, Studio argv, 48 GB (67.7% expected), two VMs x 2 launches: chat 231.9 -> 249.9 tok/s (+7.8%, 17.5 -> 16.3
                // ms per pass), chat TTFT 0.184 -> 0.150 s, 8k level (210.6 / 210.9), the same text. At 32 GB (39%) the host path is
                // needed (-15% without it), at 16 GB (11%) more so (-43%). An explicit LLAMA_MOE_CPU_MISS keeps its value
                const double hp_share = [] {
                    const char * x = getenv("LLAMA_SPEC_HOSTPATH_OFF_SHARE");
                    return x != nullptr ? atof(x) : 0.60;
                }();
                spec_chat_deep = def_on && deep_share >= 0.60 && cc_min == 1200 && cc_max == 1200;
                if (def_on && hp_share > 0.0 && deep_share >= hp_share && cc_min == 1200 && cc_max == 1200 && getenv("LLAMA_MOE_CPU_MISS") == nullptr) {
                    common_set_env("LLAMA_MOE_CPU_MISS", "0");
                    SRV_INF("MoE cache expected to hold %.1f%% of the routed experts beside the MTP drafter (compute capability 12.0): the CPU-miss "
                        "host path is off, misses are fetched (LLAMA_SPEC_HOSTPATH_OFF_SHARE=%.2f; =0: the host path rule as before)\n",
                        100.0*deep_share, hp_share);
                }
                // In the same band the MTP drafter's restricted head keeps 32768 rows instead of 65536: each draft step's lm head is
                // the largest kernel of the step there, and the deep windows draft 4-6 steps per pass. Colab G4, Qwen3.8-Flash-Next
                // UD-IQ4_XS + its MTP drafter, Studio argv, 48 GB, 2 launches: 8k 210.4 -> 215.7 tok/s (+2.5%, 16.22 -> 16.04 ms per
                // pass at 3.46 tokens), chat level (256.3 / 256.6), the same text. LLAMA_SPEC_MTP_VOCAB_WIDE=0: off; an explicit
                // LLAMA_MTP_DRAFT_VOCAB keeps its value
                const bool mtp_vocab_wide = [] {
                    const char * x = getenv("LLAMA_SPEC_MTP_VOCAB_WIDE");
                    return x == nullptr || atoi(x) != 0;
                }();
                if (def_on && mtp_vocab_wide && deep_share >= 0.60 && cc_min == 1200 && cc_max == 1200 &&
                        getenv("LLAMA_MTP_DRAFT_VOCAB") == nullptr) {
                    common_set_env("LLAMA_MTP_DRAFT_VOCAB", "32768");
                    SRV_INF("MoE cache expected to hold %.1f%% of the routed experts beside the MTP drafter (compute capability 12.0): "
                        "the drafter's restricted head keeps 32768 rows (LLAMA_SPEC_MTP_VOCAB_WIDE=0: 65536)\n", 100.0*deep_share);
                }
                if (g4t && getenv("LLAMA_MOE_PREFETCH") == nullptr && getenv("LLAMA_MOE_PREFETCH_AUTO") == nullptr) {
                    common_set_env("LLAMA_MOE_PREFETCH_AUTO", "2");
                }
                if (def_mid) {
                    SRV_INF("MoE cache expected to hold %.1f%% of the routed experts (from %.2f up to %.2f, compute capability 12.0, target in "
                        "RAM): MTP windows of %d drafts (LLAMA_SPEC_DEPTH_MID=0: 2 drafts)\n", 100.0*deep_share, g4t_hi, deep_share_min, n_mid);
                }
                if (def_mid_hi) {
                    SRV_INF("MoE cache expected to hold %.1f%% of the routed experts (from %.2f up to %.2f, compute capability 12.0): MTP "
                        "windows of %d drafts (LLAMA_SPEC_DEPTH_MID_HI=0: %d)\n", 100.0*deep_share, deep_share_min, mid_hi, n_mid, n_def);
                }
                if (g4t) {
                    SRV_INF("k19-g4tail: MoE cache expected to hold %.1f%% of the routed experts (band %.2f .. %.2f, compute capability 12.0, target "
                        "in RAM): MTP windows of %d drafts and the cache's cross-layer prefetch in its band (LLAMA_MOE_G4TAIL=0: off)\n", 100.0*deep_share,
                        g4t_lo, g4t_hi, n_def);
                }
                SRV_INF("k19-fnmtp2: MoE cache expected to hold %.1f%% of the routed experts (%.0f MiB short of %.0f MiB); target GGUF %.0f MiB %s "
                    "(%.0f MiB of RAM available): %s (LLAMA_SPEC_DEPTH_CACHED_SHARE=%.2f; LLAMA_SPEC_DEPTH_CACHED=0: 2 drafts)\n", 100.0*deep_share,
                    short_b/1048576.0, exp_b/1048576.0, tgt_bytes/1048576.0, in_ram_fs ? "on a memory file system" : resident ? "fits in RAM" :
                    "does not fit in RAM", ram_avail/1048576.0, def_on || g4t || def_mid ? "deeper MTP windows by default" : "MTP windows of 2 drafts", deep_share_min);
            }
            // The windows of the chat bucket (KV positions below LLAMA_SPEC_DEPTH_CHAT_POS, 1536) draft LLAMA_SPEC_DEPTH_CHAT_N (6; 0: off)
            // where the deep windows are on and a verify row is cheap: the full card and a cache expected to hold at least 60% of the
            // routed experts, compute capability 12.0 only. Longer contexts keep the deep windows' own depth (LLAMA_SPEC_DEPTH_MAIN_N for
            // the drafter), so an 8k request keeps its windows; the windows are sized for the deeper one. MoE targets only. A request
            // below 1536 tokens has one verify width, one that crosses 1536 two. Colab G4, Qwen3.8-Flash-Next UD-IQ4_XS + its MTP
            // drafter, 6 drafts in every window: 48 GB chat +2.5% (5.22 vs 4.06 tokens per pass), full card chat +1.7%, the same text;
            // at 16 GB chat -7% (not there)
            const int chat_n = [] {
                const char * x = getenv("LLAMA_SPEC_DEPTH_CHAT_N");
                return x != nullptr ? std::clamp(atoi(x), 0, 8) : 6;
            }();
            auto chat_deepen = [&](int n_main) -> int {
                if (chat_n <= n_main || getenv("LLAMA_SPEC_DEPTH_MAIN_N") != nullptr) {
                    return n_main;
                }
                common_set_env("LLAMA_SPEC_DEPTH_MAIN_N", std::to_string(n_main).c_str());
                SRV_INF("MTP windows below position %s draft %d, the deep windows from there %d (LLAMA_SPEC_DEPTH_CHAT_N=0: %d everywhere)\n",
                    getenv("LLAMA_SPEC_DEPTH_CHAT_POS") != nullptr ? getenv("LLAMA_SPEC_DEPTH_CHAT_POS") : "1536", chat_n, n_main, n_main);
                return chat_n;
            };
            const uint64_t expert_bytes = mtp_shape && !on && cache_expected && depth_cached >= 3 ? common_moe_expert_bytes(params.model.path) : 0;
            const bool ram_fits = ram_avail == 0 || expert_bytes <= ram_avail;
            // (the device rule as k19-tpp's: NVIDIA Ampere or newer, LLAMA_SPEC_TPP=1 forces past it, =0 turns this off too)
            const bool deep_cached = mtp_shape && !on && cache_expected && sp.has_dft() && (e != nullptr ? atoi(e) != 0 : dev_ok) &&
                depth_cached >= 3 && !exact_mtp && (ram_fits || disk_ok);
            if (mtp_shape && !on && cache_expected && sp.has_dft() && depth_cached >= 3 && !exact_mtp && !(ram_fits || disk_ok)) {
                SRV_INF("k19-fnmtp: MTP stays at 2 drafts beside the MoE cache: the routed experts (%.0f MiB) do not fit in %.0f MiB of RAM, "
                    "the cache's misses come from the disk (LLAMA_SPEC_DEPTH_CACHED_DISK=1: deeper windows anyway)\n",
                    expert_bytes/1048576.0, ram_avail/1048576.0);
            }
            if (deep_cached) {
                // k19-g4tail2: under the g4tail rule --fit admits the drafter by the verify cost of the windows it had before the rule
                // (fit_n_max). The windows of n_def drafts then stand only where the placed cache takes the prefetch band (after
                // common_init_from_params below), else they are cut back to 2. G4 16 GB ctx 262144 (r1): priced at 4 drafts the
                // requantized drafter needed a cache of 4 tokens per layer, left 3.2 and was left out, decode without drafts 87.8 / 81.7
                // tok/s chat / 8k against 132.7 / 123.8 with the rule off (LLAMA_MOE_G4TAIL_PLACED_GATE=0: priced at n_def, as before)
                const char * pg = getenv("LLAMA_MOE_G4TAIL_PLACED_GATE");
                if (spec_g4t_raised && (pg == nullptr || atoi(pg) != 0)) {
                    sp.draft.fit_n_max = sp.draft.n_max;
                }
                sp.draft.n_floor     = 2;
                sp.draft.n_max       = spec_chat_deep ? chat_deepen(std::min(depth_cached, 8)) : std::min(depth_cached, 8);
                sp.draft.deep_cached = true;
                if (getenv("LLAMA_SPEC_VERIFY4_CACHED") == nullptr) {
                    common_set_env("LLAMA_SPEC_VERIFY4_CACHED", "1");
                }
                SRV_INF("k19-fnmtp: MoE cache expected (need %.0f MiB for model + drafter, %.0f MiB free on the GPUs): MTP drafts %d in every "
                    "window below position %s, 2 and a priced 3rd from there; 4-row verify windows tagged (LLAMA_SPEC_DEPTH_CACHED=0: 2 drafts)\n", need/1048576.0,
                    dev_free/1048576.0, sp.draft.n_max, getenv("LLAMA_SPEC_DEPTH_POS") != nullptr ? getenv("LLAMA_SPEC_DEPTH_POS") : "12288");
            } else if (mtp_shape && on) {
                // k19-depth (opt-in: LLAMA_SPEC_DEPTH=1, also beside built-in MTP heads; unset or 0: off): with it on, every window
                // below position 12288 (LLAMA_SPEC_DEPTH_POS) drafts 4, the windows from there are the ones above (2, a 3rd where it pays).
                // G4 Qwen3.8-Flash-Next fn4 full card: 4 drafts +6.8% chat / +10.9% 8k, -13% 30k against 2 + tpp.
                // Built-in heads (Qwen3.6) stay at the windows above: their extra verify row costs ~20% of a pass
                const char * de = getenv("LLAMA_SPEC_DEPTH");
                // (k19-sharegate: beside the cache the deep windows stay opt-in, LLAMA_SPEC_DEPTH=1)
                const bool deep = de != nullptr && atoi(de) != 0;
                sp.draft.n_floor = 2;
                // (the chat bucket's deeper windows on a MoE target only: a dense one keeps 4)
                const bool chat_ok = cc_min == 1200 && cc_max == 1200 && de == nullptr && common_moe_expert_bytes(params.model.path) > 0;
                sp.draft.n_max   = deep ? (chat_ok ? chat_deepen(4) : 4) : 3;
                SRV_INF("k19-tpp: MTP drafts 2, a 3rd where it pays (need %.0f MiB for model + drafter, %.0f MiB free on %d GPUs%s; "
                    "LLAMA_SPEC_TPP=0: off)\n", need/1048576.0, dev_free/1048576.0, n_gpu, n_gpu == 0 ? ", no GPU: forced" : "");
                if (deep) {
                    const char * dp = getenv("LLAMA_SPEC_DEPTH_POS");
                    SRV_INF("k19-depth: MTP drafts 4 in every window below position %s (LLAMA_SPEC_DEPTH_POS; LLAMA_SPEC_DEPTH=0: off)\n",
                        dp != nullptr ? dp : "12288");
                }
                // With the deeper chat windows on the whole card (compute capability 12.0, MoE target, an MTP drafter of its own) the
                // drafter's restricted head keeps 49152 rows instead of 65536: the head is the largest kernel of a draft step (99 of 331
                // us there) and a chat window drafts 6 steps. Only the drafts change, the verify keeps the text. Colab G4, Qwen3.8-Flash-Next
                // UD-IQ4_XS + its MTP drafter, Studio argv, whole card, two VMs x 3 launches: chat +1.7% / +2.1%, 8k +0.8% / +2.4%, the
                // same tokens per pass (5.22 / 3.46) and text; 32768 rows lost tokens per pass in chat (5.22 -> 5.12).
                // LLAMA_SPEC_MTP_VOCAB_FULL=0: 65536; an explicit LLAMA_MTP_DRAFT_VOCAB keeps its value
                const bool vocab_full = [] {
                    const char * x = getenv("LLAMA_SPEC_MTP_VOCAB_FULL");
                    return x == nullptr || atoi(x) != 0;
                }();
                if (deep && chat_ok && vocab_full && getenv("LLAMA_MTP_DRAFT_VOCAB") == nullptr) {
                    common_set_env("LLAMA_MTP_DRAFT_VOCAB", "49152");
                    SRV_INF("%s", "MTP drafter on the whole card (compute capability 12.0): its restricted head keeps 49152 rows "
                        "(LLAMA_SPEC_MTP_VOCAB_FULL=0: 65536)\n");
                }
            } else if (mtp_shape && cache_expected && !exact_mtp && [] {
                    const char * a = getenv("LLAMA_SPEC_ADRAFT");
                    return a != nullptr && atoi(a) != 0;
                }()) {
                // k19-adraft (opt-in, LLAMA_SPEC_ADRAFT=1): where the windows stay at 2 drafts beside a MoE cache, a sampled request's window
                // holds up to LLAMA_SPEC_ADRAFT_MAX drafts (default 3), each past the 2nd kept while its draft probability is at least
                // LLAMA_SPEC_ADRAFT_P (0.9; the drafter's confidence gate, a function of the text). =2: priced by expected tokens per ms
                // instead; =3: one width per stretch of the request from the running share of drafts that held; =4: by position. Greedy requests keep 2 drafts. Sampled verify is sample-and-match with the target's chain and RNG, so only the
                // proposals change
                const char * am = getenv("LLAMA_SPEC_ADRAFT_MAX");
                sp.draft.adraft = sp.draft.n_max;
                sp.draft.n_max  = std::clamp(am != nullptr ? atoi(am) : 3, sp.draft.n_max + 1, 8);
                if (getenv("LLAMA_SPEC_VERIFY4_CACHED") == nullptr) {
                    common_set_env("LLAMA_SPEC_VERIFY4_CACHED", "1");
                }
                SRV_INF("k19-adraft: MoE cache expected (need %.0f MiB, %.0f MiB free on the GPUs): sampled requests draft up to %d, "
                    "greedy ones %d (LLAMA_SPEC_ADRAFT=0: off)\n", need/1048576.0, dev_free/1048576.0,
                    sp.draft.n_max, sp.draft.adraft);
            } else if (mtp_shape && n_gpu == 0 && e == nullptr) {
                SRV_INF("%s", "k19-tpp: off: no GPU (LLAMA_SPEC_TPP=1: on)\n");
            } else if (mtp_shape && (e != nullptr ? atoi(e) != 0 : dev_ok)) {
                SRV_INF("k19-tpp: off: 3-draft windows are off because the MoE cache is on (need %.0f MiB for model + drafter, %.0f MiB free on the GPUs%s%s); "
                    "MTP windows of 2 drafts, as LLAMA_SPEC_TPP=0 (LLAMA_SPEC_TPP_CACHED=1: on)\n", need/1048576.0, dev_free/1048576.0,
                    host_experts ? ", experts placed on the host" : "", fe != nullptr && atoi(fe) != 0 ? ", LLAMA_SPEC_TPP_FORCE_CACHED" : "");
            } else if (mtp_shape && e == nullptr) {
                SRV_INF("k19-tpp: MTP stays at 2 drafts here (%d GPUs, lowest compute capability %d: not every GPU is NVIDIA Ampere or newer; "
                    "LLAMA_SPEC_TPP=1 forces the wider window)\n", n_gpu, n_gpu > 0 ? cc_min : -1);
            }
            // k19-small1115: the windows above were widened past Studio's 2 drafts on a --parallel N server: cut back to 2 in the
            // rounds where another slot is active (update_slots)
            spec_active_cap = params.n_parallel > 1 && mtp_shape && sp.draft.n_max > 2 ? 2 : 0;
            if (spec_active_cap > 0) {
                SRV_INF("k19-small1115: --parallel %d: a lone request drafts up to %d as with --parallel 1, %d while another slot is active "
                    "(LLAMA_SPEC_TPP_ACTIVE=0: %d always)\n", params.n_parallel, sp.draft.n_max, spec_active_cap, spec_active_cap);
            }
        }

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        bool       has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        mtmd_memory_usage mmproj_usage = {{}, -1, false};
        int64_t mmproj_usage_t_us = 0;
        if (has_mmproj) {
            int64_t t_start = ggml_time_us();
            mmproj_usage = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            mmproj_usage_t_us = ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        if (has_mmproj && params_base.fit_params) {
            const auto & mmproj_mem = mmproj_usage.backend_mem_usage;
            const int64_t t_elapsed = mmproj_usage_t_us;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        // --fit may leave a draft model that does not fit on the GPU out, see common_init_result
        params_base.speculative.draft.fit_can_drop = true;

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        // k19-g4tail2: the rule's windows of n_def drafts stand only where the placed MoE cache took its prefetch band (the cache
        // publishes LLAMA_MOE_G4TAIL_PLACED once placed). Its estimate before the fit leaves out the KV cache of n_ctx: at 262144 the
        // placed cache is smaller than at the measured 32768, and 4-draft windows beside a cache below the band were never measured
        // to pay. Outside the band: 2 drafts per window, as without the rule (LLAMA_MOE_G4TAIL_PLACED_GATE=0: keep the windows)
        if (spec_g4t_raised && ctx_tgt != nullptr) {
            const char * pg = getenv("LLAMA_MOE_G4TAIL_PLACED_GATE");
            const char * pl = getenv("LLAMA_MOE_G4TAIL_PLACED");
            if ((pg == nullptr || atoi(pg) != 0) && (pl == nullptr || atoi(pl) == 0)) {
                spec_g4t_cap = 2;
                SRV_INF("k19-g4tail2: the placed MoE cache is outside the prefetch band (%s): MTP windows of 2 drafts "
                    "(LLAMA_MOE_G4TAIL_PLACED_GATE=0: keep the wider windows)\n", pl == nullptr ? "no verdict" : "verdict off");
            } else if (pl != nullptr && atoi(pl) != 0) {
                SRV_INF("%s", "k19-g4tail2: the placed MoE cache is in the prefetch band: the wider MTP windows stay\n");
            }
        }

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        {
            const char * e = getenv("LLAMA_SERVER_DEVICE_GREEDY");
            // on by default again with 928828cc004 (pre-sampling n_probs read the raw logits): the n_probs / logprobs check passes 4 / 4
            // (completion + chat, streaming + not) with it on. LLAMA_SERVER_DEVICE_GREEDY=0 off
            device_greedy_ok = (e == nullptr || atoi(e) != 0) && llama_model_n_devices(model_tgt) > 0;
            for (int i = 0; device_greedy_ok && i < llama_model_n_devices(model_tgt); ++i) {
                ggml_backend_dev_t dev = llama_model_get_device(model_tgt, i);
                ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
                const char * rn = reg ? ggml_backend_reg_name(reg) : "";
                // k19-igpu: an integrated GPU too (ggml-cuda reports prop.integrated as IGPU: Strix Halo gfx1151, Jetson, DGX Spark); the
                // argmax runs in the same graph on the same backend, nothing is sized from memory
                const enum ggml_backend_dev_type dev_type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
                device_greedy_ok = dev && (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                    (strcmp(rn, "CUDA") == 0 || strcmp(rn, "ROCm") == 0);
            }
        }

        vocab = llama_model_get_vocab(model_tgt);

        try {
            decision.init(model_tgt);
        } catch (const std::exception & e) {
            SRV_ERR("failed to init decision model: %s\n", e.what());
            return false;
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        // --fit placed the draft model on the CPU, where drafting is slower than none (see common_init_result)
        if (has_spec && params_base.speculative.draft.fit_off) {
            // k19-rollback: or the n-gram suffix drafts --fit put in its place (common_fit_drop_ngram, LLAMA_SPEC_DROP_NGRAM)
            const bool drop_ngram = params_base.speculative.ngram_suffix.drop_fallback;
            SRV_WRN("--fit: the draft model does not fit on the GPU beside the main model and its MoE cache, %s "
                "(LLAMA_FIT_DRAFT_CPU=1 drafts on the CPU, LLAMA_FIT_DRAFT_LAST=0 restores the old placement)\n",
                drop_ngram ? "n-gram suffix drafts instead of the draft model" : "speculative decoding is off");
            has_spec = false;
            if (!drop_ngram) {
                params_base.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
            }
            params_base.speculative.draft.mparams = {};
            load_progress_callback(1.0f, &load_progress_spec);
        }

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                // k15-small4: a draft that does not load or whose context does not allocate (device memory the fit could not see:
                // a draft it cannot measure on its own, memory another process took) is dropped and the server runs without speculative
                // decoding, as --fit does when the draft does not fit; the main model and its context are already up.
                // LLAMA_DRAFT_FAIL_HARD=1: exit, as before
                const char * env_hard = getenv("LLAMA_DRAFT_FAIL_HARD");
                const bool   hard     = env_hard != nullptr && atoi(env_hard) != 0;
                bool         failed   = (has_draft && model_dft == nullptr) || ctx_dft == nullptr;
                // k17-small: the draft is up, but device memory the fit did not price can leave the GPU too little for the CUDA pools of the
                // first prompt (Kaggle T4 4 GB, Qwen3.6-35B-A3B UD-Q4_K_XL + MTP, ctx 16384: the fit left 258 MiB, the MTP context's compute
                // buffer took 233 MiB against the 104 the fit measured and the main one grew 482 -> 514, 74.8 MiB were free after load and the
                // first prompt aborted with "CUDA error: out of memory" in every launch). A device with less than LLAMA_DRAFT_MIN_FREE_MIB (192)
                // free after the draft is created drops the draft as above; the main context stays as fitted. LLAMA_DRAFT_MIN_FREE_MIB=0: off
                // The floor is 192 MiB on NVIDIA devices below compute capability 8.0 and off elsewhere: on G4 (RTX PRO 6000, cc 12.0) free
                // memory after the draft does not predict the abort. q4x 4 GB 16k + MTP runs with 18.6 MiB free (decode 169 tok/s, dropping
                // the draft there costs 30%), while q3m 4 GB 32k + MTP aborts in its first prompt with 58.6 MiB free, the same as with the
                // draft kept, so a floor there only loses speed. AMD, Moore Threads and other GPUs report no capability: off, as before.
                // LLAMA_DRAFT_MIN_FREE_MIB=N sets one floor for every GPU, integrated ones included (ggml-cuda reports a device with
                // prop.integrated as IGPU: Strix Halo gfx1151, Jetson, DGX Spark; the check skipped them, so the floor never applied there,
                // AMD CI 41265043892: LLAMA_DRAFT_MIN_FREE_MIB=1000000 logged no floor line on the 8060S)
                if (!failed) {
                    const char * e_mf = getenv("LLAMA_DRAFT_MIN_FREE_MIB");
                    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                        const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
                        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                            continue;
                        }
                        int mib = 192;
                        if (e_mf != nullptr) {
                            mib = atoi(e_mf);
                        } else {
                            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                            auto cc_fn = reg ? (int (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_compute_capability") : nullptr;
                            const int cc = cc_fn != nullptr ? cc_fn(dev) : -1;
                            // k17-small: off where the fit priced the drafter as it runs (LLAMA_FIT_MTP_PRICE=1, exported by common):
                            // the floor stood in for that price and dropped drafts that fit (Colab T4 4 GB q4x 16k: 126.9 MiB free,
                            // chat 38.9 -> 27.1 tok/s)
                            const char * e_pr = getenv("LLAMA_FIT_MTP_PRICE");
                            const bool priced = e_pr != nullptr && atoi(e_pr) != 0 && params_base.fit_params;
                            mib = (cc >= 0 && cc < 800 && !priced) ? 192 : 0;
                        }
                        const size_t min_free = (size_t) std::max(0, mib) * 1024 * 1024;
                        if (min_free == 0) {
                            continue;
                        }
                        size_t dfree = 0, dtotal = 0;
                        ggml_backend_dev_memory(dev, &dfree, &dtotal);
                        if (dtotal > 0 && dfree < min_free) {
                            SRV_WRN("%s: %.1f MiB free after the draft was created, less than %zu MiB (LLAMA_DRAFT_MIN_FREE_MIB)\n",
                                ggml_backend_dev_name(dev), dfree/1048576.0, min_free/1048576);
                            failed = true;
                        }
                    }
                }
                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                } else if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                }
                if (failed && hard) {
                    return false;
                }
                if (failed) {
                    SRV_WRN("%s", "the draft could not be created: speculative decoding is off (LLAMA_DRAFT_FAIL_HARD=1 exits instead)\n");
                    spec_init.reset();
                    model_dft = nullptr;
                    ctx_dft   = nullptr;
                    has_spec  = false;
                    params_base.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
                    params_base.speculative.draft.mparams = {};
                } else {
                    params_base.speculative.draft.ctx_tgt = ctx_tgt;
                    params_base.speculative.draft.ctx_dft = ctx_dft;
                }
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            // the image must fit in one ubatch if the model needs non-causal attention on it
            // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
            {
                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                const bool is_mixed = use_mixed_batch();
                if (is_mixed || mmproj_usage.use_non_causal) {
                    const int n_max = is_mixed ? n_ubatch / 2 : n_ubatch;
                    if (mmproj_usage.image_max_tokens > n_max) {
                        SRV_WRN("cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", mmproj_usage.image_max_tokens, n_max, n_ubatch,
                                is_mixed ? "the model processes the prompt in one ubatch" : "model needs non-causal attention on image");
                        SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
                        mparams.image_max_tokens = n_max;
                        mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_max);
                    }
                }
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // a draft model (-md) that no requested draft type uses (none given, or only ngram types) is released before the probe below:
        // its token-only test decode can abort on a draft that needs more input (a gemma-4 assistant needs the target's state embedding)
        if (ctx_dft && !common_speculative_uses_draft_ctx(params_base.speculative.types)) {
            SRV_WRN("%s", "the draft model is not used by any --spec-type (e.g. draft-simple, draft-mtp), unloading it\n");
            params_base.speculative.draft.ctx_dft = nullptr;
            ctx_dft   = nullptr;
            model_dft = nullptr;
            spec_init.reset();
        }

        // k15-lfm2: a weak host (few threads, not faster than an upload) with most experts cached drops the CPU-miss rule, so the
        // default drafts verify exactly (llama_moe_cpu_miss_drop_slow; LLAMA_MOE_CPU_MISS_DROP=0: off)
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO && params_base.speculative.ngram_suffix.auto_mode && server_spec_auto_exact() &&
                !llama_verify_exact_placement(ctx_tgt) && llama_moe_cpu_miss_drop_slow(ctx_tgt)) {
            SRV_INF("%s", "MoE cache: CPU misses off on this host (weak host, most experts cached): drafts verified exactly\n");
        }

        // k13: default drafts where the host computes experts (server_spec_spill_mode) draft only from repeats of at least
        // server_spec_spill_min_match tokens (short repeats cost a verify window of host experts for little)
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO && params_base.speculative.ngram_suffix.auto_mode && server_spec_auto_exact() &&
                !llama_verify_exact_placement(ctx_tgt) &&
                server_spec_spill_mode(server_spec_auto_arch(params_base.model.path), llama_moe_host_mode(ctx_tgt), params_base.cpuparams.n_threads,
                    params_base.speculative.ngram_suffix.drop_spill) > 0) {
            auto & ns = params_base.speculative.ngram_suffix;
            const int mm = server_spec_spill_min_match(llama_moe_host_mode(ctx_tgt));
            if (mm > ns.n_min_match) {
                SRV_INF("n-gram drafting under the MoE cache: drafts from repeats of at least %d tokens (LLAMA_SPEC_SPILL_MIN_MATCH)\n", mm);
                ns.n_min_match = mm;
                ns.n_min_draft = std::max(ns.n_min_draft, mm);
            }
            // k15-spec: long prompts draft from shorter repeats (server_spec_spill_min_match_long)
            ns.long_prompt      = server_spec_spill_long_prompt();
            ns.n_min_match_long = server_spec_spill_min_match_long(llama_moe_host_mode(ctx_tgt), ns.n_min_match,
                server_spec_auto_arch(params_base.model.path));
            if (ns.n_min_match_long != ns.n_min_match) {
                SRV_INF("n-gram drafting under the MoE cache: prompts of %d+ tokens draft from repeats of at least %d tokens "
                        "(KLD-gated default; LLAMA_SPEC_SPILL_MIN_MATCH_LONG=16 at 4 GB, LLAMA_SPEC_SPILL_LONG_CM=65 with the CPU-miss rule: off)\n", ns.long_prompt, ns.n_min_match_long);
            }
        }

        // k15-lfm2: LFM2.5 with drafts verified exactly (output unchanged by the window): prompts of 4096+ tokens take a short window of
        // 2 drafts and the full window from a repeat of 6. Opt-in (LLAMA_SPEC_LFM_LONG_WIN=1): G4 32 GB +0.6% / +3.1% at 8k / 16k on the
        // default build, but -3.7% / -1.6% with GGML_CUDA_MMVQ_MOE_ONE=1 (k15-fn's coming default)
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO && params_base.speculative.ngram_suffix.auto_mode && server_spec_auto_exact() &&
                llama_verify_exact_placement(ctx_tgt) && server_spec_auto_arch(params_base.model.path) == "lfm2moe") {
            const char * e = getenv("LLAMA_SPEC_LFM_LONG_WIN");
            if (e != nullptr && atoi(e) != 0) {
                auto & ns = params_base.speculative.ngram_suffix;
                ns.win_long_prompt   = 4096;
                ns.n_max_short_long  = 2;
                ns.n_long_match_long = 6;
            }
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        // k13: LLAMA_MOE_CPU_MISS_FROZEN=1 freezes the small batches of a MoE cache with CPU misses (see llama_moe_freeze_small) with or
        // without drafts; unset: only where it lets the default n-gram drafts verify exactly (below, LLAMA_SPEC_EXACT_FROZEN); 0: never
        const int moe_frozen_env = getenv("LLAMA_MOE_CPU_MISS_FROZEN") != nullptr ? atoi(getenv("LLAMA_MOE_CPU_MISS_FROZEN")) : -1;
        if (moe_frozen_env == 1 && llama_moe_freeze_small(ctx_tgt)) {
            SRV_INF("%s", "MoE cache: small batches frozen (LLAMA_MOE_CPU_MISS_FROZEN=1)\n");
        }

        bool auto_verify_exact = false; // k19: the default n-gram drafts verify batch invariant (logged below)
        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
            // k11-decode: the default drafts are verified batch invariant, so greedy output stays that of plain decoding; where the
            // placement cannot give single token bytes (host experts, a MoE cache) they stay off (LLAMA_SPEC_AUTO_EXACT=0 keeps them)
            if (params_base.speculative.ngram_suffix.auto_mode && server_spec_auto_exact()) {
                // k13: a MoE cache with CPU misses verifies exactly once its small batches are frozen
                if (!llama_verify_exact_placement(ctx_tgt) && moe_frozen_env != 0 && server_spec_exact_frozen() && llama_moe_freeze_small(ctx_tgt)) {
                    SRV_INF("%s", "MoE cache: small batches frozen so the default n-gram drafts verify exactly (LLAMA_SPEC_EXACT_FROZEN=1, "
                        "LLAMA_MOE_CPU_MISS_FROZEN=0: off)\n");
                }
                const int32_t host_mode = llama_moe_host_mode(ctx_tgt);
                const int     spill     = llama_verify_exact_placement(ctx_tgt) ? -1 :
                                          server_spec_spill_mode(server_spec_auto_arch(params_base.model.path), host_mode, params_base.cpuparams.n_threads,
                                              params_base.speculative.ngram_suffix.drop_spill);
                if (spill < 0) {
                    llama_set_verify_exact(ctx_tgt, true);
                    auto_verify_exact = true;
                } else if (spill == 2 && getenv("LLAMA_MOE_CPU_MISS_FROZEN_FETCH") == nullptr && llama_moe_freeze_small(ctx_tgt)) {
                    // k13: frozen small batches, drafts verified batch invariant (exact against this frozen cache)
                    llama_set_verify_exact(ctx_tgt, true);
                    auto_verify_exact = true;
                    SRV_INF("%s", "n-gram drafting by default under the MoE cache: small batches frozen, drafts verified exactly against them "
                        "(KLD-gated default, LLAMA_SPEC_SPILL=0: off)\n");
                } else if (spill == 3 && llama_moe_pin_small(ctx_tgt)) {
                    spill_pinned = true;
                    // k13: small batches always take the host path (no timing trials), fetch share and admissions kept; drafts verified
                    // as ordinary batches: other greedy text, KLD-gated per model
                    SRV_INF("%s", "n-gram drafting by default under the MoE cache: small-batch host path pinned on, drafts verified as ordinary "
                        "batches (KLD-gated default, LLAMA_SPEC_SPILL=0: off)\n");
                } else if (spill == 1 || spill == 3) {
                    // k13: drafts verified as ordinary batches with the host path: other greedy text, KLD-gated per model
                    SRV_INF("%s", "n-gram drafting by default under the MoE cache: drafts verified as ordinary batches with the host path "
                        "(KLD-gated default, LLAMA_SPEC_SPILL=0: off)\n");
                } else {
                    SRV_INF("%s", "n-gram drafting by default stays off: host weights or a MoE cache, where verify rows are not computed as single "
                        "token decodes (LLAMA_SPEC_AUTO_EXACT=0 drafts there, with other greedy text; LLAMA_SPEC_SPILL=1 / 2)\n");
                    spec.reset();
                }
            }
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        // k19: with the reassociated verify (GGML_CUDA_VERIFY_REASSOC, default on: verify rows differ from single token decodes by f32
        // summation order equivalence rule) and windows priced from measured round times, which tokens come from a verify
        // row depends on timing, so greedy output can differ between launches at near ties
        if (spec && auto_verify_exact) {
            const char * e = getenv("GGML_CUDA_VERIFY_REASSOC");
            if (e == nullptr || atoi(e) != 0) {
                SRV_INF("%s", "n-gram drafting by default with the reassociated verify: greedy output can vary between launches near ties; "
                    "GGML_CUDA_VERIFY_REASSOC=0 (exact verify) or LLAMA_SPEC_AUTO=0 for byte-stable output\n");
            }
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        // k19-np12c: the drafter's host copy at load where the prompt groups are expected to borrow its weights (LLAMA_MTP_DRAFT_LEND_EAGER=0 off)
        if (ctx_dft != nullptr) {
            llama_mtp_draft_lend_warm(ctx_tgt);
        }


        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec_bg_join = [this]() { spec_bg_join_or_abort(); };
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        // k15-fn: every slot gets its device greedy chain once (setting a sampler reserves the graphs again): greedy requests take its
        // token and copy no logits, the others copy the raw logits as before (llama_set_sampler_raw_logits)
        if (device_greedy_ok) {
            // the EOG tokens an ignore_eos request excludes (their -inf bias): the masked chain, the larger graph, is set for the reserve and
            // the plain one is swapped in after it (llama_set_sampler_swap keeps the reserve)
            dev_greedy_mask.clear();
            for (const auto & lb : params_base.sampling.logit_bias_eog) {
                dev_greedy_mask.push_back(lb.token);
            }
            std::sort(dev_greedy_mask.begin(), dev_greedy_mask.end());
            dev_greedy_mask.erase(std::unique(dev_greedy_mask.begin(), dev_greedy_mask.end()), dev_greedy_mask.end());

            // sampled requests: a device top-k of dev_topk_k candidates, from which the host chain gives the token of the full row
            // (common_sampler_device_topk); this chain has the largest outputs, so it is the one set for the reserve
            {
                const char * ek = getenv("LLAMA_SERVER_DEVICE_TOPK");
                dev_topk_k = ek == nullptr ? 256 : atoi(ek);
                dev_topk_k = std::max(0, std::min(dev_topk_k, 4096));
            }
            bool all = true;
            for (auto & slot : slots) {
                slot.smpl_dev.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
                llama_sampler_chain_add(slot.smpl_dev.get(), llama_sampler_init_greedy());
                slot.smpl_dev_m.reset();
                if (!dev_greedy_mask.empty()) {
                    slot.smpl_dev_m.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
                    llama_sampler_chain_add(slot.smpl_dev_m.get(),
                        llama_sampler_init_greedy_masked((int32_t) dev_greedy_mask.size(), dev_greedy_mask.data()));
                }
                slot.smpl_dev_k.reset();
                if (dev_topk_k > 0) {
                    slot.smpl_dev_k.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
                    llama_sampler_chain_add(slot.smpl_dev_k.get(), llama_sampler_init_top_k(dev_topk_k));
                }
                slot.smpl_dev_set = llama_set_sampler(ctx_tgt, slot.id, slot.dev_reserve_chain());
                llama_set_sampler_raw_logits(ctx_tgt, slot.id, true);
                all = all && slot.smpl_dev_set;
            }
            if (!all) {
                for (auto & slot : slots) {
                    llama_set_sampler(ctx_tgt, slot.id, nullptr);
                    slot.smpl_dev_set = false;
                }
                device_greedy_ok = false;
            }
            llama_sched_reserve_now(ctx_tgt);
            for (auto & slot : slots) {
                if (slot.smpl_dev_set && slot.dev_reserve_chain() != slot.smpl_dev.get() && !llama_set_sampler_swap(ctx_tgt, slot.id, slot.smpl_dev.get())) {
                    slot.smpl_dev_m.reset();
                    slot.smpl_dev_k.reset();
                    slot.smpl_dev.reset();
                    llama_set_sampler(ctx_tgt, slot.id, nullptr);
                    slot.smpl_dev_set = false;
                }
            }
            SRV_INF("greedy requests are sampled on the device: %s (LLAMA_SERVER_DEVICE_GREEDY=0 to disable)\n", all ? "on" : "off");
        }

        // k19-vwidth: every verify width's graph captured ahead (server_spec_prewarm), after the slots' device samplers are set: their
        // nodes are part of the graphs (G4 Flash-Next: 7613 nodes with the greedy chain vs 7570 without), so each chain slot 0 can take
        // for a request (greedy, masked greedy, top-k) gets its own windows
        // (a set LLAMA_SPEC_PREWARM also without a draft model, e.g. n-gram drafts in CPU tests)
        if (spec && (ctx_dft != nullptr || getenv("LLAMA_SPEC_PREWARM") != nullptr) && params_base.warmup && !slots.empty()) {
            std::vector<llama_sampler *> chains;
            const auto & s0 = slots[0];
            if (s0.smpl_dev_set) {
                for (llama_sampler * c : { s0.smpl_dev.get(), s0.smpl_dev_m.get(), s0.smpl_dev_k.get() }) {
                    if (c != nullptr) {
                        chains.push_back(c);
                    }
                }
            }
            server_spec_prewarm(ctx_tgt, llama_supports_gpu_offload() && params_base.n_gpu_layers != 0,
                params_base.n_outputs_max_per_seq > 0 ? params_base.n_outputs_max_per_seq : params_base.n_outputs_max,
                chains, s0.smpl_dev.get(), true);
        }

        // k19-fnttft: the load is over (target, drafter, warm-ups): read back the PLE table pages the page cache gave up since the
        // load's part read, before the first request (llama_ple_topup; LLAMA_PLE_TOPUP=0 off)
        llama_ple_topup(ctx_tgt);

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(ctx_tgt, std::max(n_batch, params_base.n_parallel), n_embd);
        }

        // decision requests rarely repeat a state, the prompt cache is opt-in for decision models
        if (decision.type != COMMON_DECISION_TYPE_NONE && !params_base.cache_ram_explicit) {
            SRV_INF("%s", "decision model: prompt cache disabled by default, use `--cache-ram N` to enable it\n");
            params_base.cache_ram_mib    = 0;
            params_base.cache_idle_slots = false;
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
            {
                // k15-idlewait: the idle checkpoint lend and the deferred drop are for a single slot only
                static const bool lend_env  = getenv("LLAMA_PCS_IDLE_LEND") == nullptr || atoi(getenv("LLAMA_PCS_IDLE_LEND")) != 0;
                static const bool defer_env = getenv("LLAMA_PCS_DROP_DEFER") == nullptr || atoi(getenv("LLAMA_PCS_DROP_DEFER")) != 0;
                prompt_cache->drop_defer_ok = params_base.n_parallel == 1;
                SRV_INF("prompt cache: idle checkpoint lend %s, deferred drop %s (n_parallel = %d)\n",
                        lend_env && params_base.n_parallel == 1 ? "on" : "off", defer_env && params_base.n_parallel == 1 ? "on" : "off", params_base.n_parallel);
            }
            prompt_cache->wait_reads = [this]() {
                llama_state_seq_async_wait(ctx_tgt);
                if (ctx_dft) {
                    llama_state_seq_async_wait(ctx_dft);
                }
            };
        } else {
            if (!params_base.cache_ssd_dir.empty()) {
                SRV_WRN("%s", "--cache-ssd-dir requires the RAM prompt cache (--cache-ram), disabling the SSD tier\n");
            }
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        prompt_ssd.reset();
        if (prompt_cache && !params_base.cache_ssd_dir.empty()) {
            server_prompt_ssd_params sp;
            sp.dir        = params_base.cache_ssd_dir;
            sp.limit_size = params_base.cache_ssd_mib > 0 ? 1024ull*1024ull*params_base.cache_ssd_mib : 0;
            sp.readonly   = params_base.cache_ssd_readonly;
            const std::string ns = server_prompt_ssd_namespace(params_base, model_tgt, ctx_tgt, model_dft, ctx_dft);
            prompt_ssd = std::make_unique<server_prompt_ssd>(sp, ns, mctx != nullptr);
            prompt_cache->ssd = prompt_ssd.get();
        }

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);

            // the checkpoints of the target are saved asynchronously (create_checkpoint): allocate their pinned staging buffer
            // now, so the first long prompt does not allocate it between two of its ubatches
            if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS || n_swa > 0) {
                llama_state_seq_reserve_async(ctx_tgt, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (ctx_dft) {
                    llama_state_seq_reserve_async(ctx_dft, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        // k19-host30k: the staging a whole-sequence prompt-cache save needs at most, pinned now (server_state_async_ext)
        if (prompt_cache && server_state_async_ext()) {
            llama_state_seq_async_ext_reserve(ctx_tgt, 0);
            if (ctx_dft && ctx_dft != ctx_tgt) {
                llama_state_seq_async_ext_reserve(ctx_dft, 0);
            }
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_idle_work([this]() {
            spec_bg_join_or_abort();
            // k19-l4fill: the RAM tier refills the slots it freed while the server idles (the reads stop at the next batch)
            if (ctx_tgt) {
                llama_moe_host_fill_idle(ctx_tgt);
            }
            // k15-ttft: the prompt-cache save that the next request's slot selection makes (get_available_slot: the slot's prompt with its
            // checkpoints, 65 MiB for a short Qwen3.6 chat, 8-19 ms on a B200 ahead of the prompt) is made here instead, as the slot went
            // idle: the same entry, saved in the same order before the next request's load. LLAMA_PCS_IDLE_SAVE=0: at the request, as
            // before; 1: on this thread (a request sent right after the response waits for it); default 2: the host copy on the state
            // worker (async), so a request arriving meanwhile waits only where it reads the cache. A slot saved here is copied, so the
            // checkpoint move (LLAMA_PCS_MOVE_CKPT, which needs the next prompt) applies only to saves still made at a request.
            static const int idle_save = getenv("LLAMA_PCS_IDLE_SAVE") ? atoi(getenv("LLAMA_PCS_IDLE_SAVE")) : 2;
            // only where every layer weight is in device memory and no MoE cache runs (llama_verify_exact_placement): at a spilled tier the
            // save overlapped the next prompt's expert uploads (G4 q4x 12 GB chat back-to-back +11 ms TTFT) and held the entry on top
            // of the cache's host buffers (+150 MiB VmHWM); there the save stays at the request
            const bool resident = ctx_tgt && llama_verify_exact_placement(ctx_tgt) && !llama_moe_cache_active(ctx_tgt) &&
                (!ctx_dft || ctx_dft == ctx_tgt || (llama_verify_exact_placement(ctx_dft) && !llama_moe_cache_active(ctx_dft)));
            if (idle_save > 0 && resident && prompt_cache && !params_base.embedding) {
                for (auto & slot : slots) {
                    if (!slot.is_processing() && !slot.prompt_saved_idle && slot.prompt.tokens.size() > 0) {
                        // only where the save evicts nothing (an eviction could not be taken back)
                        const size_t n_state = llama_state_seq_get_size_ext(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) +
                            (slot.ctx_dft ? llama_state_seq_get_size_ext(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0);
                        size_t n_ckpt = 0;
                        for (const auto & c : slot.prompt.checkpoints) {
                            n_ckpt += c.size();
                        }
                        if ((prompt_cache->limit_size > 0 && prompt_cache->size() + n_state + n_ckpt > prompt_cache->limit_size) ||
                            (prompt_cache->limit_tokens > 0 && prompt_cache->n_tokens() + slot.prompt.tokens.size() > prompt_cache->limit_tokens)) {
                            continue;
                        }
                        // k17: default on again (LLAMA_PCS_IDLE_LEND=0: off). With the state pool fixes (LLAMA_STATE_POOL_SMALL_FRESH / _REL /
                        // _PENDING) the resident G4 multi-turn run ends below the merged opt-in defaults: Qwen3.6 MTP 3991 vs 4959 MiB, LFM2.5
                        // 1903 vs 2707 MiB, same text and VmHWM (the lend itself: +206 MiB on Qwen3.6, 3991 vs 3785; G4 q4x 32 GB chat -2 ms)
                        static const bool lend_env = getenv("LLAMA_PCS_IDLE_LEND") == nullptr || atoi(getenv("LLAMA_PCS_IDLE_LEND")) != 0;
                        if (lend_env && params_base.n_parallel == 1) {
                            common_state_ckpt_fit_enable(); // k17: lent checkpoints keep their buffers in the cache entry: fit them from now on
                        }
                        slot.prompt_saved_idle_new = slot.prompt_save(*prompt_cache, -2, idle_save == 2, lend_env && params_base.n_parallel == 1);
                        if (slot.prompt_saved_idle_new) {
                            prompt_cache->update();
                        }
                        slot.prompt_saved_idle = true;
                    }
                }
            }
            // k15-idlewait: the MoE cache re-calibration in this idle work stops (stays pending) once a task is queued
            auto idle_abort = [](void * q) { return static_cast<server_queue *>(q)->has_tasks(); };
            if (ctx_tgt) {
                llama_set_idle_abort(ctx_tgt, idle_abort, &queue_tasks);
                llama_prepare_prompt_plan(ctx_tgt);
            }
            if (ctx_dft && ctx_dft != ctx_tgt) {
                llama_set_idle_abort(ctx_dft, idle_abort, &queue_tasks);
                llama_prepare_prompt_plan(ctx_dft);
            }
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                // a busy slot is returned untouched, the caller defers the task
                if (ret->is_processing()) {
                    return ret;
                }

                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == nullptr) {
            const auto free_slots = get_free_slots(1, -1);

            if (!free_slots.empty()) {
                ret = free_slots[0];

                SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt.n_tokens(), ret->t_last_used);

                update_cache = true;
            }
        }

        if (ret) {
            // k15-ttft: a save made at idle that the old order would not have made (the slot is kept: similarity, or a stateless task)
            // is taken back, so the cache holds what it held before (LLAMA_PCS_IDLE_SAVE)
            // k15-idlewait: the checkpoints the idle save lent to its entry come back first: all of them when the entry is dropped below,
            // else a copy of the ones the next prompt keeps (past the common prefix the slot would erase them when the prompt starts)
            if (ret->ckpt_lent && prompt_cache) {
                const bool keep_entry = update_cache && !task.is_stateless();
                llama_pos kmax = std::numeric_limits<llama_pos>::max();
                if (keep_entry && task.params.n_cache_reuse == 0 && !task.tokens.has_mtmd && !ret->prompt.tokens.has_mtmd) {
                    kmax = (llama_pos) ret->prompt.tokens.get_common_prefix(task.tokens);
                }
                prompt_cache->ckpt_return(ret->prompt.tokens, ret->prompt.checkpoints, keep_entry, kmax);
            }
            ret->ckpt_lent = false;
            if (ret->prompt_saved_idle && ret->prompt_saved_idle_new && prompt_cache && !(update_cache && !task.is_stateless())) {
                prompt_cache->drop_newest(ret->prompt.tokens);
            }
            ret->prompt_saved_idle_new = false;
            update_cache = update_cache && prompt_cache;

            // cache prompts only for the tasks that reuse them
            update_cache = update_cache && !task.is_stateless();

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // the checkpoints past the common prefix with the new prompt are erased when it starts (pos_next <= n_past <= the
                // common prefix; not with chunk reuse, which can move n_past past it, nor with multimodal positions)
                llama_pos keep_max = -2;
                if (task.params.n_cache_reuse == 0 && !task.tokens.has_mtmd && !ret->prompt.tokens.has_mtmd) {
                    keep_max = (llama_pos) ret->prompt.tokens.get_common_prefix(task.tokens);
                }
                if (!ret->prompt_saved_idle) {
                    ret->prompt_save(*prompt_cache, keep_max, false, false, true);
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }
                if (getenv("LLAMA_PCS_MEM_LOG") != nullptr) {
                    size_t sz = 0, cap = 0;
                    for (const auto & c : ret->prompt.checkpoints) { sz += c.data_tgt.size() + c.data_dft.size(); cap += c.data_tgt.capacity() + c.data_dft.capacity(); }
                    SRV_INF("pcs mem: slot %d after load: %zu ckpts, %.1f / %.1f MiB (size / capacity)\n", ret->id, ret->prompt.checkpoints.size(), sz/1048576.0, cap/1048576.0);
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        slot.prompt_saved_idle = false;
        llama_moe_request_begin(ctx_tgt);
        // k15-lfm2: the pin a request without drafts lifted is restored before the next prompt (see spill_pinned)
        if (spill_pin_lifted) {
            spill_pin_lifted = !llama_moe_pin_small_set(ctx_tgt, true);
        }
        // LLAMA_SPEC_SPILL_UNPIN_LONG=2: a request that cannot draft (a long prompt, no repeat long enough) is unpinned before its prompt
        if (spill_pinned && server_spec_spill_unpin_long() == 2 && params_base.n_parallel == 1) {
            const auto & ns = params_base.speculative.ngram_suffix;
            if (ns.long_prompt > 0 && ns.n_min_match_long > 64 && (int64_t) task.tokens.size() >= ns.long_prompt) {
                spill_pin_lifted = !llama_moe_pin_small_set(ctx_tgt, false);
            }
        }
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                    slot.prompt_saved_idle = false;
                    slot.ckpt_lent = false;
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // k15-fn: a greedy request (the argmax of the raw logits, common_sampler_device_greedy) takes the token of the slot's device greedy
            // chain when every device of the model is a CUDA / ROCm GPU (its argmax returns the lowest id on a tie, as the host greedy path),
            // so the logits of the sampled rows (every verify row of speculative decoding) are not copied to the host: G4 Qwen3.8-Flash-Next
            // + MTP copied 1.08 MiB per token. Other requests copy the raw logits and sample on the host. LLAMA_SERVER_DEVICE_GREEDY=0 off
            // an ignore_eos request (every bias is the -inf of an EOG token) takes the masked chain of the slot
            // a sampled request whose chain can finish from the device top-k (common_sampler_device_topk) takes the slot's top-k chain:
            // the rows copy dev_topk_k candidates instead of the full logits. A decision task reads the raw logits
            const bool dev_ok = !use_backend_sampling && !need_pre_sample_logits && slot.smpl_dev && (slot.smpl_dev_set || device_greedy_ok) &&
                task.decision.labels.empty();
            bool dev_masked = false;
            bool dev_greedy = dev_ok && common_sampler_device_greedy(slot.smpl.get());
            if (dev_ok && !dev_greedy && slot.smpl_dev_m) {
                std::vector<llama_token> mask;
                dev_masked = common_sampler_device_greedy_masked(slot.smpl.get(), mask) && mask == dev_greedy_mask;
                dev_greedy = dev_masked;
            }
            bool dev_topk = dev_ok && !dev_greedy && slot.smpl_dev_k && common_sampler_device_topk(slot.smpl.get(), dev_topk_k) > 0;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                slot.smpl_dev_set = false;
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else if (slot.smpl_dev) {
                if (!slot.smpl_dev_set) {
                    // back from a backend sampling request: the largest chain is set for the reserve
                    slot.smpl_dev_set = llama_set_sampler(ctx_tgt, slot.id, slot.dev_reserve_chain());
                }
                if (slot.smpl_dev_set && (dev_greedy || dev_topk)) {
                    llama_sampler * want = dev_topk ? slot.smpl_dev_k.get() : dev_masked ? slot.smpl_dev_m.get() : slot.smpl_dev.get();
                    const bool ok = llama_set_sampler_swap(ctx_tgt, slot.id, want);
                    dev_greedy = dev_greedy && ok;
                    dev_topk   = dev_topk && ok;
                }
                dev_greedy = dev_greedy && slot.smpl_dev_set;
                dev_topk   = dev_topk && slot.smpl_dev_set;
                llama_set_sampler_raw_logits(ctx_tgt, slot.id, !dev_greedy && !dev_topk);
                // k19-gramdev: a grammar / reasoning budget request on the device top-k copies its raw rows too, for the positions the
                // candidates cannot decide (common_sampler_device_topk)
                llama_set_sampler_keep_logits(ctx_tgt, slot.id, dev_topk && common_sampler_device_topk_needs_raw(slot.smpl.get()));
                common_sampler_set_device_token(slot.smpl.get(), dev_greedy);
                common_sampler_set_device_topk(slot.smpl.get(), dev_topk);
            } else {
                common_sampler_set_device_token(slot.smpl.get(), false);
                common_sampler_set_device_topk(slot.smpl.get(), false);
                llama_set_sampler_keep_logits(ctx_tgt, slot.id, false);
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }
            if (use_backend_sampling) {
                llama_set_sampler_keep_logits(ctx_tgt, slot.id, false);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated.append(token_str, result.tok);
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated.text) < slot.generated.text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated.text.size());

            const std::string str_test = slot.generated.text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated.truncate(pos + stop_pos);
                pos = std::min(slot.n_sent_text, slot.generated.text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated.text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated.text.size() && (slot.generated.text[pos] == ' ' || slot.generated.text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated.text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated.text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(), tkn.text_to_send.size());
            res->tokens.assign(1, tkn.tok);
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // k19-pf debug (LLAMA_MTP_KV_HASH=1): a hash of the drafter's whole sequence state (its K / V and indexer cells) at the end of
        // the request, to compare the drafter's caches of two runs (LLAMA_MTP_PROMPT_KV_ONLY=0 / 1)
        static const bool mtp_kv_hash = getenv("LLAMA_MTP_KV_HASH") != nullptr && atoi(getenv("LLAMA_MTP_KV_HASH")) != 0;
        if (mtp_kv_hash && ctx_dft && spec_bg_join()) {
            std::vector<uint8_t> st(llama_state_seq_get_size(ctx_dft, slot.id));
            const size_t n_st = llama_state_seq_get_data(ctx_dft, st.data(), st.size(), slot.id);
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < n_st; ++i) { h ^= st[i]; h *= 1099511628211ull; }
            SLT_INF(slot, "mtp kv hash: %zu bytes %016llx\n", n_st, (unsigned long long) h);
        }

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated.text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = {};
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.tokens[i].seq_id);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;

        if (!decision.labels.empty()) {
            const float * logits = llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == nullptr) {
                send_error(slot, "failed to get logits", ERROR_TYPE_SERVER);
                return;
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : decision.labels) {
                GGML_ASSERT(label >= 0 && label < n_vocab);
                res->scores.push_back(logits[label]);
            }
            if (!decision.label_groups.empty()) {
                std::vector<float> scores;
                size_t i = 0;
                for (const int32_t n : decision.label_groups) {
                    GGML_ASSERT(n > 0 && i + n <= res->scores.size());
                    scores.push_back(*std::max_element(res->scores.begin() + i, res->scores.begin() + i + n));
                    i += n;
                }
                res->scores = std::move(scores);
            }
        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            std::vector<int32_t> idx;
            for (int i = 0; i < batch.size(); ++i) {
                if (batch.tokens[i].output && batch.tokens[i].seq_id == slot.id) {
                    idx.push_back(i);
                }
            }
            const int32_t pos_first = slot.prompt.n_tokens() - (int32_t) idx.size();
            auto get_embd = [&](int32_t pos) -> const float * {
                const int32_t i = pos - pos_first;
                return i >= 0 && i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
            };

            // joint head (decision model): the scores are the first rows
            for (int32_t i = 0; i < decision.n_scores; i++) {
                const float * embd = i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
                if (embd == nullptr) {
                    send_error(slot, "failed to get embeddings", ERROR_TYPE_SERVER);
                    return;
                }
                res->scores.push_back(embd[0]);
            }

            const int32_t n_embd_out = llama_model_n_embd_out(model_tgt);
            const int32_t n_pointer  = n_embd_out / 2;
            const float * embd_q = decision.pointer >= 0 ? get_embd(decision.pointer) : nullptr;
            GGML_ASSERT(decision.column >= 0 && decision.column < n_embd_out);

            for (const int32_t marker : decision.markers) {
                const float * embd = get_embd(marker);
                if (embd == nullptr || (decision.pointer >= 0 && embd_q == nullptr)) {
                    send_error(slot, "failed to get embeddings, the question and its options must fit in one batch", ERROR_TYPE_SERVER);
                    return;
                }
                if (decision.pointer < 0) {
                    res->scores.push_back(embd[decision.column]);
                    continue;
                }
                float dot = 0.0f;
                for (int32_t i = 0; i < n_pointer; i++) {
                    dot += embd_q[i] * embd[n_pointer + i];
                }
                res->scores.push_back(dot / sqrtf((float) n_pointer));
            }
        }

        SLT_DBG(slot, "%s", "sending decision result\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.tokens[i].seq_id);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
        }

        std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot * a, const server_slot * b) {
            if ((a->prompt.n_tokens() == 0) != (b->prompt.n_tokens() == 0)) {
                return a->prompt.n_tokens() == 0;
            }
            return a->t_last_used < b->t_last_used;
        });

        free_slots.resize(std::min(free_slots.size(), n_slots_needed));

        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    // k15-pp: the checkpoint at a merged break (n_tokens_at tokens of the slot, last position pos_at), from the state the decode captured
    // there; only where the draft and the speculative state hold nothing a checkpoint at that position would need (a draft without
    // recurrent / SWA rows, no stashed boundary row)
    void create_checkpoint_captured(server_slot & slot, int64_t n_tokens_at, llama_pos pos_at) {
        spec_bg_join_or_abort();
        if (ctx_dft) {
            // the header alone (magic, seq id): a KV cache without SWA writes nothing for a partial state
            const size_t n_dft = llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (n_dft > 16) {
                SLT_DBG(slot, "%s", "merged break: the draft holds a partial state, no checkpoint there\n");
                return;
            }
        }
        std::vector<uint8_t> ds;
        if (common_speculative_get_state(spec.get(), slot.id, ds) && !ds.empty()) {
            SLT_DBG(slot, "%s", "merged break: the speculative state is stashed per checkpoint, no checkpoint there\n");
            return;
        }
        const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
        const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
        const llama_pos pmin = std::max<llama_pos>(0, pos_at - (pos_max - pos_min));
        create_checkpoint(slot, slot.prompt.n_tokens() - n_tokens_at, pmin, pos_at, true);
    }

    // k19-ckptkeep: which checkpoint a full list keeps no matter what. Agentic clients send many requests that share the system prompt +
    // tools but diverge after it (a new conversation, a rewritten history, a sub-agent); a hybrid / recurrent model cannot roll its state
    // back, so once no checkpoint lies inside that prefix such a request re-reads the whole prompt. The prefix checkpoint is the last one
    // at or before the first user message of the current prompt (the break the server makes there), else the oldest (no message spans:
    // a raw prompt, a slot file). Only which checkpoint survives changes, never how one is made or restored, so outputs are those of a
    // full re-read. LLAMA_SERVER_CKPT_KEEP_PREFIX=0: no checkpoint is protected, the oldest is evicted first (the old behaviour).
    static bool ckpt_keep_prefix() {
        static const bool on = [] {
            const char * e = getenv("LLAMA_SERVER_CKPT_KEEP_PREFIX");
            const bool v = e == nullptr || atoi(e) != 0;
            SRV_INF("context checkpoint eviction: %s (LLAMA_SERVER_CKPT_KEEP_PREFIX=%s)\n",
                    v ? "keep the prefix checkpoint (system + tools)" : "evict the oldest", e ? e : "unset");
            return v;
        }();
        return on;
    }

    static int32_t ckpt_first_user_pos(const server_slot & slot) {
        if (slot.task) {
            for (const auto & sp : slot.task->params.message_spans.spans) {
                if (sp.role == COMMON_CHAT_ROLE_USER) {
                    return (int32_t) sp.pos;
                }
            }
        }
        return -1;
    }

    // the protected checkpoint, or end() (kill switch, or a list of at most one: keeping it would block every newer checkpoint)
    static std::list<common_prompt_checkpoint>::iterator ckpt_prefix_pos(std::list<common_prompt_checkpoint> & checkpoints, int32_t n_max,
            int32_t first_user) {
        if (!ckpt_keep_prefix() || n_max < 2 || checkpoints.empty()) {
            return checkpoints.end();
        }
        auto prot = checkpoints.begin();
        if (first_user > 0) {
            for (auto it = checkpoints.begin(); it != checkpoints.end() && it->n_tokens <= first_user; ++it) {
                prot = it;
            }
        }
        return prot;
    }

    // the checkpoint a full list gives up for a new one: the oldest one that is not protected (with -ctxcp 2: the list stays prefix + newest)
    static std::list<common_prompt_checkpoint>::iterator ckpt_evict_pos(std::list<common_prompt_checkpoint> & checkpoints,
            std::list<common_prompt_checkpoint>::iterator prot) {
        auto it = checkpoints.begin();
        if (it == prot && checkpoints.size() >= 2) {
            ++it;
        }
        return it;
    }

    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max, bool captured = false) {
        const int id_task = slot.task->id;
        // k19-host: with the drafter's pass on the worker (LLAMA_MTP_PROMPT_BG) the draft part is queued behind it; the launch completed
        // every earlier asynchronous draft save, so none is in flight for the evictions below
        const bool dft_bg = spec_bg.live && ctx_dft != nullptr;

        // the previous checkpoint's asynchronous copy may still be running (LLAMA_STATE_ASYNC_LAZY): done before any is evicted
        llama_state_seq_async_wait(ctx_tgt);
        if (ctx_dft && !dft_bg) {
            llama_state_seq_async_wait(ctx_dft); // k13-pp: the draft part is saved asynchronously too (ckpt_dft_async)
        }

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        std::list<common_prompt_checkpoint> dropped;

        // k19-ckptkeep: neither loop below drops the prefix checkpoint (ckpt_prefix_pos); list erasures keep the iterator valid
        const auto prot = ckpt_prefix_pos(slot.prompt.checkpoints, params_base.n_ctx_checkpoints, ckpt_first_user_pos(slot));

        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            if (it != prot && it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                auto next = std::next(it);
                dropped.splice(dropped.end(), slot.prompt.checkpoints, it);
                it = next;
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            // k19-ckptkeep: not the prefix checkpoint (system prompt + tools), which a new conversation or an edited history can still
            // restore; the oldest other one instead (ckpt_evict_pos)
            const auto victim = ckpt_evict_pos(slot.prompt.checkpoints, prot);
            const auto & cur = *victim;

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)%s\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024,
                    prot == slot.prompt.checkpoints.end() ? "" : ", keeping the prefix checkpoint");

            dropped.splice(dropped.end(), slot.prompt.checkpoints, victim);
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", it->n_tokens);
                    auto next = std::next(it);
                    dropped.splice(dropped.end(), slot.prompt.checkpoints, it);
                    it = next;
                } else {
                    ++it;
                }
            }
        }

        checkpoints_free_async(std::move(dropped));

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        const int64_t t_tr0 = common_mtp_handoff_trace() ? ggml_time_us() : 0;

        // the next llama_decode() is queued while the state is still being copied: the copy is complete when it returns
        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ASYNC | (captured ? LLAMA_STATE_SEQ_FLAGS_CAPTURED : 0));
        const int64_t t_tr1 = t_tr0 ? ggml_time_us() : 0;
        // k13-pp: the draft's rows are read on the draft's own stream after its queued prompt pass, so the target's next ubatch is
        // queued while the draft still computes (the synchronous read waited for the draft pass, ~1.5 ms per chat prompt on G4);
        // the save is complete when the draft's next decode() returns, or at any state call / synchronize of the draft context.
        // LLAMA_CKPT_DFT_ASYNC=0: the synchronous read as before.
        static const bool ckpt_dft_async = getenv("LLAMA_CKPT_DFT_ASYNC") == nullptr || atoi(getenv("LLAMA_CKPT_DFT_ASYNC")) != 0;
        if (dft_bg) {
            // the list node is stable, and nothing evicts or reads it before the join (decode()'s spec block, or any other user of the
            // slot's checkpoints: a later create_checkpoint, prompt save / clear, slot release, a task)
            common_prompt_checkpoint * cp = &cur;
            const int  slot_id   = slot.id;
            const int  n_ckpt    = (int) slot.prompt.checkpoints.size();
            const int  n_ckpt_mx = params_base.n_ctx_checkpoints;
            const double t_tgt_ms = t_tr0 ? (t_tr1 - t_tr0)/1e3 : 0.0;
            spec_bg_push([this, cp, slot_id, id_task, n_ckpt, n_ckpt_mx, t_tgt_ms]() {
                const int64_t t_d0 = common_mtp_handoff_trace() ? ggml_time_us() : 0;
                llama_state_seq_async_wait(ctx_dft);
                cp->update_dft(ctx_dft, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | (ckpt_dft_async ? LLAMA_STATE_SEQ_FLAGS_ASYNC : 0));
                common_speculative_get_state(spec.get(), slot_id, cp->data_spec);
                if (t_d0) {
                    LOG_INF("slot %12.*s: id %2d | task %d | mtp_trace ckpt tgt=%.1f MiB %.2f ms, dft=%.1f MiB + spec %.2f ms (draft part on the worker)\n",
                            12, "create_check", slot_id, id_task, cp->data_tgt.size()/1048576.0, t_tgt_ms, cp->data_dft.size()/1048576.0,
                            (ggml_time_us() - t_d0)/1e3);
                }
                LOG_TRC("slot %12.*s: id %2d | task %d | created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        12, "create_check", slot_id, id_task, n_ckpt, n_ckpt_mx, cp->pos_min, cp->pos_max, cp->n_tokens, (float) cp->size() / 1024 / 1024);
            });
            return;
        }
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | (ckpt_dft_async ? LLAMA_STATE_SEQ_FLAGS_ASYNC : 0));
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        if (t_tr0) {
            SLT_INF(slot, "mtp_trace ckpt tgt=%.1f MiB %.2f ms, dft=%.1f MiB + spec %.2f ms\n", cur.data_tgt.size()/1048576.0, (t_tr1 - t_tr0)/1e3,
                    cur.data_dft.size()/1048576.0, (ggml_time_us() - t_tr1)/1e3);
        }

        // k15-pp debug (LLAMA_CKPT_HASH=1): a hash of the target part, to compare checkpoints taken at one position in different ways
        static const bool ckpt_hash = getenv("LLAMA_CKPT_HASH") != nullptr && atoi(getenv("LLAMA_CKPT_HASH")) != 0;
        if (ckpt_hash) {
            llama_state_seq_async_wait(ctx_tgt);
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < cur.data_tgt.size(); ++i) { h ^= cur.data_tgt.data()[i]; h *= 1099511628211ull; }
            SLT_INF(slot, "checkpoint hash n_tokens = %" PRId64 " pos = [%d, %d] size %zu captured %d: %016llx\n", cur.n_tokens, cur.pos_min,
                cur.pos_max, cur.data_tgt.size(), (int) captured, (unsigned long long) h);
            if (const char * dir = getenv("LLAMA_CKPT_DUMP")) {
                const std::string fn = std::string(dir) + "/ckpt_" + std::to_string(cur.n_tokens) + "_" + std::to_string((int) captured) + ".bin";
                if (FILE * f = fopen(fn.c_str(), "wb")) { fwrite(cur.data_tgt.data(), 1, cur.data_tgt.size(), f); fclose(f); }
            }
        }
        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION = 1;

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    template <typename Buf>
    static bool ckpt_read_buf(std::ifstream & ifs, Buf & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    template <typename Buf>
    static void ckpt_write_buf(std::ofstream & ofs, const Buf & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        if (slot.prompt.checkpoints.empty()) {
            return true;
        }
        // asynchronous checkpoint saves (target and draft parts) must have landed before their buffers are written out
        llama_state_seq_async_wait(ctx_tgt);
        if (ctx_dft) {
            llama_state_seq_async_wait(ctx_dft);
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) slot.prompt.checkpoints.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto & cur : slot.prompt.checkpoints) {
            ckpt_write(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
            ckpt_write(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
            ckpt_write(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
            ckpt_write_buf(ofs, cur.data_tgt,  n_written);
            ckpt_write_buf(ofs, cur.data_dft,  n_written);
            ckpt_write_buf(ofs, cur.data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable appendix
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_CKPT_VERSION ||
            !ckpt_read(ifs, &count,   sizeof(count),   n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        std::list<common_prompt_checkpoint> checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            common_prompt_checkpoint cur;
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.empty()) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            checkpoints.push_back(std::move(cur));
            if (checkpoints.size() > (size_t) params_base.n_ctx_checkpoints) {
                // k19-ckptkeep: as create_checkpoint, no message spans here: the oldest is the protected one
                checkpoints.erase(ckpt_evict_pos(checkpoints, ckpt_prefix_pos(checkpoints, params_base.n_ctx_checkpoints, -1)));
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        if (ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back().data_dft.empty()) {
            const bool ok = checkpoints.back().load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            if (!ok) {
                SLT_WRN(slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", filepath.c_str());
                for (auto & cur : checkpoints) {
                    cur.clear_dft();
                }
            }
        }
        slot.prompt.checkpoints = std::move(checkpoints);
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'\n", slot.prompt.checkpoints.size(), filepath.c_str());
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only
    static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (ctx_dft != nullptr) {
            data_dft.resize(llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
            llama_state_seq_get_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        common_speculative_get_state(spec.get(), slot.id, data_spec);
        if (data_dft.empty() && data_spec.empty()) {
            return true;
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_DRAFT_MAGIC;
        const uint32_t version = SLOT_DRAFT_VERSION;
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, data_dft,  n_written);
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB) to '%s'\n", (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable draft state
    // the draft starts empty unless the file holds a draft state that fits the draft context
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        if (ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_DRAFT_VERSION ||
            !ckpt_read_buf(ifs, data_dft,  n_avail, n_read) ||
            !ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        if (!data_dft.empty()) {
            if (ctx_dft == nullptr ||
                llama_state_seq_set_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.size()) {
                if (ctx_dft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                }
                SLT_WRN(slot, "draft state in '%s' does not match the draft context - dropped\n", filepath.c_str());
                return n_read;
            }
        }
        if (!data_spec.empty()) {
            common_speculative_set_state(spec.get(), slot.id, data_spec);
        }
        SLT_INF(slot, "restored the draft state (%.3f MiB) from '%s'\n", (float) n_read / 1024 / 1024, filepath.c_str());
        return n_read;
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }
        // k19-host: any task but the loop's own next-response tick and the read-only ones ends the drafter's background pass first
        if (!is_yielding && task.type != SERVER_TASK_TYPE_NEXT_RESPONSE && task.type != SERVER_TASK_TYPE_METRICS &&
                task.type != SERVER_TASK_TYPE_SLOT_GET) {
            spec_bg_join_or_abort();
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
            case SERVER_TASK_TYPE_DECISION:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        task.t_posted_us = 0; // its prompt time starts when a slot takes it
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        task.t_posted_us = 0;
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            task.t_posted_us = 0;
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                // k15-idlewait (LLAMA_PCS_IDLE_SLOTS_ASYNC=0: synchronous, copied, as before): the save of an idle slot
                                // at a new task takes the request-save path: the state rows read into the pinned staging the context
                                // holds (no new pinned memory, else synchronous), and with a unified KV cache, where the slot is cleared
                                // right after, its checkpoints move into the entry instead of being copied. G4 fn4 16 GB -np 4 chat:
                                // 52.7 ms of synchronous reads (131.8 MiB) before the new task's prompt
                                static const bool idle_slots_async = getenv("LLAMA_PCS_IDLE_SLOTS_ASYNC") == nullptr ||
                                    atoi(getenv("LLAMA_PCS_IDLE_SLOTS_ASYNC")) != 0;
                                const bool saved = idle_slots_async ?
                                    slot.prompt_save(*prompt_cache, params_base.kv_unified ? -1 : -2, false, false, true) :
                                    slot.prompt_save(*prompt_cache);
                                if (saved) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    if (slot->ckpt_lent && prompt_cache) { // k15-idlewait: the lent checkpoints come back, the entry keeps copies (as with the lend off)
                        prompt_cache->ckpt_return(slot->prompt.tokens, slot->prompt.checkpoints, false, 0, true);
                        SLT_INF(*slot, "slot action: %zu lent checkpoint(s) back, the cache entry keeps copies\n", slot->prompt.checkpoints.size());
                        slot->ckpt_lent = false;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    if (slot->ckpt_lent && prompt_cache) { // k15-idlewait: the lent checkpoints come back, the entry keeps copies (as with the lend off)
                        prompt_cache->ckpt_return(slot->prompt.tokens, slot->prompt.checkpoints, false, 0, true);
                        SLT_INF(*slot, "slot action: %zu lent checkpoint(s) back, the cache entry keeps copies\n", slot->prompt.checkpoints.size());
                        slot->ckpt_lent = false;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    if (slot->ckpt_lent && prompt_cache) { // k15-idlewait: the lent checkpoints come back, the entry keeps copies (as with the lend off)
                        prompt_cache->ckpt_return(slot->prompt.tokens, slot->prompt.checkpoints, false, 0, true);
                        SLT_INF(*slot, "slot action: %zu lent checkpoint(s) back, the cache entry keeps copies\n", slot->prompt.checkpoints.size());
                        slot->ckpt_lent = false;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // k11 (LLAMA_LM_CKPT_MERGE): see lm_breaks. Only for a slot alone in the batch, a target with layer-major prompts,
    // no mtmd input and speculative impls that read the target rows of a segment (MTP, draft models, n-gram).
    // The prompt's ubatches and kernels are those of the separate calls, but the MoE cache is seeded once (from the routing of
    // the ubatch before the last) instead of once per call, and the checkpoint 4 + n_ubatch tokens before the end is not made (the
    // one 4 tokens before the end is). k11 saw other decode text after the merged prompt where the host computes misses or the
    // fetch overlap splits a layer by hit / miss (G4 Qwen3.6 UD-Q4_K_XL + MTP 8 GB, from token 14 / 31 / >64 of 2k / 4.6k / 8k).
    // k15-pp: on by default wherever prompts run layer-major (k13-small: was only on a small device, common_fit_small_device).
    // Separate, the last ubatch before that checkpoint ran alone and streamed every host expert a second time: G4 Qwen3.6-35B-A3B
    // Q4 + MTP at 12 GB, 2k prompt [1532] [512] [4] = 27.7 GB of uploads, 16.0 merged, 3188 -> 5226 tok/s. Merged, the output is
    // md5-equal with the misses pinned on the device (LLAMA_MOE_CPU_MISS=0) and decode speed is unchanged; with the misses on the host
    // the decode after the prompt may take another text (Qwen3.8-Flash-Next 32 GB 8k), as the cache is seeded once.
    // LLAMA_LM_CKPT_MERGE=0: the separate calls
    bool lm_ckpt_merge_ok(const server_slot & slot, int32_t n_tokens_prev, bool lm_ckpt) const {
        static const bool on = getenv("LLAMA_LM_CKPT_MERGE") == nullptr || atoi(getenv("LLAMA_LM_CKPT_MERGE")) != 0;
        return on && !lm_ckpt && n_tokens_prev == 0 && !slot.task->tokens.has_mtmd && llama_layer_major_on(ctx_tgt) &&
            common_speculative_rows_ok(spec.get());
    }

    // k15-ttft (LLAMA_CKPT_NEAR_MERGE=1: on, opt-in): the near prompt checkpoint (4 tokens before the end) from a capture inside the prompt's
    // decode. Only for a slot alone in the batch, no mtmd input, speculative impls that read the target rows of a segment, a memory
    // whose partial state is the recurrent rows (llama_state_capture_supported), and where create_checkpoint_captured can make the
    // checkpoint: a draft without a partial state and no speculative state stashed per checkpoint (else the separate call, as before)
    bool near_ckpt_merge_ok(const server_slot & slot, int32_t n_tokens_prev) const {
        // opt-in (LLAMA_CKPT_NEAR_MERGE=1): exact and multi-turn reuse unchanged, but slower on G4 (Qwen3.6 MTP 32 GB chat 1413 -> 1269
        // prompt tok/s, TTFT 54 -> 61 ms; 2k -2.4%; 12 GB chat -2.6%): the merged decode is GPU-bound for longer after its last launch
        // than the two separate calls
        static const bool on = getenv("LLAMA_CKPT_NEAR_MERGE") != nullptr && atoi(getenv("LLAMA_CKPT_NEAR_MERGE")) != 0;
        static const bool cap_env = getenv("LLAMA_LM_CKPT_CAPTURE") == nullptr || atoi(getenv("LLAMA_LM_CKPT_CAPTURE")) != 0;
        if (!on || !cap_env || n_tokens_prev != 0 || slot.task->tokens.has_mtmd || !llama_state_capture_supported(ctx_tgt) ||
                !common_speculative_rows_ok(spec.get())) {
            return false;
        }
        const_cast<server_context_impl *>(this)->spec_bg_join_or_abort(); // k19-host: the draft context is read below
        if (ctx_dft && llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) > 16) {
            return false;
        }
        std::vector<uint8_t> ds;
        if (common_speculative_get_state(spec.get(), slot.id, ds) && !ds.empty()) {
            return false;
        }
        return true;
    }

    void update_slots() {
        // k19-host: the first pass after a background drafter launch continues the slot's prompt beside it (its [4] tail); any later one joins
        if (spec_bg.live && spec_bg.hold) {
            spec_bg.hold = false;
        } else {
            spec_bg_join_or_abort();
        }
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                // k11-decode: the next prompt starts from the worst-case pp plan (a verify window may have re-planned it)
                // k13-fn: the draft context's MoE cache re-calibrates between requests too
                // k15-ttft: both run once the server has stayed idle (server_queue::on_idle_work), not before a request that is
                // already arriving (G4-class host, Qwen3.6 MTP with a CPU-miss MoE cache: a 50 ms re-calibration ahead of a chat prompt)
                queue_tasks.request_idle_work();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch.render(off, n_tokens);
                bool ok = decode(n_batch, off);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void pre_decode() {
        // k19-host: beside the drafter's background pass only the batch of the slot's last prompt tokens is built (its seq_rm and checkpoint
        // queue their draft parts behind the pass); anything else joins it first
        if (spec_bg.live) {
            // k19-studio-np4: with several slots, only while the slot of the pass is the only one processing (spec_bg_slot)
            const server_slot * bs = spec_bg_slot();
            const bool tail = bs != nullptr && bs->id == spec_bg.slot_id && bs->state == SLOT_STATE_PROCESSING_PROMPT && bs->task &&
                !bs->task->tokens.has_mtmd && bs->task->n_tokens() > bs->prompt.n_tokens() &&
                (spec_bg.group || bs->task->n_tokens() - bs->prompt.n_tokens() <= spec_bg_tail_max);
            if (!tail) {
                spec_bg_join_or_abort();
            }
        }
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt_saved_idle = false;
                    slot.ckpt_lent = false;
                    slot.prompt.tokens.insert(new_tokens);
                }

                // the remaining cells were shifted and attended to the discarded ones: not the state of these tokens
                slot.prompt.kv_shifted = true;

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();
        lm_breaks.clear();
        lm_caps.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // k19-small1115: another slot active (prompt or generation): the windows of every slot are cut to the k19-tpp floor this round
        // (the drafts of a --parallel N server without the wider windows); a lone request keeps them (see load_model)
        int32_t spec_floor_cap = 0;
        if (spec && spec_active_cap > 0) {
            int n_active = 0;
            for (const auto & s : slots) {
                n_active += s.is_processing() ? 1 : 0;
            }
            if (n_active > 1) {
                spec_floor_cap = spec_active_cap;
            }
        }

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                int n_draft_max = slot.get_n_draft_max();
                if (spec_floor_cap > 0) {
                    n_draft_max = std::min(n_draft_max, spec_floor_cap);
                }
                if (spec_g4t_cap > 0) {
                    n_draft_max = std::min(n_draft_max, spec_g4t_cap);
                }

                slot.gov_step = 0;
                // a partial draft to replay is verified whatever the governor says (it restores a state)
                if (n_draft_max > 0 && slot.spec_draft.empty() && !slot.gov.allows()) {
                    n_draft_max   = 0;
                    slot.gov_step = 1;
                }

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty() || slot.spec_rs_snap);
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        // k19-hostgap (LLAMA_SPEC_PROMPT_INCR=0: a full copy each time, as before): the drafter's copy of the prompt
                        // is extended by the tokens past its length when it is still a prefix of the slot's tokens (text-only
                        // slots, where the text tokens are the tokens): the same vector, without copying the whole context every
                        // pass (8k context on a Colab A100 host: ~17 us per pass, with the GPU idle)
                        static const bool incr_env = getenv("LLAMA_SPEC_PROMPT_INCR") == nullptr || atoi(getenv("LLAMA_SPEC_PROMPT_INCR")) != 0;
                        const llama_tokens & cur = slot.prompt.tokens.get_tokens();
                        auto & sp = slot.spec_prompt;
                        if (incr_env && !slot.prompt.tokens.has_mtmd && sp.size() <= cur.size() &&
                                (sp.empty() || memcmp(sp.data(), cur.data(), sp.size()*sizeof(llama_token)) == 0)) {
                            for (size_t k = sp.size(); k < cur.size(); ++k) {
                                if (cur[k] != LLAMA_TOKEN_NULL) {
                                    sp.push_back(cur[k]); // (as get_text_tokens: a prefix that matched holds no null token)
                                }
                            }
                        } else {
                            sp = slot.prompt.tokens.get_text_tokens();
                        }

                        const bool spec_reject = slot.use_spec_rejection();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .temp     = */ slot.task->params.sampling.temp,
                            /* .seed     = */ slot.task->params.sampling.seed,
                            /* .top_k    = */ slot.task->params.sampling.top_k,
                            /* .top_p    = */ slot.task->params.sampling.top_p,
                            /* .min_p    = */ slot.task->params.sampling.min_p,
                        };
                        common_speculative_get_draft_params(spec.get(), slot.id).shared = spec_floor_cap > 0;

                        // k19-sharednoise: the target draws with shared noise (LLAMA_SHARED_NOISE=1): the drafter takes the same noise
                        // and the verify samples and matches (LLAMA_SHARED_NOISE_DRAFT=0: the drafter as before, for attribution)
                        if (spec_sn_draft_on() && common_sampler_shared_noise_draft_init(slot.smpl.get(), slot.spec_sn)) {
                            auto & dpr = common_speculative_get_draft_params(spec.get(), slot.id);
                            dpr.sn       = &slot.spec_sn;
                            dpr.result_q = nullptr;
                            static bool logged = false;
                            if (!logged) {
                                logged = true;
                                SLT_INF(slot, "shared noise: drafts take the target's noise (seed %u, key %llu); verify by sample-and-match\n",
                                        slot.spec_sn.seed, (unsigned long long) slot.spec_sn.pos);
                            }
                        } else if (spec_draft_pen_on() && !params_base.speculative.draft.probabilistic &&
                                   common_sampler_penalty_draft_init(slot.smpl.get(), slot.spec_sn)) {
                            // k19-q4xaccept: the request has penalties: drafts are the argmax after them (spec_draft_pen_on)
                            auto & dpr = common_speculative_get_draft_params(spec.get(), slot.id);
                            dpr.sn       = &slot.spec_sn;
                            dpr.result_q = nullptr;
                            slot.n_spec_pen++;
                            static bool logged = false;
                            if (!logged) {
                                logged = true;
                                SLT_INF(slot, "%s", "penalty-aware drafts: the request's penalties apply to the draft's candidates before its argmax "
                                        "(LLAMA_SPEC_DRAFT_PEN=0: off)\n");
                            }
                        }

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            srv_pass_timer().stamp(server_pass_timer::DRAFT0);
            srv_yield_or_run(queue_tasks, true, [&]() {
                common_speculative_draft(spec.get());
            });
            srv_pass_timer().stamp(server_pass_timer::DRAFT1);
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.gov_step = draft.empty() ? 3 : 2;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                // k19-rollback: a recurrent target without rollback rows copies its state rows to host memory on the device stream instead
                // (llama_rs_snap_save: no host wait, no serialised state). A replay of the accepted tokens starts from that same copy, so
                // it keeps it. LLAMA_SPEC_RS_SNAP=0: the serialised checkpoint, as before
                static const bool rs_snap_env = getenv("LLAMA_SPEC_RS_SNAP") == nullptr || atoi(getenv("LLAMA_SPEC_RS_SNAP")) != 0;
                if (use_ckpt_tgt && slot.spec_is_replay && slot.spec_rs_snap) {
                    // the copy of the window this replay restored
                } else if (use_ckpt_tgt && rs_snap_env && llama_rs_snap_save(ctx_tgt, slot.id)) {
                    ckpt.clear_tgt();
                    slot.spec_rs_snap = true;
                } else if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();
                    slot.spec_rs_snap = false;

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        // k15-hostcut: from when the task was posted, so prompt_ms covers what delayed it on the server loop (the client's TTFT
                        // less the request's parse, tokenize and delivery)
                        static const bool from_post = getenv("LLAMA_PROMPT_TIME_FROM_POST") == nullptr || atoi(getenv("LLAMA_PROMPT_TIME_FROM_POST")) != 0;
                        slot.stats.update_prompt_start(from_post ? slot.task->t_posted_us : 0);

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not support logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // the outputs of a decision are read from one batch
                            const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;
                            if (n_decision_first >= 0 && slot.task->n_tokens() - n_decision_first > n_batch) {
                                send_error(slot,
                                           string_format("the question and its options (%d tokens) are too large to process. "
                                                         "increase the batch size (current batch size: %d)",
                                                         slot.task->n_tokens() - n_decision_first, n_batch),
                                           ERROR_TYPE_INVALID_REQUEST);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // the children start from the shared prefix, do not go past it
                                if (slot.task->n_tokens_shared > 0) {
                                    n_past = std::min(n_past, slot.task->n_tokens_shared);
                                }

                                // the outputs of a decision are read from the batch, so its first position must be evaluated
                                if (n_decision_first >= 0) {
                                    n_past = std::min(n_past, n_decision_first);
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);
                                            slot.prompt.kv_shifted = true;

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
                            // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
                            const llama_pos pos_eval = pos_next - (has_new_tokens ? 0 : 1);

                            // a state or a checkpoint is useful when its pos_min is below this threshold
                            const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        if (!it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                                            !it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                                            if (it->id_task != -1) {
                                                GGML_ABORT("failed to restore context checkpoint\n");
                                            }
                                            // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                                            SLT_WRN(slot, "%s", "failed to load context checkpoint restored from a slot file\n");
                                            do_reset = true;
                                        }
                                    }

                                    if (!do_reset) {
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                std::list<common_prompt_checkpoint> dropped;
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        auto next = std::next(it);
                                        dropped.splice(dropped.end(), slot.prompt.checkpoints, it);
                                        it = next;
                                    } else {
                                        ++it;
                                    }
                                }
                                checkpoints_free_async(std::move(dropped));
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (spec_bg.live && slot.ctx_dft) {
                        // k19-host: the draft's part behind the drafter's pass on the worker, in the order of the in-line pass
                        if (!llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, p0, -1)) {
                            GGML_ABORT("failed to remove sequence %d with p0=%d, p1=-1\n", slot.id, p0);
                        }
                        llama_context * cd = slot.ctx_dft;
                        const llama_seq_id sid = slot.id;
                        spec_bg_push([cd, sid, p0]() {
                            if (!llama_memory_seq_rm(llama_get_memory(cd), sid, p0, -1)) {
                                GGML_ABORT("failed to remove sequence %d with p0=%d, p1=-1\n", sid, p0);
                            }
                        });
                    } else {
                        slot.mem.seq_rm(slot.id, p0, -1);
                    }

                    // shared prompt prefix: once it is processed, the children continue from it with their own prompt
                    bool wait_shared = false;
                    if (slot.task->n_tokens_shared > 0) {
                        const bool is_shared_done = slot.prompt.n_tokens() == slot.task->n_tokens_shared;
                        for (auto & other : slots) {
                            if (other.state != SLOT_STATE_WAIT_OTHER || other.task->id_parent != slot.task->id) {
                                continue;
                            }
                            if (is_shared_done) {
                                SLT_TRC(slot, " - copying shared prompt (%d tokens) to child %d\n", slot.prompt.n_tokens(), other.id);
                                slot.copy_prompt_to(other);
                                other.state = SLOT_STATE_STARTED;
                            } else {
                                wait_shared = true;
                            }
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    // TODO: do the same for all models, then remove process_mtmd_chunk()
                    if (use_mixed_batch() && !slot.can_split() && input_tokens.has_mtmd) {
                        if (!add_prompt_mixed(slot)) {
                            return;
                        }
                    }

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // stop at the end of the shared prefix, the children are started from this state
                        if (wait_shared && slot.prompt.n_tokens() == slot.task->n_tokens_shared) {
                            break;
                        }

                        // the outputs of a decision are read from one batch, do not split them
                        if (slot.prompt.n_tokens() == n_decision_first && batch.size() + slot.task->n_tokens() - n_decision_first > n_batch) {
                            break;
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        if (!slot.task->decision.order.empty()) {
                            batch.set_decision_order(batch.size() - 1, slot.task->decision.order[slot.prompt.n_tokens()]);
                        }
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            // layer-major prompts read a whole batch of ubatches in one expert pass, so the last ubatch before this checkpoint
                            // costs a pass of its own. LLAMA_MOE_LAYER_MAJOR_CKPT=1 (opt-in) moves the checkpoint one batch before the end
                            // instead (faster prompts, but other ubatch boundaries, so other text, and G4 Qwen3.6 + MTP decode -4..-5%);
                            // LLAMA_SERVER_CKPT_BATCH=1 moves it without layer-major prompts, for an A/B with the same batches
                            static const bool lm = (getenv("LLAMA_MOE_LAYER_MAJOR_CKPT") != nullptr && atoi(getenv("LLAMA_MOE_LAYER_MAJOR_CKPT")) != 0) ||
                                (getenv("LLAMA_SERVER_CKPT_BATCH") != nullptr && atoi(getenv("LLAMA_SERVER_CKPT_BATCH")) != 0);
                            const int checkpoint_offsets[] = {4 + (lm ? n_batch : n_ubatch), 4};

                            // k9-prompt-pass: n_batch raised for layer-major prompts at an explicit -ub (common_moe_cache_auto_ubatch): the
                            // offsets are clamped to the n_batch given, so the prompt splits into the same batches and ubatches as before
                            const int n_batch_clamp = !lm && params_base.n_batch_ckpt > 0 ? std::min<int>(n_batch, params_base.n_batch_ckpt) : n_batch;
                            // k11: LLAMA_SERVER_CKPT_SINGLE_MAX=N (opt-in, default 0 = off): a prompt whose part still to process when its
                            // batch starts is at most N tokens (and fits one ubatch) runs in one pass, without the separate pass of its last 4
                            // tokens; its checkpoint is then made at the end of the prompt, so a next prompt that differs in those last tokens
                            // is processed again from the previous checkpoint. Other ubatch shapes, so other numerics (KLD-gated, not md5-equal)
                            // k13-pp: only where a next prompt pays at most this batch again: the whole prompt is at most N tokens, or the
                            // batch starts at the last user message (a checkpoint is made there). A single-pass prompt makes no checkpoint of
                            // its own (the memory is empty before its only pass), so without that bound a raw prompt that grows by a few
                            // tokens per turn found no usable checkpoint and ran again from the start. LLAMA_SERVER_CKPT_SINGLE_ANY=1: the
                            // k11 rule (any prompt whose rest is at most N). Still opt-in: the KLD gate  failed on G4 for
                            // Qwen3.6-35B-A3B Q4 (first-token p99 0.0058 vs 0.0024 for -fa off), LFM2.5 (first-token mean 0.24 vs 0.065) and
                            // gemma-4 26B (decode top-1 98.9 vs 99.7%)
                            // k15-ttft: on by default (N = 128) where the single pass is reassociation-only
                            // (equivalence rule, llama_single_pass_equivalent: all weights on CUDA, no host-computed expert, no F16 matmul weight) and
                            // the split pass would run at least 4 tokens before its last 4 (below that MMVF F32 / FlashAttention vec kernels
                            // take the first pass, other formats). An explicit LLAMA_SERVER_CKPT_SINGLE_MAX keeps the k11 meaning on every
                            // placement (0 = off, the kill switch)
                            static const char * ckpt_single_env = getenv("LLAMA_SERVER_CKPT_SINGLE_MAX");
                            static const bool ckpt_single_eq = ckpt_single_env == nullptr && llama_single_pass_equivalent(ctx_tgt);
                            static const int ckpt_single_max = [&] {
                                const int v = ckpt_single_env ? atoi(ckpt_single_env) : (ckpt_single_eq ? 128 : 0);
                                SRV_INF("single-pass short prompts: max %d (%s)\n", v, ckpt_single_env ? "LLAMA_SERVER_CKPT_SINGLE_MAX" :
                                        ckpt_single_eq ? "default, reassociation-only placement" : "default, off on this placement");
                                return v;
                            }();
                            static const bool ckpt_single_any = getenv("LLAMA_SERVER_CKPT_SINGLE_ANY") != nullptr && atoi(getenv("LLAMA_SERVER_CKPT_SINGLE_ANY")) != 0;
                            const int n_batch_start = slot.prompt.n_tokens() - (int) (batch.size() - n_tokens_prev);
                            const int n_rest_batch = slot.task->n_tokens() - n_batch_start;
                            const bool ckpt_single = ckpt_single_max > 0 && n_rest_batch <= std::min<int>(ckpt_single_max, n_ubatch) &&
                                (!ckpt_single_eq || n_rest_batch >= 8) &&
                                (ckpt_single_any || slot.task->n_tokens() <= ckpt_single_max || (last_user_pos > 0 && n_batch_start == last_user_pos));
                            bool should_break = false;
                            for (int k = 0; k < 2; ++k) {
                                const int offset = checkpoint_offsets[k];
                                if (offset == 4 && ckpt_single) {
                                    continue;
                                }
                                const int n_last = std::min(n_batch_clamp, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    // k11: under layer-major prompts the batch goes on past the far checkpoint, its ubatches end there
                                    // (only when the rest up to the near checkpoint fits this batch: else the batches, so the ubatches, would differ)
                                    // k15-pp (LLAMA_LM_CKPT_MERGE_SPAN, default on; =0 off): only where the group spans the break; elsewhere the separate
                                    // calls seed the cache between them and the merge only lost that (q4x 12 GB 640-token prompt: +14% uploads)
                                    // k19-mergespan: on by default again (LLAMA_LM_CKPT_MERGE_SPAN=0 = merge every eligible break). Without it the
                                    // merge changed one first token of 48 on G4 Flash-Next 32 GB long48 (first-token top-1 0.979 vs a -fa off band of
                                    // 1.000, p99 at the band edge) in two launches; with it long48 is byte-identical to the merge off and the 8k TTFT
                                    // stays the merge's (1.553 vs 1.551 s, merge off 2.030 s). Cost: G4 gemma-4 26B-A4B UD-Q3_K_M 8 GB chat TTFT
                                    // 0.151 -> 0.185 s (measured 10-25, the reason it had been off)
                                    static const bool lm_span_env = getenv("LLAMA_LM_CKPT_MERGE_SPAN") == nullptr || atoi(getenv("LLAMA_LM_CKPT_MERGE_SPAN")) != 0;
                                    if (k == 0 && offset != 4 && batch.size() + (n_last - 4) <= n_batch && lm_ckpt_merge_ok(slot, n_tokens_prev, lm) &&
                                            (!lm_span_env || llama_layer_major_spans(ctx_tgt, (int32_t) batch.size(), n_last - 4))) {
                                        lm_breaks.push_back(batch.size());
                                        if (lm_caps.empty() && do_checkpoint) {
                                            lm_caps.push_back({ slot.id, (int64_t) slot.prompt.n_tokens(), (int32_t) batch.size() });
                                        }
                                        break;
                                    }
                                    // k15-ttft: the checkpoint 4 tokens before the end is taken inside this decode (llama_set_state_capture)
                                    // instead of after a decode of its own: the same ubatches (the last 4 tokens still run as one), one call,
                                    // no host gap between them, and under layer-major prompts one expert pass (see near_ckpt_merge_ok)
                                    // k15-pp2: beside the far capture of this batch too (two captures per decode)
                                    if (k == 1 && do_checkpoint && (int) lm_caps.size() < lm_cap_max() &&
                                            (lm_caps.empty() || lm_caps[0].slot == slot.id) && batch.size() + n_last <= n_batch &&
                                            near_ckpt_merge_ok(slot, n_tokens_prev)) {
                                        lm_breaks.push_back(batch.size());
                                        lm_caps.push_back({ slot.id, (int64_t) slot.prompt.n_tokens(), (int32_t) batch.size() });
                                        break;
                                    }
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    // k15-small4: a layer-major batch merged across the far checkpoint (lm_caps[0]) ends, for this rule, where that checkpoint
                    // is captured, as the batches without the merge do: else the batch before it counted as near the end and made a checkpoint
                    // of its own at its start (q3m 4 GB, 8k prompt at n_batch 8192: 3 checkpoints of 62.8 MiB instead of 2, RssAnon +63 MiB
                    // per long prompt). LLAMA_SERVER_CKPT_MERGED_END=0: the end of the merged batch, as before
                    static const bool ckpt_merged_end = getenv("LLAMA_SERVER_CKPT_MERGED_END") == nullptr || atoi(getenv("LLAMA_SERVER_CKPT_MERGED_END")) != 0;
                    const int64_t n_tokens_end = ckpt_merged_end && !lm_caps.empty() && lm_caps[0].slot == slot.id && lm_caps[0].n_tokens > 0 ? lm_caps[0].n_tokens : slot.prompt.n_tokens();
                    const bool near_prompt_end = slot.task->n_tokens() < n_tokens_end + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // see https://github.com/ggml-org/llama.cpp/pull/29969
    // TODO @ngxson : maybe remove this once we use "mixed" batch everywhere
    bool use_mixed_batch() const {
        return !llama_get_memory(ctx_tgt) || !llama_get_causal_attn(ctx_tgt);
    }

    // add the rest of the prompt to the batch, the mtmd chunks are added as embeddings next to the text tokens
    // the caller makes sure that it fits in the batch
    // returns false on error, the slot is then released
    bool add_prompt_mixed(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        const auto n_tokens_prev = batch.size();

        while (slot.prompt.n_tokens() < slot.task->n_tokens()) {
            const auto cur_token_idx = slot.prompt.n_tokens();
            const llama_token cur_tok = input_tokens[cur_token_idx];

            if (cur_tok != LLAMA_TOKEN_NULL) {
                const bool add_ok = batch.add(slot.id,
                    cur_tok,
                    /* pos       = */ slot.prompt.tokens.pos_next(),
                    /* output    = */ slot.need_embd(),
                    /* is_prompt = */ true);
                GGML_ASSERT(add_ok);
                if (!slot.task->decision.order.empty()) {
                    batch.set_decision_order(batch.size() - 1, slot.task->decision.order[cur_token_idx]);
                }
                slot.prompt.tokens.push_back(cur_tok);
                continue;
            }

            const auto & chunk = input_tokens.find_chunk(cur_token_idx);

            float * embd = slot.mbatch ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            if (!embd) {
                // encode on the worker thread, so we can still handle metrics tasks
                int32_t res = 0;
                queue_tasks.yield_to_queue([&]() {
                    res = encode_mtmd_chunk(slot, slot.mbatch, cur_token_idx);
                });
                embd = res == 0 ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            }

            if (!embd || !add_mtmd_chunk(slot, chunk.get(), embd, batch)) {
                SLT_ERR(slot, "%s", "failed to process mtmd chunk\n");
                // the batch must not keep the entries of a released slot
                batch.truncate(n_tokens_prev);
                send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }

            slot.prompt.tokens.push_back_placeholder(chunk.get());
        }

        return true;
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd()) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        bool has_prompt = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
            has_prompt |= batch.tokens[i].is_prompt;
        }

        // k19-host: the target decode beside the drafter's background pass is at most spec_bg_tail_max tokens of the one slot: such a decode
        // is not layer-major (which may free the drafter's compute buffers) and writes at most that many next-n rows (the stash holds them),
        // it never grows the next-n store (64 rows at least, the pass's batch had more). Anything else joins first
        // k19-host (LLAMA_MTP_PROMPT_BG_GROUP): beside a pass launched in group mode the slot's next prompt batch of any size runs, the
        // target waits for the pass (its next-n hold) before it writes the rows or frees the drafter's compute buffers
        // k19-npkdp (LLAMA_MTP_PROMPT_BG_MULTI_JOIN=1: join at -np > 1 as before): with several slots, the batch of the pass's slot while it
        // is the only one processing (spec_bg_slot) runs beside the pass too; at -np 4 the join here put the drafter's pass back in line
        // before every 8192-token group of a long prompt (G4 Flash-Next 16 GB 30k: 99-144 ms between the batch build and the group)
        static const bool bg_multi_join = getenv("LLAMA_MTP_PROMPT_BG_MULTI_JOIN") != nullptr && atoi(getenv("LLAMA_MTP_PROMPT_BG_MULTI_JOIN")) != 0;
        const server_slot * bg_only = spec_bg.live && slots.size() > 1 && !bg_multi_join ? spec_bg_slot() : nullptr;
        const bool bg_other = slots.size() != 1 && (bg_only == nullptr || bg_only->id != spec_bg.slot_id);
        if (spec_bg.live && (bg_other || (batch.view.size() > spec_bg_tail_max && !spec_bg.group))) {
            if (!spec_bg_join()) {
                SRV_ERR("%s", "failed to process speculative batch\n");
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // a batch of sampled and draft tokens only is a speculative verify batch: with LLAMA_MTP_EXACT=1 its rows are
        // computed as single token decodes would compute them, so greedy speculative output equals plain decoding
        llama_set_verify(ctx_tgt, batch.has_draft && !has_prompt);

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        // k11 (LLAMA_LM_CKPT_MERGE): the breaks inside this view, relative to it
        std::vector<int32_t> seg_breaks;
        for (const int32_t b : lm_breaks) {
            if (b > off && b < off + batch.view.size()) {
                seg_breaks.push_back(b - off);
            }
        }
        // k15-pp (LLAMA_LM_CKPT_CAPTURE=0 off): the checkpoint the merged break skips is taken inside the decode
        static const bool lm_cap_env = getenv("LLAMA_LM_CKPT_CAPTURE") == nullptr || atoi(getenv("LLAMA_LM_CKPT_CAPTURE")) != 0;
        bool lm_cap_req = false;
        std::vector<lm_cap_req_t> caps_req;   // the captures asked for this view
        for (const auto & c : lm_caps) {
            if (lm_cap_env && c.batch > off && c.batch < off + batch.view.size() && llama_set_state_capture(ctx_tgt, c.slot, c.batch - off)) {
                caps_req.push_back(c);
                lm_cap_req = true;
            }
        }
        if (!seg_breaks.empty()) {
            llama_set_ubatch_breaks(ctx_tgt, seg_breaks.data(), (int32_t) seg_breaks.size());
            SRV_DBG("k11 merged prompt batch of %d tokens, %zu ubatch breaks (first at %d)\n", batch.view.size(), seg_breaks.size(), seg_breaks[0]);
        }
        // k15-ttft: the captured checkpoint is made before the wait for the batch's outputs, so its bookkeeping and the queued copies
        // overlap the ubatches still computing (made after the wait, the near checkpoint's merge cost G4 Qwen3.6 32 GB chat 6 ms of TTFT);
        // LLAMA_CKPT_CAP_EARLY=0: after the wait, as k15-pp did
        static const bool cap_early = getenv("LLAMA_CKPT_CAP_EARLY") == nullptr || atoi(getenv("LLAMA_CKPT_CAP_EARLY")) != 0;
        // k15-pp3: the overlap below makes the checkpoints up to a segment's end before it gives that segment to the drafter
        size_t caps_done = 0;
        auto cap_checkpoint_upto = [&](int64_t b_rel) {
            // in batch order: a checkpoint list keeps the order the separate calls made them in
            for (; caps_done < caps_req.size() && caps_req[caps_done].batch - off <= b_rel; ++caps_done) {
                const auto & c = caps_req[caps_done];
                llama_pos cap_pos = -1;
                if (ret == 0 && llama_state_seq_captured_at(ctx_tgt, c.slot, c.batch - off, &cap_pos)) {
                    for (auto & slot : slots) {
                        if (slot.id == c.slot && slot.task) {
                            create_checkpoint_captured(slot, c.n_tokens, cap_pos);
                        }
                    }
                }
            }
        };
        auto cap_checkpoint = [&]() {
            cap_checkpoint_upto(INT64_MAX);
            lm_caps.erase(std::remove_if(lm_caps.begin(), lm_caps.end(), [&](const lm_cap_req_t & c) {
                return c.batch > off && c.batch < off + batch.view.size(); }), lm_caps.end());
        };
        // k15-pp3 (LLAMA_MTP_PROMPT_OVERLAP=0 off): see below
        static const bool overlap_env = getenv("LLAMA_MTP_PROMPT_OVERLAP") == nullptr || atoi(getenv("LLAMA_MTP_PROMPT_OVERLAP")) != 0;
        const bool caps_sorted  = std::is_sorted(caps_req.begin(), caps_req.end(), [](const lm_cap_req_t & a, const lm_cap_req_t & b) {
            return a.batch < b.batch; });
        const bool spec_overlap = overlap_env && spec && has_output && !seg_breaks.empty() && common_speculative_rows_early_ok(spec.get()) &&
            (!lm_cap_req || (cap_early && caps_sorted));
        // the checkpoint (slot state) is made on this thread, outside the yield; with the overlap, the drafter's early segments too
        const bool sync_after = (lm_cap_req && cap_early) || spec_overlap;
        // k19-host (diagnostic, LLAMA_DECODE_PHASE_LOG=1): host time of the server's steps of a decode of more than 64 tokens
        static const bool srv_phase = getenv("LLAMA_DECODE_PHASE_LOG") != nullptr && atoi(getenv("LLAMA_DECODE_PHASE_LOG")) != 0;
        const int64_t t_ph0 = srv_phase && batch.view.size() > 64 ? ggml_time_us() : 0;
        auto srv_ph = [&](const char * what) {
            if (t_ph0) {
                SRV_INF("srv phase n=%d: %-22s %8.2f ms\n", batch.view.size(), what, (ggml_time_us() - t_ph0)/1e3);
            }
        };
        const bool pt_on = srv_pass_timer().on && batch.view.size() <= 32;
        if (pt_on) {
            srv_pass_timer().stamp(server_pass_timer::DEC0);
            srv_pass_timer().width = batch.view.size();
        }
        srv_yield_or_run(queue_tasks, batch.view.size() <= 32, [&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            srv_ph("target returned");
            if (pt_on) {
                srv_pass_timer().stamp(server_pass_timer::DEC1);
            }
            if (ret == 0 && has_output && !sync_after) {
                llama_synchronize(ctx_tgt);
                srv_ph("target synced");
            }
            if (pt_on) {
                srv_pass_timer().stamp(server_pass_timer::DEC2);
            }
        });
        // k19-host: a drafter pass beside this decode is complete before anything below reads the rows or the speculative state
        // (the target's next-n hold normally joined it already, when it wrote the rows)
        if (spec_bg.live && !spec_bg_join()) {
            SRV_ERR("%s", "failed to process speculative batch\n");
            throw std::runtime_error("failed to process speculative batch");
        }
        if (lm_cap_req && !(spec_overlap && ret == 0)) {
            cap_checkpoint();
        }
        // k15-pp3 (LLAMA_MTP_PROMPT_OVERLAP=0 off): the drafter's own pass over the segments whose target rows are already on the host
        // (a layer-major group followed by more ubatches in this decode, e.g. [1532] [512] then the [4] before the near checkpoint) runs
        // while the target still computes the rest, instead of after the target's wait. Same segments, same rows, same order: the
        // drafter's KV and outputs are those of the separate pass
        int32_t spec_seg_done = 0; // segments of seg_breaks the drafter has processed
        int32_t spec_seg_a    = 0; // where the next segment starts
        bool    spec_early_ok = true;
        {
            if (spec_overlap && ret == 0) {
                const int64_t t_ov0 = common_mtp_handoff_trace() ? ggml_time_us() : 0;
                // on this thread, outside the yield, like the checkpoints made in between: each checkpoint's asynchronous read then
                // runs under the drafter's next segment instead of under the wait of the next checkpoint or of the target
                // a segment [a, b) reads the target rows [a, b) (its first token pairs with the pending row of the one before)
                for (; spec_seg_done < (int32_t) seg_breaks.size() && spec_early_ok; ++spec_seg_done) {
                    const int32_t b = seg_breaks[spec_seg_done];
                    if (llama_nextn_rows_wait(ctx_tgt, b) < b) {
                        break;
                    }
                    if (lm_cap_req) {
                        cap_checkpoint_upto(b);
                    }
                    common_batch seg;
                    seg.n_pos = batch.view.n_pos;
                    seg.tokens.assign(batch.view.tokens.begin() + spec_seg_a, batch.view.tokens.begin() + b);
                    spec_early_ok = common_speculative_process_rows_early(spec.get(), seg, spec_seg_a);
                    spec_seg_a = b;
                }
                if (lm_cap_req) {
                    cap_checkpoint();
                }
                if (t_ov0) {
                    SRV_INF("mtp_trace prompt overlap: %d of %zu segments (%d rows) before the target's wait, %.2f ms\n", spec_seg_done,
                            seg_breaks.size() + 1, spec_seg_a, (ggml_time_us() - t_ov0)/1e3);
                }
            }
        }
        if (sync_after && ret == 0 && has_output) {
            const int64_t t_w0 = common_mtp_handoff_trace() && spec_overlap ? ggml_time_us() : 0;
            llama_synchronize(ctx_tgt);
            if (t_w0) {
                SRV_INF("mtp_trace target wait after the overlap %.2f ms\n", (ggml_time_us() - t_w0)/1e3);
            }
        }

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch.view.size(), has_output);
            // k15-idlewait: a cache entry dropped during its save goes back to the pool (the decode joined the save)
            if (prompt_cache && !prompt_cache->dropped_inflight.empty()) {
                prompt_cache->wait_inflight();
            }
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            // k19-host: the drafter's background pass of the previous prompt batch is complete before this batch's pass
            bool ok = spec_bg_join() && spec_early_ok;
            // k19-host (LLAMA_MTP_PROMPT_BG=0 off): this batch's pass on a worker, beside the slot's next (last, <= spec_bg_tail_max token)
            // prompt decode; same segments, rows and order (see spec_bg_state_t)
            static const bool bg_env = getenv("LLAMA_MTP_PROMPT_BG") == nullptr || atoi(getenv("LLAMA_MTP_PROMPT_BG")) != 0;
            // k19-host (LLAMA_MTP_PROMPT_BG_GROUP=0: only beside the last <= 64 prompt tokens): also beside the slot's next prompt batch of
            // any size (the batches of a long prompt, each a released layer-major group at 16 GB): its pre-group host work (members, measure,
            // allocation, ~65 ms with the GPU idle on G4 Flash-Next 16 GB) runs while the drafter computes. The target context waits for
            // the pass (llama_set_nextn_hold) before it writes the next-n rows the pass reads or frees the drafter's compute buffers
            static const bool bg_group_env = getenv("LLAMA_MTP_PROMPT_BG_GROUP") == nullptr || atoi(getenv("LLAMA_MTP_PROMPT_BG_GROUP")) != 0;
            // kdp (LLAMA_MTP_DRAFT_LEND=0 off; LLAMA_MTP_PROMPT_TAIL=N rows, default 2048, 0 off): a prompt batch with at least N of the
            // prompt's tokens still after it is not run through the drafter (common_speculative_process_skip: its cells are dropped, the
            // next batch pairs with this batch's last row), so the drafter computes only the batches that hold the prompt's last N rows
            // (at least N rows) and no drafter pass is in flight beside the earlier layer-major groups: those can borrow the drafter's device
            // weights for a deeper staging arena (llama_context::decode_layer_major). Drafts are verified: only the acceptance moves.
            // G4 fn4 UD-IQ4_XS 12 GB + the requantized Q8_0 drafter, ctx 32768 (kdp_g1, 3 launches): depth 2 -> 3, 8k / 16k / 30k e2e
            // 4.714 / 6.346 / 10.360 -> 4.473 / 6.145 / 9.103 s (no drafter: 4.524 / 6.417 / 9.392)
            static const bool    lend_env = getenv("LLAMA_MTP_DRAFT_LEND") == nullptr || atoi(getenv("LLAMA_MTP_DRAFT_LEND")) != 0;
            static const int32_t tail_env = getenv("LLAMA_MTP_PROMPT_TAIL") ? std::max(0, atoi(getenv("LLAMA_MTP_PROMPT_TAIL"))) : 2048;
            // kdp: drafting off for this request (a failed remap): the drafter computes nothing until the next request
            bool dft_off = false;
            for (const auto & sl : slots) {
                dft_off |= sl.is_processing() && sl.spec_off_req;
            }
            bool skip = false;
            // only where the target's groups lend (llama_mtp_draft_lend_active): elsewhere the skip would cost acceptance for nothing
            // (G4 fn4 16 GB, depth 4 without the lend: 30k e2e +0.9%)
            // k19-np12b (LLAMA_MTP_PROMPT_TAIL_MULTI=0: -np 1 only, as before): with several slots, the skip for the slot that is the only one
            // processing (spec_bg_slot). At -np 4 the target's groups lent the drafter's weights but its pass still ran over every early batch,
            // with a weight remap before each: G4 Qwen3.8-Flash-Next 12 GB -np 4 with the q3_k drafter admitted, 30k TTFT 7.35 vs 6.27 s
            static const bool skip_multi_env = getenv("LLAMA_MTP_PROMPT_TAIL_MULTI") == nullptr || atoi(getenv("LLAMA_MTP_PROMPT_TAIL_MULTI")) != 0;
            const server_slot * skip_sl = slots.size() == 1 ? &slots[0] : (skip_multi_env ? spec_bg_slot() : nullptr);
            if (!dft_off && lend_env && tail_env > 0 && ok && ret == 0 && skip_sl != nullptr && has_prompt && ctx_dft && seg_breaks.empty() &&
                    spec_seg_a == 0 && off + batch.view.size() == batch.size() && llama_mtp_draft_lend_active(ctx_tgt)) {
                const server_slot & sl = *skip_sl;
                const int32_t n_rest_s = sl.task ? sl.task->n_tokens() - sl.prompt.n_tokens() : 0;
                skip = sl.state == SLOT_STATE_PROCESSING_PROMPT && sl.task && !sl.task->tokens.has_mtmd && sl.spec_i_batch.empty() &&
                    !(sl.i_batch >= off && sl.i_batch < off + batch.view.size()) && n_rest_s >= tail_env;
                for (int i = 0; skip && i < batch.view.size(); ++i) {
                    skip = batch.view.tokens[i].seq_id == sl.id && batch.view.tokens[i].seq_ids_extra.empty() && !batch.view.tokens[i].output;
                }
                if (skip) {
                    ok = common_speculative_process_skip(spec.get(), batch.view);
                    static int64_t n_skip = 0;
                    n_skip++;
                    SRV_INF("mtp prompt pass skipped: %d rows, %d prompt tokens after them (%" PRId64 " batches; LLAMA_MTP_PROMPT_TAIL=%d, "
                            "LLAMA_MTP_DRAFT_LEND=0 off)\n", batch.view.size(), n_rest_s, n_skip, tail_env);
                }
            }
            bool restored = false;
            if (!dft_off && !skip && ok && ctx_dft && llama_mtp_draft_weights_evicted(ctx_dft)) {
                if (llama_mtp_draft_weights_restore(ctx_dft)) {
                    restored = true;
                    for (auto & sl : slots) {
                        sl.spec_lent_req |= sl.is_processing();
                    }
                } else {
                    dft_off = spec_fail_soft("the drafter's weights could not be mapped back", /*remap =*/ true);
                }
            }
            int32_t n_rest = 0;
            // k19-host (LLAMA_MTP_PROMPT_BG_MIN=N tokens, default min(n_ubatch, 1024)): only a prompt batch of at least N tokens runs its
            // drafter pass on the worker, a shorter one in line as before: G4 Flash-Next + Q8_0 MTP 16 GB, batch 17, chat TTFT +4 ms
            // (0.357 vs 0.352-0.353) with the worker on short prompts, where its pass is too short to hide anything
            static const int32_t bg_min = [&]() {
                const char * e = getenv("LLAMA_MTP_PROMPT_BG_MIN");
                const int32_t v = e ? std::max(0, atoi(e)) : std::min<int32_t>((int32_t) llama_n_ubatch(ctx_tgt), 1024);
                SRV_INF("mtp prompt pass on a worker for prompt batches of at least %d tokens (LLAMA_MTP_PROMPT_BG_MIN)\n", v);
                return v;
            }();
            // k19-studio-np4: the only slot, or the only processing one of several (spec_bg_slot, LLAMA_MTP_PROMPT_BG_MULTI=0 off)
            const server_slot * bg_sl = spec_bg_slot();
            bool bg = !dft_off && !skip && bg_env && ok && ret == 0 && bg_sl != nullptr && has_prompt && batch.view.size() > spec_bg_tail_max &&
                batch.view.size() >= bg_min && off + batch.view.size() == batch.size() && common_speculative_rows_ext_floats(spec.get()) > 0;
            if (bg) {
                const server_slot & sl = *bg_sl;
                n_rest = sl.task ? sl.task->n_tokens() - sl.prompt.n_tokens() : 0;
                bg = sl.state == SLOT_STATE_PROCESSING_PROMPT && sl.task && !sl.task->tokens.has_mtmd && sl.spec_i_batch.empty() &&
                    !(sl.i_batch >= off && sl.i_batch < off + batch.view.size()) && n_rest > 0 && (n_rest <= spec_bg_tail_max || bg_group_env);
                for (int i = 0; bg && i < batch.view.size(); ++i) {
                    bg = batch.view.tokens[i].seq_id == sl.id && batch.view.tokens[i].seq_ids_extra.empty() && !batch.view.tokens[i].output;
                }
            }
            const float * h_bg = nullptr;
            if (bg) {
                // the target's rows, resolved here (the worker calls nothing of the target); unmasked: row i at h + i * floats
                const int64_t nf = common_speculative_rows_ext_floats(spec.get());
                const int32_t n_rows = batch.view.size();
                h_bg = llama_get_embeddings_nextn(ctx_tgt);
                bg = h_bg != nullptr && llama_get_embeddings_nextn_ith(ctx_tgt, n_rows - 1) == h_bg + (size_t) (n_rows - 1) * nf;
                if (bg) {
                    // an earlier asynchronous draft save (a checkpoint's draft part) completes here, not at the worker's first decode,
                    // so the main thread's checkpoint evictions beside the pass never free a buffer still being written
                    if (ctx_dft) {
                        llama_state_seq_async_wait(ctx_dft);
                    }
                    bg = common_speculative_rows_ext_set(spec.get(), h_bg, n_rows, std::min<int32_t>(n_rows, 256));
                }
            }
            if (bg) {
                std::vector<common_batch> segs;
                std::vector<int32_t>      segs_a;
                if (seg_breaks.empty()) {
                    segs.emplace_back();
                    segs.back().n_pos = batch.view.n_pos;
                    segs.back().tokens = batch.view.tokens;
                    segs_a.push_back(0);
                } else {
                    int32_t a = spec_seg_a;
                    for (size_t k = spec_seg_done; k <= seg_breaks.size(); ++k) {
                        const int32_t b = k < seg_breaks.size() ? seg_breaks[k] : batch.view.size();
                        segs.emplace_back();
                        segs.back().n_pos = batch.view.n_pos;
                        segs.back().tokens.assign(batch.view.tokens.begin() + a, batch.view.tokens.begin() + b);
                        segs_a.push_back(a);
                        a = b;
                    }
                }
                const bool whole = seg_breaks.empty();
                spec_bg.closing  = false;
                spec_bg.q.clear();
                spec_bg.ok       = true;
                spec_bg.t_launch = ggml_time_us();
                common_speculative * sp = spec.get();
                spec_bg.thr = std::thread([this, sp, whole, segs = std::move(segs), segs_a = std::move(segs_a)]() {
                    bool ok_w = true;
                    for (size_t k = 0; k < segs.size() && ok_w; ++k) {
                        // as the in-line pass: one process() of the whole view without breaks, else one per segment at its row offset
                        ok_w = whole ? common_speculative_process(sp, segs[k]) : common_speculative_process_rows(sp, segs[k], segs_a[k]);
                    }
                    spec_bg.ok = ok_w;
                    while (true) {
                        std::function<void()> fn;
                        {
                            std::unique_lock<std::mutex> lock(spec_bg.m);
                            spec_bg.cv.wait(lock, [&]() { return spec_bg.closing || !spec_bg.q.empty(); });
                            if (spec_bg.q.empty()) {
                                return; // closing, nothing left
                            }
                            fn = std::move(spec_bg.q.front());
                            spec_bg.q.pop_front();
                        }
                        if (ok_w) {
                            fn(); // the draft-context steps the main thread queued, in its order (skipped after a failed pass)
                        }
                    }
                });
                spec_bg.live    = true;
                spec_bg.hold    = true;
                spec_bg.group   = n_rest > spec_bg_tail_max;
                spec_bg.slot_id = bg_sl->id;
                if (spec_bg.group) {
                    llama_set_nextn_hold(ctx_tgt, &server_context_impl::spec_bg_hold_cb, this);
                } else if (spec_bg_free_hold_on()) {
                    // k19-q4x262: the [<= spec_bg_tail_max] tail decode beside the pass writes no rows the pass reads (they are stashed),
                    // but it may still free the drafter's compute buffers (MoE cache banks coming back after a released prompt):
                    // the pass ends first there
                    llama_set_nextn_hold_free(ctx_tgt, &server_context_impl::spec_bg_hold_cb, this);
                    spec_bg.free_hold = true;
                }
                SRV_INF("mtp prompt pass on a worker: %d rows, %zu segments beside the next %d-token target decode%s (LLAMA_MTP_PROMPT_BG=0 off)\n",
                        batch.view.size(), seg_breaks.empty() ? (size_t) 1 : seg_breaks.size() + 1 - spec_seg_done, std::min<int32_t>(n_rest, n_batch),
                        spec_bg.group ? " (group mode, LLAMA_MTP_PROMPT_BG_GROUP=0 off)" : "");
                srv_ph("drafter launched");
            }
            if (pt_on) {
                srv_pass_timer().stamp(server_pass_timer::PROC0);
            }
            srv_yield_or_run(queue_tasks, batch.view.size() <= 32, [&]() {
                if (!ok || bg || skip || dft_off) {
                    return;
                }
                if (seg_breaks.empty()) {
                    ok = common_speculative_process(spec.get(), batch.view);
                } else {
                    // the segments one at a time, as the separate batches would have been (the draft keeps its own batches);
                    // k15-pp3: from the first one the overlap above did not process
                    int32_t a = spec_seg_a;
                    for (size_t k = spec_seg_done; k <= seg_breaks.size() && ok; ++k) {
                        const int32_t b = k < seg_breaks.size() ? seg_breaks[k] : batch.view.size();
                        common_batch seg;
                        seg.n_pos = batch.view.n_pos;
                        seg.tokens.assign(batch.view.tokens.begin() + a, batch.view.tokens.begin() + b);
                        ok = common_speculative_process_rows(spec.get(), seg, a);
                        a = b;
                    }
                }
            });

            srv_ph("drafter processed");
            if (pt_on) {
                srv_pass_timer().stamp(server_pass_timer::PROC1);
            }
            if (!ok && spec_fail_soft("the drafter's pass failed")) {
                ok = true;
            } else if (ok && restored && !bg && !skip && !dft_off) {
                llama_mtp_draft_lend_note(ctx_dft, true); // the remapped drafter computed this batch
            }
            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);
                    // children with their own prompt are started at the end of the shared prefix
                    GGML_ASSERT(slot.task->n_tokens_shared == 0);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off) {
        // for checking if a given batch index is inside the current sub-batch
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // k19-host: a slot that samples in this view uses the speculative state (begin, draft, accept): the background pass ends first
        if (spec_bg.live) {
            bool samples = false;
            for (const auto & slot : slots) {
                samples |= !slot.spec_i_batch.empty() || is_inside_view(slot.i_batch);
            }
            if (samples && !spec_bg_join()) {
                SRV_ERR("%s", "failed to process speculative batch\n");
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_DECISION) {
                    send_decision(slot, batch.view, slot.i_batch - off);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;
                // not while a MoE cache computes experts on the host: the page faults would compete with its decode
                if (!slot.task->is_child() && !slot.task->is_stateless() && llama_moe_host_mode(ctx_tgt) == 0) {
                    slot.prefault_next_save();
                }

                // k15-lfm2: a request under spill mode 3 that cannot draft (no repeat is long enough on a long prompt) gains nothing from
                // the pinned host path: its decode follows the CPU-miss rule (G4 LFM2.5 12 GB 16k, k15 re-bench: pinned 235.1 tok/s,
                // LLAMA_SPEC_SPILL=0 286.7). Its prompt ran pinned (all-device), so its first token is unchanged.
                // Opt-in, LLAMA_SPEC_SPILL_UNPIN_LONG=1 (mixed on G4, see server_spec_spill_unpin_long)
                if (spill_pinned && !spill_pin_lifted && server_spec_spill_unpin_long() == 1 && params_base.n_parallel == 1) {
                    const auto & ns = params_base.speculative.ngram_suffix;
                    if (ns.long_prompt > 0 && ns.n_min_match_long > 64 && slot.prompt.n_tokens() >= ns.long_prompt) {
                        spill_pin_lifted = !llama_moe_pin_small_set(ctx_tgt, false);
                    }
                }

                if (slot.can_speculate()) {
                    if (FILE * lf = common_spec_acc_log()) {
                        const auto & sp = slot.task->params.sampling;
                        fprintf(lf, "R %d temp=%.3f top_k=%d top_p=%.3f min_p=%.3f seed=%u n_max=%d\n", slot.task->id, sp.temp, sp.top_k, sp.top_p,
                                sp.min_p, sp.seed, common_speculative_n_max(spec.get()));
                        const auto toks = slot.prompt.tokens.get_text_tokens();
                        std::string ps = "P";
                        for (llama_token t : toks) {
                            ps += " " + std::to_string(t);
                        }
                        fprintf(lf, "%s\n", ps.c_str());
                    }
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                    slot.gov.new_sequence();
                    slot.gov_t_last = 0;
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            } else if (slot.gov_step == 1 && slot.gov_t_last > 0) {
                slot.gov.record(false, 1, t_now - slot.gov_t_last);
            }
            slot.gov_t_last = t_now;

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            {
                // the sampler is saved only for a checkpoint restore (below), which a rollback of at most n_draft tokens needs only
                // when the context cannot remove them itself; the clone copies the n_vocab candidates (LLAMA_SPEC_LAZY_CLONE=0: always)
                static const bool lazy_clone = [] {
                    const char * e = getenv("LLAMA_SPEC_LAZY_CLONE");
                    return e == nullptr || atoi(e) != 0;
                }();
                const bool may_restore = !lazy_clone ||
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > llama_n_rs_seq(ctx_tgt));
                common_sampler_ptr smpl_save(may_restore ? common_sampler_clone(slot.smpl.get()) : nullptr);

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // drafters that fill no distribution fall back here
                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                srv_pass_timer().stamp(server_pass_timer::ACC0);
                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (slot.spec_is_replay && (slot.use_spec_rejection() || !spec_replay_reverify())) {
                    // k18: greedy replays too re-accept the tokens the first verify accepted. Verifying them again assumes the
                    // replay batch computes the same argmax as the verify batch; where batch shape and routing change the rows'
                    // numerics (MoE cache host / device split, CPU misses) a near tie flips between the two batches, each replay
                    // rolls back again and the request never ends (G4 Qwen3.6-35B-A3B UD-Q4_K_XL 7 GB + Qwen3.5-0.8B draft-simple 4,
                    // FFN requantized: 12 tokens of the 2k cell, then restore / replay forever, 4 of 4 launches).
                    // LLAMA_SPEC_REPLAY_REVERIFY=1: verify greedy replays again (the old behaviour)
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                }
                srv_pass_timer().stamp(server_pass_timer::ACC1);
                if (FILE * lf = common_spec_acc_log(); lf != nullptr && synth_probs.empty()) {
                    // k19-q4xaccept diagnostic: drafts, then the tokens kept (the last one is the target's own at the first mismatch)
                    std::string vs = string_format("V %d %d %d", slot.spec_is_replay ? 1 : 0, (int) slot.spec_draft.size(), use_rejection ? 1 : 0);
                    for (llama_token t : slot.spec_draft) {
                        vs += " " + std::to_string(t);
                    }
                    vs += " |";
                    for (llama_token t : accepted) {
                        vs += " " + std::to_string(t);
                    }
                    fprintf(lf, "%s\n", vs.c_str());
                }
                // k13: the probabilities of the accepted tokens come from their verify rows, read now (a rollback below may decode
                // again); they were left empty, so drafted positions looked like a uniform distribution to the KLD gate
                slot.spec_probs.clear();
                if (slot.task->params.sampling.n_probs > 0 && !slot.spec_is_replay && synth_probs.empty()) {
                    for (size_t i = 0; i < accepted.size() && i < slot.spec_i_batch.size(); ++i) {
                        completion_token_output r;
                        r.tok  = accepted[i];
                        r.prob = 1.0f;
                        populate_token_probs(slot, r, slot.task->params.post_sampling_probs, params_base.special, slot.spec_i_batch[i]);
                        slot.spec_probs.push_back(std::move(r));
                    }
                }
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        if (slot.spec_rs_snap) {
                            GGML_ASSERT(llama_rs_snap_restore(slot.ctx_tgt, slot.id) && "host copy of the recurrent state not restored");
                        } else {
                            GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        if (slot.ctx_dft) {
                            GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        GGML_ASSERT(smpl_save && "speculative checkpoint restore without a saved sampler");
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            {
                const int64_t t_now = ggml_time_us();
                if (slot.gov_t_last > 0) {
                    slot.gov.record(true, (int) ids.size(), t_now - slot.gov_t_last);
                }
                slot.gov_t_last = t_now;
            }

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // k13: see spec_probs (filled at the verify, before the rollback)
                if (i < slot.spec_probs.size() && slot.spec_probs[i].tok == result.tok) {
                    result.prob  = slot.spec_probs[i].prob;
                    result.probs = std::move(slot.spec_probs[i].probs);
                }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();
            srv_pass_timer().stamp(server_pass_timer::TOK1);

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info              */ std::string(llama_build_info()),
        /* model_name              */ impl->model_name,
        /* model_aliases           */ impl->model_aliases,
        /* model_tags              */ impl->model_tags,
        /* model_path              */ impl->params_base.model.path,
        /* model_output_modalities */ server_model_output_modalities(common_get_decision_type(impl->model_tgt)),
        /* has_mtmd                */ impl->mctx != nullptr,
        /* has_inp_image           */ impl->chat_params.allow_image,
        /* has_inp_audio           */ impl->chat_params.allow_audio,
        /* has_inp_video           */ impl->chat_params.allow_video,
        /* json_ui_settings        */ impl->json_ui_settings,
        /* slot_n_ctx              */ impl->n_ctx_slot(),
        /* pooling_type            */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params             */ impl->chat_params,
        /* chat_template_caps      */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str           */ bos_token_str,
        /* eos_token_str           */ eos_token_str,
        /* fim_pre_token           */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token           */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token           */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token           */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token           */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token           */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog          */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type        */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens    */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train       */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp        */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params          */ llama_model_n_params(impl->model_tgt),
        /* model_size              */ llama_model_size(impl->model_tgt),
        /* model_ftype             */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.apply_chat_session(chat_session);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks), chat_session);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", server_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR,
            session);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.decision;
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);

        std::vector<raw_buffer> files;
        const json state = decision.parse_state(body, files);
        if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image)) {
            res->error(format_error_response("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // one task per variant of each question, or one task for all the questions
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, files, ctx_server.mctx, ctx_server.init_opt, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }
            if (decision.can_share_prompt()) {
                tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
        }

        auto all_results = rd.wait_for_all(req.should_stop);

        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json answers = json::object();
        int32_t n_tokens = 0;
        size_t i_result = 0;
        size_t i_score  = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                // one result with the scores of all the questions, in order
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[0].get());
                GGML_ASSERT(result != nullptr && i_score + question.options.size() <= result->scores.size());
                scores.emplace_back(result->scores.begin() + i_score, result->scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
                n_tokens = result->n_tokens;
            }
            for (size_t variant = 0; !decision.is_joint() && variant < decision.n_variants(question); variant++) {
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[i_result++].get());
                GGML_ASSERT(result != nullptr);
                scores.push_back(result->scores);
                n_tokens += result->n_tokens;
            }
            answers[question.id] = decision.format_answer(question, scores, !files.empty());
        }

        res->ok(json{
            {"model",   meta->model_name},
            {"answers", answers},
            {"usage",   {
                {"input_tokens",  n_tokens},
                {"output_tokens", 0},
            }},
        });
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.vocab, ctx_server.mctx, meta->chat_params, p.at("content"), true, true, ctx_server.init_opt);
        }
        return tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt);
    };

    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
