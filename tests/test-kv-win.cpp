// the recent-window ring beside a quantized KV cache (llama_context_params::kv_win), on the CPU, on generated models of
// several families. Every check compares logits of whole runs:
//   probe     window over every cell (q4_0 cache, q8_0 ring)        == plain q8_0 cache, bit for bit (prompt in several ubatches + decode)
//   control   window 0, sinks 0 (empty ring, the mode engaged)       == plain q4_0 cache, bit for bit
//   route     ring of the cache's own type (q4_0 beside q4_0), W 32 ~= plain q4_0 (same values, other summation order): a cell read
//             twice or dropped would move the logits far more than the tolerance
//   rollback  verify windows whose drafts are rejected (seq_rm): two runs with different drafts give equal bits afterwards (the drafts
//             leave no trace in the ring), also past the ring's size (slots reclaimed)
//   state     seq state saved and restored into a fresh context: the next decodes equal the uninterrupted run (bits while the ring
//             never wrapped, else within the tolerance: the restored slots are in another order)
//   shift     probe with a context shift (seq_rm + seq_add) == plain q8_0 doing the same, bit for bit
//   multiseq  two sequences in a unified cache, W 32: each equals its own run alone (within the tolerance)
//   stopgap   LLAMA_KV_WIN_FUSED=0 (the concat graph) ~= the two-source node (cosine)
// usage: test-kv-win --models DIR [--only NAME] [--ngl N]

#include "llama.h"

#include <algorithm>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct cfg {
    ggml_type type    = GGML_TYPE_F16;
    int32_t   win     = -1;
    int32_t   sinks   = 0;
    ggml_type wtype   = GGML_TYPE_Q8_0;
    uint32_t  n_ctx   = 512;
    uint32_t  n_ubatch = 64;
    uint32_t  n_seq   = 1;
    bool      unified = true;
};

static llama_context * make_ctx(llama_model * model, const cfg & c) {
    auto cp = llama_context_default_params();
    cp.n_ctx           = c.n_ctx;
    cp.n_batch         = 256;
    cp.n_ubatch        = c.n_ubatch;
    cp.n_seq_max       = c.n_seq;
    cp.n_rs_seq        = 4; // recurrent states of hybrid models roll back with the drafts
    cp.kv_unified      = c.unified;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = std::max(1, std::min(4, (int) std::thread::hardware_concurrency())); // no more threads than cores
    cp.n_threads_batch = cp.n_threads;
    cp.type_k          = c.type;
    cp.type_v          = c.type;
    cp.kv_win          = c.win;
    cp.kv_win_sinks    = c.sinks;
    cp.type_k_win      = c.wtype;
    cp.type_v_win      = c.wtype;
    return llama_init_from_model(model, cp);
}

static int n_vocab_of(llama_model * model) {
    return llama_vocab_n_tokens(llama_model_get_vocab(model));
}

// decode toks at p0.. for seq; appends the logits of every row with logits
static bool decode1(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, llama_seq_id seq, bool all,
        std::vector<float> * out, int n_vocab);

// in batches of at most 256 tokens (the contexts' n_batch)
static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, llama_seq_id seq, bool all,
        std::vector<float> * out, int n_vocab) {
    for (size_t i = 0; i < toks.size(); i += 256) {
        const size_t n = std::min<size_t>(256, toks.size() - i);
        const bool last = i + n == toks.size();
        std::vector<llama_token> part(toks.begin() + i, toks.begin() + i + n);
        if (!decode1(ctx, part, p0 + (llama_pos) i, seq, all, last || all ? out : nullptr, n_vocab)) {
            return false;
        }
    }
    return true;
}

static bool decode1(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, llama_seq_id seq, bool all,
        std::vector<float> * out, int n_vocab) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = p0 + (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq;
        b.logits[i]    = all || i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    if (rc == 0 && out) {
        for (size_t i = 0; i < toks.size(); ++i) {
            if (b.logits[i]) {
                const float * l = llama_get_logits_ith(ctx, (int32_t) i);
                out->insert(out->end(), l, l + n_vocab);
            }
        }
    }
    llama_batch_free(b);
    return rc == 0;
}

static std::vector<llama_token> rand_toks(std::mt19937 & gen, int n, int n_vocab) {
    std::vector<llama_token> t(n);
    for (auto & x : t) {
        x = (llama_token) (gen() % (uint32_t) std::max(2, n_vocab - 1)) + 1;
    }
    return t;
}

// prompt (several ubatches) then n_dec single-token decodes
static std::vector<float> run_plain(llama_model * model, const cfg & c, const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & dec) {
    const int nv = n_vocab_of(model);
    llama_context * ctx = make_ctx(model, c);
    std::vector<float> out;
    if (ctx == nullptr) {
        return out;
    }
    bool ok = decode(ctx, prompt, 0, 0, false, &out, nv);
    for (size_t i = 0; ok && i < dec.size(); ++i) {
        ok = decode(ctx, { dec[i] }, (llama_pos) (prompt.size() + i), 0, false, &out, nv);
    }
    llama_free(ctx);
    if (!ok) {
        out.clear();
    }
    return out;
}

static bool bits_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static double max_rel(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) {
        return INFINITY;
    }
    double mx = 0.0, ref = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        mx  = std::max(mx, (double) fabs(a[i] - b[i]));
        ref = std::max(ref, (double) fabs(b[i]));
    }
    return mx/std::max(ref, 1e-6);
}

static double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) {
        return 0.0;
    }
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += (double) a[i]*b[i]; aa += (double) a[i]*a[i]; bb += (double) b[i]*b[i];
    }
    return ab/sqrt(aa*bb + 1e-30);
}

// --ngl N: layers on the GPU. There the bit-for-bit checks against a plain cache become tolerance checks (the stopgap or the
// two-source kernel and the plain kernels differ in staging and split); rollback stays bit for bit (same shapes, same kernels), state
// within the plain cache's own save / restore difference
static int g_ngl = 0;

static int n_fail = 0;
static int n_pass = 0;

static void report(const char * model, const char * check, bool ok, const char * detail) {
    printf("%-26s %-9s %s %s\n", model, check, ok ? "PASS" : "FAIL", detail);
    fflush(stdout);
    (ok ? n_pass : n_fail)++;
}

// rollback: verify windows [tok, d1, d2] with drafts d rejected (seq_rm) after each, then plain decodes; returns all logits
static std::vector<float> run_rollback(llama_model * model, const cfg & c, const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & dec, uint32_t draft_seed) {
    const int nv = n_vocab_of(model);
    std::mt19937 gd(draft_seed);
    llama_context * ctx = make_ctx(model, c);
    std::vector<float> out;
    if (ctx == nullptr) {
        return out;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    bool ok = decode(ctx, prompt, 0, 0, false, &out, nv);
    llama_pos p = (llama_pos) prompt.size();
    for (size_t i = 0; ok && i < dec.size(); ++i) {
        std::vector<float> rows;
        auto d = rand_toks(gd, 2, nv);
        ok = decode(ctx, { dec[i], d[0], d[1] }, p, 0, true, &rows, nv);
        // the accepted row only (the draft rows see the drafts)
        out.insert(out.end(), rows.begin(), rows.begin() + nv);
        ok = ok && llama_memory_seq_rm(mem, 0, p + 1, -1);
        p += 1;
    }
    for (size_t i = 0; ok && i < 8; ++i) {
        ok = decode(ctx, { dec[i % dec.size()] }, p, 0, false, &out, nv);
        p += 1;
    }
    llama_free(ctx);
    if (!ok) {
        out.clear();
    }
    return out;
}

static std::vector<float> run_state(llama_model * model, const cfg & c, const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & dec, bool restore) {
    const int nv = n_vocab_of(model);
    llama_context * ctx = make_ctx(model, c);
    std::vector<float> out;
    if (ctx == nullptr) {
        return out;
    }
    bool ok = decode(ctx, prompt, 0, 0, false, nullptr, nv);
    if (ok && restore) {
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
        const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
        llama_free(ctx);
        ctx = make_ctx(model, c);
        ok = n > 0 && llama_state_seq_set_data(ctx, buf.data(), n, 0) == n;
    }
    for (size_t i = 0; ok && i < dec.size(); ++i) {
        ok = decode(ctx, { dec[i] }, (llama_pos) (prompt.size() + i), 0, false, &out, nv);
    }
    llama_free(ctx);
    if (!ok) {
        out.clear();
    }
    return out;
}

static std::vector<float> run_shift(llama_model * model, const cfg & c, const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & dec) {
    const int nv = n_vocab_of(model);
    llama_context * ctx = make_ctx(model, c);
    std::vector<float> out;
    if (ctx == nullptr) {
        return out;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (!llama_memory_can_shift(mem)) {
        llama_free(ctx);
        return out;
    }
    bool ok = decode(ctx, prompt, 0, 0, false, nullptr, nv);
    // keep the first 4, drop the next 32, shift the rest down
    const llama_pos n_keep = 4, n_drop = 32, n = (llama_pos) prompt.size();
    ok = ok && llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_drop);
    llama_memory_seq_add(mem, 0, n_keep + n_drop, n, -n_drop);
    for (size_t i = 0; ok && i < dec.size(); ++i) {
        ok = decode(ctx, { dec[i] }, n - n_drop + (llama_pos) i, 0, false, &out, nv);
    }
    llama_free(ctx);
    if (!ok) {
        out.clear();
    }
    return out;
}

// two sequences interleaved in one unified cache; returns the logits of seq s
static std::vector<float> run_multi(llama_model * model, const cfg & c, const std::vector<llama_token> & p0,
        const std::vector<llama_token> & p1, const std::vector<llama_token> & dec, int s_out) {
    const int nv = n_vocab_of(model);
    cfg c2 = c;
    c2.n_seq = 2;
    llama_context * ctx = make_ctx(model, c2);
    std::vector<float> o0, o1;
    if (ctx == nullptr) {
        return o0;
    }
    bool ok = decode(ctx, p0, 0, 0, false, &o0, nv) && decode(ctx, p1, 0, 1, false, &o1, nv);
    for (size_t i = 0; ok && i < dec.size(); ++i) {
        ok = decode(ctx, { dec[i] }, (llama_pos) (p0.size() + i), 0, false, &o0, nv) &&
             decode(ctx, { dec[(i + 3) % dec.size()] }, (llama_pos) (p1.size() + i), 1, false, &o1, nv);
    }
    llama_free(ctx);
    if (!ok) {
        return {};
    }
    return s_out == 0 ? o0 : o1;
}

// seq_cp of a whole sequence into another stream (the stream copy of update(): cache rows, ring rows, slot map), then decodes there
static std::vector<float> run_cp_stream(llama_model * model, const cfg & c, const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & dec) {
    const int nv = n_vocab_of(model);
    cfg c2 = c;
    c2.n_seq = 2;
    c2.unified = false;
    llama_context * ctx = make_ctx(model, c2);
    std::vector<float> out;
    if (ctx == nullptr) {
        return out;
    }
    bool ok = decode(ctx, prompt, 0, 0, false, &out, nv);
    llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);
    for (size_t i = 0; ok && i < dec.size(); ++i) {
        ok = decode(ctx, { dec[i] }, (llama_pos) (prompt.size() + i), 1, false, &out, nv);
    }
    llama_free(ctx);
    if (!ok) {
        out.clear();
    }
    return out;
}

static void test_model(const std::string & path, const std::string & name) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = g_ngl;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        report(name.c_str(), "load", false, path.c_str());
        return;
    }
    const int nv = n_vocab_of(model);
    std::mt19937 gen(7);
    const auto prompt = rand_toks(gen, 200, nv);
    const auto dec    = rand_toks(gen, 12, nv);
    char buf[256];

    cfg q8;  q8.type = GGML_TYPE_Q8_0;
    cfg q4;  q4.type = GGML_TYPE_Q4_0;
    cfg probe = q4; probe.win = 1 << 20; probe.sinks = 0;
    cfg ctrl  = q4; ctrl.win = 0; ctrl.sinks = 0;
    cfg route = q4; route.win = 32; route.sinks = 4; route.wtype = GGML_TYPE_Q4_0;
    cfg w32   = q4; w32.win = 32; w32.sinks = 4;

    const auto r_q8 = run_plain(model, q8, prompt, dec);
    const auto r_q4 = run_plain(model, q4, prompt, dec);
    if (r_q8.empty() || r_q4.empty()) {
        report(name.c_str(), "baseline", false, "plain q8_0 / q4_0 run failed");
        llama_model_free(model);
        return;
    }
    {
        // the plain runs are deterministic (else no bit-for-bit check below means anything)
        const auto r = run_plain(model, q4, prompt, dec);
        snprintf(buf, sizeof(buf), "plain q4_0 twice: max rel %.3g", max_rel(r, r_q4));
        report(name.c_str(), "determ", bits_equal(r, r_q4), buf);
    }
    // DeepSeek-V4.x: no ring; the window mode puts the raw window and indexer caches in the window type, the compressed rows stay 4-bit
    if (name.rfind("deepseek4", 0) == 0) {
        cfg wm = q4; wm.win = 1024; wm.sinks = 4;
        const auto r = run_plain(model, wm, prompt, dec);
        const double e = max_rel(r, r_q8), e4 = max_rel(r_q4, r_q8);
        snprintf(buf, sizeof(buf), "window mode vs q8_0 %.3g, plain q4_0 vs q8_0 %.3g", e, e4);
        report(name.c_str(), "dsv4", e < e4, buf);
        llama_model_free(model);
        return;
    }
    {
        const auto r = run_plain(model, probe, prompt, dec);
        const double gap = max_rel(r_q4, r_q8);
        snprintf(buf, sizeof(buf), "max rel %.3g vs q8_0 (q4_0 vs q8_0 %.3g)", max_rel(r, r_q8), gap);
        report(name.c_str(), "probe", g_ngl > 0 ? max_rel(r, r_q8) < std::max(2e-3, 0.3*gap) : bits_equal(r, r_q8), buf);
    }
    {
        const auto r = run_plain(model, ctrl, prompt, dec);
        if (getenv("TEST_KV_WIN_SEG") && r.size() == r_q4.size()) {
            for (size_t o = 0; o < r.size(); o += nv) {
                std::vector<float> x(r.begin() + o, r.begin() + o + nv), y(r_q4.begin() + o, r_q4.begin() + o + nv);
                printf("  control seg %zu: max rel %.3g\n", o/nv, max_rel(x, y));
            }
        }
        snprintf(buf, sizeof(buf), "max rel %.3g vs q4_0", max_rel(r, r_q4));
        report(name.c_str(), "control", g_ngl > 0 ? max_rel(r, r_q4) < std::max(2e-3, 0.3*max_rel(r_q4, r_q8)) : bits_equal(r, r_q4), buf);
    }
    {
        const auto r = run_plain(model, route, prompt, dec);
        const double e = max_rel(r, r_q4);
        snprintf(buf, sizeof(buf), "max rel %.3g vs q4_0 (q4_0 ring), q8_0 vs q4_0 %.3g", e, max_rel(r_q8, r_q4));
        report(name.c_str(), "route", e < 2e-3, buf);
    }
    // a prompt of many ubatches that wraps the ring (W 64, S 4: a ring of 256 slots for ub 32 / 128 against 1000 + 16 positions),
    // ring type = cache type: every row must read each visible cell exactly once (ring or cache), so the result is plain q4_0 up to
    // the summation order (a cell missing or counted twice is far larger than that)
    {
        std::mt19937 g3(11);
        const auto pl = rand_toks(g3, 1000, nv);
        const auto dl = rand_toks(g3, 16, nv);
        for (uint32_t ub : { 32u, 128u }) {
            cfg pq = q4; pq.n_ctx = 2048; pq.n_ubatch = ub;
            cfg pr = pq; pr.win = 64; pr.sinks = 4; pr.wtype = GGML_TYPE_Q4_0;
            cfg p8 = pq; p8.type = GGML_TYPE_Q8_0;
            const auto a  = run_plain(model, pq, pl, dl);
            const auto b  = run_plain(model, pr, pl, dl);
            const auto c8 = run_plain(model, p8, pl, dl);
            const double e = max_rel(b, a), gap = max_rel(c8, a);
            snprintf(buf, sizeof(buf), "1000 + 16 positions, ub %u, W 64 S 4, q4_0 ring: max rel %.3g vs q4_0 (q8_0 vs q4_0 %.3g)", ub, e, gap);
            report(name.c_str(), ub == 32 ? "wrap32" : "wrap128", e < std::max(2e-5, 0.02*gap), buf);
            // the first token after the wrapped prompt (its last row) with the q8_0 ring against f16: no farther than plain q4_0
            // (a ring that lost or doubled a cell would land far outside the q8_0 .. q4_0 range)
            cfg pw = pr; pw.wtype = GGML_TYPE_Q8_0;
            cfg pf = pq; pf.type = GGML_TYPE_F16;
            const auto w  = run_plain(model, pw, pl, dl);
            const auto f  = run_plain(model, pf, pl, dl);
            auto first = [&](const std::vector<float> & v) {
                return v.size() >= (size_t) nv ? std::vector<float>(v.begin(), v.begin() + nv) : std::vector<float>();
            };
            const double ew = max_rel(first(w), first(f)), e8 = max_rel(first(c8), first(f)), e4 = max_rel(first(a), first(f));
            snprintf(buf, sizeof(buf), "first token vs f16, ub %u: window %.3g, q8_0 %.3g, q4_0 %.3g", ub, ew, e8, e4);
            // a quarter of the q8_0 .. q4_0 gap as slack: a mixed q8_0 ring / q4_0 cache need not land strictly inside the two plain
            // errors for one token (arm64 CPU: window 0.00285 vs q4_0 0.00281, q8_0 0.00254), a lost or doubled cell lands far outside
            report(name.c_str(), ub == 32 ? "wfirst32" : "wfirst128", ew <= std::max(e4, e8) + 0.25*std::fabs(e4 - e8) + 1e-7, buf);
        }
    }
    {
        const auto a = run_rollback(model, w32, prompt, dec, 11);
        const auto b = run_rollback(model, w32, prompt, dec, 99);
        snprintf(buf, sizeof(buf), "W 32: max rel %.3g between draft sets", max_rel(a, b));
        report(name.c_str(), "rollback", bits_equal(a, b), buf);
        // past the ring (256 slots: 200 + 12*3 ... + reclaims): a longer run
        cfg small = w32; small.n_ctx = 1024;
        std::mt19937 g2(5);
        const auto prompt2 = rand_toks(g2, 600, nv);
        const auto a2 = run_rollback(model, small, prompt2, dec, 11);
        const auto b2 = run_rollback(model, small, prompt2, dec, 99);
        snprintf(buf, sizeof(buf), "600-token prompt (ring reclaimed): max rel %.3g", max_rel(a2, b2));
        report(name.c_str(), "rollback2", bits_equal(a2, b2), buf);
    }
    {
        cfg big = w32; big.win = 1 << 20; // ring = cache: never reclaimed
        const auto a = run_state(model, big, prompt, dec, false);
        const auto b = run_state(model, big, prompt, dec, true);
        // GPU: a restored cache can be read with another split / tile count than the live one (iSWA restores compact the SWA cells),
        // so there the bound is the plain q4_0 cache's own save / restore difference (the ring adds nothing to it)
        double e_plain = 0.0;
        if (g_ngl > 0 && !bits_equal(a, b)) {
            e_plain = max_rel(run_state(model, q4, prompt, dec, true), run_state(model, q4, prompt, dec, false));
        }
        snprintf(buf, sizeof(buf), "full-size ring: max rel %.3g (plain q4_0 save / restore %.3g)", max_rel(b, a), e_plain);
        report(name.c_str(), "state", bits_equal(a, b) || (g_ngl > 0 && max_rel(b, a) <= std::max(4.0*e_plain, 1e-5)), buf);
        const auto a2 = run_state(model, w32, prompt, dec, false);
        const auto b2 = run_state(model, w32, prompt, dec, true);
        const double e = max_rel(b2, a2);
        snprintf(buf, sizeof(buf), "W 32 (slots reordered): max rel %.3g (q8_0 vs q4_0 %.3g)", e, max_rel(r_q8, r_q4));
        report(name.c_str(), "state2", e < 2e-3, buf);
    }
    const enum llama_rope_type rt = llama_model_rope_type(model);
    const bool shiftable = rt != LLAMA_ROPE_TYPE_MROPE && rt != LLAMA_ROPE_TYPE_IMROPE && rt != LLAMA_ROPE_TYPE_VISION && rt != LLAMA_ROPE_TYPE_NONE;
    // (a quantized MLA latent cache has no K-shift on the CPU: the RoPE part of its rows is not contiguous, plain q8_0 aborts too)
    const bool mla = name.rfind("deepseek", 0) == 0;
    if (!shiftable || mla) {
        report(name.c_str(), "shift", true, mla ? "quantized MLA cache: no CPU K-shift (skipped)" : "multi-position RoPE / no RoPE: no K-shift (skipped)");
    } else {
        const auto a = run_shift(model, probe, prompt, dec);
        const auto b = run_shift(model, q8, prompt, dec);
        snprintf(buf, sizeof(buf), "max rel %.3g vs q8_0", max_rel(a, b));
        const bool can_shift = !a.empty() || !b.empty();
        report(name.c_str(), "shift", !can_shift || (g_ngl > 0 ? max_rel(a, b) < 2e-3 : bits_equal(a, b)), can_shift ? buf : "no K-shift for this model (skipped)");
    }
    {
        std::mt19937 g3(9);
        const auto p1 = rand_toks(g3, 150, nv);
        const auto m0 = run_multi(model, w32, prompt, p1, dec, 0);
        const auto m1 = run_multi(model, w32, prompt, p1, dec, 1);
        std::vector<llama_token> dec1(dec.size());
        for (size_t i = 0; i < dec.size(); ++i) {
            dec1[i] = dec[(i + 3) % dec.size()];
        }
        const auto s0 = run_plain(model, w32, prompt, dec);
        const auto s1 = run_plain(model, w32, p1, dec1);
        const double e = std::max(max_rel(m0, s0), max_rel(m1, s1));
        snprintf(buf, sizeof(buf), "max rel %.3g vs alone (q8_0 vs q4_0 %.3g)", e, max_rel(r_q8, r_q4));
        report(name.c_str(), "multiseq", e < 2e-3, buf);
        // one stream per sequence (rings per stream)
        cfg ws = w32; ws.unified = false;
        const auto n0 = run_multi(model, ws, prompt, p1, dec, 0);
        const auto n1 = run_multi(model, ws, prompt, p1, dec, 1);
        const double e2 = std::max(max_rel(n0, s0), max_rel(n1, s1));
        snprintf(buf, sizeof(buf), "two streams: max rel %.3g vs alone", e2);
        report(name.c_str(), "streams", e2 < 2e-3, buf);
        const auto cp = run_cp_stream(model, w32, prompt, dec);
        const double e3 = max_rel(cp, s0);
        snprintf(buf, sizeof(buf), "seq_cp into another stream: max rel %.3g vs alone", e3);
        report(name.c_str(), "seqcp", e3 < 2e-3, buf);
    }
    {
        // (keeps a caller's LLAMA_KV_WIN_FUSED for the other checks: LLAMA_KV_WIN_FUSED=0 runs the whole test on the stopgap)
        const char * prev = getenv("LLAMA_KV_WIN_FUSED");
        const std::string prev_s = prev ? prev : "";
        const auto a = run_plain(model, w32, prompt, dec);
        setenv("LLAMA_KV_WIN_FUSED", "0", 1);
        const auto b = run_plain(model, w32, prompt, dec);
        if (prev) {
            setenv("LLAMA_KV_WIN_FUSED", prev_s.c_str(), 1);
        } else {
            unsetenv("LLAMA_KV_WIN_FUSED");
        }
        const double cs = cosine(a, b);
        snprintf(buf, sizeof(buf), "cosine %.6f (max rel %.3g)", cs, max_rel(b, a));
        report(name.c_str(), "stopgap", cs > 0.9999, buf);
    }
    llama_model_free(model);
}

int main(int argc, char ** argv) {
    std::string dir;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            dir = argv[++i];
        } else if (strcmp(argv[i], "--ngl") == 0 && i + 1 < argc) {
            g_ngl = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            only = argv[++i];
        }
    }
    if (dir.empty()) {
        fprintf(stderr, "usage: %s --models DIR [--only NAME]\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    if (getenv("TEST_KV_WIN_VERBOSE") == nullptr) {
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level == GGML_LOG_LEVEL_ERROR) {
                fputs(text, stderr);
            }
        }, nullptr);
    }
    // dense GQA / MHA, hybrid (attention + recurrent), MoE hybrid, SWA (iSWA), MLA (K-only latent cache)
    const char * names[] = {
        "llama-dense", "qwen3-dense", "qwen35-dense", "qwen35moe-moe", "qwen4exp-moe", "gemma3-dense", "gemma2-dense",
        "deepseek2-moe", "phi3-dense", "deepseek41-moe",
    };
    for (const char * n : names) {
        if (!only.empty() && only != n) {
            continue;
        }
        test_model(dir + "/" + n + ".gguf", n);
    }
    printf("%s: %d pass, %d fail\n", n_fail == 0 ? "OK" : "FAIL", n_pass, n_fail);
    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}
