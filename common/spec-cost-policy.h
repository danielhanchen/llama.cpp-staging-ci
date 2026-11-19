#pragma once

// k11-decode: cost-priced window choice for n-gram (suffix) drafts drafted alone.
//
// The verify window of t tokens (the sampled token + t - 1 drafted ones) is priced by the measured wall time of the rounds of that
// size (an EMA per size, kept across requests: it is a property of the model, the GPU and the placement, not of the text). A draft
// of k tokens from a match class whose decayed per-token acceptance is q is expected to commit 1 + q + ... + q^k tokens; it is
// taken when the best expected tokens per ms beats a plain round's 1 / cost(1) by a margin
// (measured round costs, a guessed cost shape for sizes not seen yet, a few probes
// of a size before its guessed cost can veto it, a re-probe of a size not seen for a long time), with two changes for llama.cpp:
//   - only two window sizes besides 1 (a short window for short repeats, the full window for long ones): every size is a target
//     graph of its own, and a size change rebuilds it;
//   - the round right after a drafted round, if plain, carries the cost of going back to the plain graph: its time above a plain
//     round's is charged to the drafted size, so a lone drafted step between plain ones is priced with both graph changes.
// It is its own auto-off: where verify windows cost more than they commit (spilled experts read once per window row, a code answer
// whose repeats do not continue), the measured costs and acceptance stop the drafts, and a class drifts back to its prior while
// it is not drafted, so it is tried again later.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>

struct common_spec_cost_policy {
    static constexpr int MAX_T = 8;

    // relative round cost by window size before a size is measured: an RTX 5070 curve (kShape), pessimistic for a
    // resident model, so a size is priced high until measured (the probes measure it)
    static constexpr double SHAPE[MAX_T + 1] = { 0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6 };
    static constexpr double ALPHA  = 0.1;     // EMA weight of a new round time
    static constexpr int    PROBES = 3;       // a size is drafted this often (q >= P_PROBE) before its guessed cost can veto it
    static constexpr double P_PROBE = 0.85;
    static constexpr double STALE   = 200.0;  // rounds after which a measured size is re-probed when a likely draft could use it
    static constexpr double P_STALE = 0.7;    // (a size priced high in one slow stretch, a cold start or a busy host, is measured again)
    static constexpr int    SKIP    = 2;      // the first rounds of a size build its graph and capture it: not its cost

    double margin = 0.03;
    bool   gain_gate = false;  // LLAMA_SPEC_POLICY_GAIN=1: on (G4 LFM2.5 32 GB: it stopped the short window of a repeat, rep -5%; no cell gained)
    int    short_k   = 3;      // the drafts of the short window (the gain gate's window)

    std::array<double, MAX_T + 1> cost{}, cost_n{}, last{};
    std::array<int, MAX_T + 1>    seen{};
    // realized gain of drafted rounds by window size: tokens committed per ms of the round (with its switch back) over a plain round's
    // tokens per ms, an EMA; a size whose drafts lose (gain < 1 over GAIN_N rounds) is not drafted until it is stale and re-probed
    static constexpr double GAIN_ALPHA = 0.25;
    static constexpr int    GAIN_N     = 3;
    std::array<double, MAX_T + 1> gain{}, gain_n{};
    double rounds = 0.0;

    common_spec_cost_policy() {
        if (const char * e = getenv("LLAMA_SPEC_POLICY_MARGIN")) {
            margin = atof(e);
        }
        if (const char * e = getenv("LLAMA_SPEC_POLICY_GAIN")) {
            gain_gate = atoi(e) != 0;
        }
    }

    double cost_ms(int t) const {
        t = std::clamp(t, 1, MAX_T);
        if (cost_n[t] > 0) {
            return cost[t];
        }
        double num = 0.0, den = 0.0;
        for (int u = 1; u <= MAX_T; ++u) {
            if (cost_n[u] > 0) {
                const double w = std::min(cost_n[u], 20.0);
                num += w*cost[u]*SHAPE[t]/SHAPE[u];
                den += w;
            }
        }
        return den > 0 ? num/den : SHAPE[t];
    }

    bool stale(int t) const {
        return cost_n[t] > 0 && rounds - last[t] > STALE;
    }

    void observe(int t, double ms) {
        t = std::clamp(t, 1, MAX_T);
        if (!(ms > 0.0)) {
            return;
        }
        if (seen[t]++ < SKIP) {
            last[t] = rounds;
            return;
        }
        // a round far above the size's cost is not a round of decode (a request boundary, a stall of the host): left out; one well
        // above it counts at three times the cost, so a few slow rounds cannot price a size out
        if (cost_n[t] >= 4 && ms > 20.0*cost[t]) {
            return;
        }
        if (cost_n[t] >= 2) {
            ms = std::min(ms, 3.0*cost[t]);
        }
        cost[t]   = cost_n[t] > 0 ? (1.0 - ALPHA)*cost[t] + ALPHA*ms : ms;
        cost_n[t] += 1.0;
        last[t]    = rounds;
    }

    // a drafted round of window t committed `tokens` in ms (its switch back included)
    void observe_gain(int t, int tokens, double ms) {
        t = std::clamp(t, 1, MAX_T);
        // not while the size still builds and captures its graphs (its warm-up rounds are not its cost either)
        if (!(ms > 0.0) || cost_n[1] <= 0 || cost_n[t] <= 0) {
            return;
        }
        const double g = (tokens/ms)/(1.0/cost_ms(1));
        gain[t]    = gain_n[t] > 0 ? (1.0 - GAIN_ALPHA)*gain[t] + GAIN_ALPHA*g : g;
        gain_n[t] += 1.0;
    }

    bool losing(int t) const {
        t = std::clamp(t, 1, MAX_T);
        return gain_gate && gain_n[t] >= GAIN_N && gain[t] < 1.0;
    }

    // the realized gain is per request (a code answer must not stop the drafts of the next request's copy): cleared at begin()
    void reset_gain() {
        gain.fill(0.0);
        gain_n.fill(0.0);
    }

    // expected tokens per ms of a window of k drafts at per-token acceptance q
    double rate(int k, double q) const {
        double e = 1.0, qi = 1.0;
        for (int i = 0; i < k; ++i) {
            qi *= q;
            e  += qi;
        }
        return e/cost_ms(k + 1);
    }

    // the drafts to take (0 = a plain round) of the offered k_full (k_short: the short window, taken instead when it prices better)
    // probe_full (k15-lfm2): a full window not measured PROBES times yet is probed (q >= P_PROBE) even where the measured short window
    // prices better: a cheap short window measured first otherwise keeps the full one at its guessed (SHAPE) cost for good (G4 LFM2.5
    // 32 GB with GGML_CUDA_MMVQ_MOE_ONE=1 and a 2-draft short window: the 8-row window never ran, 8k 511 -> 470 tok/s)
    int choose(int k_full, int k_short, double q, bool probe_full = false) const {
        if (k_full <= 0) {
            return 0;
        }
        k_full  = std::min(k_full, MAX_T - 1);
        k_short = std::clamp(k_short, 1, k_full);
        const double base = 1.0/cost_ms(1);
        int    best_k = 0;
        double best   = 0.0;
        for (int k : { k_short, k_full }) {
            if (k <= short_k && losing(k + 1)) {   // the gate is for the short window (short repeats: chat, code); a copy keeps its window
                continue;
            }
            const double r = rate(k, q);
            if (r > best) {
                best   = r;
                best_k = k;
            }
        }
        if (best_k > 0 && best > base*(1.0 + margin)) {
            if (best_k < k_full && q >= P_PROBE && (stale(k_full + 1) || (probe_full && cost_n[k_full + 1] < PROBES))) {
                return k_full;
            }
            return best_k;
        }
        // a size not measured enough (q >= P_PROBE) or not measured for a while (q >= P_STALE) is tried: verification keeps the output,
        // only this round's speed is at stake
        for (int k : { k_full, k_short }) {
            if (k <= short_k && losing(k + 1)) {
                continue;
            }
            if ((cost_n[k + 1] < PROBES && q >= P_PROBE) || (stale(k + 1) && q >= P_STALE)) {
                return k;
            }
        }
        return 0;
    }

    // k19-lookup8 (LLAMA_SPEC_SUFFIX_SIZES): the window is any allowed size up to the offered drafts, not only the short and the full
    // one. Bit t of `sizes` allows a verify window of t rows (t - 1 drafts); every allowed size is its own target graph, so the set stays
    // small. The size with the most expected tokens per measured ms is taken when it beats a plain round by the margin; a size not
    // measured PROBES times yet (or not for STALE rounds) is probed only where the class is right at least P_PROBE of the time, and
    // the probe is the largest allowed size the drafts reach (the small ones get measured by the rounds that choose them).
    unsigned sizes = 0;

    static unsigned parse_sizes(const char * s) {
        unsigned m = 0;
        if (s == nullptr) {
            return 0;
        }
        for (const char * p = s; *p != '\0'; ) {
            const int t = atoi(p);
            if (t >= 2 && t <= MAX_T) {
                m |= 1u << t;
            }
            while (*p != '\0' && *p != ',') {
                ++p;
            }
            if (*p == ',') {
                ++p;
            }
        }
        return m;
    }

    bool allowed(int t) const {
        return t >= 2 && t <= MAX_T && ((sizes >> t) & 1u);
    }

    // drafts of the largest allowed window that k drafts fill (0: none)
    int fit(int k) const {
        for (int t = std::min(k + 1, MAX_T); t >= 2; --t) {
            if (allowed(t)) {
                return t - 1;
            }
        }
        return 0;
    }

    int choose_sizes(int k_avail, double q) const {
        const int k_full = fit(k_avail);
        if (k_full <= 0) {
            return 0;
        }
        const double base = 1.0/cost_ms(1);
        double e = 1.0, qi = 1.0, best = 0.0;
        int best_k = 0;
        for (int k = 1; k <= k_full; ++k) {
            qi *= q;
            e  += qi;
            if (!allowed(k + 1)) {
                continue;
            }
            const double r = e/cost_ms(k + 1);
            if (r > best) {
                best   = r;
                best_k = k;
            }
        }
        if (best_k > 0 && best > base*(1.0 + margin)) {
            if (best_k < k_full && q >= P_PROBE && stale(k_full + 1)) {
                return k_full;
            }
            return best_k;
        }
        if ((cost_n[k_full + 1] < PROBES && q >= P_PROBE) || (stale(k_full + 1) && q >= P_STALE)) {
            return k_full;
        }
        return 0;
    }

    std::string text() const {
        std::string out;
        char b[64];
        for (int t = 1; t <= MAX_T; ++t) {
            if (cost_n[t] > 0) {
                snprintf(b, sizeof(b), " T%d %.2f(n=%.0f)", t, cost[t], cost_n[t]);
                out += b;
            }
        }
        return out;
    }
};

// the round timer of one sequence: the time between two draft() calls is one round (verify + sampling) of the window drafted at
// the first; a drafted round's cost is booked one round late, with the extra time of a plain round that follows it
struct common_spec_round_timer {
    int64_t t_last_us = 0;   // the previous draft() call of this sequence (0: none since the request began)
    int     w_last    = 1;   // the window of that round
    int     w_pend    = 0;   // a drafted round waiting for the next one (0: none)
    double  ms_pend   = 0.0;
    int     acc_last  = 0;   // drafted tokens accepted in the last verified round (set by accept())
    int     acc_pend  = 0;

    void reset() {
        t_last_us = 0;
        w_last    = 1;
        w_pend    = 0;
        ms_pend   = 0.0;
        acc_last  = 0;
        acc_pend  = 0;
    }

    // called at a draft() call (now_us), before the choice of this round's window (then set with next())
    void tick(common_spec_cost_policy & pol, int64_t now_us) {
        if (t_last_us > 0) {
            const double ms = (now_us - t_last_us)/1000.0;
            if (w_last > 1) {
                if (w_pend > 0) {
                    pol.observe(w_pend, ms_pend);
                    pol.observe_gain(w_pend, acc_pend + 1, ms_pend);
                }
                w_pend   = w_last;
                ms_pend  = ms;
                acc_pend = acc_last;
            } else if (w_pend > 0) {
                const double extra = pol.cost_n[1] > 0 ? std::max(0.0, ms - pol.cost_ms(1)) : 0.0;
                pol.observe(w_pend, ms_pend + extra);
                pol.observe_gain(w_pend, acc_pend + 1, ms_pend + extra);
                w_pend = 0;
            } else {
                pol.observe(1, ms);
            }
            pol.rounds += 1.0;
        }
        t_last_us = now_us;
        w_last    = 1;
    }

    // the window this round verifies (1 = plain)
    void next(int w) {
        w_last = std::max(1, w);
    }
};

// k15-fn: LLAMA_SPEC_COST=1, the cost-aware MTP window.
// While the MTP drafts, each draft's top-1 probability p is mapped to a measured acceptance a(p) (8 probability
// buckets, decayed counts; before data a(p) = p), the chain's expected committed tokens E(rows) = 1 + a1 + a1 a2 + ... are
// priced by the measured round time of a window of that many rows (an EMA per context bucket and size, kept across requests),
// and the window keeps the rows with the best E / cost. Drafting stops when one or two more drafts as likely as the last could
// not beat the best so far. A size never measured is tried a few times when the chain is likely to reach it. Greedy verify keeps
// the output only where verify is batch invariant; elsewhere the window size changes the target's numerics (KLD-gated).
struct common_spec_mtp_cost {
    static constexpr int MAX_T = 8;
    static constexpr int CTX   = 4;     // context buckets: < 1536, < 6144, < 24576, longer
    static constexpr int PB    = 8;     // probability buckets
    static constexpr double SHAPE[MAX_T + 1] = { 0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6 };
    static constexpr double ALPHA    = 0.1;
    static constexpr double CAL_DECAY = 0.995;
    static constexpr double PRIOR_N  = 4.0;
    static constexpr int    PROBES   = 3;
    static constexpr int    SKIP     = 2;    // the first rounds of a size build / capture its graphs

    double margin = 0.03;
    std::array<std::array<double, MAX_T + 1>, CTX> cost{}, cost_n{};
    std::array<std::array<int, MAX_T + 1>, CTX>    seen{};
    std::array<double, PB> pa_ok{}, pa_n{};
    std::array<double, MAX_T + 1> n_rounds{};   // rounds per size (stats)
    int bk = 0;

    struct chain_state {
        double q = 1.0, e = 1.0, best_v = 0.0;
        int    best_rows = 1;
    };

    common_spec_mtp_cost() {
        if (const char * e = getenv("LLAMA_SPEC_COST_MARGIN")) {
            margin = atof(e);
        }
    }

    static int ctx_bucket(int64_t pos) {
        return pos < 1536 ? 0 : pos < 6144 ? 1 : pos < 24576 ? 2 : 3;
    }

    void set_context(int64_t pos) {
        const int nb = ctx_bucket(pos);
        if (nb == bk) {
            return;
        }
        bool has = false;
        for (int t = 1; t <= MAX_T; ++t) {
            has = has || cost_n[nb][t] > 0;
        }
        if (!has) {   // nothing measured at this length yet: the nearest bucket with data is the prior (one sample's worth)
            for (int d = 1; d < CTX && !has; ++d) {
                for (const int src : { nb - d, nb + d }) {
                    if (src < 0 || src >= CTX || has) {
                        continue;
                    }
                    for (int t = 1; t <= MAX_T; ++t) {
                        if (cost_n[src][t] > 0) {
                            cost[nb][t]   = cost[src][t];
                            cost_n[nb][t] = 1.0;
                            has = true;
                        }
                    }
                }
            }
        }
        bk = nb;
    }

    double cost_ms(int t) const {
        t = std::clamp(t, 1, MAX_T);
        if (cost_n[bk][t] > 0) {
            return cost[bk][t];
        }
        double num = 0.0, den = 0.0;
        for (int u = 1; u <= MAX_T; ++u) {
            if (cost_n[bk][u] > 0) {
                const double w = std::min(cost_n[bk][u], 20.0);
                num += w*cost[bk][u]*SHAPE[t]/SHAPE[u];
                den += w;
            }
        }
        return den > 0 ? num/den : SHAPE[t];
    }

    void observe(int t, double ms) {
        t = std::clamp(t, 1, MAX_T);
        n_rounds[t] += 1.0;
        if (!(ms > 0.0) || seen[bk][t]++ < SKIP) {
            return;
        }
        const double n = cost_n[bk][t];
        if (n >= 4 && ms > 20.0*cost[bk][t]) {   // a request boundary or a host stall, not a round
            return;
        }
        if (n >= 2) {
            ms = std::min(ms, 2.0*cost[bk][t]);
        }
        const double alpha = std::max(ALPHA, 1.0/(n + 1.0));   // a running mean for the first samples
        cost[bk][t]   = n > 0 ? (1.0 - alpha)*cost[bk][t] + alpha*ms : ms;
        cost_n[bk][t] = n + 1.0;
    }

    static int p_bucket(double p) {
        const int b = (int) (p*PB);
        return b < 0 ? 0 : b >= PB ? PB - 1 : b;
    }

    double draft_rate(double p) const {
        const int b = p_bucket(p);
        return (pa_ok[b] + PRIOR_N*std::clamp(p, 0.0, 1.0))/(pa_n[b] + PRIOR_N);
    }

    // the drafts of a verified window (their probabilities) and how many of them held
    void observe_probs(const float * p, int n, int accepted) {
        for (int j = 0; j < n && j <= accepted; ++j) {
            const int b = p_bucket(p[j]);
            pa_ok[b] = CAL_DECAY*pa_ok[b] + (j < accepted ? 1.0 : 0.0);
            pa_n[b]  = CAL_DECAY*pa_n[b] + 1.0;
        }
    }

    // after draft idx (0-based) with probability p: updates the best window (st.best_rows) and returns whether to draft another
    bool keep_draft(chain_state & st, int idx, double p, int cap) const {
        cap = std::clamp(cap, 1, MAX_T);
        if (idx == 0) {
            st = chain_state{};
            st.best_v = 1.0/cost_ms(1);
        }
        const double a = draft_rate(p);
        st.q *= a;
        st.e += st.q;
        const int rows = idx + 2;
        if (rows <= cap) {
            const double v = st.e/cost_ms(rows);
            if (v > st.best_v*(1.0 + margin/3.0)) {
                st.best_v    = v;
                st.best_rows = rows;
            }
            if (cost_n[bk][rows] < PROBES && st.q >= 0.15) {
                st.best_rows = rows;
            }
        }
        if (rows + 1 > cap) {
            return false;
        }
        const double v1 = (st.e + st.q*a)/cost_ms(rows + 1);
        const double v2 = rows + 2 <= cap ? (st.e + st.q*a + st.q*a*a)/cost_ms(rows + 2) : 0.0;
        return std::max(v1, v2) > st.best_v*(1.0 + margin/3.0) || (cost_n[bk][rows + 1] < PROBES && st.q*a >= 0.15);
    }

    // expected committed tokens of a window holding the first n drafts (probabilities p), and the chance all n hold
    double chain_e(const float * p, int n, double & q) const {
        double e = 1.0;
        q = 1.0;
        for (int j = 0; j < n; ++j) {
            q *= draft_rate(p[j]);
            e += q;
        }
        return e;
    }

    std::string text() const {
        std::string out;
        char b[96];
        for (int k = 0; k < CTX; ++k) {
            bool any = false;
            for (int t = 1; t <= MAX_T; ++t) {
                any = any || cost_n[k][t] > 0;
            }
            if (!any) {
                continue;
            }
            snprintf(b, sizeof(b), "%sctx%d:", out.empty() ? "" : " | ", k);
            out += b;
            for (int t = 1; t <= MAX_T; ++t) {
                if (cost_n[k][t] > 0) {
                    snprintf(b, sizeof(b), " T%d %.2f(n=%.0f)", t, cost[k][t], cost_n[k][t]);
                    out += b;
                }
            }
        }
        out += " | rounds:";
        for (int t = 1; t <= MAX_T; ++t) {
            if (n_rounds[t] > 0) {
                snprintf(b, sizeof(b), " T%d=%.0f", t, n_rounds[t]);
                out += b;
            }
        }
        out += " | a(p):";
        for (int i = 0; i < PB; ++i) {
            if (pa_n[i] > 0.5) {
                snprintf(b, sizeof(b), " %.2f:%.2f(%.0f)", (i + 0.5)/PB, pa_ok[i]/pa_n[i], pa_n[i]);
                out += b;
            }
        }
        return out;
    }
};

// k19-tpp (LLAMA_SPEC_TPP): the round time of the floor window (n_floor drafts) and of the floor + 1 window, per context bucket, as the
// median of the last NS rounds of each (robust to the graph captures and host stalls a mean takes in). The floor window keeps being
// measured: it runs while it has fewer than PROBES samples in the bucket and once every STALE rounds after its last sample, so one slow
// early round cannot price it out (the first G4 gate: T3 sampled once at 19.4 ms in a warm-up round, real 11 ms, then never again)
struct common_spec_tpp_cost {
    static constexpr int CTX    = 5;    // context buckets: < 1536, < 6144, < 12288, < 24576, longer
    static constexpr int NS     = 15;
    static constexpr int SKIP   = 2;    // the first rounds of a size in a bucket build / capture its graphs
    static constexpr int PROBES = 3;
    static constexpr int STALE  = 48;

    struct size_stat {
        float   ring[NS] = {};
        int     n    = 0;
        int     seen = 0;
        int64_t last = -1000000;
    };
    size_stat st[CTX][2];
    int     bk     = 0;
    int64_t rounds = 0;
    double  margin  = 0.05;
    double  probe_q = 0.5;
    int     stale   = STALE;

    // the draft past the floor: how often it held against its predicted rate (draft_rate), per context bucket, decayed. The rate
    // is calibrated over every context, and a long context can hold far less than it says (G4 fn4 30k: the wider windows added
    // ~0.05 tokens per pass), so the wider window's extra tokens are scaled by this ratio. LLAMA_SPEC_TPP_MARG_PRIOR: the prior's
    // weight in drafts (0: off, the rate as calibrated); LLAMA_SPEC_TPP_MARG_CAP: the largest ratio (a kept draft is one whose rate
    // looked high, so held / predicted runs below 1 even where the rate is right; the cap above 1 lets that noise even out)
    static constexpr double MDECAY = 0.97;
    struct marg_stat {
        double pred = 0.0;
        double held = 0.0;
    };
    marg_stat mg[CTX];
    double marg_prior  = 4.0;
    double marg_cap    = 1.25;
    // LLAMA_SPEC_TPP_MAX_POS (default 24576, 0: no limit): from this position on the window stays at the floor. G4 fn4 full card
    // (ktp_g4d): with the held / predicted pricing 30k was still 0.979 of base (0.961 without it) while 16k gained 4.1%; the
    // floor-only window there is the base's own (the kill switch arm: 0.999)
    int64_t max_pos    = 24576;
    int64_t pos_cur    = 0;
    // the round times are priced from a fixed table by default, not from live timings: live medians move with timing noise, so the
    // window sequence (and with an MoE cache, which experts are resident and how a CPU miss sums) differed launch to launch (G4 fn4
    // 16 GB, ktp_g4d: 2-3 md5 per cell, base stable). The table is c(4 rows) / c(3 rows) per context bucket, measured on G4 fn4:
    // all experts on the device 1.14 below 12288 (chat / 8k: 11.96 / 10.52, 12.50 / 10.99 ms) and 1.18 above (16k: 12.88 / 10.93);
    // with an MoE cache streaming experts 1.22 (16 / 32 GB: +5.6 / +3.1 ms per 4-row pass on ~26 / ~17 ms). The drafts' probabilities
    // and the held counts are functions of the text, so the windows are too. LLAMA_SPEC_TPP_LIVE=1: the live medians (as before);
    // LLAMA_SPEC_TPP_RATIO=x: one ratio for every bucket
    bool   live            = false;
    double cost_ratio[CTX] = { 1.14, 1.14, 1.14, 1.18, 1.18 };
    // LLAMA_SPEC_TPP_SWITCH_SKIP=1 (opt-in): the first round after a window-size switch stays out of the size's median. Off by
    // default: a size not run for a while re-captures its verify graph when the KV view has grown since (3-5 ms host
    // section 7), a real cost of switching to it, so skipping those rounds underprices the rarely used 4-row window. The switched
    // rounds' times also go to sw[], a median per kind printed in the cost table, so a run shows what a switch costs
    bool      switch_skip = false;
    int       last_k      = -1;
    size_stat sw[CTX][2];
    // k19-tppctx, LLAMA_SPEC_TPP_CTX (default 1, =0 off): the window of some context buckets is fixed instead of priced:
    // LLAMA_SPEC_TPP_CTX_EXTRA="e0,e1,e2,e3,e4" drafts past the floor per bucket (0 or 1; -1 = priced as before), default "1,-1,-1,-1,-1":
    // the chat bucket (below 1536 tokens) takes the 3rd draft in every window. G4 Qwen3.8-Flash-Next fn4 full card (kvr2_c1, x3): chat
    // 258.8 -> 278.6 tok/s (3.24 -> 3.46 tokens per pass at 12.52 / 12.42 ms per pass, the same text), 8k / 16k / 30k unchanged (0.995 /
    // 1.004 / 1.007, same text); a 4-row window runs a 3-row one's kernels (LLAMA_SPEC_TPP_VERIFY), so the text does not change.
    // Deterministic: no timing
    int  ctx_extra[CTX] = { -1, -1, -1, -1, -1 };
    bool ctx_on         = true;
    // k19-depth: below deep_pos (LLAMA_SPEC_DEPTH_POS, default 12288) every window drafts deep_extra more past floor + 1, so its width
    // is fixed at floor + 2 + deep_extra rows there; from deep_pos on the windows are the ones above (floor, floor + 1 where it pays).
    // The drafter sets deep_extra from its n_max (the server raises Studio's 2 drafts to 4 where it is on: LLAMA_SPEC_DEPTH). One width
    // per context band, as each width a request mixes pays its own CUDA graph captures. Deterministic
    int     deep_extra = 0;
    int64_t deep_pos   = 12288;
    // k19-fnmtp: the deep band also for a window of floor + 1 drafts (deep_extra 0): set beside a MoE cache by the server's rule
    bool    deep_all   = false;

    common_spec_tpp_cost() {
        if (const char * e = getenv("LLAMA_SPEC_TPP_MARG_PRIOR")) {
            marg_prior = atof(e);
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_SWITCH_SKIP")) {
            switch_skip = atoi(e) != 0;
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_LIVE")) {
            live = atoi(e) != 0;
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_RATIO")) {
            std::fill(cost_ratio, cost_ratio + CTX, atof(e));
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_MAX_POS")) {
            max_pos = atoll(e);
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_MARG_CAP")) {
            marg_cap = atof(e);
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_MARGIN")) {
            margin = atof(e);
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_PROBE_Q")) {
            probe_q = atof(e);
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_STALE")) {
            stale = std::max(1, atoi(e));
        }
        if (const char * e = getenv("LLAMA_SPEC_DEPTH_POS")) {
            deep_pos = atoll(e);
        }
        if (const char * e = getenv("LLAMA_SPEC_TPP_CTX")) {
            ctx_on = atoi(e) != 0;
        }
        if (ctx_on) {
            const char * e = getenv("LLAMA_SPEC_TPP_CTX_EXTRA");
            const std::string str = e != nullptr ? e : "1,-1,-1,-1,-1";
            size_t p = 0;
            for (int k = 0; k < CTX && p <= str.size(); ++k) {
                size_t q = str.find(',', p);
                if (q == std::string::npos) {
                    q = str.size();
                }
                ctx_extra[k] = std::clamp(atoi(str.substr(p, q - p).c_str()), -1, 1);
                p = q + 1;
            }
        }
    }

    static int bucket(int64_t pos) {
        return pos < 1536 ? 0 : pos < 6144 ? 1 : pos < 12288 ? 2 : pos < 24576 ? 3 : 4;
    }

    void set_context(int64_t pos) {
        bk      = bucket(pos);
        pos_cur = pos;
    }

    // one round of the window kind k (0 = floor, 1 = floor + 1)
    void observe(int k, double ms) {
        ++rounds;
        size_stat & s = st[bk][k];
        const bool switched = last_k >= 0 && k != last_k;
        last_k = k;
        if (!(ms > 0.0) || s.seen++ < SKIP) {
            return;
        }
        if (switched) {
            size_stat & w = sw[bk][k];
            w.ring[w.n % NS] = (float) ms;
            w.n++;
            if (switch_skip && s.n >= PROBES) {
                s.last = rounds;   // the size ran now: its median stays current
                return;
            }
        }
        s.ring[s.n % NS] = (float) ms;
        s.n++;
        s.last = rounds;
    }

    static double median_of(const size_stat & s) {
        const int m = std::min(s.n, NS);
        if (m == 0) {
            return 0.0;
        }
        float v[NS];
        std::copy(s.ring, s.ring + m, v);
        std::nth_element(v, v + m/2, v + m);
        return v[m/2];
    }

    double median(int k) const {
        return median_of(st[bk][k]);
    }

    // the floor window must run now (too few samples in this bucket, or its last one is STALE rounds old)
    // an MoE cache streams experts: a wider window reads more of them (LLAMA_SPEC_TPP_RATIO still overrides)
    void set_cached(bool cached) {
        if (cached && getenv("LLAMA_SPEC_TPP_RATIO") == nullptr) {
            std::fill(cost_ratio, cost_ratio + CTX, 1.22);
        }
    }

    bool need_floor() const {
        if (!live) {
            return false;
        }
        const size_stat & s = st[bk][0];
        return s.n < PROBES || rounds - s.last > stale;
    }

    double marg_ratio() const {
        if (!(marg_prior > 0.0)) {
            return 1.0;
        }
        const marg_stat & m = mg[bk];
        return std::min(marg_cap, (m.held + marg_prior)/(m.pred + marg_prior));
    }

    // a window that kept the draft past the floor, its predicted hold rate a, and whether it held (the floor's drafts all held)
    void observe_marginal(double a, bool held) {
        marg_stat & m = mg[bk];
        m.pred = MDECAY*m.pred + a;
        m.held = MDECAY*m.held + (held ? 1.0 : 0.0);
    }

    // k19-tppctx: the fixed drafts past the floor of the current bucket, -1 = priced
    int forced_extra() const {
        return ctx_on ? ctx_extra[bk] : -1;
    }

    // k19-depth: the current position is in the deep band (every window drafts floor + 1 + deep_extra)
    bool deep() const {
        return (deep_extra > 0 || deep_all) && pos_cur < deep_pos;
    }

    // window floor + 1 (expected tokens e_w, chance q_w that every draft holds) against the floor window (e_f)
    bool wide_pays(double e_f, double e_w, double q_w) const {
        if (deep()) {
            return true;
        }
        if (forced_extra() >= 0) {
            return forced_extra() >= 1;
        }
        if (max_pos > 0 && pos_cur >= max_pos) {
            return false;
        }
        const double r = marg_ratio();
        e_w = e_f + (e_w - e_f)*r;
        q_w *= r;
        if (!live) {
            return e_w > e_f*cost_ratio[bk]*(1.0 + margin);
        }
        const size_stat & s = st[bk][1];
        if (s.n < PROBES || rounds - s.last > stale) {
            return q_w >= probe_q;   // measure it when the chain is likely to reach it
        }
        const double c_f = median(0), c_w = median(1);
        return c_f > 0.0 && c_w > 0.0 && e_w/c_w > e_f/c_f*(1.0 + margin);
    }

    std::string text() const {
        std::string out = live ? "live " : "fixed ";
        char b[192];
        for (int k = 0; k < CTX; ++k) {
            if (st[k][0].n == 0 && st[k][1].n == 0) {
                continue;
            }
            snprintf(b, sizeof(b), "%sctx%d: floor %.2f(n=%d) wide %.2f(n=%d) held %.1f/%.1f switched %.2f(n=%d) %.2f(n=%d)",
                out.size() <= 6 ? "" : " | ", k, median_of(st[k][0]), st[k][0].n, median_of(st[k][1]), st[k][1].n, mg[k].held, mg[k].pred,
                median_of(sw[k][0]), sw[k][0].n, median_of(sw[k][1]), sw[k][1].n);
            out += b;
        }
        return out;
    }
};
