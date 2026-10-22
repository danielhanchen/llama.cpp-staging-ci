#pragma once

// k11-decode: cost-priced window choice for n-gram (suffix) drafts drafted alone.
//
// The verify window of t tokens (the sampled token + t - 1 drafted ones) is priced by the measured wall time of the rounds of that
// size (an EMA per size, kept across requests: it is a property of the model, the GPU and the placement, not of the text). A draft
// of k tokens from a match class whose decayed per-token acceptance is q is expected to commit 1 + q + ... + q^k tokens; it is
// taken when the best expected tokens per ms beats a plain round's 1 / cost(1) by a margin. This is PolyStrata's
// DraftPolicy::choose (src/spec/draft_policy.cpp: measured round costs, a guessed cost shape for sizes not seen yet, a few probes
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

struct common_spec_cost_policy {
    static constexpr int MAX_T = 8;

    // relative round cost by window size before a size is measured: PolyStrata's RTX 5070 curve (kShape), pessimistic for a
    // resident model, so a size is priced high until measured (the probes measure it)
    static constexpr double SHAPE[MAX_T + 1] = { 0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6 };
    static constexpr double ALPHA  = 0.1;     // EMA weight of a new round time
    static constexpr int    PROBES = 3;       // a size is drafted this often (q >= P_PROBE) before its guessed cost can veto it
    static constexpr double P_PROBE = 0.85;
    static constexpr double STALE   = 200.0;  // rounds after which a measured size is re-probed when a likely draft could use it
    static constexpr double P_STALE = 0.7;    // (a size priced high in one slow stretch, a cold start or a busy host, is measured again)
    static constexpr int    SKIP    = 2;      // the first rounds of a size build its graph and capture it: not its cost

    double margin = 0.03;
    bool   gain_gate = true;   // LLAMA_SPEC_POLICY_GAIN=0: off
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
    int choose(int k_full, int k_short, double q) const {
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
            if (best_k < k_full && q >= P_PROBE && stale(k_full + 1)) {
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
