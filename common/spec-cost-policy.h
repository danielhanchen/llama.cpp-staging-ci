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

// k15: window HOLD for suffix drafts drafted alone (LLAMA_SPEC_SUFFIX_HOLD, default on with the cost policy; 0 = the policy above).
//
// What the policy above prices as "the cost of a drafted window" is mostly the change of verify length: a round whose window differs
// from the previous one rebuilds and re-captures the target graph (G4 Flint Q4_K_XL: a 4-token window 10.8 ms vs a plain round 4.0 ms,
// almost all of it the switches in and out), so on long repetitive prompts it takes 5 of 55 matches, where PolyStrata, whose
// window sizes each keep a captured graph, takes almost all (21 vs 107 drafts offered, Flint 29.8K). Here the two parts are measured
// apart and priced where they occur:
//   - steady[t]: the round time of a window t whose previous round had the same window (no rebuild),
//   - sw: the extra time of a round whose window differs from the previous one (an EMA over every switch),
// and there is one drafted window size (the full window, n_max + 1): a match shorter than the window is padded to it with filler
// tokens (the verifier rejects them like any wrong draft, the output is unchanged), so consecutive drafted rounds never switch.
// A drafted round right after a drafted round costs steady[T]; entering the window from a plain round costs steady[T] + 2 sw
// (in and, later, out) spread over the expected run of drafted rounds (an EMA of past runs). A round with no match after a drafted
// round stays in the window (all filler) while that still prices better than switching out and back in for the next match.
struct common_spec_hold_policy {
    static constexpr int    MAX_T  = common_spec_cost_policy::MAX_T;
    static constexpr double ALPHA  = 0.1;
    static constexpr int    PROBES = 3;
    static constexpr double P_PROBE = 0.85;
    double margin = 0.03;
    std::array<double, MAX_T + 1> steady{}, steady_n{};
    double sw = 0.0, sw_n = 0.0;     // extra ms of a round whose window differs from the previous one
    double run = 2.0;                // EMA of drafted rounds per run (entered from a plain round)
    double hit = 0.5;                // EMA over rounds: the round found a match (1) or not (0)
    mutable int probes = 0;          // rounds drafted while the window's steady time was not measured

    double cost_ms(int t) const {
        t = std::clamp(t, 1, MAX_T);
        if (steady_n[t] > 0) {
            return steady[t];
        }
        // not measured yet: a resident mat-vec reads each weight once for up to 8 rows, so a window costs little more than a plain round
        // (1 + 0.1 per draft); the probes replace this with the measured steady time
        const double c1 = steady_n[1] > 0 ? steady[1] : 1.0;
        return c1*(1.0 + 0.1*(t - 1));
    }
    double switch_ms() const {
        return sw_n > 0 ? sw : 2.0*cost_ms(1);   // before any switch was measured: priced like two plain rounds
    }
    // one finished round: its window, the previous round's window, its wall time
    void observe(int t, int t_prev, double ms) {
        t = std::clamp(t, 1, MAX_T);
        if (!(ms > 0.0)) {
            return;
        }
        if (steady_n[t] >= 4 && ms > 20.0*steady[t] + 50.0) {
            return;   // a request boundary or a host stall, not a round
        }
        if (t == t_prev) {
            if (steady_n[t] >= 2) {
                ms = std::min(ms, 3.0*steady[t]);
            }
            steady[t]    = steady_n[t] > 0 ? (1.0 - ALPHA)*steady[t] + ALPHA*ms : ms;
            steady_n[t] += 1.0;
        } else if (steady_n[t] > 0) {
            const double extra = std::max(0.0, ms - steady[t]);
            sw    = sw_n > 0 ? (1.0 - ALPHA)*sw + ALPHA*extra : extra;
            sw_n += 1.0;
        }
    }
    void observe_run(int n_rounds) {
        run = 0.8*run + 0.2*std::max(1, n_rounds);
    }
    void observe_hit(bool h) {
        hit = 0.9*hit + 0.1*(h ? 1.0 : 0.0);
    }
    static double expect(int k, double q) {
        double e = 1.0, qi = 1.0;
        for (int i = 0; i < k; ++i) {
            qi *= q;
            e  += qi;
        }
        return e;
    }
    // drafts this round: T = the held window size; k_real = the drafts a match offers (0 = no match); q its class acceptance;
    // in_run = the previous round verified the window T. Returns true to verify the window T (padded), false for a plain round.
    bool choose(int T, int k_real, double q, bool in_run) const {
        T = std::clamp(T, 2, MAX_T);
        const double base = 1.0/cost_ms(1);
        if (k_real > 0) {
            const double e    = expect(std::min(k_real, T - 1), q);
            const double cost = in_run ? cost_ms(T) : cost_ms(T) + 2.0*switch_ms()/std::max(1.0, run);
            if (e/cost > base*(1.0 + margin)) {
                return true;
            }
            if (steady_n[T] < PROBES && probes < 4*PROBES) {
                probes++;
                return true;
            }
            return false;
        }
        if (!in_run) {
            return false;
        }
        // no match inside a run: stay on filler rounds until the next match (each costs cost(T) - cost(1) more than a plain round, for an
        // expected 1 / hit rounds) or leave (a switch out now and one back in at the next match)
        return (cost_ms(T) - cost_ms(1))/std::max(hit, 0.05) < 2.0*switch_ms();
    }
};
