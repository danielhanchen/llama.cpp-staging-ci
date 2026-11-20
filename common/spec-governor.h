// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

// k9-draft: speculation cost governor (per slot). Speculative decoding only changes speed, so it should never be slower than plain
// decoding; whether it pays depends on the model, the drafter, the device, the context and the text, so it is measured: the time from
// one emitted step to the next (draft + verify + sampling + server work; a checkpoint replay folds into the step that emits) and the
// tokens it emitted, on both sides. After TensorSharp's SpeculationCostGovernor (TensorSharp.Runtime/Speculative/
// SpeculationCostGovernor.cs): a round measures drafted steps, then a plain block, and judges by the ratio of sums (ms per token); a
// loss parks drafting for 32 steps, doubling up to 256 while it keeps losing; a win holds for 64 steps, doubling up to 1024 while it
// keeps winning, and a window of the last 8 drafted steps is watched against the round's plain cost; a probe whose first 4 measured
// steps accepted nothing ends as a loss, and so does a hold whose last 8 drafted steps accepted nothing; a new request caps a park at
// the first interval and keeps at most one counted loss, and a winning hold carries over (TensorSharp keeps it too; measured on G4,
// Qwen3.6 + MTP: re-measuring a plain block at every request cost 2-5% of the decode where drafting wins). The first sample of each side after a change is
// dropped (it pays the graph rebuild), and so is the worst drafted sample of a round.
//
// (an EMA of step costs carried the warm-up and costs of another context into later decisions): every
// verdict compares blocks of the same round, measured back to back at the same context length. A re-probe after a park is judged
// against the last parked steps (plain decodes measured just before it), never against an older plain cost, and a round's samples
// are dropped when a request ends.
//
// Steps where drafting was allowed but produced no draft (an n-gram drafter without a match) are not samples of either side.
// OPT-IN, LLAMA_SPEC_GOVERNOR=1 (unset / 0: off, drafting always allowed, the code path of the k9 tip): on G4 (RTX PRO 6000) it cost
// 0-5% decode beside a winning MTP head (Qwen3.6-35B-A3B, gemma-4-26B-A4B + its assistant, n_max 2) and did not recover the n-gram
// losses on LFM2.5. LLAMA_SPEC_GOVERNOR_MARGIN (1.03): drafting stays
// on while its ms per token is at most margin x the plain one; LLAMA_SPEC_GOVERNOR_LOG=1 logs every verdict.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct common_spec_governor {
    // drafted steps measured per round (the first round and after a win), and per re-probe after a loss
    int probe_spec   = 8;
    int reprobe_spec = 4;
    // plain steps measured per round (a re-probe uses the last parked steps instead)
    int probe_plain  = 4;
    // a probe ends as a loss when its first zero_accept measured drafted steps accepted nothing
    int zero_accept  = 4;
    int park_first   = 32;
    int park_max     = 256;
    int win_hold     = 64;
    int win_hold_max = 1024;
    static constexpr int WIN_WINDOW = 8;
    double margin      = 1.03;
    double win_margin  = 1.10;   // a held win parks when its last WIN_WINDOW drafted steps are this much slower than the round's plain
    bool   enabled     = true;
    bool   log         = false;

    // state
    int  parked       = 0;   // steps left parked (plain)
    int  hold         = 0;   // steps left in a winning hold
    int  losses_run   = 0;   // consecutive losses
    int  wins_run     = 0;   // consecutive wins (the hold doubles with them)
    int  hold_zero    = 0;   // drafted steps in a row that accepted nothing while holding
    bool spec_wins    = true;

    // round samples (ratio of sums)
    int64_t spec_us = 0, plain_us = 0;
    int64_t spec_tok = 0, plain_tok = 0;
    int     spec_n = 0, plain_n = 0;
    int64_t worst_us = 0, worst_tok = 0;
    bool    spec_skip = false, plain_skip = false;

    // parked-step tail: the last plain steps of a park, the baseline of the next re-probe
    static constexpr int TAIL = 8;
    int64_t tail_us[TAIL] = {}, tail_tok[TAIL] = {};
    int     tail_n = 0, tail_head = 0;
    bool    tail_skip = false;

    // win watch
    int64_t win_us[WIN_WINDOW] = {}, win_tok[WIN_WINDOW] = {};
    int     win_n = 0, win_head = 0;
    double  plain_ms_tok = 0.0;   // the plain cost of the round that won

    // totals for the log
    int64_t n_wins = 0, n_losses = 0, n_parked = 0, n_spec_steps = 0, n_plain_steps = 0;

    common_spec_governor() {
        const char * e = getenv("LLAMA_SPEC_GOVERNOR");
        enabled = e != nullptr && atoi(e) != 0;
        if (const char * m = getenv("LLAMA_SPEC_GOVERNOR_MARGIN")) {
            margin = std::max(0.5, atof(m));
        }
        if (const char * m = getenv("LLAMA_SPEC_GOVERNOR_WIN_MARGIN")) {
            win_margin = std::max(0.5, atof(m));
        }
        const char * l = getenv("LLAMA_SPEC_GOVERNOR_LOG");
        log = l != nullptr && atoi(l) != 0;
    }

    static double ms_tok(int64_t us, int64_t tok) {
        return tok > 0 ? (double) us / 1000.0 / (double) tok : 0.0;
    }

    int probe_len() const {
        return losses_run > 0 ? reprobe_spec : probe_spec;
    }

    // a re-probe is judged against the parked tail when it holds enough fresh plain steps
    bool tail_ready() const {
        return losses_run > 0 && tail_n >= probe_plain;
    }

    bool allows() const {
        if (!enabled) {
            return true;
        }
        if (parked > 0) {
            return false;
        }
        if (hold > 0) {
            return spec_wins;
        }
        if (spec_n < probe_len()) {
            return true;    // measuring drafted steps
        }
        if (!tail_ready() && plain_n < probe_plain) {
            return false;   // measuring the plain block
        }
        return spec_wins;
    }

    void reset_round() {
        spec_us = plain_us = spec_tok = plain_tok = 0;
        spec_n = plain_n = 0;
        worst_us = worst_tok = 0;
        spec_skip = plain_skip = false;
        win_n = win_head = 0;
    }

    void reset_tail() {
        tail_n = tail_head = 0;
        tail_skip = false;
    }

    // a new request on this slot
    void new_sequence() {
        if (!enabled) {
            return;
        }
        parked = std::min(parked, park_first);
        losses_run = std::min(losses_run, 1);
        if (hold > 0 && spec_wins) {
            // a winning hold carries over, without its watch: the round's plain cost belongs to the previous context
            plain_ms_tok = 0.0;
        } else {
            hold = 0;
        }
        spec_wins = true;
        hold_zero = 0;
        reset_round();
        reset_tail();   // plain costs of the previous context say nothing about this one
    }

    void park() {
        spec_wins = false;
        n_losses++;
        losses_run++;
        wins_run = 0;
        hold_zero = 0;
        parked = std::min(park_max, park_first << std::min(losses_run - 1, 5));
        hold = 0;
        reset_round();
        reset_tail();
    }

    void verdict(double spec_ms, double plain_ms, const char * what) {
        const bool win = spec_ms <= plain_ms*margin;
        if (log) {
            fprintf(stderr, "spec governor: %s: drafted %.3f ms/token vs plain %.3f ms/token -> %s (losses in a row %d)\n", what, spec_ms,
                    plain_ms, win ? "win" : "loss", win ? 0 : losses_run + 1);
        }
        if (win) {
            spec_wins    = true;
            n_wins++;
            losses_run   = 0;
            hold         = std::min(win_hold_max, win_hold << std::min(wins_run, 4));
            wins_run++;
            hold_zero    = 0;
            plain_ms_tok = plain_ms;
            reset_round();
            reset_tail();
        } else {
            park();
        }
    }

    // one emitted step: drafted = a draft was verified, n_tok = tokens it emitted, us = time since the previous emitted step
    void record(bool drafted, int n_tok, int64_t us) {
        if (!enabled || n_tok <= 0 || us <= 0) {
            return;
        }
        if (drafted) {
            n_spec_steps++;
        } else {
            n_plain_steps++;
        }
        if (parked > 0) {
            parked--;
            n_parked++;
            if (!drafted) {
                // the parked steps are plain decodes of this context: the last ones are the next re-probe's baseline
                if (!tail_skip) {
                    tail_skip = true;
                } else {
                    tail_us[tail_head] = us; tail_tok[tail_head] = n_tok;
                    tail_head = (tail_head + 1) % TAIL;
                    tail_n = std::min(tail_n + 1, TAIL);
                }
            }
            return;
        }
        if (hold > 0) {
            hold--;
            if (drafted) {
                hold_zero = n_tok <= 1 ? hold_zero + 1 : 0;
                if (hold_zero >= WIN_WINDOW) {
                    if (log) {
                        fprintf(stderr, "spec governor: hold: %d drafted steps in a row accepted nothing -> park\n", hold_zero);
                    }
                    park();
                    return;
                }
            }
            if (drafted && plain_ms_tok > 0.0) {
                win_us[win_head] = us; win_tok[win_head] = n_tok;
                win_head = (win_head + 1) % WIN_WINDOW;
                win_n = std::min(win_n + 1, WIN_WINDOW);
                if (win_n == WIN_WINDOW) {
                    int64_t u = 0, t = 0;
                    for (int i = 0; i < WIN_WINDOW; ++i) {
                        u += win_us[i]; t += win_tok[i];
                    }
                    if (ms_tok(u, t) > plain_ms_tok*win_margin) {
                        if (log) {
                            fprintf(stderr, "spec governor: hold: last %d drafted steps %.3f ms/token vs the round's plain %.3f -> park\n",
                                    WIN_WINDOW, ms_tok(u, t), plain_ms_tok);
                        }
                        park();
                    }
                }
            }
            if (hold == 0 && spec_wins) {
                reset_round();   // the next round measures again
            }
            return;
        }
        if (drafted) {
            if (!spec_skip) {
                spec_skip = true;
                return;
            }
            spec_us += us; spec_tok += n_tok; spec_n++;
            if (worst_tok == 0 || ms_tok(us, n_tok) > ms_tok(worst_us, worst_tok)) {
                worst_us = us; worst_tok = n_tok;
            }
            if (spec_n == zero_accept && spec_tok == spec_n && spec_n < probe_len()) {
                if (log) {
                    fprintf(stderr, "spec governor: probe: %d drafted steps accepted nothing -> loss\n", spec_n);
                }
                park();
                return;
            }
        } else {
            if (!plain_skip) {
                plain_skip = true;
            } else {
                plain_us += us; plain_tok += n_tok; plain_n++;
            }
        }
        if (spec_n < probe_len()) {
            return;
        }
        int64_t pu = plain_us, pt = plain_tok;
        if (tail_ready()) {
            pu = pt = 0;
            for (int i = 0; i < tail_n; ++i) {
                pu += tail_us[i]; pt += tail_tok[i];
            }
        } else if (plain_n < probe_plain) {
            return;
        }
        double s = ms_tok(spec_us, spec_tok);
        if (spec_n > 1 && spec_us - worst_us > 0 && spec_tok - worst_tok > 0) {
            s = std::min(s, ms_tok(spec_us - worst_us, spec_tok - worst_tok));
        }
        verdict(s, ms_tok(pu, pt), tail_ready() ? "re-probe vs the parked steps before it" : "round");
    }
};
