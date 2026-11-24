#pragma once

// the MoE cache's CPU-miss timing trials, persisted (LLAMA_MOE_CPU_MISS_PERSIST, default 1; =0 off).
//
// A trial times small batches with and without the host path and keeps the path if it is at least cmt_margin faster. Where the
// host and the upload cost about the same (T4, 4 vCPU, Qwen3.6-35B-A3B 4-bit quant 12 GB: 155 vs 188 us per expert) the first
// trial decided "off" in one launch (-2.3%) and "on" in the next (-3.6%), and the two launches wrote other texts at temperature 0
// (the host computes misses it does not fetch, so the cache, the route-tail skip and which rows run on the CPU all diverge).
//
// The verdict of every trial of a launch (on / off, the short-prompt flag, the gap to the next trial, how many small batches the trial
// ran, and the measured ratio t_on / t_off) is kept as one line in the calibration file of, under a tag naming the cache
// size (expert slots), the layer count and the small-batch limit; the file's key already names the build, the device, the threads, the
// CPU count, the cache (main / MTP) and the expert form. A later launch with the same file and tag replays the recorded trials in order:
// it runs each trial's schedule (the same blocks with and without the host path, so the cache is admitted to exactly as in the first
// launch) and takes the recorded verdict instead of its own timing. Trials beyond the record are timed, with hysteresis against the
// previous verdict, and appended, so the next launch replays them too.
//
// Header-only and free of ggml so tests/test-moe-trial-persist.cpp can test it on the CPU.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace llama_moe_tp {

struct rec {
    int    on      = 0;   // the verdict for small batches
    int    host_ok = 1;   // short prompts may take the host path (cmt_host_slow was false)
    int    gap     = 64;  // small batches until the next trial
    int    len     = 0;   // counted small batches the trial ran (it can end early)
    double ratio   = 0.0; // t_on / t_off measured by the launch that raced it (information: the margin)
    int    firm    = 1;   // replayed as recorded; 0 = a marginal timing, timed again by the next launch
    int    tries   = 1;   // launches that timed this trial
};

// the calibration file key of a pinned run (LLAMA_MOE_CPU_MISS_CAL_VALUES): the machine's key plus the pinned numbers,
// so the pinned run has a file of its own (its trial verdicts persist) and a launch without the variable never reads pinned numbers
inline std::string calv_key(const std::string & key, const double v[4]) {
    char buf[160];
    snprintf(buf, sizeof(buf), "|calv=%.17g %.17g %.17g %.17g", v[0], v[1], v[2], v[3]);
    return key + buf;
}

// margin_trust = false (LLAMA_MOE_CPU_MISS_MARGIN_TRUST=0): a history of its own, so a record made firm by trusted marginal
// timings is not replayed by a launch that turned the trust off
inline std::string tag_of(long long n_slots, int n_layers, long long max_batch, bool margin_trust = true) {
    return "trials|slots=" + std::to_string(n_slots) + "|layers=" + std::to_string(n_layers) + "|max_batch=" + std::to_string(max_batch) +
        (margin_trust ? "" : "|margin_trust=0");
}

inline std::string format_line(const std::string & tag, const std::vector<rec> & r) {
    std::string s = tag + " " + std::to_string(r.size());
    char buf[96];
    for (const rec & x : r) {
        snprintf(buf, sizeof(buf), " %d:%d:%d:%d:%.6f:%d:%d", x.on ? 1 : 0, x.host_ok ? 1 : 0, x.gap, x.len, x.ratio, x.firm ? 1 : 0, x.tries);
        s += buf;
    }
    return s;
}

// parses one line written by format_line for this tag; false if it is another tag or malformed
inline bool parse_line(const std::string & line, const std::string & tag, std::vector<rec> & out) {
    if (line.size() <= tag.size() || line.compare(0, tag.size(), tag) != 0 || line[tag.size()] != ' ') {
        return false;
    }
    const char * p = line.c_str() + tag.size() + 1;
    char * end = nullptr;
    const long n = strtol(p, &end, 10);
    if (end == p || n < 0 || n > 100000) {
        return false;
    }
    p = end;
    std::vector<rec> r;
    for (long i = 0; i < n; ++i) {
        rec x;
        int nc = 0;
        if (sscanf(p, " %d:%d:%d:%d:%lf%n", &x.on, &x.host_ok, &x.gap, &x.len, &x.ratio, &nc) != 5 || nc <= 0 ||
                x.gap <= 0 || x.len < 0) {
            return false;
        }
        p += nc;
        // ":firm:tries" (a line without them, written before, is firm)
        int nf = 0;
        if (sscanf(p, ":%d:%d%n", &x.firm, &x.tries, &nf) == 2 && nf > 0) {
            if (x.tries < 1) {
                return false;
            }
            p += nf;
        } else {
            x.firm  = 1;
            x.tries = 1;
        }
        r.push_back(x);
    }
    out = std::move(r);
    return true;
}

inline bool read_lines(const std::string & path, std::vector<std::string> & lines) {
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    lines.clear();
    std::string cur;
    char buf[4096];
    while (fgets(buf, sizeof(buf), f) != nullptr) {
        cur += buf;
        if (!cur.empty() && cur.back() == '\n') {
            while (!cur.empty() && (cur.back() == '\n' || cur.back() == '\r')) {
                cur.pop_back();
            }
            lines.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) {
        lines.push_back(cur);
    }
    fclose(f);
    return true;
}

// the recorded trials of this tag in the calibration file whose first line is `key`; false (out empty) if none
inline bool read(const std::string & path, const std::string & key, const std::string & tag, std::vector<rec> & out) {
    out.clear();
    std::vector<std::string> lines;
    if (!read_lines(path, lines) || lines.empty() || lines[0] != key) {
        return false;
    }
    for (size_t i = 1; i < lines.size(); ++i) {
        std::vector<rec> r;
        if (parse_line(lines[i], tag, r)) {
            out = std::move(r);
            return true;
        }
    }
    return false;
}

// puts this tag's line into the calibration file (key on its first line; other lines, the calibration numbers and other tags, are
// kept), written to a temporary file and renamed into place; false if the file is missing or carries another key (nothing written)
inline bool write(const std::string & path, const std::string & key, const std::string & tag, const std::vector<rec> & r, long long stamp) {
    std::vector<std::string> lines;
    if (!read_lines(path, lines) || lines.empty() || lines[0] != key) {
        return false;
    }
    std::vector<rec> dummy;
    std::string body;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0 && parse_line(lines[i], tag, dummy)) {
            continue;
        }
        body += lines[i] + "\n";
    }
    body += format_line(tag, r) + "\n";
    const std::string tmp = path + ".tmp" + std::to_string(stamp);
    FILE * f = fopen(tmp.c_str(), "w");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fwrite(body.data(), 1, body.size(), f) == body.size();
    if (fclose(f) != 0 || !ok) {
        std::remove(tmp.c_str());
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

// the verdict of a timed trial. The first trial (no earlier verdict) or hyst <= 0: on if at least `margin` faster (rule).
// Later trials keep the previous verdict inside a band of +-hyst around the margin: on -> off only if the host path is no longer
// (margin - hyst) faster, off -> on only if it is (margin + hyst) faster
inline bool decide(double t_on, double t_off, size_t n_off, size_t n_on, bool have_prev, bool prev_on, double margin, double hyst) {
    if (n_off < 3 || n_on < 3 || !(t_off > 0.0)) {
        return false;
    }
    if (!have_prev || hyst <= 0.0) {
        return t_on < (1.0 - margin)*t_off;
    }
    return prev_on ? t_on < (1.0 - margin + hyst)*t_off : t_on < (1.0 - margin - hyst)*t_off;
}

// the t_on / t_off ratio at which decide switches (the same cases as decide)
inline double threshold(bool have_prev, bool prev_on, double margin, double hyst) {
    if (!have_prev || hyst <= 0.0) {
        return 1.0 - margin;
    }
    return prev_on ? 1.0 - margin + hyst : 1.0 - margin - hyst;
}

// a timed "on" is clear only if the host path was at least min_gain faster (t_on <= (1 - min_gain) t_off; min_gain <= 0:
// every "on" is clear, rule). One noisy launch-1 trial decided "on" by 4% on an A100 40 (37.3 vs 35.7 ms), was
// persisted, and every later launch replayed the host path at 70-77 ms per pass against 50-52 ms without it
inline bool marginal_on(bool on, double t_on, double t_off, double min_gain) {
    return on && min_gain > 0.0 && t_off > 0.0 && t_on > (1.0 - min_gain)*t_off;
}

// what a timed trial keeps. A clear timing is firm (replayed as is from then on), and so is a marginal "off" (the timing and
// the safe side agree). A marginal "on" keeps the host path off (the gain is inside the noise; the host path's loss when the timing was
// wrong is not) and stays open: the next launch times it again, a clear timing then decides, and after max_tries timings without one it
// is firm "off". So only a trial that timed "on" inside the band can change in a later launch, and only to a clear "on".
// old = the open record this launch timed again (nullptr: a new trial). Returns the verdict this launch takes; out gets on / firm / tries
// (the caller fills the rest)
// trust (the caller's scope, see trial_trust_marginal): a marginal "on" is taken in the launch that timed it and recorded as an open
// "on"; a later timing that is not "on" makes it firm "off" as before, and max_tries marginal "on" timings in a row make it firm "on"
// (repeated timings that agree are a small real gain, not noise)
inline bool settle(const rec * old, bool on, bool is_marginal, int max_tries, rec & out, bool trust = false) {
    out.tries = (old != nullptr ? old->tries : 0) + 1;
    if (!is_marginal || !on) {
        out.on   = on ? 1 : 0;
        out.firm = 1;
        return on;
    }
    out.firm = out.tries >= std::max(1, max_tries) ? 1 : 0;
    out.on   = trust ? 1 : 0;
    return trust;
}

// an on-by-default switch read from the environment: unset or a nonzero number is on, "0" is off
inline bool env_on(const char * v) {
    return v == nullptr || atoi(v) != 0;
}

// a host with at least LLAMA_MOE_SINGLE_WIDE_MIN_CPUS (default 32) logical CPUs (LLAMA_MOE_SINGLE_WIDE=0 off); e / em are the two values
inline bool wide_host(int n_cpus, const char * e, const char * em) {
    const int min_cpus = em != nullptr ? atoi(em) : 32;
    return !(e != nullptr && atoi(e) == 0) && n_cpus >= min_cpus;
}

// a host with LLAMA_MOE_SINGLE_WIDE_MID_CPUS (default 12) logical CPUs or more that is not a wide host (LLAMA_MOE_SINGLE_WIDE_MID=0 off),
// so by default 12..31 CPUs; e / em are the two values, wide = wide_host for this host
inline bool wide_mid_host(int n_cpus, bool wide, const char * e, const char * em) {
    const int min_cpus = em != nullptr ? atoi(em) : 12;
    return !(e != nullptr && atoi(e) == 0) && !wide && n_cpus >= min_cpus;
}

// how many marginal "on" timings in a row make a trusted record firm "on" (value of LLAMA_MOE_CPU_MISS_TRUST_TRIES, default 2, at least 1;
// 3 = as many as the untrusted firm "off" needs). An "off" timing makes the record firm "off" at any try, so this only decides how soon
// a launch stops re-timing a host path that timed faster each time. L4 (12 vCPU), Qwen3.6-35B-A3B 4-bit quant with its MTP head:
// the first trial timed 0.919-0.952 in each of 42 launches at 16 GB on 9 VMs (a third timing never changed the verdict), at 20 GB
// 0.955-0.983 with 4 "off" timings among 22; the settled launch against the trust-off arm over 12 texts: 16 GB chat +4%, 8k +8%,
// 20 GB level
inline int trust_tries(const char * e) {
    return e != nullptr ? std::max(1, atoi(e)) : 2;
}

// a first trial record made firm "on" by agreeing marginal timings: firm, on, timed by at least max_tries launches, and each "on" by less
// than min_gain (1 - min_gain < t_on / t_off < 1)
inline bool firm_marginal_on(const rec & r, int max_tries, double min_gain) {
    return r.firm != 0 && r.on != 0 && r.tries >= max_tries && r.ratio > 1.0 - min_gain && r.ratio < 1.0;
}

// whether a replayed trial keeps the host path off in its "on" blocks too: on = LLAMA_MOE_CPU_MISS_REPLAY_OFF_FETCH (env_on), r = the
// record the trial replays. A firm "off" record takes its recorded verdict whatever this launch times, so its two blocks with the host
// path on only cost time: those small batches fetch their misses as after the verdict (the device's products, as without a cache).
// Only where the recording launch also found the host slower than calibrated (host_ok 0: t_on at least 1.25 x t_off, the short prompts
// kept off too); a record that lost by less keeps its "on" blocks, as their host products steer what the later requests generate
inline bool replay_off_fetch(bool on, const rec & r) {
    return on && r.firm != 0 && r.on == 0 && r.host_ok == 0;
}

// whether a launch skips the warm phase and the timing trials on such a record: skip_on = LLAMA_MOE_CPU_MISS_FIRM_SKIP (env_on),
// trust_scope = the marginal "on" is trusted on this host (a 12..31-CPU mailbox host, LLAMA_MOE_CPU_MISS_MARGIN_TRUST); first_rec is
// called only when both hold and returns the first trial of the calibration file (nullptr: no file or no recorded trial)
template <typename F>
inline bool firm_skip(bool skip_on, bool trust_scope, F && first_rec, int max_tries, double min_gain) {
    if (!skip_on || !trust_scope) {
        return false;
    }
    const rec * r0 = first_rec();
    return r0 != nullptr && firm_marginal_on(*r0, max_tries, min_gain);
}

// the host path's time ratio t_on / t_off of a trial whose counted small batches have different widths (a draft verify window of 2..7
// rows costs 2-3x as much at 7 rows as at 2): t[a] / n[a] are the times and the token counts of the batches of arm a (0 = host path off,
// 1 = on). A median over all batches of an arm follows which widths fell in the arm's blocks, not the path, so the ratio is the
// geometric mean of the per-width median ratios over the widths both arms ran, each weighted by the fewer batches of the two arms;
// with fewer than min_matched matched batches, the ratio of the per-token medians (time / tokens). With one width (single token
// decode, a fixed draft width) this is the ratio of the two medians, as before. Returns 0 if an arm has no batch; *matched gets the
// matched weight (0 = the per-token fallback)
inline double width_ratio(const std::vector<int64_t> t[2], const std::vector<int64_t> n[2], int min_matched, int * matched = nullptr) {
    if (matched != nullptr) {
        *matched = 0;
    }
    if (t[0].empty() || t[1].empty() || t[0].size() != n[0].size() || t[1].size() != n[1].size()) {
        return 0.0;
    }
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size()/2];
    };
    std::vector<int64_t> ws(n[0].begin(), n[0].end());
    std::sort(ws.begin(), ws.end());
    ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
    double lsum = 0.0;
    int    wsum = 0;
    for (const int64_t w : ws) {
        std::vector<double> a[2];
        for (int k = 0; k < 2; ++k) {
            for (size_t i = 0; i < t[k].size(); ++i) {
                if (n[k][i] == w) {
                    a[k].push_back((double) t[k][i]);
                }
            }
        }
        if (a[1].empty()) {
            continue;
        }
        const double m0 = med(a[0]);
        const double m1 = med(a[1]);
        if (m0 <= 0.0 || m1 <= 0.0) {
            continue;
        }
        const int wt = (int) std::min(a[0].size(), a[1].size());
        lsum += wt*std::log(m1/m0);
        wsum += wt;
    }
    if (wsum >= std::max(1, min_matched)) {
        if (matched != nullptr) {
            *matched = wsum;
        }
        return std::exp(lsum/wsum);
    }
    std::vector<double> p[2];
    for (int k = 0; k < 2; ++k) {
        for (size_t i = 0; i < t[k].size(); ++i) {
            p[k].push_back((double) t[k][i]/(double) std::max<int64_t>(1, n[k][i]));
        }
    }
    const double m0 = med(p[0]);
    return m0 > 0.0 ? med(p[1])/m0 : 0.0;
}

} // namespace llama_moe_tp
