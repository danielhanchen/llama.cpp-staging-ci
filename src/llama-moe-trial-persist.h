#pragma once

// k19-cmtpersist: the MoE cache's CPU-miss timing trials, persisted (LLAMA_MOE_CPU_MISS_PERSIST, default 1; =0 off).
//
// A trial times small batches with and without the host path and keeps the path if it is at least cmt_margin faster. Where the
// host and the upload cost about the same (Kaggle T4, 4 vCPU, Qwen3.6-35B-A3B UD-Q4_K_XL 12 GB: 155 vs 188 us per expert) the first
// trial decided "off" in one launch (-2.3%) and "on" in the next (-3.6%), and the two launches wrote other texts at temperature 0
// (the host computes misses it does not fetch, so the cache, the route-tail skip and which rows run on the CPU all diverge).
//
// The verdict of every trial of a launch (on / off, the short-prompt flag, the gap to the next trial, how many small batches the trial
// ran, and the measured ratio t_on / t_off) is kept as one line in the calibration file of k19-determinism, under a tag naming the cache
// size (expert slots), the layer count and the small-batch limit; the file's key already names the build, the device, the threads, the
// CPU count, the cache (main / MTP) and the expert form. A later launch with the same file and tag replays the recorded trials in order:
// it runs each trial's schedule (the same blocks with and without the host path, so the cache is admitted to exactly as in the first
// launch) and takes the recorded verdict instead of its own timing. Trials beyond the record are timed, with hysteresis against the
// previous verdict, and appended, so the next launch replays them too.
//
// Header-only and free of ggml so tests/test-moe-trial-persist.cpp can test it on the CPU.

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
};

inline std::string tag_of(long long n_slots, int n_layers, long long max_batch) {
    return "trials|slots=" + std::to_string(n_slots) + "|layers=" + std::to_string(n_layers) + "|max_batch=" + std::to_string(max_batch);
}

inline std::string format_line(const std::string & tag, const std::vector<rec> & r) {
    std::string s = tag + " " + std::to_string(r.size());
    char buf[96];
    for (const rec & x : r) {
        snprintf(buf, sizeof(buf), " %d:%d:%d:%d:%.6f", x.on ? 1 : 0, x.host_ok ? 1 : 0, x.gap, x.len, x.ratio);
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

// the verdict of a timed trial. The first trial (no earlier verdict) or hyst <= 0: on if at least `margin` faster (the k10 rule).
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

} // namespace llama_moe_tp
