#pragma once

// k19-autoprof: a self-learned per-model profile (the MoE routing of the prompts and the PLE rows the gather reads), recorded while
// serving and kept beside the persisted CPU-miss calibration (k19-determinism / k19-cmtpersist: the same directory, a model key of
// the same kind as the k19-ptres core profile, written to a temporary file and renamed into place). A later launch of the same model
// orders the RAM tier's page-cache fill and admissions by it and makes the PLE row cache with it. Without a file every path is the
// one of today.
//
// LLAMA_MOE_AUTOPROF: unset or 1 = record and use, 0 = off (nothing read, written or hooked), 2 = record only, 3 = use only (never
// written). LLAMA_MOE_AUTOPROF_DIR: the directory (default: the CPU-miss calibration directory, see cal_dir). Without a recorded
// profile a shipped one is read (see shipped_dirs).
//
// Header-only and free of ggml so tests/test-autoprof.cpp can test it on the CPU.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace llama_autoprof {

// the persisted calibration's directory: LLAMA_MOE_CPU_MISS_CAL_DIR, else LLAMA_CACHE, else the user cache directory; "" if none
inline std::string cal_dir() {
    auto env = [](const char * n) { const char * e = getenv(n); return std::string(e != nullptr ? e : ""); };
    std::string d = env("LLAMA_MOE_CPU_MISS_CAL_DIR");
    if (d.empty()) {
        d = env("LLAMA_CACHE");
    }
#if defined(_WIN32)
    if (d.empty() && !env("LOCALAPPDATA").empty()) {
        d = env("LOCALAPPDATA") + "\\llama.cpp";
    }
#else
    if (d.empty() && !env("XDG_CACHE_HOME").empty()) {
        d = env("XDG_CACHE_HOME") + "/llama.cpp";
    }
    if (d.empty() && !env("HOME").empty()) {
        d = env("HOME") + "/.cache/llama.cpp";
    }
#endif
    return d;
}

// 0 off, 1 record + use, 2 record only, 3 use only
inline int mode() {
    const char * e = getenv("LLAMA_MOE_AUTOPROF");
    if (e == nullptr || *e == '\0') {
        return 1;
    }
    const int m = atoi(e);
    return m >= 0 && m <= 3 ? m : 1;
}
inline bool records(int m) { return m == 1 || m == 2; }
inline bool uses(int m)    { return m == 1 || m == 3; }

inline std::string dir() {
    const char * e = getenv("LLAMA_MOE_AUTOPROF_DIR");
    return e != nullptr && *e ? std::string(e) : cal_dir();
}

// a shipped profile (read only): the profile a model can ship, used by a launch that has no recorded one of its own yet (a first
// launch, a new user). Same file names as the recorded ones (autoprof_<kind>_<model hash><suffix>), looked up in
// LLAMA_MOE_AUTOPROF_SHIPPED when it names a directory, else beside the model file, then in <calibration directory>/shipped.
// LLAMA_MOE_AUTOPROF_SHIPPED=0: never looked up. A recording launch writes its own file (dir()) with the shipped counts carried over
// at the weight of an earlier launch; from then on the own file is the one read. Never written.
inline std::vector<std::string> shipped_dirs(const std::string & model_path) {
    std::vector<std::string> v;
    const char * e = getenv("LLAMA_MOE_AUTOPROF_SHIPPED");
    if (e != nullptr && *e) {
        if (std::string(e) != "0") {
            v.emplace_back(e);
        }
        return v;
    }
    if (!model_path.empty()) {
        const std::filesystem::path p(model_path);
        v.push_back(p.has_parent_path() ? p.parent_path().string() : std::string("."));
    }
    const std::string c = cal_dir();
    if (!c.empty()) {
        v.push_back((std::filesystem::path(c) / "shipped").string());
    }
    return v;
}

// FNV-1a 64 of the model's identity (as the k19-ptres core profile: description | parameters | bytes)
inline uint64_t model_hash(const std::string & desc, uint64_t n_elements, uint64_t size) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : desc + "|" + std::to_string(n_elements) + "|" + std::to_string(size)) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

// <dir>/autoprof_<kind>_<hash><suffix>; "" when there is no directory
inline std::string path(const std::string & d, const char * kind, uint64_t hash, const std::string & suffix) {
    if (d.empty()) {
        return "";
    }
    char name[96];
    snprintf(name, sizeof(name), "autoprof_%s_%016llx", kind, (unsigned long long) hash);
#if defined(_WIN32)
    return d + "\\" + name + suffix;
#else
    return d + "/" + name + suffix;
#endif
}

inline long long pid() {
#if defined(_WIN32)
    return (long long) _getpid();
#else
    return (long long) getpid();
#endif
}

// writes `body` to path through a temporary of this process (created exclusively) that is renamed into place, so a reader sees the
// old file or the new one, never a part; creates the directory. False (the old file untouched, no temporary left) on any failure
inline bool write_atomic(const std::string & path, const std::string & body) {
    if (path.empty()) {
        return false;
    }
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    static std::atomic<int> seq{0};
    const std::string tmp = path + ".tmp." + std::to_string(pid()) + "." + std::to_string(seq++);
    FILE * f = fopen(tmp.c_str(), "wbx");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fwrite(body.data(), 1, body.size(), f) == body.size();
    if (fclose(f) != 0 || !ok) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::error_code ec2;
        std::filesystem::remove(tmp, ec2);
        return false;
    }
    return true;
}

// the first existing shipped profile of this kind (see shipped_dirs), "" if none
inline std::string find_shipped(const std::string & model_path, const char * kind, uint64_t hash, const std::string & suffix) {
    for (const std::string & d : shipped_dirs(model_path)) {
        const std::string p = path(d, kind, hash, suffix);
        std::error_code ec;
        if (!p.empty() && std::filesystem::is_regular_file(p, ec)) {
            return p;
        }
    }
    return "";
}

} // namespace llama_autoprof
