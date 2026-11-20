// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

// llama-build-id.h: an identity of the running build for keys of persisted host measurements (the MoE CPU-miss calibration).
// ggml_commit() names the build when the tree had git metadata; a build without it (a source tarball, `git archive`, a vendored copy)
// reports "unknown", and every such build would share one key, so an upgraded binary would reuse an older build's numbers. Then the
// identity also covers the content of the loaded modules (libllama and the CPU backend: FNV-1a 64 over the file, computed once per
// process) and the compiler. A "-dirty" commit gets the same treatment (local edits change the code but not the commit).
// Header-only so tests/test-cal-build-id.cpp checks the same code.

#include <cstdint>
#include <cstdio>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// the path of the module (shared library or executable) that contains `addr`; "" when the platform cannot tell
static inline std::string llama_build_id_module_path(const void * addr) {
    if (addr == nullptr) {
        return "";
    }
#if defined(_WIN32)
    HMODULE h = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) addr, &h)) {
        return "";
    }
    char buf[MAX_PATH * 2];
    DWORD n = GetModuleFileNameA(h, buf, (DWORD) sizeof(buf));
    return n > 0 && n < sizeof(buf) ? std::string(buf, n) : std::string();
#else
    Dl_info info;
    if (dladdr(addr, &info) == 0 || info.dli_fname == nullptr) {
        return "";
    }
    return info.dli_fname;
#endif
}

// FNV-1a 64 over the file's bytes and its size; 0 when it cannot be read
static inline uint64_t llama_build_id_file_hash(const std::string & path) {
    if (path.empty()) {
        return 0;
    }
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return 0;
    }
    uint64_t h = 1469598103934665603ull;
    uint64_t n = 0;
    unsigned char buf[1 << 16];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < r; ++i) {
            h ^= buf[i];
            h *= 1099511628211ull;
        }
        n += r;
    }
    fclose(f);
    return h ^ (n * 0x9e3779b97f4a7c15ull);
}

static inline const char * llama_build_id_compiler() {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#elif defined(_MSC_VER)
#define LLAMA_BUILD_ID_STR2(x) #x
#define LLAMA_BUILD_ID_STR(x) LLAMA_BUILD_ID_STR2(x)
    return "msvc " LLAMA_BUILD_ID_STR(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

// commit known and clean: the commit; else the commit plus the module content hashes (path -> hash) and the compiler
static inline std::string llama_build_id(const std::string & commit, const std::string * module_paths, int n_modules) {
    const bool unknown = commit.empty() || commit == "unknown" ||
                         (commit.size() >= 6 && commit.compare(commit.size() - 6, 6, "-dirty") == 0);
    if (!unknown) {
        return commit;
    }
    std::string id = commit.empty() ? std::string("unknown") : commit;
    for (int i = 0; i < n_modules; ++i) {
        char hex[32];
        snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) llama_build_id_file_hash(module_paths[i]));
        id += std::string("|m") + std::to_string(i) + "=" + hex;
    }
    id += std::string("|cc=") + llama_build_id_compiler();
    return id;
}
