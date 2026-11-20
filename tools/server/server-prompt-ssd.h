// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

#include "server-task.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// SSD tier for the RAM prompt cache (--cache-ssd-dir)
//
// entries evicted from the RAM cache are written by one background thread to <dir>/<namespace>/, one file per prompt
// (state blobs + context checkpoints), via a temporary file, fsync and rename. the namespace is a hash of everything that
// changes the meaning of a saved state (model file contents, KV types, context size, rope, adapters, draft model, state
// version), so a different model or configuration never sees the files of another one. a lookup compares the full token
// list before any state is restored; a file that fails its checksum or does not parse is a miss and is deleted.

struct server_prompt_ssd_params {
    std::string dir;              // root directory, the namespace subdirectory is created inside
    size_t      limit_size = 0;   // bytes, 0 = no limit
    bool        readonly   = false;
};

struct server_prompt_ssd {
    struct entry {
        std::string  path;
        llama_tokens tokens;
        uint64_t     key       = 0; // chained hash of the token blocks, also the file name
        size_t       size      = 0; // file size
        int64_t      last_used = 0; // for LRU eviction, from the file mtime at scan time
    };

    server_prompt_ssd(const server_prompt_ssd_params & params, const std::string & ns_desc, bool has_mtmd);
    ~server_prompt_ssd();

    // hand over a state evicted from the RAM cache; it is written in the background (or dropped if it cannot be saved)
    void put(server_prompt_cache_state && state);

    // wait until every queued state is written
    void flush();

    // the cached prompt with the longest common prefix with tokens_new, returns -1 if none
    // the entry is copied out, so the index may change afterwards
    int find(const server_tokens & tokens_new, entry & out) const;

    // read and validate an entry; on failure the file is deleted (unless read-only) and false is returned
    bool read(const entry & e, server_prompt_cache_state & out);

    // an entry that was read but could not be restored (incompatible state): removed like a corrupt one
    void drop(const entry & e);

    size_t size() const;
    size_t count() const;

    const std::string & dir_ns() const { return dir; }

    // chained hash of the 256-token blocks of a token list (complete blocks + the tail), namespace hash first
    static uint64_t token_key(uint64_t ns_hash, const llama_tokens & tokens);

    static constexpr int BLOCK_TOKENS = 256;

    // the entry format the reader accepts (written in every header and in the namespace)
    static constexpr uint32_t FORMAT_VERSION = 2;

private:
    server_prompt_ssd_params params;

    std::string dir;     // <root>/<namespace hex>
    uint64_t    ns_hash = 0;
    bool        has_mtmd = false;

    mutable std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable cv_done;

    std::vector<entry> index;  // guarded by mtx

    std::deque<server_prompt_cache_state> queue; // guarded by mtx
    size_t queue_bytes = 0;
    bool   writing     = false;
    bool   stop        = false;

    uint64_t n_written = 0;
    uint64_t n_tmp     = 0;

    std::thread worker;

    void scan();
    void run();
    bool write(const server_prompt_cache_state & state);
    void evict_locked();
    void remove_locked(size_t i);
};

// hash of the parts of the configuration that the saved state depends on (see the comment above)
std::string server_prompt_ssd_namespace(
        const common_params & params,
        const llama_model * model_tgt,
        const llama_context * ctx_tgt,
        const llama_model * model_dft,
        const llama_context * ctx_dft);
