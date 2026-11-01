// the SSD tier of the server prompt cache (server_prompt_ssd):
//   - the same prompt evicted several times before the writer finished the first copy keeps one entry whose file exists
//   - a corrupt entry (an untrusted token count of 2^62 + 16) is skipped and deleted at startup instead of aborting the server

#include "../tools/server/server-prompt-ssd.h"
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static int g_fail = 0;

static void check(const char * name, bool ok) {
    printf("%-64s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail = 1;
    }
}

static server_prompt_cache_state make_state(const llama_tokens & tokens, size_t bytes) {
    server_prompt_cache_state s;
    s.prompt.tokens = server_tokens(tokens, false);
    s.data.main.resize(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        s.data.main[i] = (uint8_t) (i*31 + tokens.size());
    }
    return s;
}

int main() {
    const fs::path root = fs::temp_directory_path() / ("test-server-prompt-ssd-" + std::to_string((long long) getpid()));
    std::error_code ec;
    fs::remove_all(root, ec);

    server_prompt_ssd_params params;
    // UTF-8, as the server passes it (a narrow .string() is the ANSI code page on Windows: a non-ASCII %TEMP% would not round trip)
    params.dir = fs_path_to_utf8(root);

    // the same prompt put three times in a row: the writer is still writing (or has not started) the first copy
    for (int iter = 0; iter < 4; ++iter) {
        llama_tokens tokens(300 + iter);
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = (llama_token) (i*7 + iter);
        }
        server_prompt_ssd ssd(params, "test namespace " + std::to_string(iter), false);
        for (int k = 0; k < 3; ++k) {
            ssd.put(make_state(tokens, 8u << 20));
        }
        ssd.flush();

        server_prompt_ssd::entry e;
        const int lcp = ssd.find(server_tokens(tokens, false), e);
        server_prompt_cache_state out;
        const bool found  = lcp == (int) tokens.size();
        const bool exists = found && fs::exists(fs::u8path(e.path), ec);
        const bool read   = exists && ssd.read(e, out) && out.data.main.size() == (8u << 20);
        char name[96];
        snprintf(name, sizeof(name), "duplicate puts (run %d): 1 entry, file present, readable", iter);
        check(name, ssd.count() == 1 && found && exists && read);
    }

    // a corrupt entry with a token count that overflows the size check
    {
        std::string ns_dir;
        {
            server_prompt_ssd ssd(params, "corrupt namespace", false);
            ns_dir = ssd.dir_ns();
        }
        const fs::path bad = fs::u8path(ns_dir) / "0000000000000000.lpc";
        {
            // magic "LPSC", version 1, the namespace hash (the directory name), n_tokens = 2^62 + 16, then 128 bytes
            const uint32_t magic    = 0x4353504c;
            const uint32_t version  = 1;
            const uint64_t ns       = std::stoull(fs::u8path(ns_dir).filename().string(), nullptr, 16);
            const uint64_t n_tokens = (1ull << 62) + 16;
            std::vector<uint8_t> buf(sizeof(magic) + sizeof(version) + sizeof(ns) + sizeof(n_tokens) + 128, 0);
            uint8_t * p = buf.data();
            memcpy(p, &magic, sizeof(magic));       p += sizeof(magic);
            memcpy(p, &version, sizeof(version));   p += sizeof(version);
            memcpy(p, &ns, sizeof(ns));             p += sizeof(ns);
            memcpy(p, &n_tokens, sizeof(n_tokens));
#ifdef _WIN32
            FILE * f = _wfopen(bad.wstring().c_str(), L"wb");
#else
            FILE * f = fopen(bad.string().c_str(), "wb");
#endif
            if (f == nullptr) {
                fprintf(stderr, "cannot create %s\n", fs_path_to_utf8(bad).c_str());
                return 1;
            }
            fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
        }
        bool threw = false;
        size_t n = 1;
        try {
            server_prompt_ssd ssd(params, "corrupt namespace", false);
            n = ssd.count();
        } catch (const std::exception & err) {
            fprintf(stderr, "scan threw: %s\n", err.what());
            threw = true;
        }
        check("corrupt entry: startup continues, entry skipped and deleted", !threw && n == 0 && !fs::exists(bad, ec));
    }

    fs::remove_all(root, ec);
    return g_fail;
}
