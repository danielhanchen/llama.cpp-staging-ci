#include "server-prompt-ssd.h"
#include "server-common.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <regex>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static constexpr uint32_t SSD_MAGIC     = 0x4353504c; // "LPSC"
static constexpr uint32_t SSD_MAGIC_END = 0x444e4550; // "PEND"
static constexpr uint32_t SSD_VERSION   = 1;

// a fast non-cryptographic 64-bit hash: four lanes of 8-byte words, enough to detect truncation and bit rot
struct ssd_hasher {
    uint64_t h[4] = { 0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull, 0x27D4EB2F165667C5ull };
    uint64_t n    = 0;
    uint8_t  tail[32];
    size_t   n_tail = 0;

    static uint64_t mix(uint64_t a, uint64_t w) {
        a ^= w * 0x9E3779B97F4A7C15ull;
        a  = (a << 31) | (a >> 33);
        return a * 0xBF58476D1CE4E5B9ull;
    }

    void block(const uint8_t * p) {
        uint64_t w[4];
        memcpy(w, p, 32);
        h[0] = mix(h[0], w[0]);
        h[1] = mix(h[1], w[1]);
        h[2] = mix(h[2], w[2]);
        h[3] = mix(h[3], w[3]);
    }

    void update(const void * data, size_t size) {
        const uint8_t * p = (const uint8_t *) data;
        n += size;
        if (n_tail > 0) {
            const size_t k = std::min(size, 32 - n_tail);
            memcpy(tail + n_tail, p, k);
            n_tail += k;
            p      += k;
            size   -= k;
            if (n_tail < 32) {
                return;
            }
            block(tail);
            n_tail = 0;
        }
        while (size >= 32) {
            block(p);
            p    += 32;
            size -= 32;
        }
        memcpy(tail, p, size);
        n_tail = size;
    }

    uint64_t digest() const {
        uint64_t r = n;
        for (int i = 0; i < 4; ++i) {
            r = mix(r, h[i]);
        }
        for (size_t i = 0; i < n_tail; ++i) {
            r = mix(r, tail[i]);
        }
        r ^= r >> 29;
        r *= 0x94D049BB133111EBull;
        r ^= r >> 32;
        return r;
    }
};

static uint64_t hash_str(const std::string & s) {
    ssd_hasher hs;
    hs.update(s.data(), s.size());
    return hs.digest();
}

uint64_t server_prompt_ssd::token_key(uint64_t ns_hash, const llama_tokens & tokens) {
    // H_i = H(H_{i-1}, block_i), H_0 = namespace: the key of a prompt is the key of its last (possibly partial) block
    uint64_t h = ns_hash;
    for (size_t i = 0; i < tokens.size(); i += BLOCK_TOKENS) {
        const size_t n = std::min<size_t>(BLOCK_TOKENS, tokens.size() - i);
        ssd_hasher hs;
        hs.update(&h, sizeof(h));
        hs.update(tokens.data() + i, n*sizeof(llama_token));
        h = hs.digest();
    }
    return h;
}

static int64_t file_mtime_ms(const fs::path & p) {
    std::error_code ec;
    const auto t = fs::last_write_time(p, ec);
    if (ec) {
        return 0;
    }
    // file_time_type has an unspecified epoch: only the order matters for LRU
    return std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count();
}

//
// file io: plain buffered writes from heap memory (never into a shared mapping), checksummed
//

struct ssd_writer {
    FILE * f = nullptr;
    ssd_hasher hs;
    bool ok = true;

    void put(const void * p, size_t n) {
        if (!ok || n == 0) {
            return;
        }
        if (fwrite(p, 1, n, f) != n) {
            ok = false;
            return;
        }
        hs.update(p, n);
    }

    template <typename T>
    void put_val(const T & v) { put(&v, sizeof(v)); }

    template <typename Buf>
    void put_buf(const Buf & b) {
        const uint64_t n = b.size();
        put_val(n);
        put(b.data(), b.size());
    }
};

struct ssd_reader {
    FILE * f = nullptr;
    ssd_hasher hs;
    uint64_t n_left = 0; // bytes left in the file
    bool ok = true;

    bool get(void * p, size_t n) {
        if (!ok || n > n_left || (n > 0 && fread(p, 1, n, f) != n)) {
            ok = false;
            return false;
        }
        hs.update(p, n);
        n_left -= n;
        return true;
    }

    template <typename T>
    bool get_val(T & v) { return get(&v, sizeof(v)); }

    template <typename Buf>
    bool get_buf(Buf & b) {
        uint64_t n = 0;
        // the length field may be corrupted: check it against the bytes left before allocating
        if (!get_val(n) || n > n_left) {
            ok = false;
            return false;
        }
        b.resize(n);
        return get(b.data(), n);
    }
};

// header: magic, version, namespace, n_tokens, tokens
static bool read_header(ssd_reader & r, uint64_t ns_hash, llama_tokens & tokens) {
    uint32_t magic = 0, version = 0;
    uint64_t ns = 0, n_tokens = 0;
    if (!r.get_val(magic) || magic != SSD_MAGIC ||
        !r.get_val(version) || version != SSD_VERSION ||
        !r.get_val(ns) || ns != ns_hash ||
        !r.get_val(n_tokens) || n_tokens == 0 || n_tokens*sizeof(llama_token) > r.n_left) {
        return false;
    }
    tokens.resize(n_tokens);
    return r.get(tokens.data(), n_tokens*sizeof(llama_token));
}

static FILE * open_file(const fs::path & p, const char * mode) {
#ifdef _WIN32
    return _wfopen(p.wstring().c_str(), mode[0] == 'r' ? L"rb" : L"wb");
#else
    return fopen(p.c_str(), mode);
#endif
}

static bool sync_file(FILE * f) {
    if (fflush(f) != 0) {
        return false;
    }
#ifdef _WIN32
    return _commit(_fileno(f)) == 0;
#else
    return fsync(fileno(f)) == 0;
#endif
}

//
// server_prompt_ssd
//

server_prompt_ssd::server_prompt_ssd(const server_prompt_ssd_params & params, const std::string & ns_desc, bool has_mtmd)
    : params(params), has_mtmd(has_mtmd) {
    ns_hash = hash_str(ns_desc);

    char name[32];
    snprintf(name, sizeof(name), "%016" PRIx64, ns_hash);
    dir = (fs::u8path(params.dir) / name).string();

    std::error_code ec;
    if (!params.readonly) {
        fs::create_directories(fs::u8path(dir), ec);
        if (ec) {
            SRV_WRN("prompt cache: cannot create '%s': %s - the SSD tier is read-only\n", dir.c_str(), ec.message().c_str());
            this->params.readonly = true;
        } else {
            // the description of the namespace, for whoever looks at the directory
            FILE * f = open_file(fs::u8path(dir) / "namespace.txt", "wb");
            if (f) {
                fwrite(ns_desc.data(), 1, ns_desc.size(), f);
                fclose(f);
            }
        }
    }

    scan();

    SRV_INF("prompt cache: SSD tier '%s', %zu prompts, %.3f MiB (limit %.3f MiB%s)\n",
            dir.c_str(), index.size(), size() / (1024.0*1024.0), this->params.limit_size / (1024.0*1024.0),
            this->params.readonly ? ", read-only" : "");

    if (!this->params.readonly) {
        worker = std::thread([this]() { run(); });
    }
}

server_prompt_ssd::~server_prompt_ssd() {
    flush();
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void server_prompt_ssd::scan() {
    std::error_code ec;
    fs::directory_iterator it(fs::u8path(dir), ec);
    if (ec) {
        return;
    }
    for (const auto & de : it) {
        const fs::path p = de.path();
        const std::string fname = p.filename().string();
        if (fname.rfind(".tmp", 0) == 0) {
            // left over by a writer that did not finish
            if (!params.readonly) {
                fs::remove(p, ec);
            }
            continue;
        }
        if (p.extension() != ".lpc") {
            continue;
        }
        entry e;
        e.path = p.string();
        e.size = (size_t) fs::file_size(p, ec);
        e.last_used = file_mtime_ms(p);

        bool ok = false;
        if (FILE * f = open_file(p, "rb")) {
            ssd_reader r;
            r.f = f;
            r.n_left = e.size;
            ok = read_header(r, ns_hash, e.tokens);
            fclose(f);
        }
        if (ok) {
            e.key = token_key(ns_hash, e.tokens);
            char name[32];
            snprintf(name, sizeof(name), "%016" PRIx64 ".lpc", e.key);
            ok = fname == name;
        }
        if (!ok) {
            SRV_WRN("prompt cache: '%s' is not a valid entry%s\n", e.path.c_str(), params.readonly ? "" : " - deleted");
            if (!params.readonly) {
                fs::remove(p, ec);
            }
            continue;
        }
        index.push_back(std::move(e));
    }
}

size_t server_prompt_ssd::size() const {
    std::lock_guard<std::mutex> lock(mtx);
    size_t res = 0;
    for (const auto & e : index) {
        res += e.size;
    }
    return res;
}

size_t server_prompt_ssd::count() const {
    std::lock_guard<std::mutex> lock(mtx);
    return index.size();
}

static bool is_prefix(const llama_tokens & a, const llama_tokens & b) {
    // a is a prefix of b
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// the token ids of a prompt (get_tokens() is not allowed on a multimodal context, media positions are LLAMA_TOKEN_NULL)
static llama_tokens token_ids(const server_tokens & t) {
    llama_tokens res(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        res[i] = t[i];
    }
    return res;
}

void server_prompt_ssd::put(server_prompt_cache_state && state) {
    if (params.readonly) {
        return;
    }
    const llama_tokens tokens = token_ids(state.prompt.tokens);
    if (tokens.empty() || state.data.main.empty()) {
        return;
    }
    // a state that went through a KV shift (context shift, cache reuse) does not match its token list position by
    // position, and a prompt with media has no token ids for the media: neither can be content-addressed
    if (state.prompt.kv_shifted) {
        SRV_TRC("%s", "prompt cache: not saving a KV-shifted prompt to the SSD tier\n");
        return;
    }
    if (std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end()) {
        SRV_TRC("%s", "prompt cache: not saving a prompt with media to the SSD tier\n");
        return;
    }

    const size_t bytes = state.size();
    // bound the memory held by queued states: at most the SSD limit and at most 4 GiB
    const size_t queue_limit = std::min<size_t>(params.limit_size > 0 ? params.limit_size : SIZE_MAX, 4ull << 30);

    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto & e : index) {
            if (is_prefix(tokens, e.tokens)) {
                SRV_TRC("%s", "prompt cache: prompt is already in the SSD tier\n");
                return;
            }
        }
        if (params.limit_size > 0 && bytes > params.limit_size) {
            SRV_WRN("prompt cache: state of %.3f MiB exceeds the SSD limit - not saved\n", bytes / (1024.0*1024.0));
            return;
        }
        if (queue_bytes + bytes > queue_limit) {
            SRV_WRN("prompt cache: SSD writer is behind (%.3f MiB queued) - dropping a %.3f MiB state\n",
                    queue_bytes / (1024.0*1024.0), bytes / (1024.0*1024.0));
            return;
        }
        queue_bytes += bytes;
        queue.push_back(std::move(state));
    }
    cv.notify_one();
}

void server_prompt_ssd::flush() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_done.wait(lock, [this]() { return (queue.empty() && !writing) || !worker.joinable(); });
}

void server_prompt_ssd::run() {
    while (true) {
        server_prompt_cache_state state;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return stop || !queue.empty(); });
            if (queue.empty()) {
                break; // stop
            }
            state = std::move(queue.front());
            queue.pop_front();
            writing = true;
        }

        const size_t bytes = state.size();
        const int64_t t0 = ggml_time_us();
        const bool ok = write(state);

        if (ok) {
            SRV_INF("prompt cache: saved %d tokens (%.3f MiB) to the SSD tier in %.1f ms\n",
                    state.prompt.n_tokens(), bytes / (1024.0*1024.0), (ggml_time_us() - t0) / 1000.0);
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            queue_bytes -= bytes;
            writing = false;
        }
        cv_done.notify_all();
    }
}

bool server_prompt_ssd::write(const server_prompt_cache_state & state) {
    const llama_tokens tokens = token_ids(state.prompt.tokens);
    const uint64_t key = token_key(ns_hash, tokens);

    char name[32];
    snprintf(name, sizeof(name), "%016" PRIx64 ".lpc", key);
    const fs::path path = fs::u8path(dir) / name;

    char tmp_name[64];
    snprintf(tmp_name, sizeof(tmp_name), ".tmp.%016" PRIx64 ".%" PRIu64, key, n_tmp++);
    const fs::path tmp = fs::u8path(dir) / tmp_name;

    FILE * f = open_file(tmp, "wb");
    if (!f) {
        SRV_WRN("prompt cache: cannot create '%s'\n", tmp.string().c_str());
        return false;
    }

    ssd_writer w;
    w.f = f;
    w.put_val(SSD_MAGIC);
    w.put_val(SSD_VERSION);
    w.put_val(ns_hash);
    w.put_val((uint64_t) tokens.size());
    w.put(tokens.data(), tokens.size()*sizeof(llama_token));
    w.put_buf(state.data.main);
    w.put_buf(state.data.drft);
    w.put_val((uint32_t) state.prompt.checkpoints.size());
    for (const auto & c : state.prompt.checkpoints) {
        w.put_val(c.n_tokens);
        w.put_val(c.pos_min);
        w.put_val(c.pos_max);
        w.put_buf(c.data_tgt);
        w.put_buf(c.data_dft);
        w.put_buf(c.data_spec);
    }
    const uint64_t digest = w.hs.digest();
    if (w.ok) {
        w.ok = fwrite(&digest, 1, sizeof(digest), f) == sizeof(digest) &&
               fwrite(&SSD_MAGIC_END, 1, sizeof(SSD_MAGIC_END), f) == sizeof(SSD_MAGIC_END);
    }
    const bool synced = w.ok && sync_file(f);
    fclose(f);

    std::error_code ec;
    if (!synced) {
        SRV_WRN("prompt cache: failed to write '%s' (disk full?)\n", tmp.string().c_str());
        fs::remove(tmp, ec);
        return false;
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        SRV_WRN("prompt cache: failed to rename '%s': %s\n", tmp.string().c_str(), ec.message().c_str());
        fs::remove(tmp, ec);
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx);

    // an entry that is a prefix of the new one is obsolete: the new one restores everything it did and more
    for (size_t i = 0; i < index.size();) {
        if (index[i].path != path.string() && is_prefix(index[i].tokens, tokens)) {
            remove_locked(i);
        } else {
            ++i;
        }
    }
    for (size_t i = 0; i < index.size(); ++i) {
        if (index[i].path == path.string()) {
            remove_locked(i); // replaced by the file just renamed over it
            break;
        }
    }

    entry e;
    e.path      = path.string();
    e.tokens    = tokens;
    e.key       = key;
    e.size      = (size_t) fs::file_size(path, ec);
    e.last_used = file_mtime_ms(path);
    index.push_back(std::move(e));

    evict_locked();

    n_written++;
    return true;
}

void server_prompt_ssd::remove_locked(size_t i) {
    std::error_code ec;
    if (!params.readonly) {
        fs::remove(fs::u8path(index[i].path), ec);
    }
    index.erase(index.begin() + i);
}

void server_prompt_ssd::evict_locked() {
    if (params.limit_size == 0) {
        return;
    }
    while (index.size() > 1) {
        size_t total = 0;
        size_t i_lru = 0;
        for (size_t i = 0; i < index.size(); ++i) {
            total += index[i].size;
            if (index[i].last_used < index[i_lru].last_used) {
                i_lru = i;
            }
        }
        if (total <= params.limit_size) {
            break;
        }
        SRV_INF("prompt cache: SSD limit reached, removing '%s' (%.3f MiB)\n", index[i_lru].path.c_str(), index[i_lru].size / (1024.0*1024.0));
        remove_locked(i_lru);
    }
}

int server_prompt_ssd::find(const server_tokens & tokens_new, entry & out) const {
    std::lock_guard<std::mutex> lock(mtx);

    int lcp_best = -1;
    const entry * best = nullptr;

    const size_t n_new = tokens_new.size();
    for (const auto & e : index) {
        size_t lcp = 0;
        const size_t n = std::min(n_new, e.tokens.size());
        while (lcp < n && e.tokens[lcp] == tokens_new[lcp] && tokens_new[lcp] != LLAMA_TOKEN_NULL) {
            lcp++;
        }
        // prefer the longest common prefix, then the entry that keeps more of itself
        if ((int) lcp > lcp_best || ((int) lcp == lcp_best && best && e.tokens.size() < best->tokens.size())) {
            lcp_best = (int) lcp;
            best = &e;
        }
    }

    if (best == nullptr || lcp_best <= 0) {
        return -1;
    }
    out = *best;
    return lcp_best;
}

bool server_prompt_ssd::read(const entry & e, server_prompt_cache_state & out) {
    const fs::path p = fs::u8path(e.path);

    std::error_code ec;
    const size_t file_size = (size_t) fs::file_size(p, ec);

    bool ok = false;
    llama_tokens tokens;
    if (FILE * f = (ec ? nullptr : open_file(p, "rb"))) {
        ssd_reader r;
        r.f = f;
        r.n_left = file_size;
        ok = read_header(r, ns_hash, tokens) && tokens == e.tokens;

        uint32_t n_ckpt = 0;
        ok = ok && r.get_buf(out.data.main) && !out.data.main.empty() && r.get_buf(out.data.drft) && r.get_val(n_ckpt);
        std::list<common_prompt_checkpoint> checkpoints;
        for (uint32_t i = 0; ok && i < n_ckpt; ++i) {
            common_prompt_checkpoint c;
            c.id_task = -1;
            ok = r.get_val(c.n_tokens) && r.get_val(c.pos_min) && r.get_val(c.pos_max) &&
                 r.get_buf(c.data_tgt) && r.get_buf(c.data_dft) && r.get_buf(c.data_spec) && !c.data_tgt.empty();
            if (ok) {
                checkpoints.push_back(std::move(c));
            }
        }
        const uint64_t digest = r.hs.digest();
        uint64_t digest_file = 0;
        uint32_t magic_end = 0;
        ok = ok && r.get_val(digest_file) && r.get_val(magic_end) &&
             digest_file == digest && magic_end == SSD_MAGIC_END && r.n_left == 0;
        fclose(f);

        if (ok) {
            out.prompt.tokens      = server_tokens(tokens, has_mtmd);
            out.prompt.checkpoints = std::move(checkpoints);
            out.prompt.kv_shifted  = false;
        }
    }

    std::lock_guard<std::mutex> lock(mtx);
    for (size_t i = 0; i < index.size(); ++i) {
        if (index[i].path != e.path) {
            continue;
        }
        if (!ok) {
            SRV_WRN("prompt cache: '%s' is corrupt or truncated%s\n", e.path.c_str(), params.readonly ? "" : " - deleted");
            remove_locked(i);
        } else {
            // a hit is the most recently used entry; the mtime carries that across restarts
            if (!params.readonly) {
                fs::last_write_time(p, fs::file_time_type::clock::now(), ec);
            }
            index[i].last_used = std::max(file_mtime_ms(p), index[i].last_used + 1);
        }
        break;
    }
    if (!ok) {
        out = server_prompt_cache_state();
    }
    return ok;
}

void server_prompt_ssd::drop(const entry & e) {
    std::lock_guard<std::mutex> lock(mtx);
    for (size_t i = 0; i < index.size(); ++i) {
        if (index[i].path == e.path) {
            SRV_WRN("prompt cache: '%s' cannot be restored%s\n", e.path.c_str(), params.readonly ? "" : " - deleted");
            remove_locked(i);
            break;
        }
    }
}

//
// namespace
//

// identity of a model file from its contents: size, the first 16 MiB (header, metadata, tensor infos) and samples of the
// tensor data, so a different file with the same name (or the same file under another name) is recognised
static std::string file_identity(const std::string & path) {
    std::error_code ec;
    const fs::path p = fs::u8path(path);
    const uint64_t size = fs::file_size(p, ec);
    if (ec) {
        return "missing";
    }
    ssd_hasher hs;
    hs.update(&size, sizeof(size));
    if (FILE * f = open_file(p, "rb")) {
        std::vector<uint8_t> buf(16u << 20);
        size_t n = fread(buf.data(), 1, buf.size(), f);
        hs.update(buf.data(), n);
        const int n_samples = 32;
        const size_t sample = 64u << 10;
        buf.resize(sample);
        for (int i = 0; i < n_samples && size > (16u << 20) + sample; ++i) {
            const uint64_t off = (16u << 20) + (size - (16u << 20) - sample) / (n_samples - 1) * i;
#ifdef _WIN32
            _fseeki64(f, (long long) off, SEEK_SET);
#else
            fseeko(f, (off_t) off, SEEK_SET);
#endif
            n = fread(buf.data(), 1, sample, f);
            hs.update(&off, sizeof(off));
            hs.update(buf.data(), n);
        }
        fclose(f);
    }
    char out[32];
    snprintf(out, sizeof(out), "%016" PRIx64, hs.digest());
    return out;
}

// all the shards of a split model, in order
static std::vector<std::string> model_files(const std::string & path) {
    static const std::regex re("-(\\d{5})-of-(\\d{5})\\.gguf$");
    std::smatch m;
    if (!std::regex_search(path, m, re)) {
        return { path };
    }
    const int n_split = std::stoi(m[2].str());
    const std::string prefix = path.substr(0, m.position(0));
    std::vector<std::string> res;
    for (int i = 0; i < n_split; ++i) {
        char buf[4096];
        llama_split_path(buf, sizeof(buf), prefix.c_str(), i, n_split);
        res.emplace_back(buf);
    }
    return res;
}

static std::string model_desc(const llama_model * model) {
    char buf[256] = {};
    llama_model_desc(model, buf, sizeof(buf));
    char arch[128] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    return string_format("%s|%s|%" PRIu64 "|%" PRIu64, arch, buf, llama_model_size(model), llama_model_n_params(model));
}

std::string server_prompt_ssd_namespace(
        const common_params & params,
        const llama_model * model_tgt,
        const llama_context * ctx_tgt,
        const llama_model * model_dft,
        const llama_context * ctx_dft) {
    std::string s;
    auto add = [&](const std::string & k, const std::string & v) { s += k + "=" + v + "\n"; };

    add("format", std::to_string(SSD_VERSION));
    add("state_seq_version", std::to_string(LLAMA_STATE_SEQ_VERSION));

    add("model", model_desc(model_tgt));
    for (const auto & f : model_files(params.model.path)) {
        add("model_file", fs::u8path(f).filename().string() + ":" + file_identity(f));
    }
    add("n_swa", std::to_string(llama_model_n_swa(model_tgt)));

    add("n_ctx", std::to_string(llama_n_ctx(ctx_tgt)));
    add("n_ctx_seq", std::to_string(llama_n_ctx_seq(ctx_tgt)));
    add("n_seq_max", std::to_string(llama_n_seq_max(ctx_tgt)));
    add("n_rs_seq", std::to_string(llama_n_rs_seq(ctx_tgt)));
    add("kv_unified", std::to_string(params.kv_unified));
    add("swa_full", std::to_string(params.swa_full));
    add("type_k", ggml_type_name(params.cache_type_k));
    add("type_v", ggml_type_name(params.cache_type_v));
    add("flash_attn", std::to_string((int) params.flash_attn_type));

    add("rope", string_format("%d|%a|%a|%a|%a|%a|%a|%d", (int) params.rope_scaling_type, params.rope_freq_base,
            params.rope_freq_scale, params.yarn_ext_factor, params.yarn_attn_factor, params.yarn_beta_fast,
            params.yarn_beta_slow, params.yarn_orig_ctx));

    for (const auto & la : params.lora_adapters) {
        add("lora", la.path + ":" + file_identity(la.path) + string_format(":%a", la.scale));
    }
    for (const auto & cv : params.control_vectors) {
        add("control_vector", cv.fname + ":" + file_identity(cv.fname) + string_format(":%a", cv.strength));
    }
    if (!params.control_vectors.empty()) {
        add("control_vector_layers", string_format("%d:%d", params.control_vector_layer_start, params.control_vector_layer_end));
    }

    if (model_dft) {
        add("draft_model", model_desc(model_dft));
    }
    if (ctx_dft) {
        add("draft_n_ctx", std::to_string(llama_n_ctx(ctx_dft)));
        add("draft_type_k", ggml_type_name(params.speculative.draft.cache_type_k));
        add("draft_type_v", ggml_type_name(params.speculative.draft.cache_type_v));
    }
    std::string spec_types;
    for (const auto t : params.speculative.types) {
        spec_types += std::to_string((int) t) + ",";
    }
    add("speculative", spec_types);

    return s;
}
