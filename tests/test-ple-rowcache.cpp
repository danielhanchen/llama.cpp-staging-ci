// the PLE row cache (src/llama-row-cache.*) in front of a lazily mapped table must not change any output.
//   1. ggml level, per table type (F32, F16, BF16, Q8_0, Q4_0, IQ4_NL): a GET_ROWS over random row sets (duplicates included) gives
//      bit-identical results with the cache as the row source (cold, then warm, with evictions, with a skip range) and without it;
//      every cached row's bytes equal the table's; a saved profile preloads static rows with the table's bytes; a stale profile (one
//      byte changed) or another table's profile is refused.
//   2. model level (optional argument: a tiny qwen4exp, test-llama-archs -a qwen4exp): logits bit-identical with the cache off
//      (LLAMA_PLE_ROW_CACHE=0), on with a small dynamic part (evictions), and on with a profile written by the previous run preloaded;
//      the cache is engaged (hits, inserts, static hits) and every hit equals the mapping (LLAMA_PLE_ROW_CACHE_VERIFY=1).
// usage: test-ple-rowcache [qwen4exp.gguf]

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../src/llama-row-cache.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
static int setenv(const char * name, const char * value, int /*overwrite*/) { return _putenv_s(name, value); }
#endif

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

// ------------------------------------------------------------------------------------------------------------------------------------
// 1. ggml level

struct gather_env {
    ggml_backend_t        be  = nullptr;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         tab = nullptr;
    ggml_tensor *         ids = nullptr;
    ggml_tensor *         out = nullptr;
    ggml_cgraph *         gf  = nullptr;
};

static bool gather_init(gather_env & g, ggml_type type, int64_t ncols, int64_t nrows, int64_t nids, std::mt19937 & rng) {
    // through the registry: the CPU backend can be a loadable module (GGML_BACKEND_DL), whose functions are not linked
    g.be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(g.be != nullptr && "no CPU backend");
    {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(g.be));
        auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads != nullptr) {
            set_n_threads(g.be, 4);
        }
    }
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.tab = ggml_new_tensor_2d(g.ctx, type, ncols, nrows);
    g.ids = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, nids);
    g.out = ggml_get_rows(g.ctx, g.tab, g.ids);
    g.gf  = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.be);
    if (!g.buf) {
        return false;
    }
    // random table content: floats quantized to the type
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f(ncols*nrows);
    for (auto & x : f) {
        x = nd(rng);
    }
    std::vector<uint8_t> q(ggml_nbytes(g.tab));
    if (type == GGML_TYPE_F32) {
        memcpy(q.data(), f.data(), q.size());
    } else {
        ggml_quantize_chunk(type, f.data(), q.data(), 0, nrows, ncols, nullptr);
    }
    ggml_backend_tensor_set(g.tab, q.data(), 0, q.size());
    return true;
}

static void gather_free(gather_env & g) {
    ggml_backend_buffer_free(g.buf);
    ggml_free(g.ctx);
    ggml_backend_free(g.be);
}

static std::vector<float> gather_run(gather_env & g, const std::vector<int32_t> & ids) {
    ggml_backend_tensor_set(g.ids, ids.data(), 0, ids.size()*sizeof(int32_t));
    ggml_backend_graph_compute(g.be, g.gf);
    std::vector<float> out(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, out.size()*sizeof(float));
    return out;
}

static void test_gather(ggml_type type) {
    std::mt19937 rng(1234 + type);
    const int64_t ncols = 256, nrows = 6000, nids = 4096;
    gather_env g;
    if (!gather_init(g, type, ncols, nrows, nids, rng)) {
        CHECK(false, "%s: alloc failed", ggml_type_name(type));
        return;
    }
    const size_t row_b = ggml_row_size(type, ncols);
    std::vector<uint8_t> tab(ggml_nbytes(g.tab));
    ggml_backend_tensor_get(g.tab, tab.data(), 0, tab.size());

    // skewed random row sets: half from a hot range of 300 rows, half uniform
    auto make_ids = [&]() {
        std::vector<int32_t> v(nids);
        for (auto & x : v) {
            x = (rng() & 1) ? (int32_t) (rng() % 300) * 17 % nrows : (int32_t) (rng() % nrows);
        }
        return v;
    };
    std::vector<std::vector<int32_t>> sets;
    for (int i = 0; i < 6; ++i) {
        sets.push_back(make_ids());
    }
    std::vector<std::vector<float>> ref;
    for (const auto & s : sets) {
        ref.push_back(gather_run(g, s));
    }

    // a cache of ~1500 slots for 6000 rows (evictions), rows < 500 not cached
    const int64_t skip = 500;
    const size_t  slot = llama_row_cache::slot_bytes(row_b);
    llama_row_cache rc(g.tab, 1500*slot, 0, skip);
    rc.verify = true;
    rc.load_profile("");
    CHECK(llama_row_cache_attach(&rc), "%s: attach failed", ggml_type_name(type));
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < sets.size(); ++i) {
            const auto out = gather_run(g, sets[i]);
            CHECK(out.size() == ref[i].size() && memcmp(out.data(), ref[i].data(), out.size()*sizeof(float)) == 0,
                  "%s: pass %d set %zu: gather through the cache differs", ggml_type_name(type), pass, i);
        }
    }
    // every cached row holds the table's bytes
    std::vector<uint8_t> tmp(row_b);
    size_t n_cached = 0;
    for (int64_t r = 0; r < nrows; ++r) {
        if (rc.contains(r)) {
            n_cached++;
            CHECK(r >= skip, "%s: row %" PRId64 " under the skip range was cached", ggml_type_name(type), r);
            CHECK(rc.get(r, tmp.data()) && memcmp(tmp.data(), tab.data() + r*row_b, row_b) == 0, "%s: row %" PRId64 " bytes differ",
                  ggml_type_name(type), r);
        }
    }
    CHECK(rc.st.n_hits > 0 && rc.st.n_inserts > 0 && rc.st.n_evicts > 0 && n_cached > 0, "%s: cache not engaged: %s",
          ggml_type_name(type), rc.describe().c_str());
    CHECK(rc.st.n_verify_bad == 0 && rc.st.n_verified > 0, "%s: verified hits differ: %s", ggml_type_name(type), rc.describe().c_str());
    // the hint filter drops exactly the cached rows
    {
        std::vector<int32_t> v = sets[0];
        std::vector<int32_t> keep;
        for (int32_t r : v) {
            if (!(r >= skip && rc.contains(r))) {
                keep.push_back(r);
            }
        }
        v.resize(rc.filter_uncached(v.data(), v.size()));
        CHECK(v == keep, "%s: filter_uncached", ggml_type_name(type));
    }

    // profile: save, preload into a new cache (static rows), gather equal; a stale profile is refused
    const std::string prof = "test-ple-rowcache-" + std::string(ggml_type_name(type)) + ".prof";
    CHECK(rc.save_profile(prof), "%s: save failed", ggml_type_name(type));
    llama_row_cache_detach(g.tab);
    {
        llama_row_cache rc2(g.tab, 3000*slot, 1500*slot, skip);
        rc2.verify = true;
        const size_t n = rc2.load_profile(prof);
        CHECK(n > 0 && n == rc2.n_static(), "%s: profile preloaded %zu rows", ggml_type_name(type), n);
        for (int64_t r = 0; r < nrows; ++r) {
            if (rc2.contains(r)) {
                CHECK(rc2.get(r, tmp.data()) && memcmp(tmp.data(), tab.data() + r*row_b, row_b) == 0, "%s: preloaded row %" PRId64 " differs",
                      ggml_type_name(type), r);
            }
        }
        rc2.st.n_hits_static = 0;
        CHECK(llama_row_cache_attach(&rc2), "attach 2");
        for (size_t i = 0; i < sets.size(); ++i) {
            const auto out = gather_run(g, sets[i]);
            CHECK(memcmp(out.data(), ref[i].data(), out.size()*sizeof(float)) == 0, "%s: set %zu with a preloaded profile differs",
                  ggml_type_name(type), i);
        }
        CHECK(rc2.st.n_hits_static > 0 && rc2.st.n_verify_bad == 0, "%s: static rows not hit: %s", ggml_type_name(type), rc2.describe().c_str());
        llama_row_cache_detach(g.tab);
    }
    {
        // one byte of the last row changed: the spot check (every row of a small profile) refuses the file
        FILE * f = fopen(prof.c_str(), "r+b");
        CHECK(f != nullptr, "reopen");
        if (f) {
            fseek(f, -1, SEEK_END);
            const int c = fgetc(f);
            fseek(f, -1, SEEK_END);
            fputc(c ^ 0x5a, f);
            fclose(f);
        }
        llama_row_cache rc3(g.tab, 3000*slot, 3000*slot, skip); // every row of the file kept, the changed one included
        CHECK(rc3.load_profile(prof) == 0 && rc3.n_static() == 0, "%s: a stale profile was accepted", ggml_type_name(type));
    }
    {
        // a profile of more rows than the spot check covers (5000+), one byte of a middle row changed: the row checksums refuse it;
        // a profile cut short is refused before anything is allocated
        llama_row_cache big(g.tab, 7000*slot, 0, 0);
        std::vector<int32_t> all(nrows);
        for (int64_t r = 0; r < nrows; ++r) {
            all[r] = (int32_t) r;
        }
        llama_row_cache_attach(&big);
        gather_run(g, std::vector<int32_t>(all.begin(), all.begin() + nids));
        gather_run(g, std::vector<int32_t>(all.begin() + 1904, all.begin() + 1904 + nids));
        llama_row_cache_detach(g.tab);
        CHECK(big.save_profile(prof), "save big");
        {
            llama_row_cache ok2(g.tab, 0, 8000*slot, 0);
            CHECK(ok2.load_profile(prof) > 4096, "%s: big profile: %zu rows preloaded", ggml_type_name(type), ok2.n_static());
        }
        FILE * f = fopen(prof.c_str(), "r+b");
        if (f) {
            // the entry count from the header, then a byte of entry 2500's row
            uint64_t n_ent = 0;
            fseek(f, 40, SEEK_SET);
            CHECK(fread(&n_ent, 8, 1, f) == 1, "header");
            const long at = (long) (48 + 12*n_ent + 2500*row_b + 3);
            fseek(f, at, SEEK_SET);
            const int c = fgetc(f);
            fseek(f, at, SEEK_SET);
            fputc(c ^ 0x01, f);
            fclose(f);
        }
        llama_row_cache bad(g.tab, 0, 8000*slot, 0);
        CHECK(bad.load_profile(prof) == 0, "%s: a big profile with a corrupt middle row was accepted", ggml_type_name(type));
        // truncated: drop the last row
        {
            FILE * in = fopen(prof.c_str(), "rb");
            std::vector<uint8_t> bytes;
            if (in) {
                int c;
                while ((c = fgetc(in)) != EOF) {
                    bytes.push_back((uint8_t) c);
                }
                fclose(in);
            }
            FILE * out = fopen(prof.c_str(), "wb");
            if (out) {
                fwrite(bytes.data(), 1, bytes.size() - row_b, out);
                fclose(out);
            }
        }
        llama_row_cache cut(g.tab, 0, 8000*slot, 0);
        CHECK(cut.load_profile(prof) == 0, "%s: a truncated profile was accepted", ggml_type_name(type));
    }
    remove(prof.c_str());
    printf("gather %-7s row %4zu B: %zu sets x %d ids bit-identical (cold, warm, evictions, skip range, preloaded profile); %s\n",
           ggml_type_name(type), row_b, sets.size(), (int) nids, rc.describe().c_str());
    gather_free(g);
}

// ------------------------------------------------------------------------------------------------------------------------------------
// 2. model level

struct log_capture {
    std::string text;
};

static void log_cb(enum ggml_log_level level, const char * text, void * ud) {
    ((log_capture *) ud)->text += text;
    if (level >= GGML_LOG_LEVEL_WARN || getenv("TEST_PLE_VERBOSE")) {
        fputs(text, stderr);
    }
}

static bool run_model(const char * path, std::vector<float> & logits, std::string & log) {
    log_capture lc;
    llama_log_set(log_cb, &lc);
    llama_model_params mp = llama_model_default_params();
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode    = LLAMA_LAZY_MODE_ON;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        llama_log_set(nullptr, nullptr);
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 64; cp.n_seq_max = 1; cp.n_threads = 4; cp.n_threads_batch = 4; cp.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    bool ok = ctx != nullptr;
    std::mt19937 rng(7);
    // a skewed token stream (repeated n-grams), a prompt then single-token decode
    std::vector<llama_token> toks;
    for (int i = 0; i < 400; ++i) {
        toks.push_back((rng() % 3 == 0) ? (llama_token) (1 + rng() % (n_vocab - 1)) : (llama_token) (1 + (i % 11) * 7 % (n_vocab - 1)));
    }
    int pos = 0;
    auto decode = [&](const std::vector<llama_token> & t) {
        llama_batch b = llama_batch_init((int32_t) t.size(), 0, 1);
        for (size_t i = 0; i < t.size(); ++i) {
            b.token[i] = t[i]; b.pos[i] = pos + (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (i % 5 == 0) || i + 1 == t.size();
        }
        b.n_tokens = (int32_t) t.size();
        bool r = llama_decode(ctx, b) == 0;
        for (size_t i = 0; r && i < t.size(); ++i) {
            if (b.logits[i]) {
                const float * l = llama_get_logits_ith(ctx, (int32_t) i);
                logits.insert(logits.end(), l, l + n_vocab);
            }
        }
        pos += (int) t.size();
        llama_batch_free(b);
        return r;
    };
    ok = ok && decode(std::vector<llama_token>(toks.begin(), toks.begin() + 300));
    for (int i = 300; ok && i < 400; ++i) {
        ok = decode({ toks[i] });
    }
    if (ctx) {
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_log_set(nullptr, nullptr);
    log = lc.text;
    return ok;
}

static uint64_t field(const std::string & s, const char * key) {
    const size_t p = s.rfind(key);
    return p == std::string::npos ? UINT64_MAX : strtoull(s.c_str() + p + strlen(key), nullptr, 10);
}

static void test_model(const char * path) {
    const std::string prof = "test-ple-rowcache-model.prof";
    remove(prof.c_str());
    setenv("LLAMA_PLE_ROW_CACHE_VERIFY", "1", 1);
    setenv("LLAMA_PLE_ROW_PROFILE", prof.c_str(), 1);

    std::vector<float> l_off, l_dyn, l_pre;
    std::string g_off, g_dyn, g_pre;

    setenv("LLAMA_PLE_ROW_CACHE", "0", 1);
    CHECK(run_model(path, l_off, g_off), "off: run failed");
    CHECK(g_off.find("PLE row cache") == std::string::npos, "off: the cache was made");

    // forced on (the tiny table would be resident), ~400 slots for its rows, the profile written at the end
    setenv("LLAMA_PLE_ROW_CACHE", "2", 1);
    setenv("LLAMA_PLE_ROW_CACHE_MIB", "0.1", 1);
    setenv("LLAMA_PLE_ROW_PROFILE_SAVE", "1", 1);
    CHECK(run_model(path, l_dyn, g_dyn), "dynamic: run failed");
    CHECK(l_dyn.size() == l_off.size() && !l_off.empty() && memcmp(l_dyn.data(), l_off.data(), l_off.size()*sizeof(float)) == 0,
          "dynamic: logits differ from the cache off");
    CHECK(field(g_dyn, ", hits = ") > 0 && field(g_dyn, ", hits = ") != UINT64_MAX && field(g_dyn, "inserts = ") > 0 && field(g_dyn, "mismatched = ") == 0,
          "dynamic: not engaged or mismatched");

    // the profile preloaded (static rows), the profile not rewritten
    setenv("LLAMA_PLE_ROW_PROFILE_SAVE", "0", 1);
    setenv("LLAMA_PLE_ROW_CACHE_MIB", "0.05", 1);
    CHECK(run_model(path, l_pre, g_pre), "preload: run failed");
    CHECK(l_pre.size() == l_off.size() && memcmp(l_pre.data(), l_off.data(), l_off.size()*sizeof(float)) == 0, "preload: logits differ from the cache off");
    const uint64_t pre_rows = field(g_pre, "rows preloaded (");
    CHECK(g_pre.find("rows preloaded (") != std::string::npos && field(g_pre, "(static ") > 0 && field(g_pre, "mismatched = ") == 0,
          "preload: no static hits or mismatched");
    printf("model %s: logits bit-identical off / dynamic / preloaded (%zu outputs); dynamic: hits %" PRIu64 ", inserts %" PRIu64
           ", evictions %" PRIu64 "; preloaded: static hits %" PRIu64 "; verified hits mismatched: %" PRIu64 " / %" PRIu64 "\n", path,
           l_off.size(), field(g_dyn, ", hits = "), field(g_dyn, "inserts = "), field(g_dyn, "evictions = "), field(g_pre, "(static "),
           field(g_dyn, "mismatched = "), field(g_pre, "mismatched = "));
    (void) pre_rows;
    remove(prof.c_str());
}

int main(int argc, char ** argv) {
    llama_backend_init();
    ggml_backend_load_all();
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL }) {
        test_gather(t);
    }
    for (int a = 1; a < argc; ++a) {
        test_model(argv[a]);
    }
    llama_backend_free();
    printf("%s\n", n_fail == 0 ? "ALL OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
