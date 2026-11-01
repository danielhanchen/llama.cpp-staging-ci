// The advisory PLE prefetch predicts the rows of a ubatch before the ubatch before it is applied.
// This test checks, on a model with a PLE table (qwen4exp n-gram hash, gemma4 token rows), that
//   - every predicted row equals the row set_input computes for that ubatch (LLAMA_PLE_PREFETCH_VERIFY=1),
//     over multi-ubatch prompts, generation, two interleaved sequences, an EOS inside the prompt and a seq_rm,
//   - the prefetcher is engaged for each mode when the table is lazy (jobs > 0),
//   - the logits are bit-identical across off / madvise / populate / pread.
// usage: test-ple-prefetch <model.gguf> [more models]
// a tiny qwen4exp model: test-llama-archs -a qwen4exp -o <dir>

#include "llama.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
// POSIX environment helpers for the MSVC CRT (getenv sees _putenv_s; an empty value removes the variable)
static int setenv(const char * name, const char * value, int /*overwrite*/) { return _putenv_s(name, value); }
static int unsetenv(const char * name) { return _putenv_s(name, ""); }
#endif

struct log_capture {
    std::string text;
};

static void log_cb(enum ggml_log_level level, const char * text, void * ud) {
    auto * lc = (log_capture *) ud;
    lc->text += text;
    if (level >= GGML_LOG_LEVEL_WARN || getenv("TEST_PLE_VERBOSE")) {
        fputs(text, stderr);
    }
}

static bool parse_u64(const std::string & text, const char * key, uint64_t & out) {
    const size_t p = text.rfind(key);
    if (p == std::string::npos) {
        return false;
    }
    out = strtoull(text.c_str() + p + strlen(key), nullptr, 10);
    return true;
}

struct run_result {
    std::vector<float> logits; // the logits of every output, concatenated
    uint64_t checked  = 0;
    uint64_t mismatch = 0;
    uint64_t jobs     = 0;
    uint64_t predicted = 0;
    bool     engaged  = false;
};

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, const std::vector<llama_seq_id> & seqs,
        const std::vector<llama_pos> & pos, std::vector<float> & out, int n_vocab) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = pos[i];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seqs[i];
        batch.logits[i]    = (i % 7 == 0) || i + 1 == toks.size();
    }
    batch.n_tokens = (int32_t) toks.size();
    const int ret = llama_decode(ctx, batch);
    if (ret == 0) {
        for (size_t i = 0; i < toks.size(); ++i) {
            if (batch.logits[i]) {
                const float * l = llama_get_logits_ith(ctx, (int32_t) i);
                out.insert(out.end(), l, l + n_vocab);
            }
        }
    }
    llama_batch_free(batch);
    return ret == 0;
}

static bool run(llama_model * model, enum llama_ple_prefetch_mode mode, run_result & res) {
    log_capture lc;
    llama_log_set(log_cb, &lc);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 512;
    cp.n_ubatch        = 32;
    cp.n_seq_max       = 2;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.ple_prefetch    = mode;
    cp.kv_unified      = true;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return false;
    }

    std::mt19937 rng(42);
    auto tok = [&]() { return (llama_token) (1 + rng() % (n_vocab - 1)); };

    bool ok = true;

    // 1. a 200-token prompt with an EOS (0 for the tiny model, the PLE EOS) inside: 7 ubatches of 32
    {
        std::vector<llama_token> t; std::vector<llama_seq_id> s; std::vector<llama_pos> p;
        for (int i = 0; i < 200; ++i) {
            t.push_back(i == 70 ? 0 : tok()); s.push_back(0); p.push_back(i);
        }
        ok = ok && decode_tokens(ctx, t, s, p, res.logits, n_vocab);
    }
    // 2. generation: 8 single tokens (the first ubatch of each decode is predicted from the cells)
    for (int i = 0; ok && i < 8; ++i) {
        ok = decode_tokens(ctx, { tok() }, { 0 }, { 200 + i }, res.logits, n_vocab);
    }
    // 3. two sequences interleaved in one batch, seq 1 starts fresh, seq 0 continues
    {
        std::vector<llama_token> t; std::vector<llama_seq_id> s; std::vector<llama_pos> p;
        for (int i = 0; i < 100; ++i) {
            t.push_back(tok()); s.push_back(1); p.push_back(i);
            t.push_back(tok()); s.push_back(0); p.push_back(208 + i);
        }
        ok = ok && decode_tokens(ctx, t, s, p, res.logits, n_vocab);
    }
    // 4. drop seq 1 (recurrent states cannot be cut partially) and decode a new seq 1 over the same positions
    if (ok) {
        ok = llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);
        std::vector<llama_token> t; std::vector<llama_seq_id> s; std::vector<llama_pos> p;
        for (int i = 0; i < 70; ++i) {
            t.push_back(tok()); s.push_back(1); p.push_back(i);
        }
        ok = ok && decode_tokens(ctx, t, s, p, res.logits, n_vocab);
    }

    llama_free(ctx);
    llama_log_set(nullptr, nullptr);

    if (!ok) {
        fprintf(stderr, "decode failed\n");
        return false;
    }

    if (!parse_u64(lc.text, "PLE prefetch verify: checked = ", res.checked) ||
        !parse_u64(lc.text, "mismatched rows = ", res.mismatch) ||
        !parse_u64(lc.text, ", predicted = ", res.predicted)) {
        fprintf(stderr, "no PLE prefetch verify line in the log\n");
        return false;
    }
    parse_u64(lc.text, "PLE prefetch: jobs = ", res.jobs);
    res.engaged = lc.text.find("PLE prefetch of '") != std::string::npos;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [more models]\n", argv[0]);
        return 1;
    }

    setenv("LLAMA_PLE_PREFETCH_VERIFY", "1", 1);
    setenv("LLAMA_PLE_PREFETCH_STATS",  "1", 1);
    unsetenv("LLAMA_PLE_PREFETCH");

    llama_backend_init();

    int n_fail = 0;

    for (int a = 1; a < argc; ++a) {
        llama_model_params mp = llama_model_default_params();
        mp.load_mode  = LLAMA_LOAD_MODE_MMAP;
        mp.lazy_mode  = LLAMA_LAZY_MODE_ON;
        mp.n_gpu_layers = 0;

        llama_model * model = llama_model_load_from_file(argv[a], mp);
        if (!model) {
            fprintf(stderr, "%s: failed to load\n", argv[a]);
            n_fail++;
            continue;
        }

        run_result base;
        for (auto mode : { LLAMA_PLE_PREFETCH_OFF, LLAMA_PLE_PREFETCH_MADVISE, LLAMA_PLE_PREFETCH_POPULATE, LLAMA_PLE_PREFETCH_PREAD }) {
            run_result r;
            bool pass = run(model, mode, r);
            std::string why = pass ? "" : "run failed";
            if (pass && (r.checked == 0 || r.predicted == 0)) { pass = false; why = "nothing was verified"; }
            if (pass && r.mismatch != 0)                       { pass = false; why = "predicted rows differ from set_input"; }
            if (pass && mode != LLAMA_PLE_PREFETCH_OFF && (!r.engaged || r.jobs == 0)) { pass = false; why = "the prefetcher did not run"; }
            if (pass && mode == LLAMA_PLE_PREFETCH_OFF && (r.engaged || r.jobs != 0))  { pass = false; why = "off still prefetched"; }
            if (pass && mode == LLAMA_PLE_PREFETCH_OFF) {
                base = r;
            } else if (pass && (r.logits.size() != base.logits.size() ||
                                memcmp(r.logits.data(), base.logits.data(), r.logits.size()*sizeof(float)) != 0)) {
                pass = false; why = "logits differ from off";
            }
            printf("%-60s %-9s checked = %4" PRIu64 " predicted = %4" PRIu64 " mismatched rows = %" PRIu64 " jobs = %4" PRIu64 " outputs = %zu : %s %s\n",
                    argv[a], llama_ple_prefetch_mode_name(mode), r.checked, r.predicted, r.mismatch, r.jobs,
                    r.logits.size() / llama_vocab_n_tokens(llama_model_get_vocab(model)), pass ? "OK" : "FAIL", why.c_str());
            n_fail += !pass;
        }

        llama_model_free(model);
    }

    llama_backend_free();

    printf("%s\n", n_fail == 0 ? "ALL OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
