// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// the QSA selection of qwen4exp without indexer scoring while every sequence of the ubatch holds at most top_k / kpool pools
// (LLAMA_QSA_ALLFIT, llama_memory_hybrid_idx_context::get_kpool_allfit) against the scored selection (LLAMA_QSA_ALLFIT=0), byte for byte.
// The generated qwen4exp (MoE) model with the indexer top_k overridden to 32 (8 pools of 4 cells), so the switch between the two graphs happens
// at 32..36 cells instead of 2048: two sequences in a unified KV cache, prompts in several ubatches (all-fit, scored, and an all-fit
// ubatch of one sequence next to a scored one of the other), single-token decodes across the boundary, a two-sequence decode batch, and a
// cleared sequence prompted again below the boundary (all-fit again). Each setting runs in a child process (the switch is read once per
// process) that dumps the logits of every output row of every decode; the parent compares the dumps and the graph kinds the children
// logged (LLAMA_QSA_ALLFIT_LOG=1): the all-fit child must build both kinds, the scored child only scored graphs.
// usage: test-qsa-allfit-exact qwen4exp-model.gguf   (not a .gguf file: SKIP, exit 0)
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int n_allfit = 0;
static int n_scored = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "QSA graph: allfit") != nullptr) {
        n_allfit++;
    } else if (strstr(text, "QSA graph: scored") != nullptr) {
        n_scored++;
    } else if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

struct dump {
    std::vector<uint8_t> bytes;
    void add(const float * p, size_t n) {
        const uint8_t * b = (const uint8_t *) p;
        bytes.insert(bytes.end(), b, b + n*sizeof(float));
    }
};

// decode tokens [p0, p0 + n) of sequence s (token ids from a fixed pattern), every row an output; dump the logits
static bool decode_seq(llama_context * ctx, const std::vector<std::pair<llama_seq_id, llama_pos>> & rows, int n_vocab, dump & d) {
    llama_batch batch = llama_batch_init((int32_t) rows.size(), 0, 1);
    for (size_t i = 0; i < rows.size(); ++i) {
        const llama_seq_id s = rows[i].first;
        const llama_pos    p = rows[i].second;
        batch.token[i]     = (llama_token) ((p*7 + s*13 + 3) % (n_vocab - 1) + 1);
        batch.pos[i]       = p;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = s;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = (int32_t) rows.size();
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        llama_batch_free(batch);
        return false;
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        d.add(llama_get_logits_ith(ctx, (int32_t) i), (size_t) n_vocab);
    }
    llama_batch_free(batch);
    return true;
}

static std::vector<std::pair<llama_seq_id, llama_pos>> span(llama_seq_id s, llama_pos p0, llama_pos n) {
    std::vector<std::pair<llama_seq_id, llama_pos>> r;
    for (llama_pos p = p0; p < p0 + n; ++p) {
        r.push_back({ s, p });
    }
    return r;
}

static int run_child(const char * model_path, const char * out_path) {
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    std::vector<llama_model_kv_override> kvo(2);
    memset(kvo.data(), 0, kvo.size()*sizeof(kvo[0]));
    kvo[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(kvo[0].key, sizeof(kvo[0].key), "qwen4exp.attention.indexer.top_k");
    // (QSA_ALLFIT_TEST_TOPK: another top_k, for a manual control: 131072 keeps every pool, so its logits must differ past 32 cells)
    kvo[0].val_i64 = getenv("QSA_ALLFIT_TEST_TOPK") ? atoll(getenv("QSA_ALLFIT_TEST_TOPK")) : 32;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.kv_overrides = kvo.data();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 512;
    cp.n_batch    = 64;
    cp.n_ubatch   = 16;
    cp.n_seq_max  = 2;
    cp.kv_unified = true;
    cp.n_threads  = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);

    dump d;
    bool ok = true;
    // sequence 0: a 20-token prompt (two ubatches, 5 pools), then single tokens to 60 (the 9th pool, past the top 8, at 36)
    ok = ok && decode_seq(ctx, span(0, 0, 20), n_vocab, d);
    for (llama_pos p = 20; ok && p < 60; ++p) {
        ok = ok && decode_seq(ctx, span(0, p, 1), n_vocab, d);
    }
    // sequence 1: a 44-token prompt in one batch (ubatches of 16: all-fit, all-fit, scored)
    ok = ok && decode_seq(ctx, span(1, 0, 44), n_vocab, d);
    // both sequences in one batch (scored), then sequence 0 cleared (a recurrent state cannot be cut partially) and prompted again
    // with 26 tokens in one ubatch-split batch (6 pools: all-fit)
    ok = ok && decode_seq(ctx, { { 0, 60 }, { 1, 44 } }, n_vocab, d);
    ok = ok && llama_memory_seq_rm(mem, 0, -1, -1);
    ok = ok && decode_seq(ctx, span(0, 0, 26), n_vocab, d);
    // sequence 0 alone (6 pools again: all-fit), across the boundary once more, a 3-row verify-like batch each step
    for (llama_pos p = 26; ok && p < 50; p += 3) {
        ok = ok && decode_seq(ctx, span(0, p, 3), n_vocab, d);
    }
    // sequence 1 again below the boundary while sequence 0 is past it: a batch of each, the sequence-1 one all-fit
    ok = ok && llama_memory_seq_rm(mem, 1, -1, -1);
    ok = ok && decode_seq(ctx, span(1, 0, 17), n_vocab, d);
    ok = ok && decode_seq(ctx, span(0, 50, 2), n_vocab, d);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    if (!ok) {
        return 1;
    }

    FILE * f = fopen(out_path, "wb");
    if (f == nullptr) {
        return 1;
    }
    const uint64_t n = d.bytes.size();
    const int32_t counts[2] = { n_allfit, n_scored };
    ok = fwrite(&n, sizeof(n), 1, f) == 1 && fwrite(d.bytes.data(), 1, n, f) == n && fwrite(counts, sizeof(counts), 1, f) == 1;
    fclose(f);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[2], argv[3]);
    }
    const std::string model = argc > 1 ? argv[1] : "";
    FILE * mf = model.size() > 5 && model.compare(model.size() - 5, 5, ".gguf") == 0 ? fopen(model.c_str(), "rb") : nullptr;
    if (mf == nullptr) {
        printf("SKIP: no qwen4exp .gguf model given\n");
        return 0;
    }
    fclose(mf);

    std::vector<uint8_t> outs[2];
    int32_t counts[2][2] = {};
    for (int mode = 0; mode < 2; ++mode) {
        const std::string path = model + ".allfit_exact_" + std::to_string(mode) + ".bin";
        set_env("LLAMA_QSA_ALLFIT", mode ? "1" : "0");
        set_env("LLAMA_QSA_ALLFIT_LOG", "1");
        fflush(stdout);
        std::string cmd = "\"" + std::string(argv[0]) + "\" --child \"" + model + "\" \"" + path + "\"";
#if defined(_WIN32)
        cmd = "\"" + cmd + "\""; // cmd.exe /c strips the outer pair of quotes of a line with more than two
#endif
        const int rc = std::system(cmd.c_str());
        FILE * f = rc == 0 ? fopen(path.c_str(), "rb") : nullptr;
        if (f == nullptr) {
            printf("FAIL: child LLAMA_QSA_ALLFIT=%d rc %d\n", mode, rc);
            return 1;
        }
        uint64_t n = 0;
        bool ok = fread(&n, sizeof(n), 1, f) == 1;
        outs[mode].resize(ok ? n : 0);
        ok = ok && fread(outs[mode].data(), 1, n, f) == n && fread(counts[mode], sizeof(counts[mode]), 1, f) == 1;
        fclose(f);
        std::remove(path.c_str());
        if (!ok) {
            printf("FAIL: short dump from child LLAMA_QSA_ALLFIT=%d\n", mode);
            return 1;
        }
        printf("LLAMA_QSA_ALLFIT=%d: %zu logit bytes, graphs built: %d all-fit, %d scored\n", mode, outs[mode].size(), counts[mode][0], counts[mode][1]);
    }
    size_t first_diff = outs[0].size();
    for (size_t i = 0; i < std::min(outs[0].size(), outs[1].size()); ++i) {
        if (outs[0][i] != outs[1][i]) {
            first_diff = i;
            break;
        }
    }
    const bool eq = outs[0] == outs[1] && !outs[0].empty();
    const bool engaged = counts[1][0] > 0 && counts[1][1] > 0 && counts[0][0] == 0 && counts[0][1] > 0;
    if (!eq || !engaged) {
        printf("FAIL: logits %s (first differing byte %zu of %zu), engagement all-fit child %d / %d, scored child %d / %d "
            "(want > 0 / > 0, 0 / > 0)\n", eq ? "equal" : "DIFFER", first_diff, outs[0].size(), counts[1][0], counts[1][1],
            counts[0][0], counts[0][1]);
        return 1;
    }
    printf("OK: every logit byte-identical with and without the all-fit selection (%zu bytes)\n", outs[0].size());
    return 0;
}
