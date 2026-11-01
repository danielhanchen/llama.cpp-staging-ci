// -sm tensor (a meta device) with the delta-net rollback by replay requested (LLAMA_RS_REPLAY=1): the replay commit graph fed mirrored
// replay inputs and split states to the meta backend, whose graph allocation aborted (split_dim_view_src assert). Under a meta device the
// recurrent memory keeps the snapshot rows: a decode with rollbacks on Meta(CPU, CPU) matches the CPU decode
#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// a 12-token prompt, then 6 times a 4-token window, a rollback of 3 tokens and one more token; the logits of every token
static bool run(const char * path, bool meta, std::vector<float> & out) {
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[3] = { cpu, cpu, nullptr };
    if (meta) {
        mp.devices    = devs;
        mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    }
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 1; cp.n_rs_seq = 7; cp.n_threads = 4; cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        llama_model_free(model);
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);
    int pos = 0;
    bool ok = true;
    auto dec = [&](int n, int step) {
        llama_batch b = llama_batch_init(64, 0, 1);
        for (int i = 0; i < n; i++) {
            b.token[i] = (llama_token) ((pos * 17 + step * 7) % (n_vocab - 10) + 5);
            b.pos[i] = pos++; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) {
            ok = false;
        } else {
            for (int i = 0; i < n; i++) {
                const float * l = llama_get_logits_ith(ctx, i);
                out.insert(out.end(), l, l + n_vocab);
            }
        }
        llama_batch_free(b);
    };
    dec(12, 0);
    for (int step = 1; step <= 6 && ok; step++) {
        dec(4, step);
        ok = ok && llama_memory_seq_rm(mem, 0, pos - 3, -1);
        pos -= 3;
        dec(1, step + 10);
    }
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s qwen35-model.gguf\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);

    std::vector<float> ref, met;
    set_env("LLAMA_RS_REPLAY", "0");
    if (!run(argv[1], false, ref)) {
        printf("CPU decode failed\nFAIL\n");
        return 1;
    }
    set_env("LLAMA_RS_REPLAY", "1");
    if (!run(argv[1], true, met)) {
        printf("Meta(CPU, CPU) decode with LLAMA_RS_REPLAY=1 failed\nFAIL\n");
        return 1;
    }
    double max_diff = ref.size() == met.size() ? 0.0 : INFINITY;
    for (size_t i = 0; i < ref.size() && i < met.size(); ++i) {
        max_diff = std::fmax(max_diff, std::fabs((double) ref[i] - met[i]));
    }
    printf("%zu logits, max |CPU - Meta(CPU, CPU)| = %g\n", ref.size(), max_diff);
    const bool pass = max_diff <= 1e-4;
    printf("%s\n", pass ? "OK" : "FAIL");
    llama_backend_free();
    return pass ? 0 : 1;
}
