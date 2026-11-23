// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// the recent-window ring of a quantized KV cache (LLAMA_KV_WIN / LLAMA_KV_WIN_SINKS): LLAMA_KV_WIN_SINKS alone makes a sinks-only
// ring, an explicit LLAMA_KV_WIN=-1 keeps it off beside it, and (with a sparse attention model, argv[2]) a cache with a stream per
// sequence decodes a ubatch that holds two streams
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static bool g_ring = false;

static void log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    if (strstr(text, "ring of ") != nullptr && strstr(text, " slots x ") != nullptr) {
        g_ring = true;
    }
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static llama_context * make_ctx(llama_model * model, uint32_t n_seq) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = n_seq; cp.kv_unified = false;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    return llama_init_from_model(model, cp);
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) { setenv(k, v, 1); } else { unsetenv(k); }
#endif
}

static int ring_on(llama_model * model, const char * win, const char * sinks) {
    set_env("LLAMA_KV_WIN", win);
    set_env("LLAMA_KV_WIN_SINKS", sinks);
    g_ring = false;
    llama_context * ctx = make_ctx(model, 1);
    if (ctx == nullptr) {
        return -1;
    }
    llama_free(ctx);
    return g_ring ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s dense.gguf [qsa.gguf]\n", argv[0]);
        return 1;
    }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        return 1;
    }
    const int sinks_only = ring_on(model, nullptr, "4");
    const int killed     = ring_on(model, "-1", "4");
    llama_model_free(model);
    printf("sinks only: ring %d, LLAMA_KV_WIN=-1 beside sinks: ring %d\n", sinks_only, killed);
    if (sinks_only != 1 || killed != 0) {
        return 1;
    }

    if (argc >= 3) {
        model = llama_model_load_from_file(argv[2], mp);
        if (model == nullptr) {
            return 1;
        }
        set_env("LLAMA_KV_WIN", "64");
        set_env("LLAMA_KV_WIN_SINKS", nullptr);
        llama_context * ctx = make_ctx(model, 2);
        if (ctx == nullptr) {
            return 1;
        }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        bool ok = true;
        // a prompt per sequence, then one token of each in the same batch
        for (int step = 0; step < 3 && ok; step++) {
            const int n = step < 2 ? 8 : 2;
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; i++) {
                const int s = step < 2 ? step : i;
                b.token[i] = (llama_token) ((i * 13 + s * 7 + 3) % (n_vocab - 10) + 5);
                b.pos[i] = step < 2 ? i : 8; b.n_seq_id[i] = 1; b.seq_id[i][0] = s; b.logits[i] = step == 2 || i == n - 1;
            }
            b.n_tokens = n;
            ok = llama_decode(ctx, b) == 0;
            llama_batch_free(b);
        }
        llama_free(ctx);
        llama_model_free(model);
        printf("two-stream sparse attention decode: %s\n", ok ? "ok" : "FAILED");
        if (!ok) {
            return 1;
        }
    }
    return 0;
}
