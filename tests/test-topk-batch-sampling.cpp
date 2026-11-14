// k19-p3topk: backend sampler chains that start with a top-k give the same per-row outputs (candidate ids and their order, the
// top logits, the sampled token and probabilities) whether build_sampling runs that top-k once per verify window
// (LLAMA_DEV_TOPK_BATCH, default on) or once per row (LLAMA_DEV_TOPK_BATCH=0). Each mode runs in a child process (the engagement
// log line is printed once per process) that decodes a prompt and then windows of 1 .. 8 rows with every row an output, a window
// whose outputs are not consecutive (the per-row fallback) and two sequences in one batch; chains: the server's device top-k
// (top-k 256 / 20 alone) and a sampling chain (top-k 20, temp 0.7, dist with a fixed seed). Gate: the dumps are equal and the
// batched top-k engaged in the default mode only (no batched windows: FAIL, the comparison would prove nothing).
//
// Usage: test-topk-batch-sampling <model.gguf> [cpu|gpu]

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool g_engaged = false;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "k19-p3topk:") != nullptr) {
        g_engaged = true;
    }
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static void dump_row(FILE * f, llama_context * ctx, int32_t idx) {
    const uint32_t nc = llama_get_sampled_candidates_count_ith(ctx, idx);
    const llama_token * c = llama_get_sampled_candidates_ith(ctx, idx);
    const uint32_t nl = llama_get_sampled_logits_count_ith(ctx, idx);
    const float * l = llama_get_sampled_logits_ith(ctx, idx);
    const uint32_t np = llama_get_sampled_probs_count_ith(ctx, idx);
    const float * p = llama_get_sampled_probs_ith(ctx, idx);
    const llama_token t = llama_get_sampled_token_ith(ctx, idx);
    fwrite(&nc, sizeof(nc), 1, f);
    if (c != nullptr) fwrite(c, sizeof(llama_token), nc, f);
    fwrite(&nl, sizeof(nl), 1, f);
    if (l != nullptr) fwrite(l, sizeof(float), nl, f);
    fwrite(&np, sizeof(np), 1, f);
    if (p != nullptr) fwrite(p, sizeof(float), np, f);
    fwrite(&t, sizeof(t), 1, f);
}

static int run_child(const char * model_path, const char * device, const char * out_path) {
    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    auto mparams = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    if (strcmp(device, "gpu") == 0) {
        devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (devs[0] == nullptr) {
            return 3;
        }
        mparams.n_gpu_layers = 999;
    } else {
        devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        mparams.n_gpu_layers = 0;
    }
    mparams.devices = devs;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        return 4;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    FILE * f = fopen(out_path, "wb");
    if (f == nullptr) {
        return 5;
    }

    for (int chain_kind = 0; chain_kind < 3; ++chain_kind) {
        // 0: device top-k 256 (the server's dev_topk chain), 1: top-k 20, 2: top-k 20 + temp 0.7 + dist(seed 1234)
        llama_sampler * ch[2];
        for (int s = 0; s < 2; ++s) {
            ch[s] = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(ch[s], llama_sampler_init_top_k(chain_kind == 0 ? std::min(256, n_vocab) : 20));
            if (chain_kind == 2) {
                llama_sampler_chain_add(ch[s], llama_sampler_init_temp(0.7f));
                llama_sampler_chain_add(ch[s], llama_sampler_init_dist(1234 + s));
            }
        }
        llama_sampler_seq_config cfg[2] = { { 0, ch[0] }, { 1, ch[1] } };

        auto cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        cparams.n_batch = 64;
        cparams.n_ubatch = 64;
        cparams.n_seq_max = 2;
        cparams.n_outputs_max_per_seq = 8;
        cparams.kv_unified = true;
        cparams.samplers = cfg;
        cparams.n_samplers = 2;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            return 6;
        }

        llama_batch batch = llama_batch_init(64, 0, 1);
        auto add = [&](llama_token t, llama_pos pos, llama_seq_id s, bool out) {
            batch.token[batch.n_tokens] = t;
            batch.pos[batch.n_tokens] = pos;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = s;
            batch.logits[batch.n_tokens] = out;
            batch.n_tokens++;
        };
        auto decode_dump = [&]() -> bool {
            if (llama_decode(ctx, batch) != 0) {
                return false;
            }
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                if (batch.logits[i]) {
                    dump_row(f, ctx, i);
                }
            }
            batch.n_tokens = 0;
            return true;
        };

        llama_pos pos[2] = { 0, 0 };
        // prompts, last token an output
        for (int s = 0; s < 2; ++s) {
            for (int i = 0; i < 10; ++i) {
                add((llama_token) ((7 * i + 3 + s * 11) % n_vocab), pos[s]++, s, i == 9);
            }
            if (!decode_dump()) return 7;
        }
        uint32_t tok = 17;
        auto next_tok = [&]() { tok = tok * 1103515245u + 12345u; return (llama_token) ((tok >> 8) % (uint32_t) n_vocab); };
        // verify windows of 1 .. 8 rows on seq 0, every row an output
        for (int w = 1; w <= 8; ++w) {
            for (int i = 0; i < w; ++i) {
                add(next_tok(), pos[0]++, 0, true);
            }
            if (!decode_dump()) return 8;
        }
        // outputs not consecutive: rows 0, 2, 3 of a 4-token batch (per-row fallback)
        for (int i = 0; i < 4; ++i) {
            add(next_tok(), pos[0]++, 0, i != 1);
        }
        if (!decode_dump()) return 9;
        // two sequences in one batch: 3 rows of seq 0 then 5 rows of seq 1 (a window each)
        for (int i = 0; i < 3; ++i) {
            add(next_tok(), pos[0]++, 0, true);
        }
        for (int i = 0; i < 5; ++i) {
            add(next_tok(), pos[1]++, 1, true);
        }
        if (!decode_dump()) return 10;
        // the same window width again (a reused graph)
        for (int r = 0; r < 2; ++r) {
            for (int i = 0; i < 5; ++i) {
                add(next_tok(), pos[0]++, 0, true);
            }
            if (!decode_dump()) return 11;
        }

        llama_batch_free(batch);
        llama_free(ctx);
        llama_sampler_free(ch[0]);
        llama_sampler_free(ch[1]);
    }
    const int32_t eng = g_engaged ? 1 : 0;
    fwrite(&eng, sizeof(eng), 1, f);
    fclose(f);
    llama_model_free(model);
    return 0;
}

static std::vector<char> slurp(const std::string & path) {
    std::vector<char> v;
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return v;
    }
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        v.insert(v.end(), buf, buf + n);
    }
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    if (argc == 5 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[2], argv[3], argv[4]);
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [cpu|gpu]\n", argv[0]);
        return 1;
    }
    const std::string device = argc > 2 ? argv[2] : "cpu";
    if (device == "gpu") {
        ggml_backend_load_all();
        if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
            printf("SKIP: no GPU\n");
            return 0;
        }
    }
    std::vector<char> dumps[2];
    for (int mode = 0; mode < 2; ++mode) {
        const std::string path = std::string(argv[1]) + ".topk_batch_" + device + "_" + std::to_string(mode) + ".bin";
        set_env("LLAMA_DEV_TOPK_BATCH", mode == 0 ? "1" : "0");
        const std::string cmd = "\"" + std::string(argv[0]) + "\" --child \"" + argv[1] + "\" " + device + " \"" + path + "\"";
        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            printf("FAIL: child mode %d rc %d\n", mode, rc);
            return 1;
        }
        dumps[mode] = slurp(path);
        std::remove(path.c_str());
    }
    if (dumps[0].size() < 4 || dumps[1].size() < 4) {
        printf("FAIL: empty dump\n");
        return 1;
    }
    int32_t eng[2];
    for (int m = 0; m < 2; ++m) {
        memcpy(&eng[m], dumps[m].data() + dumps[m].size() - 4, 4);
    }
    // the dumps hold the engagement flag last: compare the rows only
    const bool rows_equal = dumps[0].size() == dumps[1].size() && memcmp(dumps[0].data(), dumps[1].data(), dumps[0].size() - 4) == 0;
    printf("row dumps %zu / %zu bytes: %s; batched top-k engaged: default %d, LLAMA_DEV_TOPK_BATCH=0 %d\n", dumps[0].size(), dumps[1].size(),
            rows_equal ? "equal" : "DIFFER", eng[0], eng[1]);
    if (!rows_equal || eng[0] != 1 || eng[1] != 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
