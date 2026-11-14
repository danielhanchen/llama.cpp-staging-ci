// k19-p3topk: backend sampler chains that start with a top-k give the same per-row outputs (candidate ids and their order, the
// top logits, the sampled token and probabilities) whether build_sampling runs that top-k once per verify window
// (LLAMA_DEV_TOPK_BATCH, default on) or once per row (LLAMA_DEV_TOPK_BATCH=0). Each mode runs in a child process (the engagement
// log line is printed once per process) that decodes a prompt and then windows of 1 .. 8 and 17 rows with every row an output, a window
// whose outputs are not consecutive (the per-row fallback) and two sequences in one batch; chains: the server's device top-k
// (top-k 256 / 20 alone) and a sampling chain (top-k 20, temp 0.7, dist with a fixed seed). Gate: the dumps are equal and the
// batched top-k engaged in the default mode only (no batched windows: FAIL, the comparison would prove nothing); a third child
// repeats the per-row mode, and the two per-row runs must agree (else the device is not repeatable and nothing is proven).
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

static uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const unsigned char *) p)[i]) * 1099511628211ULL;
    }
    return h;
}

static int g_chain = 0;
static int g_call  = 0;

static void dump_row(FILE * f, llama_context * ctx, int32_t idx) {
    const uint32_t nc = llama_get_sampled_candidates_count_ith(ctx, idx);
    const llama_token * c = llama_get_sampled_candidates_ith(ctx, idx);
    const uint32_t nl = llama_get_sampled_logits_count_ith(ctx, idx);
    const float * l = llama_get_sampled_logits_ith(ctx, idx);
    const uint32_t np = llama_get_sampled_probs_count_ith(ctx, idx);
    const float * p = llama_get_sampled_probs_ith(ctx, idx);
    const llama_token t = llama_get_sampled_token_ith(ctx, idx);
    fprintf(f, "chain %d decode %2d row %2d: cand %u %016llx logits %u %016llx probs %u %016llx token %d first %d\n", g_chain, g_call, idx,
            nc, (unsigned long long) (c ? fnv(c, nc * sizeof(llama_token)) : 0), nl, (unsigned long long) (l ? fnv(l, nl * sizeof(float)) : 0),
            np, (unsigned long long) (p ? fnv(p, np * sizeof(float)) : 0), t, c && nc > 0 ? c[0] : -1);
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

    FILE * f = fopen(out_path, "w");
    if (f == nullptr) {
        return 5;
    }

    for (int chain_kind = 0; chain_kind < 3; ++chain_kind) {
        g_chain = chain_kind;
        g_call  = 0;
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
        cparams.n_outputs_max_per_seq = 20;
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
            g_call++;
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
        // a window of 17 rows (CUDA: launches of 15 + 2 rows)
        for (int i = 0; i < 17; ++i) {
            add(next_tok(), pos[0]++, 0, true);
        }
        if (!decode_dump()) return 12;
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
    fprintf(f, "engaged %d\n", g_engaged ? 1 : 0);
    fclose(f);
    llama_model_free(model);
    return 0;
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
    // mode 0: default (batched), 1: LLAMA_DEV_TOPK_BATCH=0, 2: LLAMA_DEV_TOPK_BATCH=0 again (the device's own repeatability)
    std::vector<std::string> lines[3];
    int eng[3] = { -1, -1, -1 };
    for (int mode = 0; mode < 3; ++mode) {
        const std::string path = std::string(argv[1]) + ".topk_batch_" + device + "_" + std::to_string(mode) + ".txt";
        set_env("LLAMA_DEV_TOPK_BATCH", mode == 0 ? "1" : "0");
        const std::string cmd = "\"" + std::string(argv[0]) + "\" --child \"" + argv[1] + "\" " + device + " \"" + path + "\"";
        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            printf("FAIL: child mode %d rc %d\n", mode, rc);
            return 1;
        }
        FILE * f = fopen(path.c_str(), "r");
        char buf[512];
        while (f != nullptr && fgets(buf, sizeof(buf), f)) {
            if (strncmp(buf, "engaged ", 8) == 0) {
                eng[mode] = atoi(buf + 8);
            } else {
                lines[mode].push_back(buf);
            }
        }
        if (f != nullptr) {
            fclose(f);
        }
        std::remove(path.c_str());
    }
    auto ndiff = [&](int a, int b, bool show) {
        int n = (int) std::max(lines[a].size(), lines[b].size()) - (int) std::min(lines[a].size(), lines[b].size());
        for (size_t i = 0; i < std::min(lines[a].size(), lines[b].size()); ++i) {
            if (lines[a][i] != lines[b][i]) {
                if (show && n < 6) {
                    printf("  mode %d: %s  mode %d: %s", a, lines[a][i].c_str(), b, lines[b][i].c_str());
                }
                n++;
            }
        }
        return n;
    };
    const int d_rep = ndiff(1, 2, true);
    const int d_on  = ndiff(0, 1, true);
    const bool rows_equal = d_on == 0 && !lines[0].empty();
    printf("rows %zu: batched vs per-row %d differ, per-row vs per-row %d differ; batched top-k engaged: default %d, "
           "LLAMA_DEV_TOPK_BATCH=0 %d / %d\n", lines[0].size(), d_on, d_rep, eng[0], eng[1], eng[2]);
    if (d_rep != 0) {
        printf("FAIL: the per-row runs differ from each other on this device (no exactness claim can be checked)\n");
        return 1;
    }
    if (!rows_equal || eng[0] != 1 || eng[1] != 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
