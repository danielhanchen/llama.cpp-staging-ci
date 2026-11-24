// the QSA indexer chain (ggml_lightning_indexer, n_head 4, then ggml_top_k) with the live pool count on the device
// (ggml_lightning_indexer_set_live_len / ggml_top_k_set_live_len, the padded pools past it: mask -inf for every token) against the
// same graph without it, and the multi-row vec indexer kernel (GGML_CUDA_LI_VEC_MULTI, default on) against the per-token kernel
// (=0): every score byte (the padded columns included, -inf) and every top-k id in order must be equal across the four arms
// (live on / off x multi on / off). Each multi mode runs in a child process (this binary with --child MODE FILE; the switch is read
// once per process). Cases: 1..8 tokens (8: the token tile kernel), 1088 / 4160 columns (the 8k bucket), real pools below / at / above
// the top-k count, live below / equal / above the column count, K F16 / F32 / Q8_0, visible prefixes per token. Engagement: the
// backend's "kpool live" lines (indexer and top-k) in every child, "indexer vec multi" in mode 1 only. No CUDA device: SKIP, exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static bool g_live_li   = false;
static bool g_live_topk = false;
static bool g_multi     = false;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "kpool live: the indexer") != nullptr) {
        g_live_li = true;
    }
    if (strstr(text, "kpool live: top-k") != nullptr) {
        g_live_topk = true;
    }
    if (strstr(text, "indexer vec multi") != nullptr) {
        g_multi = true;
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct lcase {
    int       n_tok;
    int       n_kv;
    int       n_real;
    int       n_top;
    int       live; // value uploaded (the graph clamps nothing: the backend takes max(k, min(ncols, live)))
    ggml_type tk;
    std::string str() const {
        char b[160];
        snprintf(b, sizeof(b), "tok %d kv %5d real %5d top %4d live %5d K %s", n_tok, n_kv, n_real, n_top, live, ggml_type_name(tk));
        return b;
    }
};

static std::vector<lcase> cases() {
    std::vector<lcase> v;
    for (int t = 1; t <= 8; ++t) {
        for (ggml_type tk : { GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0 }) {
            if (tk != GGML_TYPE_F16 && t != 1 && t != 7 && t != 8) {
                continue;
            }
            v.push_back({ t, 4160, 2048, 512, 2048, tk });  // the 8k bucket: real pools far below the padded count
            v.push_back({ t, 4160, 100,  512, 512,  tk });  // fewer real pools than the top-k: -inf ties inside the live columns
            v.push_back({ t, 4160, 4159, 512, 4159, tk });  // live just below the column count
            v.push_back({ t, 1088, 1088, 64,  1088, tk });  // live == columns
            v.push_back({ t, 1088, 700,  64,  5000, tk });  // live above the columns (clamped)
            v.push_back({ t, 1088, 3,    64,  64,   tk });  // a short context
        }
    }
    return v;
}

static ggml_backend_dev_t gpu_dev() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "CUDA") == 0) {
            return d;
        }
    }
    return nullptr;
}

// scores (n_kv x n_tok f32 bits) then top-k ids (n_top x n_tok)
static std::vector<uint32_t> run_case(ggml_backend_t backend, const lcase & c, const bool live_on, const int seed) {
    ggml_init_params ip = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int D = 128, H = 4;
    ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, c.n_tok);
    ggml_tensor * k  = ggml_new_tensor_3d(ctx, c.tk, D, 1, c.n_kv);
    ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, c.n_tok);
    ggml_tensor * m  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv, c.n_tok);
    ggml_tensor * lv = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * score = ggml_lightning_indexer(ctx, q, k, w, m);
    ggml_tensor * top   = ggml_top_k(ctx, score, c.n_top);
    if (live_on) {
        ggml_lightning_indexer_set_live_len(score, lv);
        ggml_top_k_set_live_len(top, lv);
    }
    ggml_set_output(score);
    ggml_set_output(top);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, score);
    ggml_build_forward_expand(gf, top);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> vq((size_t) D*H*c.n_tok), vk((size_t) D*c.n_kv), vw((size_t) H*c.n_tok, 1.0f/sqrtf((float) D));
    for (float & x : vq) x = nd(rng);
    for (float & x : vk) x = nd(rng);
    // coarse keys in part of the rows: tied scores
    for (int r = 0; r < c.n_kv; r += 7) {
        for (int i = 0; i < D; ++i) vk[(size_t) r*D + i] = std::round(vk[(size_t) r*D + i]);
    }
    std::vector<ggml_fp16_t> vm((size_t) c.n_kv*c.n_tok);
    for (int t = 0; t < c.n_tok; ++t) {
        // token t sees a prefix of the real pools (the last tokens see all), a few real pools dropped (other sequences)
        const int nv = std::max(0, c.n_real - (c.n_tok - 1 - t)*3);
        for (int i = 0; i < c.n_kv; ++i) {
            const bool keep = i < nv && (rng() % 17) != 0;
            vm[(size_t) t*c.n_kv + i] = ggml_fp32_to_fp16(keep ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(q, vq.data(), 0, ggml_nbytes(q));
    if (c.tk == GGML_TYPE_F32) {
        ggml_backend_tensor_set(k, vk.data(), 0, ggml_nbytes(k));
    } else {
        std::vector<uint8_t> qk(ggml_nbytes(k));
        if (c.tk == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(vk.data(), (ggml_fp16_t *) qk.data(), (int64_t) vk.size());
        } else {
            ggml_quantize_chunk(c.tk, vk.data(), qk.data(), 0, c.n_kv, D, nullptr);
        }
        ggml_backend_tensor_set(k, qk.data(), 0, qk.size());
    }
    ggml_backend_tensor_set(w, vw.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(m, vm.data(), 0, ggml_nbytes(m));
    const int32_t live = c.live;
    ggml_backend_tensor_set(lv, &live, 0, sizeof(live));
    // fill the score with a sentinel first, so an unwritten column shows
    {
        std::vector<uint32_t> junk(ggml_nelements(score), 0x7fc01234u);
        ggml_backend_tensor_set(score, junk.data(), 0, ggml_nbytes(score));
    }
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    std::vector<uint32_t> out(ggml_nelements(score) + ggml_nelements(top));
    ggml_backend_tensor_get(score, out.data(), 0, ggml_nbytes(score));
    ggml_backend_tensor_get(top, out.data() + ggml_nelements(score), 0, ggml_nbytes(top));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static int run_child(const char * path) {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t dev = gpu_dev();
    if (dev == nullptr) return 3;
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) return 3;
    FILE * f = fopen(path, "wb");
    if (f == nullptr) return 4;
    const auto cs = cases();
    for (size_t i = 0; i < cs.size(); ++i) {
        for (int lo = 0; lo < 2; ++lo) {
            const std::vector<uint32_t> o = run_case(backend, cs[i], lo == 1, 11 + (int) i);
            const uint64_t n = o.size();
            fwrite(&n, sizeof(n), 1, f);
            fwrite(o.data(), sizeof(uint32_t), o.size(), f);
        }
    }
    const int32_t eng[3] = { g_live_li ? 1 : 0, g_live_topk ? 1 : 0, g_multi ? 1 : 0 };
    fwrite(eng, sizeof(eng), 1, f);
    fclose(f);
    ggml_backend_free(backend);
    return 0;
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[3]);
    }
    ggml_log_set(log_cb, nullptr);
    if (gpu_dev() == nullptr) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    const auto cs = cases();
    // outs[mode][case][live off / on]
    std::vector<std::vector<uint32_t>> outs[2][2];
    int32_t eng[2][3] = {};
    for (int mode = 0; mode < 2; ++mode) {
        const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/kpool_live_exact_" + std::to_string(mode) + ".bin";
        set_env("GGML_CUDA_LI_VEC_MULTI", mode == 1 ? "1" : "0");
        set_env("GGML_CUDA_TOPK_DETERMINISTIC", "1");
        const std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\"";
        const int rc = std::system(cmd.c_str());
        FILE * f = rc == 0 ? fopen(path.c_str(), "rb") : nullptr;
        if (f == nullptr) {
            printf("FAIL: child %d rc %d\n", mode, rc);
            return 1;
        }
        for (size_t i = 0; i < cs.size(); ++i) {
            for (int lo = 0; lo < 2; ++lo) {
                uint64_t n = 0;
                bool ok = fread(&n, sizeof(n), 1, f) == 1;
                outs[mode][lo].emplace_back(ok ? n : 0);
                ok = ok && fread(outs[mode][lo].back().data(), sizeof(uint32_t), n, f) == n;
                if (!ok) {
                    printf("FAIL: short dump from child %d\n", mode);
                    return 1;
                }
            }
        }
        if (fread(eng[mode], sizeof(eng[mode]), 1, f) != 1) {
            printf("FAIL: short dump from child %d\n", mode);
            return 1;
        }
        fclose(f);
        std::remove(path.c_str());
    }
    int n_bad = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const auto & ref = outs[0][0][i]; // per-token kernel, every key scored
        bool ok = true;
        std::string tag;
        for (int mode = 0; mode < 2; ++mode) {
            for (int lo = 0; lo < 2; ++lo) {
                const bool eq = outs[mode][lo][i] == ref;
                ok = ok && eq;
                tag += std::string(" multi") + std::to_string(mode) + (lo ? "/live" : "/all ") + (eq ? " =" : " DIFF");
            }
        }
        // no sentinel left: every score column was written
        for (int64_t j = 0; j < (int64_t) cs[i].n_kv*cs[i].n_tok; ++j) {
            ok = ok && ref[j] != 0x7fc01234u;
        }
        n_bad += !ok;
        if (!ok || i % 16 == 0) {
            printf("  %s:%s%s\n", cs[i].str().c_str(), tag.c_str(), ok ? "" : "  <- FAIL");
        }
    }
    printf("engagement: indexer live %d / %d, top-k live %d / %d, vec multi %d / %d (mode 0 / 1)\n",
            eng[0][0], eng[1][0], eng[0][1], eng[1][1], eng[0][2], eng[1][2]);
    const bool eng_ok = eng[0][0] && eng[1][0] && eng[0][1] && eng[1][1] && !eng[0][2] && eng[1][2];
    if (n_bad || !eng_ok) {
        printf("FAIL: %d / %zu cases differ, engagement %s\n", n_bad, cs.size(), eng_ok ? "ok" : "WRONG");
        return 1;
    }
    printf("OK: %zu / %zu cases byte-identical (scores + top-k ids) across live on / off x vec multi on / off\n", cs.size(), cs.size());
    return 0;
}
