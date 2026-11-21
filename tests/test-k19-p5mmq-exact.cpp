// k19-p5mmq: the cached-tier MoE prompt MMQ changes, on the GPU backend, at the Qwen3.6-35B-A3B / Flash-Next MoE prompt shapes:
//   exact tiles: a MUL_MAT_ID over a bank of MoE cache slots (op param 2 > 0, skip_empty) runs one block per non-empty tile (the
//                exact prefix of ceil(count/J) tiles) when those fill the SMs, else the stream-k split over one block per SM as before
//                (GGML_CUDA_MMQ_EXACT_TILES=0: always the split, =2: always one block per tile);
//   memo down:   the fused-SwiGLU down product reuses the routing maps its gate / up built (GGML_CUDA_MMID_MEMO_DOWN=0: built again);
//   match any:   the stable scatter ranks the lanes of one expert by __match_any_sync (GGML_CUDA_MMID_MATCH_ANY=0: 32 shuffles).
// Three children of this binary run every case and dump the outputs: "head" (defaults), "kill" (the three switches off = the code
// before k19-p5mmq) and "force" (one block per tile for every bank product). Checks, byte for byte:
//   - force: bank product == the same product without the bank (op param 2 = 0: the compact tiling of the products without skip_empty,
//     one block per tile at the large sizes: >= 10 waves on any card up to 204 SMs). Both sum all of K per tile: same bytes. The small
//     cases are not compared (their plain product is split too).
//   - head:  bank product == the force one or the kill one (the device picked one of the two forms); both forms seen (stats lines).
//   - kill:  negative control: a bank product split over the SMs differs from the one-block-per-tile result in at least one case
//            (the split changes the rounding: the comparison above can see a K split).
//   - the gate / up / SwiGLU / down graphs: head == kill (the routing maps are the same integers); head reuses the maps (log line),
//     kill does not; the SwiGLU is fused into the down product in both.
//   - all slots skipped (no column at all): zeros in every child.
// No CUDA device: SKIP, exit 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct p5_case {
    std::string name;
    bool        ffn;       // gate / up / SwiGLU / down graph (type = gate / up type, type_d = down type) instead of one product
    ggml_type   type;
    ggml_type   type_d;
    int64_t     K, N, E, topk;
    bool        bcast;     // one product: src1 [K, 1, T] (gate / up) instead of [K, topk, T] (down)
    int64_t     tokens;
    float       skip_frac; // share of slots (after the first of each token) set to -1; > 1: every slot
    float       skew;
    uint32_t    seed;
};

static std::vector<p5_case> cases() {
    std::vector<p5_case> cs;
    // Qwen3.6 MoE (35B-A3B class): 256 experts, 8 used, n_embd 2048, n_ff_exp 512
    cs.push_back({ "q36_gate_q4k_8192",   false, GGML_TYPE_Q4_K,   GGML_TYPE_COUNT, 2048,  512, 256,  8, true,  8192, 0.0f,  0.5f, 1 });
    cs.push_back({ "q36_down_q6k_4096",   false, GGML_TYPE_Q6_K,   GGML_TYPE_COUNT,  512, 2048, 256,  8, false, 4096, 0.0f,  0.5f, 2 });
    cs.push_back({ "q36_gate_iq4xs_skip", false, GGML_TYPE_IQ4_XS, GGML_TYPE_COUNT, 2048,  512, 256,  8, true,  8192, 0.3f,  1.0f, 3 });
    // Flash-Next (qwen4exp): 512 experts, 10 used, n_embd 2560, n_ff_exp 640
    cs.push_back({ "fn_gate_iq3s_4096",   false, GGML_TYPE_IQ3_S,  GGML_TYPE_COUNT, 2560,  640, 512, 10, true,  4096, 0.0f,  0.5f, 4 });
    cs.push_back({ "fn_down_iq4nl_4096",  false, GGML_TYPE_IQ4_NL, GGML_TYPE_COUNT,  640, 2560, 512, 10, false, 4096, 0.1f,  0.5f, 5 });
    // small products (16 experts, 4 used, 32 tokens: 16 column tiles of J 8 x 4 or 8 row tiles, at most 80% of the last wave on T4 /
    // L4 / A100 / G4 / B200, so the split stays for the gate)
    cs.push_back({ "small_gate_q4k_32",   false, GGML_TYPE_Q4_K,   GGML_TYPE_COUNT, 2048,  512,  16,  4, true,    32, 0.0f,  0.0f, 6 });
    cs.push_back({ "small_down_q6k_32",   false, GGML_TYPE_Q6_K,   GGML_TYPE_COUNT,  512, 2048,  16,  4, false,   32, 0.2f,  0.0f, 7 });
    // no column at all
    cs.push_back({ "q36_gate_q4k_none",   false, GGML_TYPE_Q4_K,   GGML_TYPE_COUNT, 2048,  512, 256,  8, true,   512, 2.0f,  0.0f, 8 });
    // gate / up / SwiGLU / down: the maps of gate / up reused by the down (prompt ubatches)
    cs.push_back({ "q36_ffn_q4k_q6k_2048", true,  GGML_TYPE_Q4_K,  GGML_TYPE_Q6_K,  2048,  512, 256,  8, true,  2048, 0.0f,  0.5f, 9 });
    cs.push_back({ "q36_ffn_skip_1024",    true,  GGML_TYPE_Q4_K,  GGML_TYPE_Q6_K,  2048,  512, 256,  8, true,  1024, 0.2f,  0.5f, 10 });
    cs.push_back({ "fn_ffn_iq3s_iq4nl_1024", true, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, 2560, 640, 512, 10, true, 1024, 0.0f,  0.5f, 11 });
    return cs;
}

static void fill_weights(ggml_tensor * w, std::mt19937 & rng) {
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    const int64_t K     = w->ne[0];
    const int64_t nrow  = w->ne[1]*w->ne[2];
    const int64_t npool = std::min<int64_t>(nrow, 2048);
    const size_t  rs    = ggml_row_size(w->type, K);
    std::vector<float> f(K*npool);
    for (auto & v : f) {
        v = ud(rng);
    }
    std::vector<uint8_t> pool(rs*npool);
    ggml_quantize_chunk(w->type, f.data(), pool.data(), 0, npool, K, nullptr);
    std::vector<uint8_t> q(rs*nrow);
    std::uniform_int_distribution<int64_t> pick(0, npool - 1);
    for (int64_t r = 0; r < nrow; ++r) {
        memcpy(q.data() + r*rs, pool.data() + pick(rng)*rs, rs);
    }
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
}

static std::vector<int32_t> make_ids(const p5_case & c, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<int32_t> id(c.topk*c.tokens);
    for (int64_t t = 0; t < c.tokens; ++t) {
        std::vector<bool> used(c.E, false);
        for (int64_t s = 0; s < c.topk; ++s) {
            int32_t e;
            do {
                const float r = u01(rng);
                e = (int32_t) (c.E * (c.skew > 0.0f ? std::pow(r, 1.0f + c.skew) : r));
                e = std::min<int32_t>(e, (int32_t) c.E - 1);
            } while (used[e]);
            used[e] = true;
            const bool skip = c.skip_frac > 1.0f || (s > 0 && u01(rng) < c.skip_frac);
            id[t*c.topk + s] = skip ? -1 : e;
        }
    }
    return id;
}

// one product: two outputs (the bank product, op param 2 = E, and the same product without the bank); ffn: one output
static std::vector<std::vector<uint8_t>> run_case(ggml_backend_t backend, const p5_case & c) {
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> outs;
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.topk, c.tokens);
    std::vector<ggml_tensor *> ws;
    ggml_tensor * x = nullptr;
    ggml_cgraph * g = ggml_new_graph(ctx);
    if (!c.ffn) {
        ggml_tensor * w = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, c.E);
        x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.bcast ? 1 : c.topk, c.tokens);
        ws.push_back(w);
        for (int bank = 1; bank >= 0; --bank) {
            ggml_tensor * o = ggml_mul_mat_id(ctx, w, x, ids);
            ((int32_t *) o->op_params)[2] = bank ? (int32_t) c.E : 0;
            ggml_set_output(o);
            outs.push_back(o);
            ggml_build_forward_expand(g, o);
        }
    } else {
        ggml_tensor * wg = ggml_new_tensor_3d(ctx, c.type,   c.K, c.N, c.E);
        ggml_tensor * wu = ggml_new_tensor_3d(ctx, c.type,   c.K, c.N, c.E);
        ggml_tensor * wd = ggml_new_tensor_3d(ctx, c.type_d, c.N, c.K, c.E);
        ws = { wg, wu, wd };
        x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, 1, c.tokens);
        ggml_tensor * gt = ggml_mul_mat_id(ctx, wg, x, ids);
        ggml_tensor * up = ggml_mul_mat_id(ctx, wu, x, ids);
        ggml_tensor * a  = ggml_swiglu_split(ctx, gt, up);
        ggml_tensor * d  = ggml_mul_mat_id(ctx, wd, a, ids);
        ggml_set_output(d);
        outs.push_back(d);
        ggml_build_forward_expand(g, d);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (ggml_tensor * w : ws) {
        fill_weights(w, rng);
    }
    {
        std::vector<float> f(ggml_nelements(x));
        for (auto & v : f) {
            v = ud(rng);
        }
        ggml_backend_tensor_set(x, f.data(), 0, f.size()*sizeof(float));
        const std::vector<int32_t> id = make_ids(c, rng);
        ggml_backend_tensor_set(ids, id.data(), 0, id.size()*sizeof(int32_t));
    }
    std::vector<std::vector<uint8_t>> res;
    for (ggml_tensor * o : outs) {
        std::vector<uint8_t> p(ggml_nbytes(o), 0x7f); // poison: an element nothing writes differs
        ggml_backend_tensor_set(o, p.data(), 0, p.size());
    }
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "case %s: compute failed\n", c.name.c_str());
        res.assign(outs.size(), std::vector<uint8_t>());
    } else {
        for (ggml_tensor * o : outs) {
            std::vector<uint8_t> v(ggml_nbytes(o));
            ggml_backend_tensor_get(o, v.data(), 0, v.size());
            res.push_back(std::move(v));
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static ggml_backend_dev_t gpu_dev() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && strstr(ggml_backend_dev_name(d), "CUDA") != nullptr) {
            return d;
        }
    }
    return nullptr;
}

static int run_child(const char * path) {
    ggml_backend_dev_t dev = gpu_dev();
    if (dev == nullptr) {
        return 2;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        return 3;
    }
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        return 4;
    }
    for (const p5_case & c : cases()) {
        const std::vector<std::vector<uint8_t>> o = run_case(backend, c);
        for (const auto & v : o) {
            const uint64_t n = v.size();
            fwrite(&n, sizeof(n), 1, f);
            fwrite(v.data(), 1, v.size(), f);
        }
    }
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

static bool read_dump(const std::string & path, std::vector<std::vector<uint8_t>> & outs, size_t n) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        uint64_t m = 0;
        if (fread(&m, sizeof(m), 1, f) != 1) {
            fclose(f);
            return false;
        }
        outs.emplace_back(m);
        if (m > 0 && fread(outs.back().data(), 1, m, f) != m) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

static std::string slurp(const std::string & path) {
    std::string s;
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return s;
    }
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof(b), f)) > 0) {
        s.append(b, n);
    }
    fclose(f);
    return s;
}

static bool same(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()) == 0;
}

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[3]);
    }
    if (gpu_dev() == nullptr) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    const std::vector<p5_case> cs = cases();
    size_t n_out = 0;
    for (const auto & c : cs) {
        n_out += c.ffn ? 1 : 2;
    }
    static const char * names[3] = { "head", "kill", "force" };
    std::vector<std::vector<uint8_t>> outs[3];
    std::string err[3];
    const std::string dir = argc > 1 ? argv[1] : ".";
    for (int mode = 0; mode < 3; ++mode) {
        const std::string path = dir + "/k19_p5mmq_exact_" + std::to_string(mode) + ".bin";
        const std::string errp = path + ".err";
        set_env("GGML_CUDA_MMQ_EXACT_TILES_STATS", "1");
        set_env("GGML_CUDA_SWIGLU_MMID_LOG", "1");
        set_env("GGML_CUDA_MMQ_EXACT_TILES", mode == 1 ? "0" : mode == 2 ? "2" : "1");
        set_env("GGML_CUDA_MMID_MEMO_DOWN",  mode == 1 ? "0" : "1");
        set_env("GGML_CUDA_MMID_MATCH_ANY",  mode == 1 ? "0" : "1");
        std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\" 2> \"" + errp + "\"";
#ifdef _WIN32
        cmd = "\"" + cmd + "\"";
#endif
        const int rc = std::system(cmd.c_str());
        err[mode] = slurp(errp);
        if (rc != 0 || !read_dump(path, outs[mode], n_out)) {
            fprintf(stderr, "FAIL: child %s rc %d (dump missing, see %s)\n", names[mode], rc, errp.c_str());
            return 1;
        }
        std::remove(path.c_str());
        std::remove(errp.c_str());
    }
    int n_fail = 0;
    bool neg_fired = false;
    size_t k = 0;
    for (const auto & c : cs) {
        if (c.ffn) {
            const bool ok = same(outs[0][k], outs[1][k]) && same(outs[0][k], outs[2][k]);
            printf("  %-24s ffn  head == kill == force: %s\n", c.name.c_str(), ok ? "memcmp == 0" : "DIFFER");
            n_fail += !ok;
            k += 1;
            continue;
        }
        const auto & hb = outs[0][k]; const auto & hp = outs[0][k + 1];
        const auto & kb = outs[1][k]; const auto & kp = outs[1][k + 1];
        const auto & fb = outs[2][k]; const auto & fp = outs[2][k + 1];
        // the products without the bank do not change in any child
        const bool plain_ok = same(hp, kp) && same(hp, fp);
        // force: one block per tile == the plain compact tiling (all of K per tile in both). Only where the plain product runs one block per
        // tile, i.e. the large cases (>= 10 waves); the small ones run the plain product split as well
        const bool big = c.tokens >= 512;
        const bool force_ok = !big || same(fb, fp);
        // head: one of the two forms
        const bool head_tiles = same(hb, fb);
        const bool head_split = same(hb, kb);
        const bool head_ok = head_tiles || head_split;
        const bool kill_differs = !same(kb, fb);
        neg_fired = neg_fired || kill_differs;
        const bool ok = plain_ok && force_ok && head_ok;
        printf("  %-24s plain same in all %s, force bank == plain %s, head bank == %s, kill split %s\n", c.name.c_str(),
            plain_ok ? "yes" : "NO", !big ? (same(fb, fp) ? "yes (not checked)" : "no (not checked: plain split too)") : force_ok ? "yes" : "NO",
            head_tiles && head_split ? "tiles == split" : head_tiles ? "tiles" : head_split ? "split" : "NEITHER",
            kill_differs ? "differs (other rounding)" : "same bytes");
        n_fail += !ok;
        k += 2;
    }
    const bool eng_tiles = err[0].find("device form: one block per tile") != std::string::npos;
    const bool eng_split = err[0].find("device form: stream-k split") != std::string::npos;
    const bool eng_memo  = err[0].find("MMID routing maps reused by the fused-SwiGLU down") != std::string::npos;
    const bool kill_none = err[1].find("MMQ exact tiles") == std::string::npos && err[1].find("MMID routing maps reused") == std::string::npos;
    const bool force_all = err[2].find("device form: one block per tile") != std::string::npos;
    const bool swiglu    = err[0].find("swiglu mmid:") != std::string::npos && err[1].find("swiglu mmid:") != std::string::npos;
    printf("engagement: head one block per tile %s, head split %s, head maps reused by the down %s, kill none %s, force tiles %s, "
           "SwiGLU fused into the down %s\n", eng_tiles ? "yes" : "NO", eng_split ? "yes" : "NO", eng_memo ? "yes" : "NO",
           kill_none ? "yes" : "NO", force_all ? "yes" : "NO", swiglu ? "yes" : "NO");
    printf("negative control (a bank product split over the SMs differs from one block per tile): %s\n", neg_fired ? "fired" : "DID NOT FIRE");
    if (!eng_tiles || !eng_split || !eng_memo || !kill_none || !force_all || !swiglu || !neg_fired) {
        printf("FAIL: engagement / negative control\n");
        return 1;
    }
    if (n_fail) {
        printf("FAIL: %d / %zu cases\n", n_fail, cs.size());
        return 1;
    }
    printf("OK: %zu / %zu cases\n", cs.size(), cs.size());
    return 0;
}
