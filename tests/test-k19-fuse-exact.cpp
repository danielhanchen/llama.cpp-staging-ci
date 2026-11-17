// Bit identity of the k19-fuse prompt fusions of qwen4exp (Qwen3.8-Flash-Next), at the real prompt shapes:
//   F4: the PLE conv taps read from a channels-first padded input (llama_model_qwen4exp::graph::build_ple, LLAMA_QWEN4EXP_PLE_CF) vs
//       the token-major input with one transposed copy per tap: both graphs run on the CPU backend (always) and on the GPU backend (in
//       both children), the outputs and the written conv-state rows must match byte for byte;
//   F5: the CUDA prompt conv-input concat elide (GGML_CUDA_DISABLE_CONV_CAT_PROMPT): state ++ transposed tokens -> SSM_CONV + SILU and
//       the per-rollback-slot tail copies into the cache (slot 0 over the state row the convolution reads);
//   F6: the CUDA QSA prompt mask built in one launch (GGML_CUDA_DISABLE_QSA_PROMPT_MASK): the dense selection mask of build_qsa_sel
//       feeding flash attention;
//   k19-fu2 F1: the gated hc combine (SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST) computed in the next mix's grouped norm, which writes
//       the q8_1 MMQ copy of its Q8_0 product (GGML_CUDA_DISABLE_FUSE_HC_POST_CONV);
//   k19-fu2 F2: the gated delta net output norm writing the q8_1 MMQ copy of its out projection (GGML_CUDA_DISABLE_FUSE_GATED_NORM_MMQ);
//   k19-fu2 F3: the routed SWIGLU computed in the gathered quantize of the down MUL_MAT_ID (GGML_CUDA_DISABLE_FUSE_SWIGLU_MMID); the SWIGLU
//       output itself is not written when fused, so only the product is compared.
// The CUDA kill switches are read once per process, so each mode runs in a child process (this binary with --child MODE FILE) that
// dumps every case's outputs; the parent compares the dumps byte for byte. GGML_CUDA_CONV_CAT_PROMPT_LOG=1 / GGML_CUDA_QSA_PROMPT_LOG=1
// (and GGML_CUDA_HC_POST_CONV_LOG / GGML_CUDA_GATED_NORM_MMQ_LOG / GGML_CUDA_SWIGLU_MMID_LOG) make the backend log every fused launch: the fused child must log exactly the expected launches, the other none, so a silent
// fallback fails the test. No CUDA / ROCm device: the F4 CPU check runs, then SKIP for the rest, exit 0.

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

// fused launches logged per kind: F5, F6, F1, F2, F3
enum { L_CONV, L_QSA, L_HCN, L_GN, L_SWM, L_COUNT };
static int n_logs[L_COUNT] = {};

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "conv cat prompt:") != nullptr && strstr(text, "read in place") != nullptr) {
        n_logs[L_CONV]++;
        return;
    }
    if (strstr(text, "qsa prompt mask:") != nullptr) {
        n_logs[L_QSA]++;
        return;
    }
    if (strstr(text, "hc post conv:") != nullptr) {
        n_logs[L_HCN]++;
        return;
    }
    if (strstr(text, "gated norm mmq:") != nullptr) {
        n_logs[L_GN]++;
        return;
    }
    if (strstr(text, "swiglu mmid:") != nullptr) {
        n_logs[L_SWM]++;
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

enum case_kind { K_PLE, K_CONV, K_QSA, K_HCN, K_GN, K_SWM };

struct fuse_case {
    case_kind kind;
    // PLE: hc_dim, n_t, n_slots, w_bf16; CONV: channels, n_t, n_slots, cpy_after; QSA: kv, n_t, nr, n_sel; HCN: n_embd, n_t, gated, -;
    // GN: head dim, n_t, heads, out type (0 Q8_0 = D4, 1 Q4_K = DS4: not fused); SWM: n_ff, n_t, n_used, skipped slots
    int64_t   a, b, c, d;
    bool      cf = false; // PLE: channels-first graph
    bool      fused() const { // the fused child must log one launch
        switch (kind) {
            case K_CONV: return d == 0;
            case K_QSA:  return true;
            case K_HCN:  return c != 0;
            case K_GN:   return d == 0;
            case K_SWM:  return true;
            default:     return false;
        }
    }
    int       log_kind() const { return kind == K_CONV ? L_CONV : kind == K_QSA ? L_QSA : kind == K_HCN ? L_HCN : kind == K_GN ? L_GN : L_SWM; }
    std::string str() const {
        char buf[160];
        if (kind == K_PLE) {
            snprintf(buf, sizeof(buf), "F4 PLE conv  hc_dim=%lld n_t=%lld slots=%lld w=%s %s", (long long) a, (long long) b,
                     (long long) c, d ? "bf16" : "f32", cf ? "channels-first" : "token-major");
        } else if (kind == K_CONV) {
            snprintf(buf, sizeof(buf), "F5 conv cat  channels=%lld n_t=%lld slots=%lld%s", (long long) a, (long long) b, (long long) c,
                     d ? " cpy_after (refused)" : "");
        } else if (kind == K_QSA) {
            snprintf(buf, sizeof(buf), "F6 QSA mask  kv=%lld n_t=%lld nr=%lld n_sel=%lld", (long long) a, (long long) b, (long long) c,
                     (long long) d);
        } else if (kind == K_HCN) {
            snprintf(buf, sizeof(buf), "F1 hc post + norm  n_embd=%lld n_t=%lld %s", (long long) a, (long long) b,
                     c ? "gated" : "ungated (refused)");
        } else if (kind == K_GN) {
            snprintf(buf, sizeof(buf), "F2 gated norm mmq  hd=%lld n_t=%lld heads=%lld %s", (long long) a, (long long) b, (long long) c,
                     d ? "Q4_K (DS4, refused)" : "Q8_0");
        } else {
            snprintf(buf, sizeof(buf), "F3 swiglu mmid  n_ff=%lld n_t=%lld n_used=%lld%s", (long long) a, (long long) b, (long long) c,
                     d ? " skipped slots" : "");
        }
        return buf;
    }
};

static std::vector<fuse_case> gpu_cases() {
    std::vector<fuse_case> s;
    // GDN conv input of Flash-Next (10240 channels, d_conv 4): across the 32-token long-kernel boundary, a partial last tile, the
    // ubatch sizes; one slot (no drafting) and three (two rollback slots)
    for (int64_t nt : {33, 100, 512, 2048}) {
        for (int64_t slots : {1, 3}) {
            s.push_back({K_CONV, 10240, nt, slots, 0});
        }
    }
    s.push_back({K_CONV, 8192, 512, 3, 0});  // Qwen3.6 conv channels
    s.push_back({K_CONV, 10240, 512, 3, 1}); // the copies after the convolution: refused, the concat runs
    // QSA layers: 2 KV heads x 12, head 256, 2051 selected cells, prompt ubatches at 8k / 30k of context
    for (int64_t kv : {8192, 30720}) {
        for (int64_t nt : {512, 2048}) {
            s.push_back({K_QSA, kv, nt, 12, 2051});
        }
    }
    s.push_back({K_QSA, 4096, 40, 8, 2051});
    // k19-fu2: the Flash-Next prompt shapes (hc 4 x 2560 into the Q8_0 hc down 10240 -> 320; GDN 48 heads x 128 into the Q8_0 out
    // projection 6144 -> 2560; 8 of 64 experts, n_ff 640, IQ4_NL down 640 -> 2560), a partial tile and the ubatch sizes
    for (int64_t nt : {33, 512, 2048}) {
        s.push_back({K_HCN, 2560, nt, 1, 0});
        s.push_back({K_GN, 128, nt, 48, 0});
        s.push_back({K_SWM, 640, nt, 8, 0});
        s.push_back({K_SWM, 640, nt, 8, 1});
    }
    s.push_back({K_HCN, 2560, 512, 0, 0}); // ungated post: not taken
    s.push_back({K_GN, 128, 512, 48, 1});  // DS4 product: not taken
    return s;
}

static std::vector<fuse_case> ple_cases() {
    std::vector<fuse_case> s;
    // the PLE conv of Flash-Next: 4 hc streams x 2560, kernel 4, dilation 3 (9 history columns)
    for (int64_t nt : {32, 512, 2048}) {
        for (int64_t slots : {1, 3}) {
            for (int64_t bf : {0, 1}) {
                fuse_case t = {K_PLE, 10240, nt, slots, bf};
                s.push_back(t);
                t.cf = true;
                s.push_back(t);
            }
        }
    }
    return s;
}

struct built {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    std::vector<ggml_tensor *> outs;
};

// the graphs as qwen4exp builds them (build_conv_state_at, build_ple, build_qsa_sel + build_attn_qsa)
static built build_case(const fuse_case & t) {
    built r;
    ggml_init_params ip = { 512 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    r.ctx = ggml_init(ip);
    ggml_context * ctx = r.ctx;
    r.gf = ggml_new_graph(ctx);
    if (t.kind == K_CONV || t.kind == K_PLE) {
        const bool ple = t.kind == K_PLE;
        const int64_t C = t.a, n_t = t.b, n_slots = t.c, mem_size = 2, head = 1;
        const int64_t kern = 4, dil = 3;
        const int64_t sc = ple ? (kern - 1) * dil : 3;
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sc * C, n_slots * mem_size);
        ggml_set_name(cache, "cache");
        ggml_tensor * state = ggml_reshape_3d(ctx, ggml_view_2d(ctx, cache, sc * C, 1, cache->nb[1], head * cache->nb[1]), sc, C, 1);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, n_t, 1);
        ggml_set_name(x, "x");
        ggml_tensor * input;
        if (ple && t.cf) {
            input = ggml_concat(ctx, ggml_cont(ctx, ggml_transpose(ctx, state)), x, 1);
        } else {
            input = ggml_concat(ctx, state, ggml_transpose(ctx, x), 0);
        }
        std::vector<ggml_tensor *> cpys;
        for (int64_t slot = 0; slot < n_slots; ++slot) {
            ggml_tensor * dst = ggml_view_3d(ctx, cache, sc, C, 1, sc * sizeof(float), cache->nb[1], (slot * mem_size + head) * cache->nb[1]);
            if (ple && t.cf) {
                const int64_t s_idx = std::max<int64_t>(0, input->ne[1] - sc - slot);
                ggml_tensor * tail = ggml_view_3d(ctx, input, C, sc, 1, input->nb[1], input->nb[2], s_idx * input->nb[1]);
                cpys.push_back(ggml_cpy(ctx, ggml_transpose(ctx, tail), dst));
            } else {
                const int64_t s_idx = std::max<int64_t>(0, input->ne[0] - sc - slot);
                ggml_tensor * tail = ggml_view_3d(ctx, input, sc, C, 1, input->nb[1], input->nb[2], s_idx * sizeof(float));
                cpys.push_back(ggml_cpy(ctx, tail, dst));
            }
        }
        const bool cpy_after = !ple && t.d;
        if (!cpy_after) {
            for (ggml_tensor * c : cpys) {
                ggml_build_forward_expand(r.gf, c);
            }
        }
        ggml_tensor * out;
        if (!ple) {
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, C);
            ggml_set_name(w, "w");
            out = ggml_silu(ctx, ggml_ssm_conv(ctx, input, w));
        } else {
            ggml_tensor * w = ggml_new_tensor_2d(ctx, t.d ? GGML_TYPE_BF16 : GGML_TYPE_F32, kern, C);
            ggml_set_name(w, "w");
            out = nullptr;
            for (int64_t k = 0; k < kern; ++k) {
                const int64_t start = sc - (kern - 1 - k) * dil;
                ggml_tensor * shifted = t.cf ?
                    ggml_view_3d(ctx, input, C, n_t, 1, input->nb[1], input->nb[2], start * input->nb[1]) :
                    ggml_cont(ctx, ggml_transpose(ctx, ggml_view_3d(ctx, input, n_t, C, 1, input->nb[1], input->nb[2], start * sizeof(float))));
                ggml_tensor * wk = ggml_reshape_1d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, w, 1, C, w->nb[1], k * w->nb[0])), C);
                if (wk->type != GGML_TYPE_F32) {
                    wk = ggml_cast(ctx, wk, GGML_TYPE_F32);
                }
                ggml_tensor * term = ggml_mul(ctx, shifted, wk);
                out = out ? ggml_add(ctx, out, term) : term;
            }
            out = ggml_silu(ctx, out);
            if (!t.cf) {
                out = ggml_cont(ctx, out);
            }
        }
        ggml_build_forward_expand(r.gf, out);
        if (cpy_after) {
            for (ggml_tensor * c : cpys) {
                ggml_build_forward_expand(r.gf, c);
            }
        }
        r.outs = { out, cache };
    } else if (t.kind == K_HCN) {
        // qwen4exp build_hc_combine (gated) of block i, then build_hc_mix's grouped norm of block i + 1 and its hc down / inject products
        const int64_t n_embd = t.a, hc = 4, n_t = t.b;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_t);
        ggml_set_name(x, "x");
        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_t);
        ggml_set_name(residual, "residual");
        ggml_tensor * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_t);
        ggml_set_name(post, "post");
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, hc);
        ggml_set_name(w, "w");
        ggml_tensor * wd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd*hc, 320);
        ggml_set_name(wd, "wd");
        ggml_tensor * wi = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd*hc, hc);
        ggml_set_name(wi, "wi");
        if (t.c) {
            post = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, post, 1.0f / (float) hc)), 2.0f);
        }
        ggml_tensor * hc_out = ggml_dsv4_hc_post(ctx, x, residual, post, nullptr);
        ggml_tensor * normed = ggml_mul(ctx, ggml_rms_norm(ctx, hc_out, 1e-6f), w);
        ggml_tensor * flat = ggml_reshape_2d(ctx, normed, n_embd*hc, n_t);
        ggml_tensor * mm_i = ggml_mul_mat(ctx, wi, flat);
        ggml_tensor * mm_d = ggml_mul_mat(ctx, wd, flat);
        ggml_build_forward_expand(r.gf, mm_i);
        ggml_build_forward_expand(r.gf, mm_d);
        r.outs = { hc_out, normed, mm_d, mm_i };
    } else if (t.kind == K_GN) {
        // qwen4exp build_norm_gated (gated delta net output) into ssm_out
        const int64_t hd = t.a, n_t = t.b, heads = t.c;
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, n_t);
        ggml_set_name(x, "x");
        ggml_tensor * z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, n_t);
        ggml_set_name(z, "z");
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_set_name(w, "w");
        ggml_tensor * wo = ggml_new_tensor_2d(ctx, t.d ? GGML_TYPE_Q4_K : GGML_TYPE_Q8_0, hd*heads, 2560);
        ggml_set_name(wo, "wo");
        ggml_tensor * gated = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w), ggml_sigmoid(ctx, z));
        ggml_tensor * out = ggml_mul_mat(ctx, wo, ggml_reshape_2d(ctx, gated, hd*heads, n_t));
        ggml_build_forward_expand(r.gf, out);
        r.outs = { gated, out };
    } else if (t.kind == K_SWM) {
        // build_moe_ffn: SWIGLU of the gate / up products, the down MUL_MAT_ID (gate / up are inputs here)
        const int64_t n_ff = t.a, n_t = t.b, n_used = t.c, n_exp = 64;
        ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_used, n_t);
        ggml_set_name(g, "g");
        ggml_tensor * u = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_used, n_t);
        ggml_set_name(u, "u");
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_t);
        ggml_set_name(ids, t.d ? "ids_skip" : "ids");
        ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_NL, n_ff, 2560, n_exp);
        ggml_set_name(down, "down");
        ggml_tensor * out = ggml_mul_mat_id(ctx, down, ggml_swiglu_split(ctx, g, u), ids);
        ggml_build_forward_expand(r.gf, out);
        r.outs = { out };
    } else {
        const int64_t hs = 256, nh = 2, kv = t.a, nb = t.b, nr = t.c, n_sel = t.d;
        ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, nb, nh * nr, 1);
        ggml_set_name(q, "q");
        ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hs, kv, nh, 1);
        ggml_set_name(k, "k");
        ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hs, kv, nh, 1);
        ggml_set_name(v, "v");
        ggml_tensor * kq = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb, 1, 1);
        ggml_set_name(kq, "kq");
        ggml_tensor * idx = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_sel, nb, 1);
        ggml_set_name(idx, "idx");
        ggml_tensor * mask_all = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv + n_sel, 1, 1, 1), -INFINITY);
        mask_all = ggml_repeat_4d(ctx, mask_all, kv + n_sel, nb, 1, 1);
        mask_all = ggml_reshape_3d(ctx, mask_all, 1, kv + n_sel, nb);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_sel, 1, 1, 1), 0.0f);
        zeros = ggml_repeat_4d(ctx, zeros, n_sel, nb, 1, 1);
        zeros = ggml_reshape_3d(ctx, zeros, 1, n_sel, nb);
        ggml_tensor * sel = ggml_set_rows(ctx, mask_all, zeros, idx);
        const size_t row = sel->nb[2];
        sel = ggml_view_4d(ctx, sel, kv, nb, 1, 1, row, row * nb, row * nb, 0);
        sel = ggml_add(ctx, sel, kq);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, sel, 1.0f / sqrtf((float) hs), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_n_kv_max(out, (int32_t) n_sel);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        ggml_build_forward_expand(r.gf, out);
        r.outs = { out };
    }
    return r;
}

static void init_inputs(const fuse_case & t, ggml_context * ctx, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (ggml_tensor * x = ggml_get_first_tensor(ctx); x != nullptr; x = ggml_get_next_tensor(ctx, x)) {
        if (x->buffer == nullptr || x->view_src != nullptr || x->op != GGML_OP_NONE) {
            continue;
        }
        const int64_t n = ggml_nelements(x);
        if (strcmp(x->name, "kq") == 0) {
            // causal: token j sees the cells up to kv - nb + j
            const int64_t kv = x->ne[0], nb = x->ne[1];
            std::vector<ggml_fp16_t> m(kv * nb);
            for (int64_t j = 0; j < nb; ++j) {
                for (int64_t c = 0; c < kv; ++c) {
                    m[j * kv + c] = ggml_fp32_to_fp16(c <= kv - nb + j ? 0.0f : -INFINITY);
                }
            }
            ggml_backend_tensor_set(x, m.data(), 0, ggml_nbytes(x));
        } else if (strcmp(x->name, "idx") == 0) {
            // live slots: distinct cells (some past the causal limit), one duplicate, every 7th slot dead (its dump row >= kv)
            const int64_t n_sel = x->ne[0], nb = x->ne[1], kv = t.a;
            std::vector<int32_t> ids(n_sel * nb);
            std::vector<int32_t> order(kv);
            for (int64_t c = 0; c < kv; ++c) {
                order[c] = (int32_t) c;
            }
            for (int64_t j = 0; j < nb; ++j) {
                std::shuffle(order.begin(), order.end(), rng);
                for (int64_t s = 0; s < n_sel; ++s) {
                    ids[j * n_sel + s] = s % 7 == 3 ? (int32_t) (kv + s) : order[s];
                }
                ids[j * n_sel + 1] = ids[j * n_sel + 0];
                ids[j * n_sel + 2] = (int32_t) (kv - 1);
            }
            ggml_backend_tensor_set(x, ids.data(), 0, ggml_nbytes(x));
        } else if (strncmp(x->name, "ids", 3) == 0) {
            // n_used distinct experts of 64 per token; ids_skip: every 5th token's last two slots are -1 (skipped)
            const int64_t n_used = x->ne[0], n_t = x->ne[1];
            std::vector<int32_t> ids(n_used * n_t);
            std::vector<int32_t> order(64);
            for (int e = 0; e < 64; ++e) {
                order[e] = e;
            }
            for (int64_t j = 0; j < n_t; ++j) {
                std::shuffle(order.begin(), order.end(), rng);
                for (int64_t s = 0; s < n_used; ++s) {
                    ids[j * n_used + s] = (x->name[3] == '_' && j % 5 == 2 && s >= n_used - 2) ? -1 : order[s];
                }
            }
            ggml_backend_tensor_set(x, ids.data(), 0, ggml_nbytes(x));
        } else if (ggml_is_quantized(x->type)) {
            const int64_t n_per_row = x->ne[0];
            const int64_t nrows = n / n_per_row;
            std::vector<float> f(n);
            for (auto & v : f) {
                v = u(rng);
            }
            std::vector<uint8_t> q(ggml_nbytes(x));
            ggml_quantize_chunk(x->type, f.data(), q.data(), 0, nrows, n_per_row, nullptr);
            ggml_backend_tensor_set(x, q.data(), 0, q.size());
        } else if (x->type == GGML_TYPE_F32) {
            std::vector<float> d(n);
            for (auto & v : d) {
                v = u(rng);
            }
            ggml_backend_tensor_set(x, d.data(), 0, ggml_nbytes(x));
        } else if (x->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> d(n);
            for (auto & v : d) {
                v = ggml_fp32_to_fp16(u(rng));
            }
            ggml_backend_tensor_set(x, d.data(), 0, ggml_nbytes(x));
        } else if (x->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> d(n);
            for (auto & v : d) {
                v = ggml_fp32_to_bf16(u(rng));
            }
            ggml_backend_tensor_set(x, d.data(), 0, ggml_nbytes(x));
        }
    }
}

// runs case t on backend, returns the concatenated output bytes (empty on failure)
static std::vector<uint8_t> run_case(ggml_backend_t backend, const fuse_case & t, int seed) {
    built b = build_case(t);
    std::vector<uint8_t> res;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    if (buf != nullptr) {
        init_inputs(t, b.ctx, seed);
        if (ggml_backend_graph_compute(backend, b.gf) == GGML_STATUS_SUCCESS) {
            for (ggml_tensor * o : b.outs) {
                const size_t off = res.size();
                res.resize(off + ggml_nbytes(o));
                ggml_backend_tensor_get(o, res.data() + off, 0, ggml_nbytes(o));
            }
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(b.ctx);
    return res;
}

// the first CUDA / ROCm GPU: the fused launches and their logs are ggml-cuda's (a Metal / Vulkan / SYCL / OpenCL device is not used)
static ggml_backend_dev_t gpu_dev() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        if (strcmp(reg, "CUDA") == 0 || strcmp(reg, "ROCm") == 0) {
            return d;
        }
    }
    return nullptr;
}

static bool same_bytes(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, size_t & first) {
    first = 0;
    if (!a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()) == 0) {
        return true;
    }
    while (first < a.size() && first < b.size() && a[first] == b[first]) {
        first++;
    }
    return false;
}

// F4: the channels-first and the token-major PLE graphs on one backend, pairwise (cases come in token-major / channels-first pairs)
static int check_ple(ggml_backend_t backend, const char * name) {
    const std::vector<fuse_case> sh = ple_cases();
    int n_fail = 0;
    for (size_t i = 0; i + 1 < sh.size(); i += 2) {
        const std::vector<uint8_t> a = run_case(backend, sh[i], 77 + (int) i);
        const std::vector<uint8_t> b = run_case(backend, sh[i + 1], 77 + (int) i);
        size_t first = 0;
        const bool ok = same_bytes(a, b, first);
        printf("  %s %-60s bytes=%zu: %s", name, sh[i].str().c_str(), a.size(), ok ? "memcmp == 0 (channels-first)\n" : "DIFFER");
        if (!ok) {
            printf(" (first differing byte %zu)\n", first);
            n_fail++;
        }
    }
    return n_fail;
}

static int run_child(const char * path) {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t dev = gpu_dev();
    if (dev == nullptr) {
        return 3;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        return 3;
    }
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        return 4;
    }
    const std::vector<fuse_case> sh = gpu_cases();
    for (size_t i = 0; i < sh.size(); ++i) {
        const std::vector<uint8_t> o = run_case(backend, sh[i], 1234 + (int) i);
        const uint64_t n = o.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(o.data(), 1, o.size(), f);
    }
    fwrite(n_logs, sizeof(n_logs), 1, f);
    fclose(f);
    const int n_fail = check_ple(backend, ggml_backend_name(backend));
    ggml_backend_free(backend);
    return n_fail ? 7 : 0;
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static bool read_dump(const std::string & path, std::vector<std::vector<uint8_t>> & outs, int * logs, size_t n_cases) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    for (size_t i = 0; i < n_cases; ++i) {
        uint64_t n = 0;
        if (fread(&n, sizeof(n), 1, f) != 1) {
            fclose(f);
            return false;
        }
        outs.emplace_back(n);
        if (n > 0 && fread(outs.back().data(), 1, n, f) != n) {
            fclose(f);
            return false;
        }
    }
    const bool ok = fread(logs, sizeof(int), L_COUNT, f) == L_COUNT;
    fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        return run_child(argv[3]);
    }

    // F4 on the CPU backend: the reference graph rewrite, no GPU needed
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        printf("FAIL: no CPU backend\n");
        return 1;
    }
    const int n_fail_cpu = check_ple(cpu, "CPU");
    ggml_backend_free(cpu);
    if (n_fail_cpu) {
        printf("FAIL: %d F4 cases differ on the CPU backend\n", n_fail_cpu);
        return 1;
    }

    // the parent never initialises a GPU: only decide whether there is one (the registry enumerates without a context)
    if (gpu_dev() == nullptr) {
        printf("OK: F4 bit-identical on the CPU backend; SKIP: no CUDA / ROCm device for F4 / F5 / F6 on the GPU\n");
        return 0;
    }

    const std::vector<fuse_case> sh = gpu_cases();
    std::vector<std::vector<uint8_t>> outs[2];
    int logs[2][L_COUNT];
    for (int m = 0; m < 2; ++m) {
        for (int k = 0; k < L_COUNT; ++k) {
            logs[m][k] = -1;
        }
    }
    for (int mode = 0; mode < 2; ++mode) {
        const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/k19_fuse_exact_" + std::to_string(mode) + ".bin";
        set_env("GGML_CUDA_DISABLE_CONV_CAT_PROMPT", mode ? "0" : "1");
        set_env("GGML_CUDA_DISABLE_QSA_PROMPT_MASK", mode ? "0" : "1");
        // both modes on the f16 kernel (the INT8 Q*K^T route of f16 prompt nodes is not byte-equal)
        set_env("GGML_CUDA_FATTN_I8", "0");
        set_env("GGML_CUDA_FATTN_I8_F16", "0");
        set_env("GGML_CUDA_DISABLE_FUSE_HC_POST_CONV", mode ? "0" : "1");
        set_env("GGML_CUDA_DISABLE_FUSE_GATED_NORM_MMQ", mode ? "0" : "1");
        set_env("GGML_CUDA_DISABLE_FUSE_SWIGLU_MMID", mode ? "0" : "1");
        set_env("GGML_CUDA_CONV_CAT_PROMPT_LOG", "1");
        set_env("GGML_CUDA_QSA_PROMPT_LOG", "1");
        set_env("GGML_CUDA_HC_POST_CONV_LOG", "1");
        set_env("GGML_CUDA_GATED_NORM_MMQ_LOG", "1");
        set_env("GGML_CUDA_SWIGLU_MMID_LOG", "1");
        std::string cmd = "\"" + std::string(argv[0]) + "\" --child " + std::to_string(mode) + " \"" + path + "\"";
#ifdef _WIN32
        cmd = "\"" + cmd + "\""; // cmd.exe /c strips the outer pair of quotes of a line with more than two
#endif
        const int rc = std::system(cmd.c_str());
        if (rc != 0 || !read_dump(path, outs[mode], logs[mode], sh.size())) {
            fprintf(stderr, "FAIL: child mode %d rc %d\n", mode, rc);
            return 1;
        }
        std::remove(path.c_str());
    }

    int n_fail = 0;
    int exp_logs[L_COUNT] = {};
    for (size_t i = 0; i < sh.size(); ++i) {
        const auto & a = outs[0][i];
        const auto & b = outs[1][i];
        size_t first = 0;
        const bool ok = same_bytes(a, b, first);
        if (sh[i].fused()) {
            exp_logs[sh[i].log_kind()]++;
        }
        printf("  fused vs graph: %-60s bytes=%zu: %s", sh[i].str().c_str(), a.size(), ok ? "memcmp == 0\n" : "DIFFER");
        if (!ok) {
            printf(" (first differing byte %zu)\n", first);
            n_fail++;
        }
    }
    static const char * log_names[L_COUNT] = { "F5 conv cat", "F6 QSA mask", "F1 hc post conv", "F2 gated norm mmq", "F3 swiglu mmid" };
    // F1 / F2 / F3 are fused into MMQ, which ggml_cuda_should_use_mmq picks for every case only on NVIDIA Turing and newer (MMA): an
    // AMD CDNA1 / CDNA2 device (MI100, MI210) takes BLAS for the Q8_0 products of 512 / 2048 tokens, an older NVIDIA GPU above its DP4A
    // batch limit. Elsewhere the fused child must log at least one of their launches (the 33-token cases take MMQ there too) and at
    // most the expected count; the outputs are compared byte for byte either way
    typedef int (*cc_fn_t)(ggml_backend_dev_t);
    ggml_backend_dev_t gdev = gpu_dev();
    auto cc_fn = (cc_fn_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(gdev), "ggml_backend_dev_compute_capability");
    const int cc = cc_fn != nullptr ? cc_fn(gdev) : -1; // NVIDIA only, -1 for ROCm
    const bool mmq_all = cc >= 750;
    bool logs_ok = true;
    for (int k = 0; k < L_COUNT; ++k) {
        const bool mmq_kind = k == L_HCN || k == L_GN || k == L_SWM;
        const bool exact    = !mmq_kind || mmq_all;
        printf("fused launches logged: %-18s graph child %d, fused child %d (expected 0 / %s%d)\n", log_names[k], logs[0][k], logs[1][k],
               exact ? "" : "1..", exp_logs[k]);
        logs_ok = logs_ok && logs[0][k] == 0 &&
            (exact ? logs[1][k] == exp_logs[k] : logs[1][k] >= std::min(1, exp_logs[k]) && logs[1][k] <= exp_logs[k]);
    }
    if (!logs_ok) {
        printf("FAIL: a fusion did not run exactly once per case in the fused child (or ran in the graph child)\n");
        return 1;
    }
    if (n_fail) {
        printf("FAIL: %d / %zu cases differ\n", n_fail, sh.size());
        return 1;
    }
    printf("OK: %zu / %zu GPU cases bit-identical, F4 bit-identical on the CPU and the GPU backend\n", sh.size(), sh.size());
    return 0;
}
