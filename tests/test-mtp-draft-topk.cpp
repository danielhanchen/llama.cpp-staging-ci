// the MTP draft step's backend top-k over the restricted draft head's logits (gathered in token order, the unwritten
// request rows on a -inf column) must give the candidates, their logits and their order that the top-k over the scatter into the
// full vocabulary gives.
//   part 1 (ops, every backend of this build): random logits with many ties (at the k-th value too), +0.0 / -0.0, frequent-token
//          rows + request rows in ring order + unwritten rows; candidates and logit bits equal. Control: the same top-k over the rows
//          in row order (no token-order gather) must differ somewhere, else the cases do not exercise the order.
//   part 2 (qwen35moe-mtp model, CPU, children with LLAMA_MTP_DRAFT_TOPK_FULL=0 / 1): greedy MTP speculative decoding gives the same
//          tokens and every draft, with request rows (written and unwritten) and p_min; the default child logs the draft top-k line,
//          the kill-switch child does not.
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
// the child processes run on POSIX only (main), but the test is built on Windows static builds too
#define popen  _popen
#define pclose _pclose
static void setenv(const char * k, const char * v, int) { _putenv_s(k, v); }
#endif

// ---------------------------------------------------------------------------------------------------------------------- part 1

struct head_case {
    int64_t n_vocab;
    int32_t n_base;
    int32_t n_ctx;
    int32_t n_written; // request rows written (ring order), the rest stay copies of row 0
    int     levels;    // distinct logit values (0: continuous)
    uint32_t seed;
};

// the same row map as llama_model::draft_head_order_update
static void order_rows(const std::vector<int64_t> & ids, int32_t n_base, bool sorted, std::vector<int32_t> & perm, std::vector<int32_t> & cand) {
    const int32_t n = (int32_t) ids.size();
    std::vector<int32_t> rows;
    for (int32_t r = 0; r < n; ++r) {
        if (r < n_base || ids[r] != ids[0]) {
            rows.push_back(r);
        }
    }
    if (sorted) {
        std::sort(rows.begin(), rows.end(), [&](int32_t a, int32_t b) { return ids[a] < ids[b]; });
    }
    perm.clear();
    cand.clear();
    for (int32_t r : rows) {
        perm.push_back(r);
        cand.push_back((int32_t) ids[r]);
    }
    while ((int32_t) perm.size() < n) {
        perm.push_back(n);
        cand.push_back((int32_t) ids[0]);
    }
}

struct topk_out {
    std::vector<int32_t> ids;
    std::vector<float>   logits;
};

// mode 0: scatter into the full vocabulary (build_draft_head) + top-k; mode 1: token-order gather (build_sampling)
static bool run_topk(ggml_backend_t backend, const head_case & hc, const std::vector<float> & sub, const std::vector<int64_t> & ids,
                     const std::vector<int32_t> & perm, const std::vector<int32_t> & cand, int mode, int k, topk_out & out) {
    const int64_t n_sub = (int64_t) ids.size();
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t_sub  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_sub, 1);
    ggml_tensor * t_d2t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_sub);
    ggml_tensor * t_perm = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sub);
    ggml_tensor * t_cand = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sub);
    ggml_tensor * cands  = nullptr;
    ggml_tensor * vals   = nullptr;
    if (mode == 0) {
        ggml_tensor * full = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, hc.n_vocab, 1), -INFINITY);
        full = ggml_set_rows(ctx, full, ggml_reshape_3d(ctx, t_sub, 1, n_sub, 1), ggml_reshape_3d(ctx, t_d2t, n_sub, 1, 1));
        ggml_tensor * lt  = ggml_pad(ctx, ggml_reshape_2d(ctx, full, hc.n_vocab, 1), 0, 1, 0, 0);
        ggml_tensor * row = ggml_view_1d(ctx, lt, lt->ne[0], 0);
        ggml_tensor * l1  = ggml_reshape_1d(ctx, row, ggml_nelements(row));
        ggml_tensor * tk  = ggml_top_k(ctx, l1, k);
        cands = tk;
        vals  = ggml_get_rows(ctx, ggml_reshape_2d(ctx, l1, 1, l1->ne[0]), tk);
    } else {
        ggml_tensor * ninf = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1), -INFINITY);
        ggml_tensor * lt   = ggml_pad(ctx, ggml_concat(ctx, t_sub, ninf, 0), 0, 1, 0, 0);
        ggml_tensor * row  = ggml_view_1d(ctx, lt, lt->ne[0], 0);
        ggml_tensor * l1   = ggml_reshape_1d(ctx, ggml_get_rows(ctx, ggml_reshape_2d(ctx, row, 1, row->ne[0]), t_perm), n_sub);
        ggml_tensor * tk   = ggml_top_k(ctx, l1, k);
        cands = ggml_get_rows(ctx, ggml_reshape_2d(ctx, t_cand, 1, n_sub), tk);
        vals  = ggml_get_rows(ctx, ggml_reshape_2d(ctx, l1, 1, l1->ne[0]), tk);
    }
    ggml_set_output(cands);
    ggml_set_output(vals);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cands);
    ggml_build_forward_expand(gf, vals);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(t_sub, sub.data(), 0, sub.size()*sizeof(float));
    ggml_backend_tensor_set(t_d2t, ids.data(), 0, ids.size()*sizeof(int64_t));
    ggml_backend_tensor_set(t_perm, perm.data(), 0, perm.size()*sizeof(int32_t));
    ggml_backend_tensor_set(t_cand, cand.data(), 0, cand.size()*sizeof(int32_t));
    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    out.ids.resize(k);
    out.logits.resize(k);
    if (ok) {
        ggml_backend_tensor_get(cands, out.ids.data(), 0, k*sizeof(int32_t));
        ggml_backend_tensor_get(vals, out.logits.data(), 0, k*sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static bool same(const topk_out & a, const topk_out & b) {
    return a.ids == b.ids && memcmp(a.logits.data(), b.logits.data(), a.logits.size()*sizeof(float)) == 0;
}

// ggml_top_k returns its k indices "in no particular order": a backend other than the CPU and CUDA ones (whose order the draft path
// relies on and this test checks bit for bit) may order them, and pick among the logits tied at the k-th value, differently in the
// two graphs. 0: equal bit for bit; 1: the same candidates in another order; 2: the same logits, the candidates differing only among
// the tokens tied at the k-th logit; 3: a real difference (another logit, a token that does not have its logit, a duplicate)
static int topk_diff_kind(const topk_out & a, const topk_out & b, const std::vector<float> & tok_logit) {
    if (same(a, b)) {
        return 0;
    }
    const size_t k = a.ids.size();
    if (b.ids.size() != k) {
        return 3;
    }
    for (const topk_out * o : { &a, &b }) {
        std::vector<int32_t> s_ids = o->ids;
        std::sort(s_ids.begin(), s_ids.end());
        if (std::adjacent_find(s_ids.begin(), s_ids.end()) != s_ids.end()) {
            return 3;
        }
        for (size_t i = 0; i < k; ++i) {
            const int32_t t = o->ids[i];
            if (t < 0 || (size_t) t >= tok_logit.size() || memcmp(&o->logits[i], &tok_logit[t], sizeof(float)) != 0) {
                return 3;
            }
        }
    }
    std::vector<float> la = a.logits, lb = b.logits;
    std::sort(la.begin(), la.end(), std::greater<float>());
    std::sort(lb.begin(), lb.end(), std::greater<float>());
    if (la != lb) {
        return 3;
    }
    std::vector<int32_t> ia = a.ids, ib = b.ids;
    std::sort(ia.begin(), ia.end());
    std::sort(ib.begin(), ib.end());
    if (ia == ib) {
        return 1;
    }
    // every candidate above the k-th logit is in both
    const float kth = la.back();
    for (const topk_out * o : { &a, &b }) {
        const topk_out & other = o == &a ? b : a;
        for (size_t i = 0; i < k; ++i) {
            if (o->logits[i] > kth && std::find(other.ids.begin(), other.ids.end(), o->ids[i]) == other.ids.end()) {
                return 3;
            }
        }
    }
    return 2;
}

static int part_ops(int & n_control_diff) {
    int fails = 0;
    n_control_diff = 0;
    const head_case cases[] = {
        { 248320, 65536, 4096, 4096, 0,  1 },  // Qwen3.6 shape, every request row written, continuous logits
        { 248320, 65536, 4096, 1500, 6,  2 },  // unwritten request rows, 6 levels: ties everywhere
        { 248320, 65536, 4096, 4000, 40, 3 },
        { 248320, 65536, 4096, 0,    3,  4 },  // no request row written
        { 248320, 65536, 0,    0,    9,  5 },
        { 151936, 32768, 2048, 700,  12, 6 },
        { 6000,   2048,  512,  100,  5,  7 },  // small (CPU-shaped; CUDA takes the sort paths here, see build_draft_head)
        { 512,    64,    16,   7,    3,  8 },
    };
    for (size_t bi = 0; bi < ggml_backend_dev_count(); ++bi) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(bi);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        const bool is_cpu = type == GGML_BACKEND_DEVICE_TYPE_CPU;
        // bit for bit (order and ties too) where the draft path relies on it; elsewhere the op's contract: same logits, ties free
        const bool exact = is_cpu || strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0;
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            continue;
        }
        int n_run = 0, n_bad = 0, n_ctl = 0, n_order = 0, n_tie = 0;
        for (const head_case & hc : cases) {
            const int32_t n_sub = hc.n_base + hc.n_ctx;
            if (!is_cpu && n_sub <= 4096) {
                continue; // the graph keeps the full-vocabulary top-k there (build_draft_head)
            }
            for (int rep = 0; rep < 6; ++rep) {
                std::mt19937 rng(hc.seed * 1000 + rep);
                // frequent-token rows: sorted unique ids; request rows: other tokens in ring (arbitrary) order
                std::vector<int64_t> all(hc.n_vocab);
                for (int64_t i = 0; i < hc.n_vocab; ++i) {
                    all[i] = i;
                }
                std::shuffle(all.begin(), all.end(), rng);
                std::vector<int64_t> ids(all.begin(), all.begin() + hc.n_base);
                std::sort(ids.begin(), ids.end());
                for (int32_t r = 0; r < hc.n_ctx; ++r) {
                    ids.push_back(r < hc.n_written ? all[hc.n_base + r] : ids[0]);
                }
                std::vector<float> sub(n_sub);
                std::uniform_real_distribution<float> ud(-8.0f, 8.0f);
                for (int32_t r = 0; r < n_sub; ++r) {
                    if (hc.levels > 0) {
                        const int lv = (int) (rng() % hc.levels);
                        sub[r] = lv == 0 ? ((rng() & 1) ? -0.0f : 0.0f) : (float) lv * 0.5f;
                    } else {
                        sub[r] = ud(rng);
                    }
                }
                // an unwritten request row is a copy of row 0 (same weights, same logit)
                for (int32_t r = hc.n_base + hc.n_written; r < n_sub; ++r) {
                    sub[r] = sub[0];
                }
                std::vector<int32_t> perm, cand, perm_u, cand_u;
                order_rows(ids, hc.n_base, true, perm, cand);
                order_rows(ids, hc.n_base, false, perm_u, cand_u);
                // each token's logit in the full vocabulary (-inf where the head has no row)
                std::vector<float> tok_logit(hc.n_vocab, -INFINITY);
                for (int32_t r = 0; r < n_sub; ++r) {
                    tok_logit[ids[r]] = sub[r];
                }
                for (int k : { 1, 10, 40 }) {
                    topk_out ref, got, ctl;
                    if (!run_topk(backend, hc, sub, ids, perm, cand, 0, k, ref) ||
                        !run_topk(backend, hc, sub, ids, perm, cand, 1, k, got) ||
                        !run_topk(backend, hc, sub, ids, perm_u, cand_u, 1, k, ctl)) {
                        fprintf(stderr, "%s: compute failed\n", ggml_backend_dev_name(dev));
                        n_bad++;
                        continue;
                    }
                    n_run++;
                    const int kind = topk_diff_kind(ref, got, tok_logit);
                    n_order += kind == 1;
                    n_tie   += kind == 2;
                    if (kind == 3 || (kind != 0 && exact)) {
                        n_bad++;
                        if (n_bad <= 3) {
                            fprintf(stderr, "%s: V %lld n_sub %d levels %d k %d: candidates differ (first %d vs %d, %s)\n", ggml_backend_dev_name(dev),
                                    (long long) hc.n_vocab, n_sub, hc.levels, k, ref.ids[0], got.ids[0],
                                    kind == 1 ? "same candidates, other order" : kind == 2 ? "same logits, other tokens among the ties at the k-th logit" :
                                    "another logit or token");
                        }
                    }
                    n_ctl += !same(ref, ctl);
                }
            }
        }
        printf("part 1 %-8s: %d top-k cases, %d differ from the full-vocabulary top-k (%s; order-only %d, tie-only %d); control (row order, "
                "no gather) differs in %d\n", ggml_backend_dev_name(dev), n_run, n_bad, exact ? "bit for bit" : "tie-aware", n_order, n_tie, n_ctl);
        fails += n_bad + (n_run == 0);
        n_control_diff += n_ctl;
        ggml_backend_free(backend);
    }
    return fails;
}

// ---------------------------------------------------------------------------------------------------------------------- part 2

static bool g_engaged = false;

static bool run_spec(const char * path, int n_predict, int n_max, float p_min, std::string & out) {
    common_params params;
    params.model.path   = path;
    params.n_ctx        = 1024;
    params.n_batch      = 256;
    params.n_ubatch     = 256;
    params.n_parallel   = 1;
    params.fit_params   = false;
    params.n_gpu_layers = 0;
    params.warmup       = false;
    params.load_mtp     = true;
    params.cpuparams.n_threads       = std::max(1, std::min(4, (int) std::thread::hardware_concurrency())); // no more threads than cores
    params.cpuparams_batch.n_threads = params.cpuparams.n_threads;
    params.sampling.temp  = 0.0f;
    params.sampling.top_k = 1;
    params.speculative.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max   = n_max;
    params.speculative.draft.p_min   = p_min;
    params.devices    = { nullptr };

    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    auto init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = init_tgt->model();
    llama_context * ctx_tgt   = init_tgt->context();
    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        return false;
    }
    common_speculative_init_result_ptr spec_init;
    {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    if (ctx_dft == nullptr) {
        return false;
    }
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // a prompt of 40 tokens: request tokens outside the draft vocabulary get rows of their own (llama_model_draft_head_add)
    std::vector<llama_token> inp;
    for (int i = 0; i < 40; i++) {
        inp.push_back((llama_token) ((i * 7919 + 13) % (n_vocab - 10) + 5));
    }

    const llama_seq_id seq_id = 0;
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        return false;
    }
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0 || !common_speculative_process(spec, batch_prompt)) {
            common_speculative_free(spec);
            return false;
        }
    }
    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = inp.size() - 1;
    common_speculative_begin(spec, seq_id, prompt_tgt);

    common_batch batch_tgt(ctx_tgt);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;
    int n_out = 0, n_drafted = 0, n_accept = 0;
    std::string s_out, s_drafts;
    bool ok = true;
    while (ok) {
        if (draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(), llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));
            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ std::max(0, n_predict - n_out - 1),
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft,
            };
            common_speculative_draft(spec);
            for (llama_token t : draft) {
                s_drafts += std::to_string(t) + " ";
            }
            s_drafts += "| ";
            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (use_ckpt_dft) {
                ok = ok && ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
        }
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            batch_tgt.add(draft[i], n_past + i, seq_id, true);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0 || !common_speculative_process(spec, batch_tgt)) {
            ok = false;
            break;
        }
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }
        const size_t n_draft = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            draft = std::move(ids);
            ok = ok && ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            ok = ok && ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }
        common_speculative_accept(spec, seq_id, ids.size() - 1);
        n_past    += ids.size() - 1;
        n_drafted += n_draft;
        n_accept  += ids.size() - 1;
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            s_out += std::to_string(id_last) + " ";
            n_out++;
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
        if (n_out >= n_predict) {
            break;
        }
    }
    common_speculative_free(spec);
    out = "drafted " + std::to_string(n_drafted) + " accepted " + std::to_string(n_accept) + "\ntokens " + s_out + "\ndrafts " + s_drafts + "\n";
    return ok && n_out >= n_predict && n_drafted > 0;
}

struct variant { const char * vocab_ctx; int n_max; float p_min; int n_predict; };
static const variant g_variants[] = {
    { "64", 3, 0.00f, 120 }, // request rows: the prompt's tokens written, the rest unwritten
    { "16", 4, 0.00f, 200 }, // every request row written
    { "0",  2, 0.00f, 80  },
    { "64", 6, 0.08f, 160 },
    { "16", 3, 0.11f, 120 },
};

static int child(const char * model, int vi) {
    const variant & v = g_variants[vi];
    setenv("LLAMA_MTP_DRAFT_VOCAB", "64", 1);
    setenv("LLAMA_MTP_DRAFT_VOCAB_CTX", v.vocab_ctx, 1);
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (strstr(text, "device top-k: MTP draft top-k over the") != nullptr) {
            g_engaged = true;
        }
        if (level >= GGML_LOG_LEVEL_ERROR || getenv("TEST_MTP_DTOPK_LOG")) {
            fputs(text, stderr);
        }
    }, nullptr);
    std::string out;
    const bool ok = run_spec(model, v.n_predict, v.n_max, v.p_min, out);
    printf("%s%s\nengaged %d\n", ok ? "ok\n" : "FAIL\n", out.c_str(), (int) g_engaged);
    llama_backend_free();
    return ok ? 0 : 1;
}

static bool run_child(const char * self, const char * model, int vi, const char * full, std::string & out) {
    const std::string cmd = std::string("LLAMA_MTP_DRAFT_TOPK_FULL=") + full + " '" + self + "' '" + model + "' --child " + std::to_string(vi);
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) {
        return false;
    }
    char buf[4096];
    out.clear();
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p)) > 0) {
        out.append(buf, n);
    }
    return pclose(p) == 0;
}

int main(int argc, char ** argv) {
    if (argc >= 4 && strcmp(argv[2], "--child") == 0) {
        return child(argv[1], atoi(argv[3]));
    }
    ggml_backend_load_all();
    int n_ctl = 0;
    int ret = part_ops(n_ctl) != 0 ? 1 : 0;
    if (n_ctl == 0) {
        printf("part 1: the control never differed: the cases do not exercise the order\n");
        ret = 1;
    }
    if (argc >= 2 && getenv("TEST_MTP_DTOPK_OPS_ONLY") == nullptr) {
#ifndef _WIN32
        for (int vi = 0; vi < (int) (sizeof(g_variants)/sizeof(g_variants[0])); ++vi) {
            const variant & v = g_variants[vi];
            std::string on, off;
            const bool ok_on  = run_child(argv[0], argv[1], vi, "0", on);
            const bool ok_off = run_child(argv[0], argv[1], vi, "1", off);
            const bool eng_on  = on.find("engaged 1") != std::string::npos;
            const bool eng_off = off.find("engaged 1") != std::string::npos;
            const std::string res_on  = on.substr(0, on.find("engaged"));
            const std::string res_off = off.substr(0, off.find("engaged"));
            const bool eq = ok_on && ok_off && res_on == res_off;
            printf("part 2 request rows %-2s n_max %d p_min %.2f n_predict %3d: dtopk %s (%s), full %s (%s): %s; %s",
                    v.vocab_ctx, v.n_max, v.p_min, v.n_predict, ok_on ? "ok" : "FAIL", eng_on ? "engaged" : "NOT engaged",
                    ok_off ? "ok" : "FAIL", eng_off ? "ENGAGED" : "off", eq ? "tokens and drafts equal" : "DIFFER",
                    on.substr(3, on.find('\n', 3) - 2).c_str());
            if (!eq || !eng_on || eng_off) {
                ret = 1;
                if (getenv("TEST_MTP_DTOPK_LOG")) {
                    fprintf(stderr, "--- on\n%s\n--- off\n%s\n", on.c_str(), off.c_str());
                }
            }
        }
#endif
    }
    printf("%s\n", ret == 0 ? "OK" : "FAIL");
    return ret;
}
