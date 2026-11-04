// -sm tensor (a meta device) with MTP drafting: the restricted-vocabulary draft head read its request-token rows from the
// vocabulary-split LM head with one-row reads that the meta buffer refused (GGML_ASSERT(offset % chunk_size_full == 0)), and the
// full head aborted in the draft decode. Greedy MTP speculative decoding on Meta(CPU, CPU) must run and give the tokens of the same
// run on the CPU, with the restricted head (rows of the request tokens added, LLAMA_MTP_DRAFT_VOCAB_CTX) and with the full head
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

struct run_result {
    bool ok = false;
    std::vector<llama_token> out;
    std::vector<llama_token> drafts; // every draft, in order
    int n_drafted = 0;
    int n_accept  = 0;
};

static run_result run(const char * path, bool meta, int n_predict) {
    run_result res;

    common_params params;
    params.model.path   = path;
    params.n_ctx        = 512;
    params.n_batch      = 256;
    params.n_ubatch     = 256;
    params.n_parallel   = 1;
    params.fit_params   = false;
    params.n_gpu_layers = 999;
    params.warmup       = false;
    params.load_mtp     = true;
    params.cpuparams.n_threads       = 4;
    params.cpuparams_batch.n_threads = 4;
    params.sampling.temp  = 0.0f;
    params.sampling.top_k = 1;
    params.speculative.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max   = 2;
    params.speculative.draft.p_min   = 0.0f;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (meta) {
        params.devices    = { cpu, cpu, nullptr };
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    } else {
        params.devices    = { nullptr };
    }

    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    auto init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = init_tgt->model();
    llama_context * ctx_tgt   = init_tgt->context();
    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        fprintf(stderr, "%s: failed to load the target (meta = %d)\n", __func__, meta);
        return res;
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
        fprintf(stderr, "%s: no MTP draft context (meta = %d)\n", __func__, meta);
        return res;
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
        return res;
    }
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0 || !common_speculative_process(spec, batch_prompt)) {
            common_speculative_free(spec);
            return res;
        }
    }
    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = inp.size() - 1;
    common_speculative_begin(spec, seq_id, prompt_tgt);

    common_batch batch_tgt(ctx_tgt);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;
    int n_out = 0;
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
            res.drafts.insert(res.drafts.end(), draft.begin(), draft.end());
            res.drafts.push_back(-1);
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
        n_past        += ids.size() - 1;
        res.n_drafted += n_draft;
        res.n_accept  += ids.size() - 1;
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            res.out.push_back(id_last);
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
    res.ok = ok && n_out >= n_predict;
    return res;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s qwen35moe-mtp-model.gguf\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN || getenv("TEST_MTP_META_LOG")) {
            fputs(text, stderr);
        }
    }, nullptr);
    // a restricted head small enough to engage on a test vocabulary; with request rows (on the CPU a mapped head reads none, so only
    // the target tokens compare) and without (the drafts compare too), and the full head
    struct variant { const char * vocab; const char * ctx_rows; bool drafts; };
    const int n_predict = 24;
    int ret = 0;
    for (const variant & v : { variant { "64", "16", false }, variant { "64", "0", true }, variant { "0", "0", true } }) {
        set_env("LLAMA_MTP_DRAFT_VOCAB", v.vocab);
        set_env("LLAMA_MTP_DRAFT_VOCAB_CTX", v.ctx_rows);
        const run_result ref = run(argv[1], false, n_predict);
        const run_result met = run(argv[1], true,  n_predict);
        const bool same = ref.ok && met.ok && ref.out == met.out && (!v.drafts || ref.drafts == met.drafts);
        printf("draft vocab %-3s request rows %-3s: cpu %s (%d/%d accepted), meta %s (%d/%d accepted), tokens%s %s\n", v.vocab, v.ctx_rows,
                ref.ok ? "ok" : "FAIL", ref.n_accept, ref.n_drafted, met.ok ? "ok" : "FAIL", met.n_accept, met.n_drafted,
                v.drafts ? " and drafts" : "", same ? "equal" : "DIFFER");
        if (getenv("TEST_MTP_META_LOG")) {
            for (const auto * r : { &ref, &met }) {
                for (llama_token t : r->out) {
                    printf(" %d", t);
                }
                printf("\n");
            }
        }
        if (!same || met.n_drafted == 0) {
            ret = 1;
        }
    }
    llama_backend_free();
    printf("%s\n", ret == 0 ? "OK" : "FAIL");
    return ret;
}
