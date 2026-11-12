// k19-hostgap: the host-side changes of a speculative pass leave every token, every draft and the memory state unchanged.
// Greedy MTP speculative decoding (the generated qwen35moe-mtp model, a 2048-cell cache of which a few hundred cells are used, drafts
// rejected and rewound every pass: llama_memory_seq_rm on the target and the drafter) runs in a child process per setting of the
// switches; the parent compares the outputs: generated tokens, every draft, and a hash of the target's and the drafter's sequence
// state (llama_state_seq_get_data) after every pass.
//   LLAMA_KV_SEQ_RM_USED=0 / 1: KV seq_rm over every cell / over the cells below the last used one
// usage: test-hostgap-exact qwen35moe-mtp-model.gguf
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const uint8_t * p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

static uint64_t state_hash(llama_context * ctx, llama_seq_id s) {
    const size_t n = llama_state_seq_get_size(ctx, s);
    std::vector<uint8_t> buf(n);
    const size_t w = llama_state_seq_get_data(ctx, buf.data(), buf.size(), s);
    return fnv1a(buf.data(), w);
}

// one run; prints "tok ...", "draft ...", "state ..." lines and "RUN OK" / "RUN FAIL"
static int child(const char * path) {
    const int n_predict = 160;

    common_params params;
    params.model.path   = path;
    params.n_ctx        = 2048;
    params.n_batch      = 256;
    params.n_ubatch     = 256;
    params.n_parallel   = 1;
    params.fit_params   = false;
    params.n_gpu_layers = 0;
    params.warmup       = false;
    params.load_mtp     = true;
    params.cpuparams.n_threads       = 2;
    params.cpuparams_batch.n_threads = 2;
    params.sampling.temp  = 0.0f;
    params.sampling.top_k = 1;
    params.speculative.types       = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 3;
    params.speculative.draft.p_min = 0.0f;
    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    auto init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = init_tgt->model();
    llama_context * ctx_tgt   = init_tgt->context();
    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        printf("RUN FAIL (target)\n");
        return 1;
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
        printf("RUN FAIL (drafter)\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
    std::vector<llama_token> inp;
    for (int i = 0; i < 48; i++) {
        inp.push_back((llama_token) ((i * 7919 + 13) % (n_vocab - 10) + 5));
    }
    const llama_seq_id seq_id = 0;
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        printf("RUN FAIL (spec)\n");
        return 1;
    }
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0 || !common_speculative_process(spec, batch_prompt)) {
            printf("RUN FAIL (prompt)\n");
            return 1;
        }
    }
    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = inp.size() - 1;
    common_speculative_begin(spec, seq_id, prompt_tgt);

    common_batch batch_tgt(ctx_tgt);
    llama_tokens draft;
    int n_out = 0, n_pass = 0, n_drafted = 0, n_accept = 0;
    std::string toks, drafts, states;
    while (n_out < n_predict) {
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
            drafts += " " + std::to_string(t);
        }
        drafts += " |";
        // the drafter's cells past the accepted ones go before the verify (as the server's rewind)
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            batch_tgt.add(draft[i], n_past + i, seq_id, true);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0 || !common_speculative_process(spec, batch_tgt)) {
            printf("RUN FAIL (verify)\n");
            return 1;
        }
        const size_t n_draft = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        common_speculative_accept(spec, seq_id, ids.size() - 1);
        n_past    += ids.size() - 1;
        n_drafted += n_draft;
        n_accept  += ids.size() - 1;
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            toks += " " + std::to_string(id_last);
            n_out++;
        }
        draft.clear();
        // the rejected drafts' cells go (target and drafter)
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
        char b[64];
        snprintf(b, sizeof(b), " %016llx/%016llx", (unsigned long long) state_hash(ctx_tgt, seq_id), (unsigned long long) state_hash(ctx_dft, seq_id));
        states += b;
        n_pass++;
    }
    common_speculative_free(spec);
    printf("tok%s\ndraft%s\nstate%s\n", toks.c_str(), drafts.c_str(), states.c_str());
    printf("RUN OK passes %d drafted %d accepted %d cells used %d of %u\n", n_pass, n_drafted, n_accept,
           (int) llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id) + 1, llama_n_ctx(ctx_tgt));
    return 0;
}

static std::string run_child(const char * argv0, const char * model, const char * env) {
    std::string cmd = std::string(env) + " '" + argv0 + "' --child '" + model + "' 2>/dev/null";
    std::string out;
    FILE * f = popen(cmd.c_str(), "r");
    if (f == nullptr) {
        return out;
    }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    pclose(f);
    return out;
}

static std::string line_of(const std::string & s, const char * key) {
    const size_t p = s.find(std::string("\n") + key);
    const size_t q = p == std::string::npos ? (s.rfind(key, 0) == 0 ? 0 : std::string::npos) : p + 1;
    if (q == std::string::npos) {
        return "";
    }
    return s.substr(q, s.find('\n', q) - q);
}

int main(int argc, char ** argv) {
    if (argc >= 3 && strcmp(argv[1], "--child") == 0) {
        llama_backend_init();
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        const int r = child(argv[2]);
        llama_backend_free();
        return r;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s qwen35moe-mtp-model.gguf\n", argv[0]);
        return 1;
    }
#ifdef _WIN32
    printf("SKIP (POSIX only)\n");
    return 0;
#else
    const std::string ref = run_child(argv[0], argv[1], "LLAMA_KV_SEQ_RM_USED=0");
    const std::string hed = run_child(argv[0], argv[1], "LLAMA_KV_SEQ_RM_USED=1");
    int ret = 0;
    for (const char * key : { "RUN OK", "tok", "draft", "state" }) {
        const std::string a = line_of(ref, key), b = line_of(hed, key);
        const bool same = !a.empty() && (strcmp(key, "RUN OK") == 0 ? !b.empty() : a == b);
        printf("%-6s: %s\n", key, same ? "equal" : "DIFFER");
        ret |= same ? 0 : 1;
    }
    printf("ref: %s\nhead: %s\n", line_of(ref, "RUN").c_str(), line_of(hed, "RUN").c_str());
    printf("%s\n", ret == 0 ? "OK" : "FAIL");
    return ret;
#endif
}
