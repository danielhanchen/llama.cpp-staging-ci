// same-text (teacher-forced) acceptance of a draft model.
//
// The target decodes a FIXED continuation instead of its own greedy text, and the drafter drafts at each window start as it would in
// a speculative run: the prompt is decoded and fed to the drafter, then per window the drafter drafts from the anchor C[j] (position
// N + j), the accepted count is the number of leading drafts equal to C[j + 1 ..], and the target decodes the window's committed rows
// C[j .. j'] (the anchor and the accepted drafts, as a verify batch would leave them) which are fed to the drafter as its context.
// With TFA_STARTS the window starts are given (another engine's windows on the same text); otherwise j' = j + accepted + 1.
// So two drafters, or one drafter under two settings, are compared on identical tokens at identical positions.
//
// env: TFA_PROMPT_IDS=file, TFA_CONT_IDS=file (token ids, any separators), TFA_STARTS=file (optional, window anchor indices into the
//      continuation, ascending), TFA_OUT=file (one JSON line per window), TFA_MAX_ACC=n (optional: count at most n accepted drafts).
// args: as llama-speculative-simple (-m, -md, --spec-type draft-dflash, --spec-draft-n-max, -c, -b, -ub, -ngl, -fa ...)

#include "arg.h"
#include "common.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int64_t> read_ints(const char * path) {
    std::vector<int64_t> out;
    if (path == nullptr || path[0] == '\0') {
        return out;
    }
    std::ifstream f(path);
    if (!f) {
        LOG_ERR("cannot open %s\n", path);
        exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '-' || isdigit((unsigned char) s[i])) {
            size_t e = 0;
            out.push_back(std::stoll(s.substr(i), &e));
            i += e;
        } else {
            ++i;
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    std::vector<int64_t> P64 = read_ints(getenv("TFA_PROMPT_IDS"));
    std::vector<int64_t> C64 = read_ints(getenv("TFA_CONT_IDS"));
    std::vector<int64_t> S64 = read_ints(getenv("TFA_STARTS"));
    const int max_acc = getenv("TFA_MAX_ACC") ? atoi(getenv("TFA_MAX_ACC")) : 1 << 30;
    if (P64.empty() || C64.size() < 2) {
        LOG_ERR("TFA_PROMPT_IDS / TFA_CONT_IDS: need a prompt and at least 2 continuation tokens\n");
        return 1;
    }
    const llama_tokens P(P64.begin(), P64.end());
    const llama_tokens C(C64.begin(), C64.end());
    const int32_t N = (int32_t) P.size();
    const int32_t M = (int32_t) C.size();

    FILE * fo = getenv("TFA_OUT") ? fopen(getenv("TFA_OUT"), "w") : nullptr;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt   = llama_init_tgt->context();
    if (ctx_tgt == nullptr) {
        LOG_ERR("failed to create the target context\n");
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
        LOG_ERR("no draft context\n");
        return 1;
    }
    if ((int64_t) llama_n_ctx(ctx_tgt) < N + M + 16) {
        LOG_ERR("context %u too small for %d + %d tokens\n", llama_n_ctx(ctx_tgt), N, M);
        return 1;
    }

    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        LOG_ERR("failed to initialize speculative decoding\n");
        return 1;
    }

    const llama_seq_id seq = 0;
    const int32_t n_batch  = (int32_t) llama_n_batch(ctx_tgt);
    const int64_t t0 = ggml_time_us();

    // the prompt, as the server decodes it: every row, the last one with an output, each batch fed to the drafter
    for (int32_t b = 0; b < N; b += n_batch) {
        const int32_t n = std::min(n_batch, N - b);
        common_batch batch(ctx_tgt);
        for (int32_t i = 0; i < n; ++i) {
            batch.add(P[b + i], b + i, seq, b + i == N - 1);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("prompt decode failed at %d\n", b);
            return 1;
        }
        if (!common_speculative_process(spec, batch)) {
            LOG_ERR("drafter prompt pass failed at %d\n", b);
            return 1;
        }
    }
    const int64_t t1 = ggml_time_us();

    llama_tokens hist(P.begin(), P.end());
    common_speculative_begin(spec, seq, hist);

    size_t si = 0;
    while (si < S64.size() && S64[si] < 0) {
        ++si;
    }
    int32_t j = 0;
    if (!S64.empty() && si < S64.size() && S64[si] > 0) {
        // the given windows start later (another engine's first window covered C[0 .. S[0] - 1]): those rows are decoded without a draft
        j = (int32_t) std::min<int64_t>(S64[si], M - 1);
        common_batch batch(ctx_tgt);
        for (int32_t i = 0; i < j; ++i) {
            batch.add(C[i], N + i, seq, i == j - 1);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 || !common_speculative_process(spec, batch)) {
            LOG_ERR("decode of the first %d continuation rows failed\n", j);
            return 1;
        }
        for (int32_t i = 0; i < j; ++i) {
            hist.push_back(C[i]);
        }
    }

    int64_t n_win = 0, n_acc = 0, n_off = 0, n_tok = 0;
    std::vector<int64_t> acc_hist(16, 0);
    llama_tokens draft;
    while (j < M - 1) {
        draft.clear();
        common_speculative_get_draft_params(spec, seq) = {
            /* .drafting = */ true,
            /* .n_max    = */ std::max(0, M - 1 - j),
            /* .pos0     = */ N + j,
            /* .id_last  = */ C[j],
            /* .prompt   = */ &hist,
            /* .result   = */ &draft,
        };
        common_speculative_draft(spec);
        // the noise block's cells go, as in the server (the checkpoint's pos_max + 1)
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, N + j, -1);

        int32_t a = 0;
        while (a < (int32_t) draft.size() && j + 1 + a < M && draft[a] == C[j + 1 + a] && a < max_acc) {
            ++a;
        }

        int32_t jn = j + a + 1;
        if (!S64.empty()) {
            ++si;
            jn = si < S64.size() ? (int32_t) S64[si] : M;
        }
        jn = std::min(jn, M);
        if (jn <= j) {
            LOG_ERR("TFA_STARTS not ascending at %d\n", j);
            return 1;
        }

        if (fo) {
            fprintf(fo, "{\"j\": %d, \"pos\": %d, \"a\": %d, \"n\": %zu, \"next\": %d, \"d\": [", j, N + j, a, draft.size(), jn);
            for (size_t k = 0; k < draft.size(); ++k) {
                fprintf(fo, "%s%d", k ? ", " : "", draft[k]);
            }
            fprintf(fo, "]}\n");
        }
        n_win++;
        n_acc += a;
        n_off += (int64_t) draft.size();
        n_tok += a + 1;
        acc_hist[std::min<int32_t>(a, 15)]++;

        // the committed rows C[j .. jn - 1] on the target, then to the drafter (a verify batch leaves the same rows)
        common_batch batch(ctx_tgt);
        for (int32_t i = j; i < jn; ++i) {
            batch.add(C[i], N + i, seq, i == jn - 1);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("decode failed at %d\n", j);
            return 1;
        }
        if (!common_speculative_process(spec, batch)) {
            LOG_ERR("drafter pass failed at %d\n", j);
            return 1;
        }
        common_speculative_accept(spec, seq, (uint16_t) a);
        for (int32_t i = j; i < jn; ++i) {
            hist.push_back(C[i]);
        }
        j = jn;
    }
    const int64_t t2 = ggml_time_us();

    std::string h;
    for (int k = 0; k < 16; ++k) {
        h += (k ? "," : "") + std::to_string(acc_hist[k]);
    }
    LOG_INF("TFA prompt %d cont %d windows %lld accepted %lld offered %lld tok/pass %.4f acc/window %.4f hist [%s] prompt %.1f s walk %.1f s%s\n",
            N, M, (long long) n_win, (long long) n_acc, (long long) n_off, n_win ? (double) n_tok / n_win : 0.0,
            n_win ? (double) n_acc / n_win : 0.0, h.c_str(), (t1 - t0) / 1e6, (t2 - t1) / 1e6, S64.empty() ? "" : " (given starts)");
    if (fo) {
        fprintf(fo, "{\"summary\": true, \"prompt\": %d, \"cont\": %d, \"windows\": %lld, \"accepted\": %lld, \"offered\": %lld, \"tok_per_pass\": %.4f, \"given_starts\": %s}\n",
                N, M, (long long) n_win, (long long) n_acc, (long long) n_off, n_win ? (double) n_tok / n_win : 0.0, S64.empty() ? "false" : "true");
        fclose(fo);
    }

    common_speculative_free(spec);
    llama_backend_free();
    return 0;
}
