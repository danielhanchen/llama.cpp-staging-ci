// Teacher-forced check of batch invariant speculative verify (LLAMA_MTP_EXACT=1, llama_set_verify).
//
// A reference context decodes the prompt, then greedy one token at a time (plain decode, as llama-server without
// speculation: no recurrent rollback snapshots, no nextn embeddings) and keeps the logits of every step. A verify context
// (the configuration of a speculative target: n_rs_seq snapshots, nextn embeddings) decodes the same prompt, then the same
// tokens in verify batches of w tokens (every draft accepted), for every width w. Every logits row must be byte-equal to the
// reference row of its position.
//
// usage: llama-verify-exact -m model.gguf [-ngl N] [-c N] (-p PROMPT | -f FILE) [-n N_TOKENS]
// env:   VERIFY_WIDTHS=1,2,3,4   widths of the verify batches (default 1,2,3,4)
//        VERIFY_RS=4             recurrent snapshots of the verify context (0 = plain)
//        VERIFY_NEXTN=1          the verify context outputs the nextn embeddings (as the MTP target does)
//        VERIFY_ONE_CTX=1        one context for the reference and the verify runs (less memory: one MoE cache): the fitted
//                                context of the command line, add --spec-type draft-mtp --spec-draft-n-max N for the
//                                snapshots of a speculative target (VERIFY_RS is then ignored)
//        VERIFY_REF_RS=0         recurrent snapshots of the reference context (two contexts)
//        VERIFY_PROMPT_PAD=N     repeat the prompt until it has at least N tokens (to cross a 256 cell boundary)
//        VERIFY_DUMP=1           on the first mismatching width, compare every graph node of the first verify batch
//                                against the single token decodes and name the first node whose row differs
// exit code: 0 = every row byte-equal, 1 = a row differs, 2 = setup error

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

static std::vector<int> env_list(const char * name, const std::vector<int> & def) {
    const char * v = getenv(name);
    if (!v) {
        return def;
    }
    std::vector<int> out;
    std::string s = v;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) {
            q = s.size();
        }
        out.push_back(atoi(s.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return out;
}

// node dump: the output bytes of every computed node, keyed by name and occurrence
struct node_rec {
    std::string name;
    ggml_op     op;
    int64_t     ne[4];
    std::vector<uint8_t> data;
};

struct dump_state {
    bool enabled = false;
    std::vector<node_rec> nodes;
    std::map<std::string, int> seen;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    if (!st->enabled) {
        return false;
    }
    if (ask) {
        return t->op != GGML_OP_NONE && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE &&
               t->op != GGML_OP_TRANSPOSE;
    }
    if (!ggml_is_contiguous(t) || t->type != GGML_TYPE_F32 || ggml_nbytes(t) > (size_t) 512*1024*1024) {
        return true;
    }
    node_rec r;
    const int occ = st->seen[t->name]++;
    r.name = std::string(t->name) + "#" + std::to_string(occ) + " (" + ggml_op_desc(t) + ")";
    r.op   = t->op;
    for (int i = 0; i < 4; ++i) {
        r.ne[i] = t->ne[i];
    }
    r.data.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, r.data.data(), 0, r.data.size());
    st->nodes.push_back(std::move(r));
    return true;
}

static void batch_clear(llama_batch & b) {
    b.n_tokens = 0;
}

static void batch_add(llama_batch & b, llama_token id, llama_pos pos, bool logits) {
    b.token   [b.n_tokens] = id;
    b.pos     [b.n_tokens] = pos;
    b.n_seq_id[b.n_tokens] = 1;
    b.seq_id  [b.n_tokens][0] = 0;
    b.logits  [b.n_tokens] = logits;
    b.n_tokens++;
}

static bool decode_prompt(llama_context * ctx, const std::vector<llama_token> & prompt, int n_batch) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (size_t i0 = 0; i0 < prompt.size(); i0 += n_batch) {
        batch_clear(batch);
        const size_t n = std::min(prompt.size() - i0, (size_t) n_batch);
        for (size_t i = 0; i < n; ++i) {
            batch_add(batch, prompt[i0 + i], (llama_pos) (i0 + i), i0 + i == prompt.size() - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
    }
    llama_batch_free(batch);
    return true;
}

static llama_token argmax(const float * x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    common_params params;
    params.n_predict = 64;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 2;
    }
    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    const std::vector<int> widths = env_list("VERIFY_WIDTHS", { 1, 2, 3, 4 });
    const int  n_rs      = env_int("VERIFY_RS", 4);
    const bool nextn     = env_int("VERIFY_NEXTN", 0) != 0;
    const bool one_ctx   = env_int("VERIFY_ONE_CTX", 0) != 0;
    const int  pad_to    = env_int("VERIFY_PROMPT_PAD", 0);
    const bool want_dump = env_int("VERIFY_DUMP", 0) != 0;

    dump_state dump;
    params.cb_eval           = dump_cb;
    params.cb_eval_user_data = &dump;
    // the eval callback splits the graph at every node: only when the dump is requested
    if (!want_dump) {
        params.cb_eval           = nullptr;
        params.cb_eval_user_data = nullptr;
    }

    const int ref_rs = env_int("VERIFY_REF_RS", 0);

    // one context: the fitted context of the command line (pass --spec-type draft-mtp --spec-draft-n-max N for the
    // snapshots and the fit of a speculative target); two contexts: unfitted contexts with VERIFY_REF_RS / VERIFY_RS
    common_init_result_ptr init = common_init_from_params(params, /*model_only =*/ !one_ctx);
    llama_model * model = init->model();
    if (!model) {
        fprintf(stderr, "failed to load the model\n");
        return 2;
    }
    llama_context * ctx_ref = one_ctx ? init->context() : nullptr;
    if (!one_ctx) {
        llama_context_params cp = common_context_params_to_llama(params);
        cp.n_rs_seq = ref_rs;
        ctx_ref = llama_init_from_model(model, cp);
    }
    if (!ctx_ref) {
        fprintf(stderr, "failed to create the reference context\n");
        return 2;
    }

    llama_context * ctx_ver = ctx_ref;
    if (!one_ctx) {
        llama_context_params cp = common_context_params_to_llama(params);
        cp.n_rs_seq = n_rs;
        ctx_ver = llama_init_from_model(model, cp);
        if (!ctx_ver) {
            fprintf(stderr, "failed to create the verify context\n");
            return 2;
        }
    }
    if (nextn) {
        llama_set_embeddings_nextn(ctx_ver, true, false);
    }
    printf("reference ctx n_rs_seq %u nextn %d\n", llama_n_rs_seq(ctx_ref), (one_ctx && nextn) ? 1 : 0);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt = common_tokenize(ctx_ref, params.prompt, true, true);
    if (prompt.empty()) {
        fprintf(stderr, "empty prompt\n");
        return 2;
    }
    if (pad_to > 0) {
        const std::vector<llama_token> body = common_tokenize(ctx_ref, params.prompt, false, true);
        while ((int) prompt.size() < pad_to && !body.empty()) {
            prompt.insert(prompt.end(), body.begin(), body.end());
        }
        prompt.resize(std::min<size_t>(prompt.size(), (size_t) pad_to));
    }
    const int n_gen   = params.n_predict > 0 ? params.n_predict : 64;
    const int n_batch = llama_n_batch(ctx_ref);
    const int P       = (int) prompt.size();

    printf("verify-exact: prompt %d tokens, %d steps (positions %d..%d), widths", P, n_gen, P, P + n_gen - 1);
    for (int w : widths) {
        printf(" %d", w);
    }
    printf(", verify ctx n_rs_seq %u nextn %d%s, LLAMA_MTP_EXACT=%d\n", llama_n_rs_seq(ctx_ver), nextn ? 1 : 0,
            one_ctx ? " (same ctx)" : "", llama_exact_verify_enabled() ? 1 : 0);

    // reference: plain greedy decode
    std::vector<llama_token> toks;  // toks[t] is decoded at position P + t
    std::vector<float> ref((size_t) n_gen*n_vocab);
    std::vector<std::vector<node_rec>> ref_nodes; // per step, the first max(widths) steps
    int max_w = 1;
    for (int w : widths) {
        max_w = std::max(max_w, w);
    }
    {
        dump.enabled = false;
        if (!decode_prompt(ctx_ref, prompt, n_batch)) {
            fprintf(stderr, "reference prompt decode failed\n");
            return 2;
        }
        toks.push_back(argmax(llama_get_logits_ith(ctx_ref, -1), n_vocab));
        llama_batch batch = llama_batch_init(1, 0, 1);
        for (int t = 0; t < n_gen; ++t) {
            batch_clear(batch);
            batch_add(batch, toks[t], P + t, true);
            dump.enabled = want_dump && t < max_w;
            dump.nodes.clear();
            dump.seen.clear();
            if (llama_decode(ctx_ref, batch) != 0) {
                fprintf(stderr, "reference decode failed at step %d\n", t);
                return 2;
            }
            if (dump.enabled) {
                ref_nodes.push_back(dump.nodes);
            }
            const float * lg = llama_get_logits_ith(ctx_ref, -1);
            memcpy(ref.data() + (size_t) t*n_vocab, lg, n_vocab*sizeof(float));
            toks.push_back(argmax(lg, n_vocab));
        }
        dump.enabled = false;
        llama_batch_free(batch);
    }

    std::string text;
    for (int t = 0; t < std::min(n_gen, 24); ++t) {
        text += common_token_to_piece(ctx_ref, toks[t]);
    }
    printf("reference text: %s\n", text.c_str());

    bool all_equal = true;
    bool dumped    = false;
    for (int w : widths) {
        if (!decode_prompt(ctx_ver, prompt, n_batch)) {
            fprintf(stderr, "verify prompt decode failed\n");
            return 2;
        }
        const llama_token first = argmax(llama_get_logits_ith(ctx_ver, -1), n_vocab);
        if (first != toks[0]) {
            printf("width %d: the prompt logits of the verify context give another first token (%d vs %d)\n", w, first, toks[0]);
        }
        llama_batch batch = llama_batch_init(std::max(w, 1), 0, 1);
        int n_eq = 0, n_rows = 0, first_bad = -1, n_argmax_bad = 0;
        double max_diff = 0.0;
        for (int t0 = 0; t0 < n_gen; t0 += w) {
            const int n = std::min(w, n_gen - t0);
            batch_clear(batch);
            for (int i = 0; i < n; ++i) {
                batch_add(batch, toks[t0 + i], P + t0 + i, true);
            }
            const bool do_dump = want_dump && !dumped && t0 == 0 && n > 1 && !ref_nodes.empty();
            dump.enabled = do_dump;
            dump.nodes.clear();
            dump.seen.clear();
            llama_set_verify(ctx_ver, n > 1);
            const int rc = llama_decode(ctx_ver, batch);
            llama_set_verify(ctx_ver, false);
            dump.enabled = false;
            if (rc != 0) {
                fprintf(stderr, "verify decode failed at step %d width %d\n", t0, w);
                return 2;
            }
            bool batch_bad = false;
            for (int i = 0; i < n; ++i) {
                const float * lg = llama_get_logits_ith(ctx_ver, i);
                const float * rf = ref.data() + (size_t) (t0 + i)*n_vocab;
                n_rows++;
                if (memcmp(lg, rf, n_vocab*sizeof(float)) == 0) {
                    n_eq++;
                    continue;
                }
                batch_bad = true;
                if (first_bad < 0) {
                    first_bad = t0 + i;
                }
                for (int v = 0; v < n_vocab; ++v) {
                    max_diff = std::max(max_diff, (double) std::fabs(lg[v] - rf[v]));
                }
                n_argmax_bad += argmax(lg, n_vocab) != argmax(rf, n_vocab);
            }
            if (do_dump && batch_bad) {
                dumped = true;
                // match the nodes by name: a verify node with the token dimension n and the single token node with 1
                std::map<std::string, const node_rec *> by_name[16];
                for (int i = 0; i < n && i < (int) ref_nodes.size(); ++i) {
                    for (const auto & r : ref_nodes[i]) {
                        by_name[i][r.name] = &r;
                    }
                }
                int n_reported = 0;
                for (const auto & vr : dump.nodes) {
                    const node_rec * r0 = by_name[0].count(vr.name) ? by_name[0][vr.name] : nullptr;
                    if (!r0) {
                        continue;
                    }
                    int dim = -1;
                    for (int d = 0; d < 4; ++d) {
                        bool ok = vr.ne[d] == n && r0->ne[d] == 1;
                        for (int e = 0; e < 4 && ok; ++e) {
                            if (e != d && vr.ne[e] != r0->ne[e]) {
                                ok = false;
                            }
                        }
                        if (ok) {
                            dim = d;
                            break;
                        }
                    }
                    if (dim < 0) {
                        continue;
                    }
                    // the slice of row i along dim: blocks of (product of lower dims) elements
                    int64_t inner = 1;
                    for (int d = 0; d < dim; ++d) {
                        inner *= vr.ne[d];
                    }
                    int64_t outer = 1;
                    for (int d = dim + 1; d < 4; ++d) {
                        outer *= vr.ne[d];
                    }
                    std::string bad_rows;
                    for (int i = 0; i < n && i < (int) ref_nodes.size(); ++i) {
                        const node_rec * ri = by_name[i].count(vr.name) ? by_name[i][vr.name] : nullptr;
                        if (!ri) {
                            continue;
                        }
                        bool eq = true;
                        for (int64_t o = 0; o < outer && eq; ++o) {
                            const uint8_t * a = vr.data.data() + (size_t) ((o*n + i)*inner)*4;
                            const uint8_t * b = ri->data.data() + (size_t) (o*inner)*4;
                            eq = memcmp(a, b, inner*4) == 0;
                        }
                        if (!eq) {
                            bad_rows += " " + std::to_string(i);
                        }
                    }
                    if (!bad_rows.empty()) {
                        const float * a = (const float *) vr.data.data();
                        const float * b = (const float *) by_name[0][vr.name]->data.data();
                        printf("  node %s [%lld %lld %lld %lld] differs in rows%s (row 0: %g %g vs %g %g)\n", vr.name.c_str(),
                                (long long) vr.ne[0], (long long) vr.ne[1], (long long) vr.ne[2], (long long) vr.ne[3], bad_rows.c_str(),
                                a[0], a[1], b[0], b[1]);
                        if (++n_reported >= 12) {
                            break;
                        }
                    }
                }
                if (n_reported == 0) {
                    printf("  node dump: no node with a token dimension differs (%zu nodes captured)\n", dump.nodes.size());
                }
            }
        }
        llama_batch_free(batch);
        const bool eq = n_eq == n_rows;
        all_equal &= eq;
        printf("width %d: %d/%d rows byte-equal%s", w, n_eq, n_rows, eq ? "" : " MISMATCH");
        if (!eq) {
            printf(" (first at step %d = position %d, max |diff| %.3g, argmax differs in %d rows)", first_bad, P + first_bad,
                    max_diff, n_argmax_bad);
        }
        printf("\n");
    }

    printf("verify-exact: %s\n", all_equal ? "PASS (every row byte-equal to the single token decode)" : "FAIL");

    if (ctx_ver != ctx_ref) {
        llama_free(ctx_ver);
    }
    if (!one_ctx) {
        llama_free(ctx_ref);
    }
    return all_equal ? 0 : 1;
}
