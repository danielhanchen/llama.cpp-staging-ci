// Row-count invariance of a speculative verify batch (k19-fn-tpp): after the same prompt, the first verify batch (llama_set_verify)
// of width A and of width B > A over the same tokens. Rows 0 .. A-1 of both must be byte-equal (logits, and with VERIFY_DUMP=1
// every graph node, sliced along its token axis); the first node whose rows differ names the row-count dependent kernel.
//
// usage: llama-verify-width -m model.gguf [-ngl N] [-c N] [-fa on] [-ub N] ...
// env:   VW_LENS=2048,8192,16384,30720   prompt lengths (random tokens, seed VW_SEED)
//        VW_A=3 VW_B=4                   the two widths
//        VERIFY_DUMP=1                   node dump (the eval callback splits the graph at every node, so fusions differ from a
//                                        plain run: the logits compare runs without it first)
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
#include <random>
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

struct node_rec {
    std::string name;
    std::string desc;
    ggml_type   type;
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
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32 && t->type != GGML_TYPE_F16) ||
            ggml_nbytes(t) > (size_t) 512*1024*1024) {
        return true;
    }
    node_rec r;
    const int occ = st->seen[t->name]++;
    r.name = std::string(t->name) + "#" + std::to_string(occ);
    r.desc = ggml_op_desc(t);
    r.type = t->type;
    for (int i = 0; i < 4; ++i) {
        r.ne[i] = t->ne[i];
    }
    r.data.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, r.data.data(), 0, r.data.size());
    st->nodes.push_back(std::move(r));
    return true;
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
        batch.n_tokens = 0;
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

// prompt, then one verify batch of w tokens; logits of all w rows
static bool run_width(llama_context * ctx, const std::vector<llama_token> & prompt, const std::vector<llama_token> & drafts, int w,
                      int n_batch, int n_vocab, dump_state * dump, std::vector<float> & logits) {
    if (!decode_prompt(ctx, prompt, n_batch)) {
        return false;
    }
    llama_batch batch = llama_batch_init(w, 0, 1);
    for (int i = 0; i < w; ++i) {
        batch_add(batch, drafts[i], (llama_pos) (prompt.size() + i), true);
    }
    if (dump) {
        dump->nodes.clear();
        dump->seen.clear();
        dump->enabled = true;
    }
    llama_set_verify(ctx, true);
    const int rc = llama_decode(ctx, batch);
    llama_set_verify(ctx, false);
    if (dump) {
        dump->enabled = false;
    }
    llama_batch_free(batch);
    if (rc != 0) {
        return false;
    }
    logits.resize((size_t) w * n_vocab);
    for (int i = 0; i < w; ++i) {
        memcpy(logits.data() + (size_t) i * n_vocab, llama_get_logits_ith(ctx, i), n_vocab * sizeof(float));
    }
    return true;
}

// rows 0 .. a-1 of node x (width a) against node y (width b): the token axis is the one dim that is a in x and b in y
// returns -1 = not comparable, 0 = equal, else the number of differing elements
static int64_t cmp_rows(const node_rec & x, const node_rec & y, int a, int b, int & axis) {
    axis = -1;
    for (int d = 0; d < 4; ++d) {
        if (x.ne[d] == y.ne[d]) {
            continue;
        }
        if (x.ne[d] == a && y.ne[d] == b && axis < 0) {
            axis = d;
            continue;
        }
        return -1;
    }
    if (x.type != y.type) {
        return -1;
    }
    const size_t es = ggml_type_size(x.type);
    int64_t nd = 0;
    int64_t sx[4], sy[4];
    sx[0] = sy[0] = 1;
    for (int d = 1; d < 4; ++d) {
        sx[d] = sx[d - 1]*x.ne[d - 1];
        sy[d] = sy[d - 1]*y.ne[d - 1];
    }
    for (int64_t i3 = 0; i3 < x.ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < x.ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < x.ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < x.ne[0]; ++i0) {
        const int64_t ix = i0*sx[0] + i1*sx[1] + i2*sx[2] + i3*sx[3];
        const int64_t iy = i0*sy[0] + i1*sy[1] + i2*sy[2] + i3*sy[3];
        if (memcmp(x.data.data() + ix*es, y.data.data() + iy*es, es) != 0) {
            nd++;
        }
    }
    return nd;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 2;
    }
    common_init();
    llama_backend_init();

    const std::vector<int> lens = env_list("VW_LENS", { 2048, 8192, 16384, 30720 });
    const int  A         = env_int("VW_A", 3);
    const int  B         = env_int("VW_B", 4);
    const bool want_dump = env_int("VERIFY_DUMP", 0) != 0;
    const int  seed      = env_int("VW_SEED", 1234);

    dump_state dump;
    if (want_dump) {
        params.cb_eval           = dump_cb;
        params.cb_eval_user_data = &dump;
    }
    common_init_result_ptr init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to load the model / context\n");
        return 2;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_batch = (int) llama_n_batch(ctx);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(1000, std::min(n_vocab - 1, 150000));
    int bad = 0;
    for (int len : lens) {
        std::vector<llama_token> prompt(len), drafts(B);
        for (auto & t : prompt) {
            t = tok(rng);
        }
        for (auto & t : drafts) {
            t = tok(rng);
        }
        std::vector<float> la, la2, lb;
        if (!run_width(ctx, prompt, drafts, A, n_batch, n_vocab, nullptr, la) ||
            !run_width(ctx, prompt, drafts, A, n_batch, n_vocab, nullptr, la2) ||
            !run_width(ctx, prompt, drafts, B, n_batch, n_vocab, nullptr, lb)) {
            fprintf(stderr, "decode failed at len %d\n", len);
            return 2;
        }
        const bool aa = memcmp(la.data(), la2.data(), la.size()*sizeof(float)) == 0;
        int rows_eq = 0;
        float maxd = 0.0f;
        for (int i = 0; i < A; ++i) {
            const float * x = la.data() + (size_t) i*n_vocab;
            const float * y = lb.data() + (size_t) i*n_vocab;
            rows_eq += memcmp(x, y, n_vocab*sizeof(float)) == 0;
            for (int j = 0; j < n_vocab; ++j) {
                maxd = std::max(maxd, std::fabs(x[j] - y[j]));
            }
        }
        printf("VW len %d: width %d vs %d: %d / %d rows byte-equal, max |dlogit| %.3g, A/A %s\n", len, A, B, rows_eq, A, maxd,
               aa ? "equal" : "DIFFERS");
        const bool force_dump = env_int("VW_FORCE_DUMP", 0) != 0;
        if (rows_eq != A) {
            bad = 1;
        }
        if ((rows_eq == A && !force_dump) || !want_dump) {
            continue;
        }
        std::vector<float> l;
        run_width(ctx, prompt, drafts, A, n_batch, n_vocab, &dump, l);
        std::vector<node_rec> na = std::move(dump.nodes);
        run_width(ctx, prompt, drafts, B, n_batch, n_vocab, &dump, l);
        std::vector<node_rec> nb = std::move(dump.nodes);
        std::map<std::string, const node_rec *> mb;
        for (const auto & r : nb) {
            mb[r.name] = &r;
        }
        int n_cmp = 0, n_skip = 0, n_diff = 0, n_tok_diff = 0, idx = -1;
        const int max_print = env_int("VW_MAX_PRINT", 60);
        for (const auto & r : na) {
            idx++;
            auto it = mb.find(r.name);
            if (it == mb.end()) {
                n_skip++;
                continue;
            }
            int axis = -1;
            const int64_t nd = cmp_rows(r, *it->second, A, B, axis);
            if (nd < 0) {
                n_skip++;
                continue;
            }
            n_cmp++;
            if (nd > 0) {
                // whole-state copies (recurrent / KV caches, no token axis) differ by construction: the state after A vs B tokens
                const bool cache = axis < 0 && r.name.rfind("cache_", 0) == 0;
                if (!cache && n_tok_diff++ < max_print) {
                    printf("VW len %d: [%d] node %s (%s) [%lld %lld %lld %lld] vs [%lld %lld %lld %lld] token axis %d: %lld elements differ\n",
                           len, idx, r.name.c_str(), r.desc.c_str(), (long long) r.ne[0], (long long) r.ne[1], (long long) r.ne[2],
                           (long long) r.ne[3], (long long) it->second->ne[0], (long long) it->second->ne[1],
                           (long long) it->second->ne[2], (long long) it->second->ne[3], axis, (long long) nd);
                }
                n_diff++;
            }
        }
        printf("VW len %d: dump: %d nodes compared, %d differ (%d outside the cache copies), %d not comparable (%zu / %zu recorded)\n",
               len, n_cmp, n_diff, n_tok_diff, n_skip, na.size(), nb.size());
    }
    printf("VW %s\n", bad ? "ROWS DIFFER" : "ALL ROWS EQUAL");
    return bad;
}
